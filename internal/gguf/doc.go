// Package gguf reads GGUF model files.
//
// GGUF is the container format used across the ggml ecosystem -- llama.cpp, Ollama,
// Hugging Face's GGUF releases. A file is a header, a metadata key/value section, a
// tensor directory, and then the raw tensor data. This package parses the first three and
// exposes the fourth through TensorData, so a caller that only needs the metadata never
// reads the weights and a caller that loads weights streams them rather than holding a
// second copy of the file.
//
// The implementation follows gguf-spec.md from the ggml repository: every integer is
// little-endian, strings are a uint64 length followed by that many bytes of UTF-8.
//
// # Decisions worth knowing about
//
// Versions 2 and 3 are accepted, and nothing else is. A newer GGUF revision may mean a
// different tensor encoding, and parsing one as if it were version 3 would produce a
// plausible-looking file full of wrong numbers; rejecting it names the version instead.
//
// Version 2 is in the range because much of the published corpus is still written in it and
// version 3 changed nothing it could break: the only edit was dropping big-endian support, so a
// little-endian version 2 file has the same header, metadata and directory. A big-endian version
// 2 file reads little-endian as version 0x02000000 and falls outside the range, which is the
// right answer for a revision this package cannot parse. Version 1 used 32-bit counts for both
// directories and stays refused for that reason.
//
// Tensors are described by offset, not loaded. TensorData returns a reader for one
// tensor's bytes, so the decoder and the compute layer decide how much of the file is in
// memory. This is why loading a 47B-parameter Mixtral file does not need 47B parameters
// worth of RAM: the quantized bytes are what stay resident.
//
// The file is read with os.Open and ReadAt, not mmap. A memory map would be faster
// to start and would let the kernel page weights in on demand, but it fails differently:
// a truncated or replaced file becomes a fatal fault inside a matmul rather than an error
// at load. On Linux that fault is SIGBUS, not SIGSEGV -- the address is valid and the
// backing file is gone -- which is worth naming because the two are easy to conflate and
// only the second is what a reader would guess from "memory map went wrong". A Go program
// sees both as "fatal error: fault", so nothing in this package could catch either.
// An error a caller can report is worth more here than the start-up time, and the whole
// file is read once anyway.
//
// Metadata is parsed into a map, not a schema. Keys differ between architectures and
// between converter versions; a struct with one field per known key would reject files it
// could otherwise read. The typed accessors (Get, GetArray, and the arch helpers) are
// where a key's meaning is asserted, so an unknown key is inert rather than fatal.
//
// Alignment is read from the file. general.alignment defaults to 32, and the tensor
// data section starts at the next multiple after the directory. Treating it as a constant
// would misread any file that chose a different value, and the failure would look like
// corrupt weights rather than a header mistake.
//
// # How the parse was checked
//
// Every decision above rests on one assumption: the tensors this package hands out are
// the tensors the file contains, at the offsets the file states. That was verified against
// an implementation written by someone else rather than against this package's own idea of
// the format. Six checkpoints were parsed twice -- once here, once by the independent
// Python gguf package -- and each tensor's bytes compared by SHA-256: a 15M Llama
// (Q4_0/Q8_0/F32), a TinyLlama in K-quants, Qwen2 (Q8_0/F32), a Mixtral-format MoE file
// whose expert weights are 3-D tensors, Llama 3.2 (Q2_K through Q6_K), and a 1.9 GB
// all-F32 checkpoint. 1009 tensors across eight formats, every one byte-identical, as were
// the shapes, the quantisation types and the byte counts.
//
// The comparison is not a test in this package. It needs the third-party reader and
// multi-gigabyte checkpoints, and a unit test may depend on neither, so it is a check a
// person runs -- which is why the result is written down here rather than left as the
// method's possibility. The tests that do run here cover the error paths instead: wrong
// magic, unsupported version, every truncation up to and past the end of the directory, a
// tensor whose offset would wrap when converted, and an out-of-range tensor request.
//
// The one format the comparison did not reach is F16, because none of the available
// checkpoints stores weights that way. Its conversion is covered by unit tests over the
// boundaries and subnormals, but not by an external reader.
//
// # The file is untrusted input
//
// A GGUF file is usually downloaded, so its length fields cannot be believed. Every count
// that a naive parser would turn into an allocation is bounded here: one string or byte run
// at 64 MiB, one metadata array at 64 Mi elements, array elements at 64 Mi too. Without
// those bounds a sixteen-byte header claims a gibibyte and the parser allocates it before
// discovering there is no such data -- measured at 1024 MiB for one string claim and 6144
// MiB for an array claim, both from a file under a kilobyte.
//
// The same reasoning covers arithmetic rather than sizes. A tensor offset above MaxInt64
// wraps negative when cast to int64, and `start + n > size` then passes for a range that
// begins before the file; both are checked in an order that cannot overflow. Strings must
// be valid UTF-8, because the alternative is a replacement character appearing silently in
// metadata the caller reads.
//
// None of this is reachable from a well-formed file. The point is that the parser is the
// boundary between a network download and the rest of the program, and a failure here is
// supposed to be an error naming the field rather than a panic or an allocation.
package gguf
