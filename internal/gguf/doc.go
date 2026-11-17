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
// Only the version this package implements is accepted. A newer GGUF revision may
// mean a different tensor encoding, and parsing one as if it were version 2 would produce
// a plausible-looking file full of wrong numbers. Rejecting it names the version instead.
//
// Tensors are described by offset, not loaded. TensorData returns a reader for one
// tensor's bytes, so the decoder and the compute layer decide how much of the file is in
// memory. This is why loading a 47B-parameter Mixtral file does not need 47B parameters
// worth of RAM: the quantized bytes are what stay resident.
//
// The file is read with os.Open and ReadAt, not mmap. A memory map would be faster
// to start and would let the kernel page weights in on demand, but it fails differently:
// a truncated or replaced file becomes a segmentation fault inside a matmul rather than
// an error at load. An error a caller can report is worth more here than the start-up
// time, and the whole file is read once anyway.
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
package gguf
