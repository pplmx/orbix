// Package gguf reads GGUF model files.
//
// GGUF is the container format used across the GGUF ecosystem (llama.cpp,
// Ollama, Hugging Face GGUF releases). A file is a header, a metadata
// key/value section, a tensor directory, and then the raw tensor data. This
// package parses the first three and exposes the fourth through TensorData,
// so callers can stream weights without loading the whole file.
//
// The implementation follows gguf-spec.md from the ggml repository:
// every integer is little-endian, strings are a uint64 length followed by
// that many bytes of UTF-8.
package gguf
