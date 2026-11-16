// Package quant decodes ggml's block-quantized tensor formats into float32.
//
// GGUF stores quantized tensors as sequences of fixed-size blocks. Each block
// packs a small number of scale factors together with the quantized weights,
// so the decoded values depend on data from neighbouring bytes. This package
// turns those blocks into plain float32 slices, which is the form the compute
// kernels work in.
//
// Decoding is deliberately separate from the kernels: it is the part that
// mirrors the file format and can be checked against ggml byte for byte,
// while the kernels only see ordinary floats.
//
// Layouts implemented here are documented in ggml's ggml-quants.c, which is
// the reference this package is written against.
package quant
