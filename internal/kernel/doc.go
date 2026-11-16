// Package kernel holds the numerical operations a transformer forward pass is
// built from.
//
// Everything here is plain Go operating on float32 slices. The functions are
// written to be read: each one states the shape it expects, what it computes,
// and why, in the order a reader would work it out on paper. That is the
// point of this project — a matmul you can follow line by line.
//
// Shapes follow ggml's convention, because that is how the tensors arrive from
// a GGUF file: the first dimension is the innermost (contiguous) one. So a
// weight matrix stored as [in, out] is a slice of out rows, each holding in
// contiguous weights.
package kernel
