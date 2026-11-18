// Package kernel holds the numerical operations a transformer forward pass is built from.
//
// Everything here is plain Go operating on float32 slices. The functions are written to
// be read: each one states the shape it expects, what it computes, and why, in the order a
// reader would work it out on paper. That is the point of this project -- a matmul you can
// follow line by line.
//
// Shapes follow ggml's convention, because that is how the tensors arrive from a GGUF
// file: the first dimension is the innermost (contiguous) one. So a weight matrix stored
// as [in, out] is a slice of out rows, each holding in contiguous weights.
//
// # Decisions worth knowing about
//
// Parallelism is over independent outputs, never over a reduction. Every output element is
// computed start to finish by one goroutine, in the same order the serial version uses. The
// split decides which goroutine computes which element -- rows of the output for MatVec,
// columns for MatMulTransposed, where a decode step is one row against the whole vocabulary
// and a row split would leave the machine idle -- and never splits the dot product itself.
//
// A split reduction would be faster on paper -- more cores on the same sum -- and would
// silently change every number, because float32 addition is not associative. The rule
// that follows is the one this project leans on everywhere: a batched or parallel run
// must equal the serial reference exactly, not approximately, which is what makes a
// regression visible as inequality rather than as text that got slightly worse.
//
// There is no assembly, and no SIMD. The speed comes from using more than one core
// and from the work being independent per output element. A kernel that cannot be read
// in the repository's own language would defeat the reason the repository exists.
//
// **A kernel that takes more than one slice checks that their shapes agree, and returns
// an error when they do not.** A mismatched length in a matmul is a bug in the layer
// above, and the alternative to checking is a slice bounds panic three frames deep in a
// decode loop with no indication of which tensor was wrong. The in-place elementwise
// kernels -- Softmax, SiLU -- take one slice and have no shape to disagree with, so they
// return nothing and a caller cannot get them wrong.
//
// The reference is the readable implementation, not a faster one with a fallback.
// There is no "fast path" to drift from a slow path: one implementation, used by
// everything from a single-token forward pass to a batched prefill.
package kernel
