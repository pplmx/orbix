// Package llama implements the Llama-family transformer: the architecture a
// GGUF file with general.architecture = "llama" describes.
//
// The package owns the mapping from GGUF metadata and tensor names onto a
// forward pass. It sits above the kernels (which know nothing about models) and
// below the engine (which knows nothing about tensors): the shapes and the
// order of operations live here, and nothing else.
//
// The forward pass is written as one readable function rather than fused kernels.
// For a 15M-parameter model that is fast enough to be useful; for a 7B model it
// is the honest reference the optimized paths will be checked against.
package llama
