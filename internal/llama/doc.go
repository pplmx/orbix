// Package llama implements the Llama-family transformer: the architecture a GGUF file with
// general.architecture = "llama" describes, plus the variants that share its layout.
//
// The package owns the mapping from GGUF metadata and tensor names onto a forward pass. It
// sits above the kernels (which know nothing about models) and below the engine (which
// knows nothing about tensors): the shapes and the order of operations live here, and
// nothing else.
//
// The forward pass is written as one readable function rather than fused kernels. For a
// 15M-parameter model that is fast enough to be useful; for a 7B model it is the honest
// reference the optimized paths are checked against.
//
// # Decisions worth knowing about
//
// Architecture differences are a table, not a type hierarchy. archSpecs maps an
// architecture name to the handful of properties that actually differ -- the RoPE pairing
// convention, whether attention carries a bias, the sliding-window length. A Go type per
// family would duplicate the forward pass per family; the table keeps one forward pass and
// makes the differences enumerable in eleven lines. The cost is that a genuine structural
// difference has no home here, which is the point at which a second implementation is
// cheaper than more flags.
//
// Two RoPE layouts, because the models disagree on how dimensions pair. RopeNormal
// pairs dimension i with i+1, as Llama does; RopeNeox pairs i with i+dims/2, as the
// Qwen family does. Getting this wrong rotates the wrong pairs, which is silent: nothing
// errors, the run completes, and every id it produces is a valid id. What comes out is
// not plausible text, though -- measured on Qwen2.5 with the layout swapped, "The capital
// of France is" continued as "1000000000000000000" where the right layout gives "Paris. It
// is the largest city in Europe...". A degenerate repetition rather than noise is what
// makes it worth stating: the output looks like a model that went wrong, not like a
// program that did, so there is no crash to search for. That is why the layout is a
// property of the architecture spec rather than a default.
//
// The KV cache is an interface with two implementations. A contiguous cache is one
// arena per sequence; a paged cache is blocks shared from a pool, which is what makes
// prefix reuse possible and what bounds memory when sequences have different lengths.
// They must produce identical numbers -- the tests assert exact equality, not a tolerance
// -- because a cache is storage, and storage that changes the result is a bug.
//
// Mixture of experts is a property of the feed-forward layer, not a separate model.
// A file carrying expert_count replaces its single gated FFN with one per expert plus a
// router, and runs expert_used_count of them per token. Both the prefill and decode paths
// group rows by chosen expert so one matmul serves every row that picked the same one.
//
// The prefix index records which block continues which. Two requests that share a
// prompt prefix should not recompute it, and a prompt prefix is a chain: block k+1 is only
// reachable by walking from block k. Recording the link is what lets eviction give up the
// end of a chain instead of its middle, which would strand the rest while still holding
// memory for it.
package llama
