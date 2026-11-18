// Package quant decodes ggml's block-quantized tensor formats into float32.
//
// GGUF stores quantized tensors as sequences of fixed-size blocks. Each block packs a
// small number of scale factors together with the quantized weights, so a decoded value
// depends on data from neighbouring bytes: a nibble in one byte is meaningless without
// the scale stored elsewhere in the block. This package turns those blocks into plain
// float32 slices, which is the form the compute kernels work in.
//
// Layouts implemented here are documented in ggml's ggml-quants.c, which is the reference
// this package is written against.
//
// # Decisions worth knowing about
//
// Decoding is separate from the kernels. It is the part that mirrors the file format
// and can be checked against ggml byte for byte, while the kernels only ever see ordinary
// floats. Keeping them apart is what lets the numerics be tested against a reference
// implementation rather than against output text, which was how several decoder bugs here
// were found.
//
// One format at a time, behind a table. Each format is a decoder function in a map
// keyed by gguf.QuantType. A decode with an unknown type reports the type rather than
// guessing, because the alternative -- treating an unreadable block as some other format
// -- produces weights that load, run, and generate nonsense.
//
// The K-quants are the hard ones, and they are hard for a specific reason. Q2_K
// through Q6_K pack a 6-bit scale and a 6-bit minimum into 12 bytes, split across two
// nibble halves and four bit-planes: the scale for sub-block sb lives in a byte whose
// index and shift both depend on sb. Reading them as an array of 6-bit values, which is
// the obvious approach, produces values of a plausible magnitude in the right range and
// still generates text -- it is simply wrong for a large fraction of sub-blocks, and only
// a comparison against a known-good decode shows it. The tests here check exact values
// from a known block rather than a round trip, because a round trip through an encoder
// written alongside the decoder agrees with whatever misreading both share.
//
// The last sentence of that paragraph is a claim about method, and it has been carried
// out rather than assumed. Six checkpoints were parsed twice -- once here and once by
// the independent Python gguf package -- and every tensor's bytes were compared by
// SHA-256: stories15M (Q4_0, Q8_0, F32), tinyllama-q2k (Q2_K, Q3_K, Q6_K, F32),
// qwen2-0.5b (Q8_0, F32), a Mixtral-format MoE model whose expert weights are 3-D
// tensors (Q2_K, Q3_K, Q5_K, Q6_K, Q8_0, F32), llama3.2 (Q2_K, Q3_K, Q4_K, Q6_K, F32),
// and a 1.9 GB all-F32 checkpoint. That is 1009 tensors across eight formats, all
// byte-identical. The comparison is not a test in this package: it needs the
// third-party reader and a multi-gigabyte checkpoint, and a unit test may depend on
// neither. It is a check a person runs, which is why the result is written down here --
// otherwise the next reader has only the claim.
//
// The per-format tests in this package are a different kind of evidence, and worth
// distinguishing. TestKQuantRoundTrip pins exact values from a known block. The
// MODEL_KQUANT-gated test decodes a real tensor and checks that its spread is a weight
// matrix's rather than a garbage decode's -- deliberately wide bounds, so it catches a
// scale field read from the wrong offset and not a bit-plane off by one. The SHA-256
// comparison above is what covers the finer errors, which is why both exist.
//
// The one format this did not reach is F16: none of the available checkpoints is stored
// in it. Its conversion has unit tests over the boundaries and subnormals, but no
// external cross-check, and saying so is more useful than implying the coverage is total.
//
// IQ4_NL uses a lookup table, not arithmetic. Its 4-bit values are a non-linear
// codebook of sixteen signed values, so decoding is a table lookup. Deriving the values
// from a formula would be a reimplementation of the table, and the table is the format.
//
// Decode writes into a caller-provided slice, and validates its length first. The
// number of elements comes from len(dst) and the required byte count is checked against
// len(src) before any decoding, so a mismatch is an error rather than a slice bounds
// panic. Keeping the destination caller-owned is what keeps the allocation policy with
// the caller: a model's largest tensor is hundreds of megabytes, and a decoder that
// returned a fresh slice would force a copy at exactly the wrong moment.
package quant
