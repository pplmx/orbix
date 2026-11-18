// Package f16 converts between IEEE 754 binary16 (half precision) and float32.
//
// GGUF files store F16 tensors and every block quant's scale factor as
// binary16. Go has no native half-precision type, so converting explicitly is
// the boundary between the file format and the float32 world the kernels
// compute in.
//
// The conversion is written out rather than reached for via float64 math
// because the rounding and the subnormal/overflow edges are the whole point:
// the code below is meant to be read alongside IEEE 754-2008 §3.6.
//
// The domain is small enough to check exhaustively, and it is checked that way
// rather than by sampling. ToFloat32 is compared against a second implementation
// written from the standard -- sign, biased exponent, mantissa, with math.Ldexp
// doing the scaling -- for every one of the 65536 half-precision patterns,
// including the sign of zero and the quiet NaNs. FromFloat32 is checked by
// round-tripping that same set: binary16 is a subset of binary32, so every finite
// value has to come back unchanged. The two directions are not independent of
// each other -- a round trip cannot catch an error both functions share -- which
// is why ToFloat32 is compared against something other than FromFloat32.
//
// What that pair does not cover is a float32 that is not representable in
// binary16: rounding, overflow to infinity, and underflow to a subnormal or zero
// are pinned by their own tests over chosen values rather than exhaustively,
// since the float32 domain is not small enough to enumerate. gguf stores only
// F16 tensors and block scale factors, so the values this package has to produce
// come from the narrow side.
package f16
