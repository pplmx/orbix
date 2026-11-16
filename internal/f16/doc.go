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
package f16
