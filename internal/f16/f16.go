package f16

import "math"

// MAX is the largest finite value representable in binary16 (65504).
const MAX = 65504.0

// Smallest positive subnormal in binary16, 2^-24.
const smallestSubnormal = 5.9604644775390625e-08

// ToFloat32 converts a binary16 bit pattern to the nearest float32.
//
// Every binary16 value is exactly representable in binary32, so this
// direction is lossless and has no rounding cases: the work is unpacking the
// 1/5/10-bit fields and handling infinities, NaNs, and subnormals.
func ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32((h >> 10) & 0x1f)
	mant := uint32(h & 0x3ff)

	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign) // +/- zero
		}
		// Subnormal: value = mant * 2^-24. Normalizing here rather than
		// scaling by a constant keeps the result exact, because mant has at
		// most 10 significant bits and 2^-24 is a power of two.
		f := float32(mant) * float32(smallestSubnormal)
		if sign != 0 {
			return -f
		}
		return f
	case 0x1f:
		if mant == 0 {
			return math.Float32frombits(sign | 0x7f800000) // +/- Inf
		}
		// NaN. Preserve the payload so a NaN stays a NaN, and set the quiet
		// bit so arithmetic on it does not signal.
		return math.Float32frombits(sign | 0x7fc00000 | mant<<13)
	default:
		// Normal: rebias the exponent from 15 to 127 and left-align the
		// mantissa into 23 bits.
		bits := sign | (exp+112)<<23 | mant<<13
		return math.Float32frombits(bits)
	}
}

// FromFloat32 converts a float32 to binary16, rounding to nearest with ties
// to even. Values too large for binary16 become infinity, and values below
// the subnormal range round to zero.
func FromFloat32(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16(bits>>16) & 0x8000
	exp := int32(bits>>23) & 0xff
	mant := bits & 0x7fffff

	switch exp {
	case 0xff:
		if mant == 0 {
			return sign | 0x7c00 // Inf
		}
		// NaN: keep it a NaN. A payload of zero would turn into infinity, so
		// force at least one mantissa bit.
		if mant&0x7fe000 == 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00 | uint16(mant>>13) | 0x0200
	case 0:
		// float32 zero or subnormal. binary16's subnormal range starts at
		// 2^-24, far below float32's smallest subnormal (2^-149), so these
		// values underflow to zero.
		return sign
	}

	// Rebias: float32 bias 127, binary16 bias 15.
	e := exp - 127 + 15
	if e >= 0x1f {
		return sign | 0x7c00 // overflow to infinity
	}
	if e <= 0 {
		// Result is subnormal or zero. Shift the implicit leading 1 into the
		// mantissa and shift right by the amount that would have been the
		// exponent deficit, rounding as we go.
		if e < -10 {
			return sign // underflow to zero
		}
		mant |= 0x800000
		shift := uint32(14 - e)
		half := uint32(1) << (shift - 1)
		round := mant & half
		rem := mant & (half - 1)
		mant >>= shift
		// Round to nearest, ties to even.
		if round != 0 && (rem != 0 || mant&1 != 0) {
			mant++
		}
		return sign | uint16(mant)
	}

	// Normal case: drop 13 mantissa bits with round-to-nearest-even.
	mantRound := mant & 0x1fff
	mant >>= 13
	if mantRound > 0x1000 || (mantRound == 0x1000 && mant&1 != 0) {
		mant++
		if mant == 0x400 {
			// Rounding carried into the exponent.
			mant = 0
			e++
			if e >= 0x1f {
				return sign | 0x7c00
			}
		}
	}
	return sign | uint16(e)<<10 | uint16(mant)
}

// IsNaN reports whether the binary16 pattern is a NaN.
func IsNaN(h uint16) bool {
	return h&0x7c00 == 0x7c00 && h&0x03ff != 0
}

// IsInf reports whether the binary16 pattern is an infinity.
func IsInf(h uint16) bool {
	return h&0x7fff == 0x7c00
}
