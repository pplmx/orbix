package f16

import (
	"math"
	"testing"
)

func TestToFloat32Specials(t *testing.T) {
	tests := []struct {
		h    uint16
		want float32
	}{
		{0x0000, 0},                             // +0
		{0x8000, float32(math.Copysign(0, -1))}, // -0
		{0x3c00, 1},                             // 1.0
		{0xbc00, -1},                            // -1.0
		{0x4000, 2},                             // 2.0
		{0x7bff, MAX},                           // largest finite
		{0xfbff, -MAX},                          // largest finite, negative
		{0x0001, smallestSubnormal},             // smallest positive subnormal
		{0x8001, -smallestSubnormal},            // smallest negative subnormal
		{0x03ff, 6.097555160522461e-05},         // largest subnormal
		{0x0400, 6.103515625e-05},               // smallest positive normal
		{0x3555, 0.333251953125},                // a value that is not exact in decimal
	}
	for _, tc := range tests {
		got := ToFloat32(tc.h)
		if got != tc.want && !(got == 0 && tc.want == 0 && math.Signbit(float64(got)) == math.Signbit(float64(tc.want))) {
			t.Errorf("ToFloat32(%#04x) = %v, want %v", tc.h, got, tc.want)
		}
	}

	if got := ToFloat32(0x7c00); !math.IsInf(float64(got), 1) {
		t.Errorf("ToFloat32(+Inf) = %v", got)
	}
	if got := ToFloat32(0xfc00); !math.IsInf(float64(got), -1) {
		t.Errorf("ToFloat32(-Inf) = %v", got)
	}
	if got := ToFloat32(0x7e00); !math.IsNaN(float64(got)) {
		t.Errorf("ToFloat32(NaN) = %v", got)
	}
	// A NaN payload must survive the widening.
	if got := ToFloat32(0x7c01); !math.IsNaN(float64(got)) {
		t.Errorf("ToFloat32(NaN with payload) = %v, want NaN", got)
	}
}

func TestToFloat32NegativeZero(t *testing.T) {
	got := ToFloat32(0x8000)
	if got != 0 {
		t.Errorf("ToFloat32(-0) = %v, want 0", got)
	}
	if !math.Signbit(float64(got)) {
		t.Error("ToFloat32(-0) lost the sign bit")
	}
}

func TestFromFloat32RoundTrip(t *testing.T) {
	// Every finite binary16 value must survive a round trip exactly, because
	// binary16 is a strict subset of binary32.
	for h := 0; h <= 0xffff; h++ {
		v := uint16(h)
		f := ToFloat32(v)
		if IsNaN(v) {
			continue
		}
		got := FromFloat32(f)
		if got != v {
			t.Fatalf("round trip %#04x -> %v -> %#04x", v, f, got)
		}
	}
}

func TestFromFloat32Specials(t *testing.T) {
	if got := FromFloat32(0); got != 0x0000 {
		t.Errorf("FromFloat32(0) = %#04x", got)
	}
	if got := FromFloat32(float32(math.Copysign(0, -1))); got != 0x8000 {
		t.Errorf("FromFloat32(-0) = %#04x, want 0x8000", got)
	}
	if got := FromFloat32(1); got != 0x3c00 {
		t.Errorf("FromFloat32(1) = %#04x, want 0x3c00", got)
	}
	if got := FromFloat32(MAX); got != 0x7bff {
		t.Errorf("FromFloat32(MAX) = %#04x, want 0x7bff", got)
	}
	// Beyond the representable range the result is infinity, not a wrapped or
	// clamped value: silently clamping would corrupt a model's weights.
	if got := FromFloat32(65520); got != 0x7c00 {
		t.Errorf("FromFloat32(65520) = %#04x, want +Inf", got)
	}
	if got := FromFloat32(-65520); got != 0xfc00 {
		t.Errorf("FromFloat32(-65520) = %#04x, want -Inf", got)
	}
	if got := FromFloat32(float32(math.Inf(1))); got != 0x7c00 {
		t.Errorf("FromFloat32(+Inf) = %#04x", got)
	}
	if got := FromFloat32(float32(math.Inf(-1))); got != 0xfc00 {
		t.Errorf("FromFloat32(-Inf) = %#04x", got)
	}
	if got := FromFloat32(float32(math.NaN())); !IsNaN(got) {
		t.Errorf("FromFloat32(NaN) = %#04x, want NaN", got)
	}
	// The NaN path has two arms, chosen by whether the payload survives the shift into the
	// ten-bit binary16 mantissa. math.NaN() is 0x7fc00000, whose payload sits in the high
	// bits, so it always takes the second arm; the branch for a payload that would shift to
	// zero had no test. Truncating such a payload without forcing a bit turns NaN into
	// infinity, which is a different value and a much worse one to meet in weights.
	for _, bits := range []uint32{
		0x7f800001, // signaling NaN: payload is one bit, below the shift
		0x7f801fff, // payload 0x1fff, still entirely below the shift
	} {
		f := math.Float32frombits(bits)
		got := FromFloat32(f)
		if !IsNaN(got) {
			t.Errorf("FromFloat32(%#08x) = %#04x, want NaN: a payload that would shift away "+
				"must still produce a NaN, not infinity", bits, got)
		}
		if got == 0x7c00 || got == 0xfc00 {
			t.Errorf("FromFloat32(%#08x) = %#04x, which is infinity", bits, got)
		}
	}
	// Values below the subnormal range underflow to zero rather than to a
	// denormal that binary16 cannot hold.
	if got := FromFloat32(1e-10); got != 0x0000 {
		t.Errorf("FromFloat32(1e-10) = %#04x, want +0", got)
	}
}

func TestFromFloat32Rounding(t *testing.T) {
	// Round to nearest, ties to even, checked at values where the 13 dropped
	// mantissa bits land exactly on a halfway point and just off it.
	//
	// 1 + 2^-11 is exactly halfway between 1.0 (0x3c00) and the next binary16
	// value 1 + 2^-10 (0x3c01). Ties go to even, so it rounds down to 1.0.
	half := math.Float32frombits(math.Float32bits(1) + 1<<12)
	if got := FromFloat32(half); got != 0x3c00 {
		t.Errorf("FromFloat32(1+2^-11) = %#04x, want 0x3c00 (round to even)", got)
	}
	// Just above the halfway point rounds up.
	above := math.Float32frombits(math.Float32bits(1) + 1<<12 + 1)
	if got := FromFloat32(above); got != 0x3c01 {
		t.Errorf("FromFloat32(1+2^-11+eps) = %#04x, want 0x3c01", got)
	}
	// The next halfway point sits between 0x3c01 (1 + 2^-10) and 0x3c02
	// (1 + 2^-9): it is 1 + 3*2^-11. A tie rounds to the even mantissa,
	// which is 0x3c02.
	mid := float32(1) + 3.0/2048.0
	if got := FromFloat32(mid); got != 0x3c02 {
		t.Errorf("FromFloat32(1+3*2^-11 = %v) = %#04x, want 0x3c02 (round to even)", mid, got)
	}
}

func TestFromFloat32Subnormals(t *testing.T) {
	// The smallest positive binary16 subnormal is 2^-24. Half of it is below
	// the representable range and must round to zero; the value itself must
	// convert exactly.
	if got := FromFloat32(float32(smallestSubnormal)); got != 0x0001 {
		t.Errorf("FromFloat32(2^-24) = %#04x, want 0x0001", got)
	}
	if got := FromFloat32(float32(smallestSubnormal / 2)); got != 0x0000 {
		t.Errorf("FromFloat32(2^-25) = %#04x, want 0x0000", got)
	}
	// 3 * 2^-24 is exactly representable as a subnormal.
	if got := FromFloat32(float32(3 * smallestSubnormal)); got != 0x0003 {
		t.Errorf("FromFloat32(3*2^-24) = %#04x, want 0x0003", got)
	}
}

func TestIsNaNIsInf(t *testing.T) {
	if !IsNaN(0x7e00) || !IsNaN(0x7c01) || !IsNaN(0xfe00) {
		t.Error("IsNaN missed a NaN pattern")
	}
	if IsNaN(0x7c00) || IsNaN(0x3c00) {
		t.Error("IsNaN accepted a non-NaN pattern")
	}
	if !IsInf(0x7c00) || !IsInf(0xfc00) {
		t.Error("IsInf missed an infinity")
	}
	if IsInf(0x7e00) || IsInf(0x3c00) {
		t.Error("IsInf accepted a non-infinity")
	}
}

func BenchmarkToFloat32(b *testing.B) {
	var sum float32
	for i := 0; i < b.N; i++ {
		sum += ToFloat32(uint16(i))
	}
	_ = sum
}

func BenchmarkFromFloat32(b *testing.B) {
	var sum uint16
	for i := 0; i < b.N; i++ {
		sum += FromFloat32(float32(i) * 0.001)
	}
	_ = sum
}

// TestToFloat32Exhaustive checks all 65,536 bit patterns against an independently written
// decoder, which is the claim the README makes.
//
// Rounding is the part of the conversion that is easy to get subtly wrong and hard to see:
// a subnormal normalised with the wrong power of two, or a mantissa shifted into the wrong
// place, still produces a plausible number. The reference below reconstructs the value from
// the field layout with math.Ldexp rather than reusing the implementation's bit
// manipulation, so the two agreeing means the arithmetic, not just the shift, is right.
//
// Verified once against numpy's float16 as a third opinion; this version is the one that
// runs in the suite.
func TestToFloat32Exhaustive(t *testing.T) {
	ref := func(h uint16) float32 {
		sign := 1.0
		if h&0x8000 != 0 {
			sign = -1.0
		}
		exp := int(h>>10) & 0x1f
		mant := int(h & 0x3ff)
		switch exp {
		case 0:
			// Subnormal: mant * 2^-24, and zero when the mantissa is zero.
			return float32(sign * float64(mant) * math.Ldexp(1, -24))
		case 0x1f:
			if mant == 0 {
				return float32(sign * math.Inf(1))
			}
			return float32(math.NaN())
		default:
			// Normal: (1 + mant/1024) * 2^(exp-15).
			return float32(sign * (1 + float64(mant)/1024) * math.Ldexp(1, exp-15))
		}
	}
	for h := range 0x10000 {
		v := uint16(h)
		got := ToFloat32(v)
		want := ref(v)
		if math.IsNaN(float64(want)) {
			if !math.IsNaN(float64(got)) {
				t.Errorf("ToFloat32(%#04x) = %v, want NaN", v, got)
			}
			continue
		}
		if got != want || math.Signbit(float64(got)) != math.Signbit(float64(want)) {
			t.Fatalf("ToFloat32(%#04x) = %v, want %v", v, got, want)
		}
	}
}

// TestToFloat32QuietsEveryNaN covers the one thing the exhaustive test cannot see.
//
// That test asks only whether the result IsNaN, so every NaN in the half's range passes no
// matter which NaN it is: verified by dropping the quiet bit from the implementation, after
// which the whole package still passed. What the code deliberately does is wider than
// "produces a NaN" -- it keeps the payload and sets the quiet bit, so arithmetic on the
// result does not raise a signalling-NaN trap, and a NaN produced from one half is not
// silently interchangeable with one produced from another.
//
// Cross-checked against numpy before writing this: of the 2046 NaN halves, the conversion
// agrees with numpy.float16 on every value except the quiet bit, which numpy leaves clear
// and this sets on purpose. The other 63490 halves agree bit for bit.
func TestToFloat32QuietsEveryNaN(t *testing.T) {
	nanCount := 0
	for h := range 0x10000 {
		v := uint16(h)
		exp := (v >> 10) & 0x1f
		mant := v & 0x3ff
		if exp != 0x1f || mant == 0 {
			continue // finite or +/-Inf
		}
		nanCount++
		got := math.Float32bits(ToFloat32(v))
		if got&0x00400000 == 0 {
			t.Errorf("ToFloat32(%#04x) = %#08x, which is a signalling NaN: the quiet bit "+
				"has to be set", v, got)
		}
		// The payload survives, shifted into the float32 mantissa. A NaN carrying a payload
		// is how a vocabulary records where it came from, so dropping it loses information
		// that costs nothing to keep.
		if wantPayload := uint32(mant) << 13; got&0x007fffff != 0x00400000|wantPayload {
			t.Errorf("ToFloat32(%#04x) mantissa = %#07x, want %#07x",
				v, got&0x007fffff, 0x00400000|wantPayload)
		}
		// The sign of the NaN is preserved, as it is for every other value.
		if sign := v&0x8000 != 0; sign != (got&0x80000000 != 0) {
			t.Errorf("ToFloat32(%#04x) = %#08x: sign bit %v, want %v",
				v, got, got&0x80000000 != 0, sign)
		}
	}
	// 2 * (1024 - 1): both signs, every mantissa except zero, which is infinity.
	if want := 2 * 1023; nanCount != want {
		t.Fatalf("walked %d NaN halves, want %d -- the filter is wrong, not the code",
			nanCount, want)
	}
}

// TestFromFloat32OverflowBoundary pins where a value stops rounding to the largest finite
// half and starts rounding to infinity.
//
// The reference test walks midpoints between adjacent halves up to 0x7bff, so the largest
// value it ever encodes is (0x7bfe + 0x7bff)/2 -- just below 65504. Everything above the
// largest finite half is outside its range, which is exactly where this decision is made, so
// nothing exercised it. Two bugs in the reference helper are hidden by the same gap: its
// overflow branch computes a midpoint and discards it (`_ = mid`, then unconditionally
// returns infinity), and it returns zero for a NaN because no finite candidate is within any
// distance of one. Neither is a defect in FromFloat32, but both mean the helper cannot be
// consulted near this boundary.
//
// The rule: 65504 is 0x7bff and the next value up is infinity, so the midpoint is one half
// ULP above 65504. ULP there is 2^5 = 32, so values below 65520 round down and 65520 rounds
// to infinity. Ties go to even, and infinity's significand is even, so the midpoint itself
// goes up.
func TestFromFloat32OverflowBoundary(t *testing.T) {
	const maxFinite = 65504.0
	const halfULP = 16.0 // 2^5 / 2

	if got := FromFloat32(float32(maxFinite)); got != 0x7bff {
		t.Errorf("FromFloat32(65504) = %#04x, want 0x7bff", got)
	}
	// Everything strictly below the midpoint rounds back down to the largest finite value.
	for _, extra := range []float64{1, 8, halfULP - 1, halfULP - 0.5} {
		v := float32(maxFinite + extra)
		if got := FromFloat32(v); got != 0x7bff {
			t.Errorf("FromFloat32(%v) = %#04x, want 0x7bff: it is below the midpoint so it rounds "+
				"to the nearest finite half", maxFinite+extra, got)
		}
	}
	// The midpoint and everything above it overflow.
	for _, extra := range []float64{halfULP, halfULP + 1, 100, 1e10} {
		v := float32(maxFinite + extra)
		if got := FromFloat32(v); got != 0x7c00 {
			t.Errorf("FromFloat32(%v) = %#04x, want 0x7c00 (infinity)", maxFinite+extra, got)
		}
	}
	// The negative side mirrors it, with the sign bit.
	if got := FromFloat32(float32(-(maxFinite + halfULP - 1))); got != 0xfbff {
		t.Errorf("FromFloat32(-(65504+15)) = %#04x, want 0xfbff", got)
	}
	if got := FromFloat32(float32(-(maxFinite + halfULP))); got != 0xfc00 {
		t.Errorf("FromFloat32(-(65504+16)) = %#04x, want 0xfc00", got)
	}
}

// TestFromFloat32NaNBecomesNaN pins the encoding of a NaN input, which the reference helper
// gets wrong (it returns zero) and which no test asserted.
//
// A NaN has to stay a NaN in the other precision: converting it to zero would turn a corrupt
// weight into a plausible one silently. The exact payload is not specified -- FromFloat32
// produces the canonical quiet NaN 0x7e00 -- so the assertion is IsNaN rather than equality
// with that constant.
func TestFromFloat32NaNBecomesNaN(t *testing.T) {
	for _, v := range []float32{
		float32(math.NaN()),
		float32(math.Inf(1) - math.Inf(1)), // NaN by arithmetic rather than by constructor
	} {
		got := FromFloat32(v)
		if !IsNaN(got) {
			t.Errorf("FromFloat32(%v) = %#04x, want a NaN pattern", v, got)
		}
	}
	// A negative NaN keeps its sign, which is what makes the payload preservation meaningful
	// rather than incidental.
	negNaN := math.Float32frombits(math.Float32bits(float32(math.NaN())) | 1<<31)
	if got := FromFloat32(negNaN); got&0x8000 == 0 {
		t.Errorf("FromFloat32(negative NaN) = %#04x, lost the sign bit", got)
	}
}
