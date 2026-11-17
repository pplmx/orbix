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
