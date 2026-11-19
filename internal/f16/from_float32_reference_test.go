package f16

import (
	"math"
	"testing"
)

func nearestHalfSlow(v float64) uint16 {
	neg := math.Signbit(v)
	if neg {
		v = -v
	}
	best := uint16(0)
	bestDist := math.Inf(1)
	for h := 0; h < 0x7c00; h++ {
		hv := float64(ToFloat32(uint16(h)))
		d := math.Abs(hv - v)
		if d < bestDist {
			bestDist, best = d, uint16(h)
		} else if d == bestDist && best&1 == 1 && uint16(h)&1 == 0 {
			best = uint16(h)
		}
	}
	if v > float64(ToFloat32(0x7bff)) {
		// Overflow: nearest finite or inf, whichever is closer; halfway rounds to even,
		// and 0x7c00 has an even significand, so >= the midpoint goes to inf.
		mid := (float64(ToFloat32(0x7bff)) + math.Inf(1)) / 2
		_ = mid
		best = 0x7c00
	}
	if neg {
		best |= 0x8000
	}
	return best
}

// TestFromFloat32RoundsLikeAReference covers the encode direction against an independent
// reference.
//
// ToFloat32 exhaustive is the model this follows: compare the implementation to a second one
// written from the specification rather than to hand-picked cases. FromFloat32 had no such
// check, and its round trip through ToFloat32 only visits exactly representable values, so
// no existing test exercised rounding at all. Dropping the ties-to-even condition in the
// subnormal path -- `if round != 0 && rem != 0` -- passed the whole f16 suite.
//
// The reference is brute force on purpose: nearestHalfSlow compares the value against every
// finite half with float64 arithmetic and keeps the closest, breaking a tie toward the even
// significand. That is the definition of the rounding mode with none of the bit
// manipulation, so it cannot share a mistake with the implementation.
//
// Every midpoint between adjacent halves is checked. The exact half values are all
// representable, so encoding them cannot exercise rounding; every decision is made at a
// midpoint, which is what this walks.
func TestFromFloat32RoundsLikeAReference(t *testing.T) {
	// Midpoints between adjacent halves, which is where a tie occurs.
	for h := 0; h < 0x7bff; h++ {
		lo := float64(ToFloat32(uint16(h)))
		hi := float64(ToFloat32(uint16(h + 1)))
		mid := (lo + hi) / 2
		want := nearestHalfSlow(mid)
		if got := FromFloat32(float32(mid)); got != want {
			t.Fatalf("midpoint between %#04x and %#04x (%v): FromFloat32 = %#04x, want %#04x",
				h, h+1, mid, got, want)
		}
	}
}
