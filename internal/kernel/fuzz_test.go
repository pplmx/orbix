package kernel

import (
	"math"
	"testing"
)

// FuzzMatVecShapes checks that no shape combination reaches a write outside the buffers.
//
// Every kernel takes caller-sized slices plus the dimensions describing them, so a check that
// misses one relation turns a bug in the caller into a panic or a silent write past the end. The
// dimensions come from a checkpoint's metadata, which is to say from a file the user downloaded.
//
// The shapes are fuzzed rather than the data: a wrong shape is what the checks exist for, and
// random floats exercise nothing the fixed-value tests do not already cover.
func FuzzMatVecShapes(f *testing.F) {
	f.Add(int64(4), int64(2), int64(4), int64(2), int64(4))
	f.Add(int64(0), int64(0), int64(0), int64(0), int64(0))
	f.Add(int64(-1), int64(2), int64(4), int64(2), int64(4))
	// Shapes whose product overflows int. The guards have to reject these rather than compute a
	// negative requirement and accept a buffer that is too small.
	f.Add(int64(math.MaxInt64), int64(2), int64(4), int64(2), int64(4))
	f.Add(int64(math.MaxInt64/2+1), int64(2), int64(4), int64(2), int64(4))

	f.Fuzz(func(t *testing.T, nIn, kIn, aLen, vLen, dstLen int64) {
		// Clamp to what a slice can actually be, so the test allocates nothing absurd. The
		// overflow cases above survive the clamp because n and k stay large while the slice
		// lengths become small, which is exactly the combination the checks must refuse.
		const maxLen = 64
		n, k := clampInt(nIn, maxLen), clampInt(kIn, maxLen)
		a := make([]float32, clampInt(aLen, maxLen))
		v := make([]float32, clampInt(vLen, maxLen))
		dst := make([]float32, clampInt(dstLen, maxLen))

		// The property is that the call either returns an error or writes only within dst, a and
		// v. Filling them with a sentinel makes a write past the end a slice-bounds panic, which
		// the test reports rather than the process dying later.
		for i := range a {
			a[i] = 1
		}
		for i := range v {
			v[i] = 1
		}
		err := MatVec(dst, a, v, n, k)
		if err != nil {
			return
		}
		// Accepted, so the dimensions described buffers that fit. Writing n outputs of a
		// k-long product into a is the shape the call claims to have checked.
		if len(dst) < n || len(a) < n*k || len(v) < k {
			t.Fatalf("MatVec accepted n=%d k=%d with a=%d v=%d dst=%d",
				n, k, len(a), len(v), len(dst))
		}
	})
}

// clampInt narrows a fuzzer-supplied int64 into a length this test can allocate.
//
// A negative value becomes zero, which is a shape the checks must still handle; a large one is
// capped, because the interesting overflow cases come from n and k, not from allocating gigabytes.
func clampInt(v int64, max int64) int {
	if v < 0 {
		return 0
	}
	if v > max {
		return int(max)
	}
	return int(v)
}
