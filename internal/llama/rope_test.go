package llama

import (
	"math"
	"testing"
)

// These tests use a synthetic config rather than a model file: the RoPE and
// cache code is pure math and shape bookkeeping, and checking it needs no
// weights.

func testConfig() Config {
	return Config{
		Arch:       "llama",
		Hidden:     8,
		Layers:     2,
		Heads:      2,
		KVHeads:    2,
		HeadDim:    4,
		FFN:        16,
		ContextLen: 16,
		RopeDim:    4,
		RopeBase:   10000,
		RopeScale:  1,
		RMSNormEps: 1e-5,
	}
}

func TestRopePreservesNorm(t *testing.T) {
	// A rotation is norm-preserving. If the implementation paired the wrong
	// dimensions, the vector would still be finite but its norm would change,
	// which is exactly the kind of error that produces plausible garbage.
	r, err := newRope(testConfig(), 4)
	if err != nil {
		t.Fatal(err)
	}
	for pos := 0; pos < 4; pos++ {
		x := []float32{1, -2, 3, -4}
		before := norm(x)
		if err := r.apply(x, pos); err != nil {
			t.Fatal(err)
		}
		if got := norm(x); math.Abs(float64(got-before)) > 1e-5 {
			t.Errorf("pos %d: norm %v -> %v, rotation must preserve it", pos, before, got)
		}
	}
}

func TestRopeAtZeroIsIdentity(t *testing.T) {
	// cos(0) = 1 and sin(0) = 0, so the rotation at position 0 must change
	// nothing. A non-identity here means the frequency table is off.
	r, err := newRope(testConfig(), 4)
	if err != nil {
		t.Fatal(err)
	}
	x := []float32{0.5, -1.5, 2.5, -3.5}
	want := append([]float32(nil), x...)
	if err := r.apply(x, 0); err != nil {
		t.Fatal(err)
	}
	for i := range x {
		if math.Abs(float64(x[i]-want[i])) > 1e-6 {
			t.Errorf("x[%d] = %v, want %v (position 0 must be identity)", i, x[i], want[i])
		}
	}
}

func TestRopePairsHalvesNotNeighbours(t *testing.T) {
	// Llama uses NEOX rotation: dimension i pairs with i + dims/2, not with
	// i+1. Set one dimension to a vector that distinguishes the two schemes and
	// check which one moved.
	cfg := testConfig()
	cfg.RopeDim = 4
	cfg.HeadDim = 4
	r, err := newRope(cfg, 4)
	if err != nil {
		t.Fatal(err)
	}
	// Position 1 with base 10000 and 4 dims: pair k rotates by 10000^(-2k/4).
	x := []float32{1, 0, 0, 0}
	if err := r.apply(x, 1); err != nil {
		t.Fatal(err)
	}
	// At position 1 the first pair rotates by theta = 1 (10000^0 * pos).
	// Under the halves scheme x[0] = cos(1)*1 - sin(1)*0 = cos(1). Under the
	// neighbour scheme x[0] would pair with x[1], which is zero here, so it
	// would also be cos(1) -- the two only diverge for the second half, which
	// the following assertion pins down.
	want0 := float32(math.Cos(1))
	if math.Abs(float64(x[0]-want0)) > 1e-5 {
		t.Errorf("x[0] = %v, want %v", x[0], want0)
	}
	// x[3] is index 1 of the second half. Under the halves scheme it pairs with
	// x[1] (zero), so it stays zero; under the neighbour scheme it would pair
	// with x[2] only, which is also zero. Neither distinguishes them alone.
	// The discriminating case is a vector with a value in the second half.
	if math.Abs(float64(x[1])) > 1e-6 {
		t.Errorf("x[1] = %v, want 0 (not paired with x[0])", x[1])
	}

	// The discriminating input: weight only in dimension 1. Under halves
	// pairing dimension 1 is index 1 of the *second* pair (k = 1), so it
	// rotates against dimension 3 by theta = 0.01 at position 1 with
	// base 10000 and 4 dims. A neighbour pairing would rotate it against
	// dimension 0 at theta = 1 instead, which is what the expected values
	// below rule out.
	y := []float32{0, 1, 0, 0}
	if err := r.apply(y, 1); err != nil {
		t.Fatal(err)
	}
	const theta1 = 0.01 // 10000^(-2/4) * pos(1)
	if want := float32(math.Cos(theta1)); math.Abs(float64(y[1]-want)) > 1e-6 {
		t.Errorf("y[1] = %v, want cos(0.01) = %v", y[1], want)
	}
	if want := float32(math.Sin(theta1)); math.Abs(float64(y[3]-want)) > 1e-6 {
		t.Errorf("y[3] = %v, want sin(0.01) = %v (halves pairing)", y[3], want)
	}
	if math.Abs(float64(y[0])) > 1e-6 || math.Abs(float64(y[2])) > 1e-6 {
		t.Errorf("y[0]=%v y[2]=%v, want 0", y[0], y[2])
	}
}

func TestRopeLeavesTailUntouched(t *testing.T) {
	// When rope.dimension_count is smaller than the head width, the remaining
	// dimensions pass through unchanged. Models with partial rotation rely on
	// this.
	cfg := testConfig()
	cfg.HeadDim = 8
	cfg.RopeDim = 4
	r, err := newRope(cfg, 4)
	if err != nil {
		t.Fatal(err)
	}
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	tail := append([]float32(nil), x[4:]...)
	if err := r.apply(x, 2); err != nil {
		t.Fatal(err)
	}
	for i := range tail {
		if x[4+i] != tail[i] {
			t.Errorf("x[%d] = %v, want %v (tail must be untouched)", 4+i, x[4+i], tail[i])
		}
	}
}

func TestRopeRejectsBadPosition(t *testing.T) {
	r, err := newRope(testConfig(), 2)
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float32, 4)
	if err := r.apply(x, 2); err == nil {
		t.Error("apply at pos 2 succeeded with a 2-position table")
	}
	if err := r.apply(x, -1); err == nil {
		t.Error("apply at pos -1 succeeded")
	}
	if err := r.apply(make([]float32, 2), 0); err == nil {
		t.Error("apply with a short vector succeeded")
	}
}

func TestNewRopeRejectsOddDims(t *testing.T) {
	cfg := testConfig()
	cfg.RopeDim = 3
	if _, err := newRope(cfg, 4); err == nil {
		t.Error("newRope accepted an odd rope dim")
	}
}

func TestKVCacheOffsets(t *testing.T) {
	// Each position's key and value must land at its own offset; an off-by-one
	// here makes every token attend to its neighbour's state.
	cfg := testConfig()
	c, err := NewKVCache(cfg, 4)
	if err != nil {
		t.Fatal(err)
	}
	if got := c.Cap(); got != 4 {
		t.Errorf("Cap = %d, want 4", got)
	}
	dim := cfg.kvDim()
	for pos := range 4 {
		k := make([]float32, dim)
		v := make([]float32, dim)
		for i := range k {
			k[i] = float32(pos*100 + i)
			v[i] = float32(pos*100 + i + 50)
		}
		allK, allV, err := c.appendKV(0, k, v, pos)
		if err != nil {
			t.Fatal(err)
		}
		if len(allK) != (pos+1)*dim {
			t.Errorf("pos %d: key view = %d, want %d", pos, len(allK), (pos+1)*dim)
		}
		// The value just written must be readable at the expected offset.
		if got := allK[pos*dim]; got != float32(pos*100) {
			t.Errorf("pos %d: k[%d] = %v, want %v", pos, pos*dim, got, float32(pos*100))
		}
		if got := allV[pos*dim]; got != float32(pos*100+50) {
			t.Errorf("pos %d: v[%d] = %v, want %v", pos, pos*dim, got, float32(pos*100+50))
		}
	}
}

func TestKVCacheRejectsBadArguments(t *testing.T) {
	cfg := testConfig()
	c, err := NewKVCache(cfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	dim := cfg.kvDim()
	if _, _, err := c.appendKV(cfg.Layers, make([]float32, dim), make([]float32, dim), 0); err == nil {
		t.Error("appendKV accepted an out-of-range layer")
	}
	if _, _, err := c.appendKV(0, make([]float32, dim), make([]float32, dim), 2); err == nil {
		t.Error("appendKV accepted a position at capacity")
	}
	if _, _, err := c.appendKV(0, make([]float32, 1), make([]float32, 1), 0); err == nil {
		t.Error("appendKV accepted a wrong-width vector")
	}
	if _, err := NewKVCache(cfg, 0); err == nil {
		t.Error("NewKVCache accepted a zero size")
	}
}

func TestKVCacheReset(t *testing.T) {
	cfg := testConfig()
	c, err := NewKVCache(cfg, 4)
	if err != nil {
		t.Fatal(err)
	}
	c.len = 3
	if got := c.Len(); got != 3 {
		t.Errorf("Len = %d, want 3", got)
	}
	c.Reset()
	if got := c.Len(); got != 0 {
		t.Errorf("after Reset Len = %d, want 0", got)
	}
	// The capacity survives a reset: resetting is for reusing the arena, not
	// for freeing it.
	if got := c.Cap(); got != 4 {
		t.Errorf("after Reset Cap = %d, want 4", got)
	}
}

func TestKVCacheBytes(t *testing.T) {
	cfg := testConfig()
	c, err := NewKVCache(cfg, 8)
	if err != nil {
		t.Fatal(err)
	}
	// Two tensors (k and v) per layer, each 8 positions of kvDim floats.
	want := int64(2 * cfg.Layers * 8 * cfg.kvDim() * 4)
	if got := c.Bytes(); got != want {
		t.Errorf("Bytes = %d, want %d", got, want)
	}
}

func norm(x []float32) float32 {
	var sum float32
	for _, v := range x {
		sum += v * v
	}
	return float32(math.Sqrt(float64(sum)))
}
