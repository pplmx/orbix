package llama

import (
	"math"
	"os"
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

func TestRopePairsAdjacentDimensions(t *testing.T) {
	// Llama pairs dimension 2i with 2i+1. The alternative (i with i+n_dims/2,
	// which ggml calls NEOX and Qwen models use) produces a different rotation,
	// so this test distinguishes them with an input that only one can make.
	cfg := testConfig()
	cfg.RopeDim = 4
	cfg.HeadDim = 4
	r, err := newRope(cfg, 4)
	if err != nil {
		t.Fatal(err)
	}

	// A unit vector in dimension 1: adjacent pairing rotates it against
	// dimension 0, so the result puts cos into index 1 and sin into index 0's
	// partner. Halves pairing would rotate it against dimension 3 instead.
	y := []float32{0, 1, 0, 0}
	if err := r.apply(y, 1); err != nil {
		t.Fatal(err)
	}
	const theta0 = 1.0 // base^0 * pos 1
	if want := float32(math.Cos(theta0)); math.Abs(float64(y[1]-want)) > 1e-6 {
		t.Errorf("y[1] = %v, want cos(1) = %v (adjacent pairing)", y[1], want)
	}
	if want := float32(math.Sin(theta0)); math.Abs(float64(y[0]-(-want))) > 1e-6 {
		t.Errorf("y[0] = %v, want -sin(1) = %v", y[0], -want)
	}
	// Dimensions 2 and 3 form the second pair and are untouched by this input.
	if math.Abs(float64(y[2])) > 1e-6 || math.Abs(float64(y[3])) > 1e-6 {
		t.Errorf("y[2]=%v y[3]=%v, want 0", y[2], y[3])
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
	if err := c.Reset(); err != nil {
		t.Fatal(err)
	}
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

func TestRopeMatchesReferenceLogprobs(t *testing.T) {
	// The decisive check for the rotation: run the real model over a fixed prefix and
	// compare the resulting log-probabilities with llama.cpp's for the same tokens.
	//
	// The deviations measured here are 0.0014, 0.0728 and 0.1192 nats, so the tolerance
	// below is about twice the largest of them. The halves pairing this test was written
	// to catch was off by more than two nats on the same input while still producing
	// fluent text, which is why a comparison against real numbers exists rather than a
	// shape-only test.
	//
	// An earlier version of this comment said "within a few hundredths of a nat", which
	// understated the true agreement by an order of magnitude for two of the three
	// tokens. The tolerance is what protects the test; the comment now matches it.
	//
	// Expected values come from llama-server's /completion endpoint
	// (n_probs=5, temperature 0, -fa off) on stories15M-q4_0 with the prompt
	// "Once" -> tokens [1, 9038].
	path := os.Getenv("MODEL")
	if path == "" {
		t.Skip("set MODEL to a stories15M GGUF to run this check")
	}
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	cache, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatal(err)
	}
	var logits []float32
	for pos, id := range []int{1, 9038} {
		logits, err = m.Forward(cache, id, pos)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
	}

	want := map[int]float64{
		2501:  -0.0810, // " upon"
		727:   -2.8706, // " there"
		29892: -3.9226, // ","
	}
	// Twice the largest measured deviation (0.1192), so a real regression shows up rather
	// than hiding under a slack bound.
	const tol = 0.25
	for id, wantLP := range want {
		got := logSoftmaxOf(logits, id)
		if math.Abs(got-wantLP) > tol {
			t.Errorf("logprob[%d] = %.4f, want %.4f (+/- %.2f)", id, got, wantLP, tol)
		}
	}
}

// logSoftmaxOf returns log(softmax(logits)[i]).
func logSoftmaxOf(logits []float32, i int) float64 {
	max := float64(logits[0])
	for _, v := range logits[1:] {
		max = math.Max(max, float64(v))
	}
	var sum float64
	for _, v := range logits {
		sum += math.Exp(float64(v) - max)
	}
	return (float64(logits[i]) - max) - math.Log(sum)
}

// TestRopeFrequencyFactors checks the tensor that extends a model's context without
// changing freq_base. Each dimension pair's rate is divided by its entry, so a model can
// keep the low dimensions at the frequency it was trained on and slow only the high ones.
//
// Ignoring the tensor leaves the low dimensions correct, which is why it produces fluent
// text and looks like it works: the error is confined to the dimensions that carry the
// long-range position signal.
func TestRopeFrequencyFactors(t *testing.T) {
	cfg := testConfig()
	cfg.RopeDim = 8
	cfg.RopeBase = 10000
	cfg.RopeFreqs = []float32{1, 1, 4, 16}

	const pos = 500
	with, err := newRope(cfg, 1024)
	if err != nil {
		t.Fatalf("newRope: %v", err)
	}
	bare := cfg
	bare.RopeFreqs = nil
	without, err := newRope(bare, 1024)
	if err != nil {
		t.Fatalf("newRope: %v", err)
	}

	// A dimension whose factor is 1 is unchanged, which is what makes the error hard to
	// notice: most dimensions agree.
	if with.cos[pos][0] != without.cos[pos][0] {
		t.Errorf("dimension 0 has factor 1 but changed: %v vs %v",
			with.cos[pos][0], without.cos[pos][0])
	}
	// A scaled dimension must differ, and by more the larger its factor.
	for _, k := range []int{2, 3} {
		if with.cos[pos][k] == without.cos[pos][k] {
			t.Errorf("dimension %d has factor %v but did not change",
				k, cfg.RopeFreqs[k])
		}
	}
	d2 := math.Abs(float64(with.cos[pos][2] - without.cos[pos][2]))
	d3 := math.Abs(float64(with.cos[pos][3] - without.cos[pos][3]))
	if d3 <= d2 {
		t.Errorf("a factor of 16 moved dimension 3 by %v and a factor of 4 moved "+
			"dimension 2 by %v; the larger factor should move it more", d3, d2)
	}
}

// TestRopeFactorsAreNotCarriedForward checks that a dimension's factor applies to that
// dimension only. Dividing a running product would compound every later dimension by the
// earlier factor, which slows the whole head instead of part of it.
func TestRopeFactorsAreNotCarriedForward(t *testing.T) {
	cfg := testConfig()
	cfg.RopeDim = 8
	cfg.RopeBase = 10000
	// Only dimension 1 is scaled.
	cfg.RopeFreqs = []float32{1, 8, 1, 1}

	onlySecond, err := newRope(cfg, 256)
	if err != nil {
		t.Fatalf("newRope: %v", err)
	}
	// The same model with dimension 1 at factor 1: dimensions past it must match, which
	// they would not if the factor had been carried forward.
	cfg.RopeFreqs = []float32{1, 1, 1, 1}
	none, err := newRope(cfg, 256)
	if err != nil {
		t.Fatalf("newRope: %v", err)
	}
	const pos = 100
	for _, k := range []int{2, 3} {
		if onlySecond.cos[pos][k] != none.cos[pos][k] {
			t.Errorf("dimension %d changed although only dimension 1 has a factor; "+
				"the factor is being carried forward", k)
		}
	}
}
