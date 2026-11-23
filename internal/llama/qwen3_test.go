package llama

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguftest"
)

// TestQwen3ConfigRecordsItsDifferences pins what Qwen3 changes relative to Qwen2.
//
// The two are close enough to be confusable: same rotation, same grouped-query shapes. What Qwen3
// does instead is normalise each head's query and key over the head width, and drop the attention
// bias Qwen2 carries. Loading one as the other runs cleanly, which is why the table records both.
func TestQwen3ConfigRecordsItsDifferences(t *testing.T) {
	path := gguftest.Model{Arch: "qwen3", Layers: 2, Hidden: 32, Heads: 4, FFN: 64,
		QKNorm: true}.Write(t, t.TempDir())
	m, err := NewWithOptions(path, Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	cfg := m.Config()
	if cfg.Arch != "qwen3" {
		t.Errorf("Arch = %q", cfg.Arch)
	}
	if !cfg.QKNorm {
		t.Error("QKNorm is false on a qwen3 model")
	}
	if cfg.AttnBias {
		t.Error("AttnBias is true; Qwen3 dropped the bias vectors Qwen2 has")
	}
	if cfg.RopeLayout != RopeNeox {
		t.Errorf("RopeLayout = %v, want neox", cfg.RopeLayout)
	}
	// The norm vectors are the shape the forward pass indexes by head width.
	for l := range cfg.Layers {
		for _, w := range [][]float32{m.W.Layers[l].qNorm, m.W.Layers[l].kNorm} {
			if len(w) != cfg.HeadDim {
				t.Errorf("layer %d norm holds %d values, want head dim %d", l, len(w), cfg.HeadDim)
			}
		}
	}
	if !strings.Contains(cfg.String(), "qk norm") {
		t.Errorf("summary %q does not mention the qk norm", cfg.String())
	}
}

// TestNormQKNormalisesEachHead covers the arithmetic the architecture turns on.
//
// The property is per head: a norm over the whole projection would give a different answer
// whenever heads differ in scale, which they do. The weights here are all ones, so the result is
// each head scaled to unit RMS, and the assertion is on that number rather than on the
// implementation's intermediate state.
func TestNormQKNormalisesEachHead(t *testing.T) {
	path := gguftest.Model{Arch: "qwen3", Layers: 1, Hidden: 32, Heads: 4, FFN: 64,
		QKNorm: true}.Write(t, t.TempDir())
	m, err := NewWithOptions(path, Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	cfg := m.Config()
	d := cfg.HeadDim

	// Two heads with very different scales: a norm over the whole vector would leave the
	// second head much smaller than the first.
	q := make([]float32, 2*d)
	for i := range d {
		q[i] = 1     // head 0
		q[d+i] = 100 // head 1
	}
	k := append([]float32(nil), q...)
	if err := m.normQK(&m.W.Layers[0], q, k, 2, 2); err != nil {
		t.Fatalf("normQK: %v", err)
	}
	for h := range 2 {
		head := q[h*d : (h+1)*d]
		var sum float64
		for _, v := range head {
			sum += float64(v) * float64(v)
		}
		if rms := math.Sqrt(sum / float64(d)); math.Abs(rms-1) > 1e-5 {
			t.Errorf("q head %d has RMS %v after the norm, want 1", h, rms)
		}
	}
	// Each head kept its own direction: head 1 was uniform at 100 and is uniform afterwards.
	for i := d; i < 2*d; i++ {
		if math.Abs(float64(q[i]-q[d])) > 1e-6 {
			t.Errorf("head 1 is no longer uniform at index %d: %v against %v", i, q[i], q[d])
			break
		}
	}
}

// TestNormQKIsSkippedWithoutTheFlag keeps the branch honest for the other three architectures.
//
// Qwen2 has no norm vectors at all, so a forward pass that normalised unconditionally would divide
// by whichever vector happened to be there -- none -- and change every Qwen2 answer.
func TestNormQKIsSkippedWithoutTheFlag(t *testing.T) {
	path := gguftest.Model{Layers: 1, Hidden: 32, Heads: 4, FFN: 64}.Write(t, t.TempDir())
	m, err := NewWithOptions(path, Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	l := &m.W.Layers[0]
	q := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	orig := append([]float32(nil), q...)
	if err := m.normQK(l, q, q, 1, 1); err != nil {
		t.Fatalf("normQK: %v", err)
	}
	for i := range q {
		if q[i] != orig[i] {
			t.Errorf("a model without the flag changed value %d: %v, want %v", i, q[i], orig[i])
		}
	}
}

// TestNormQKRejectsShortBuffers covers the two shapes the helper refuses.
//
// Both are upstream bugs rather than caller mistakes, but the alternative is a slice-bounds panic
// inside a forward pass, which names neither the layer nor the tensor.
func TestNormQKRejectsShortBuffers(t *testing.T) {
	path := gguftest.Model{Arch: "qwen3", Layers: 1, Hidden: 32, Heads: 4, FFN: 64,
		QKNorm: true}.Write(t, t.TempDir())
	m, err := NewWithOptions(path, Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	l := &m.W.Layers[0]
	d := m.Config().HeadDim
	if err := m.normQK(l, make([]float32, d-1), nil, 1, 0); err == nil {
		t.Error("a query shorter than one head was accepted")
	} else if !strings.Contains(err.Error(), "needs") {
		t.Errorf("error = %v, want it to name the sizes", err)
	}
	// A weight shorter than the head width would read past its end.
	l.qNorm = l.qNorm[:d-1]
	if err := m.normQK(l, make([]float32, d), nil, 1, 0); err == nil {
		t.Error("a norm weight shorter than the head was accepted")
	}
}

// TestQwen3PrefillMatchesPerTokenForward covers the batched path, which the gated comparison
// above cannot see.
//
// That test drives Forward one token at a time, so it exercises the single-token normalisation
// only: the batch prefill and the batch decode each have their own call, and one of them skipping
// the norm would leave every existing assertion green. Prefill's own documentation promises "the
// result is identical to calling Forward once per token", so the invariant is already stated --
// this is the test that holds it for an architecture whose forward pass has a step Llama's does
// not.
func TestQwen3PrefillMatchesPerTokenForward(t *testing.T) {
	path := gguftest.Model{Arch: "qwen3", Layers: 2, Hidden: 32, Heads: 4, FFN: 64,
		QKNorm: true, NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	m, err := NewWithOptions(path, Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	cfg := m.Config()
	tokens := []int{1, 2, 3, 4}

	batched, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatalf("cache: %v", err)
	}
	want, err := m.Prefill(batched, toInt32(tokens), 0)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	want = append([]float32(nil), want...)

	single, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatalf("cache: %v", err)
	}
	var got []float32
	for pos, tok := range tokens {
		got, err = m.Forward(single, tok, pos)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
	}

	if len(got) != len(want) {
		t.Fatalf("logit counts differ: %d batched, %d one at a time", len(want), len(got))
	}
	var worst float64
	for i := range want {
		if d := math.Abs(float64(want[i] - got[i])); d > worst {
			worst = d
		}
	}
	// Not an equality check: the two paths accumulate in a different order, so the last bits
	// differ. A step applied on one path and not the other does not differ by bits.
	if worst > 1e-3 {
		t.Errorf("batched and per-token logits differ by up to %v; a step is applied on one path"+
			" and not the other", worst)
	}
}

// TestQwen3MatchesReferenceLogprobs is the decisive check, and the only one that can see the
// normalisation: every fixture test above pins a piece of it, but only the real checkpoint's
// distribution says whether the pieces were put together the way the model was trained.
//
// Set QWEN3_MODEL to a Qwen3 GGUF to run it. The expected values come from llama.cpp 0.3.37 on
// Qwen3-0.6B-Q4_K_M, prompt "The capital of France is", top-8 at the first generated position. The
// prompt encodes to the same five ids on both sides.
//
// Measured deviations: 0.020, 0.139, 0.045, 0.203, 0.286, 0.213, 0.192 nats over the seven tokens
// the two lists share, and the same seven in the same order. The eighth differs -- ' Vers' against
// ' ____' -- which is what a dense tail does when the two lists disagree by a fifth of a nat. The
// tolerance is set from the nearest wrong variant rather than from the measurement: removing the
// normalisation changes the greedy token, so the argmax assertion below is the discriminating one.
func TestQwen3MatchesReferenceLogprobs(t *testing.T) {
	path := os.Getenv("QWEN3_MODEL")
	if path == "" {
		t.Skip("set QWEN3_MODEL to a Qwen3 GGUF file")
	}
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	if cfg.VocabSize != 151936 || cfg.Layers != 28 {
		t.Skipf("the expected values are Qwen3 0.6B's (vocab 151936, 28 layers); this file has"+
			" vocab %d and %d layers", cfg.VocabSize, cfg.Layers)
	}

	cache, err := NewKVCache(cfg, 128)
	if err != nil {
		t.Fatal(err)
	}
	var logits []float32
	for pos, id := range []int{785, 6722, 315, 9625, 374} {
		logits, err = m.Forward(cache, id, pos)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
	}

	want := []struct {
		id   int32
		text string
		lp   float64
	}{
		{12095, " Paris", -0.474920},
		{7407, " located", -3.762811},
		{1112, "...", -3.913126},
		{279, " the", -3.939489},
		{304, " in", -3.958619},
		{55201, " Lyon", -4.443489},
		{32671, " ______", -4.519934},
	}
	const tolerance = 0.5
	for _, w := range want {
		got := logSoftmax(logits, w.id)
		if d := math.Abs(got - w.lp); d > tolerance {
			t.Errorf("logprob of %q (id %d) = %.4f, llama.cpp says %.4f, off by %.4f",
				w.text, w.id, got, w.lp, d)
		}
	}
	argmax := int32(0)
	for i, v := range logits {
		if v > logits[argmax] {
			argmax = int32(i)
		}
	}
	if argmax != 12095 {
		t.Errorf("greedy token is %d, want 12095 (\" Paris\")", argmax)
	}
	// The gate itself, checked rather than assumed: a file promising qk norms that did not load
	// would make the comparison above fail for a reason nobody could read off the numbers.
	if !cfg.QKNorm {
		t.Error("QKNorm is false on a qwen3 model")
	}
}

// toInt32 converts a token list for Prefill, which takes int32 ids where Forward takes int.
func toInt32(v []int) []int32 {
	out := make([]int32, len(v))
	for i, x := range v {
		out[i] = int32(x)
	}
	return out
}

// TestQwen3DecodeStepMatchesForward covers the third path that normalises: the batched decode.
//
// TestDecodeStepMatchesForward already holds this contract for the default fixture, which is a
// Llama graph with no per-head norm, so it cannot fail for the reason this one can. Two sequences
// with different histories make the batch advance positions independently, and the reference is
// one Forward per sequence -- the same shape as the prefill test above, one step later.
func TestQwen3DecodeStepMatchesForward(t *testing.T) {
	path := gguftest.Model{Arch: "qwen3", Layers: 2, Hidden: 32, Heads: 4, FFN: 64,
		QKNorm: true, NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	m, err := NewWithOptions(path, Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	cfg := m.Config()
	ids := fixtureTokens(cfg, 6)
	histories := [][]int32{{ids[0], ids[1]}, {ids[0], ids[1], ids[2], ids[3]}}
	step := ids[4:]

	want := make([][]float32, len(histories))
	for i, hist := range histories {
		cache, err := NewKVCache(cfg, cfg.ContextLen)
		if err != nil {
			t.Fatalf("cache: %v", err)
		}
		if _, err := m.Prefill(cache, hist, 0); err != nil {
			t.Fatalf("Prefill(%d): %v", i, err)
		}
		last, err := m.Forward(cache, int(step[i]), cache.Len())
		if err != nil {
			t.Fatalf("Forward(%d): %v", i, err)
		}
		want[i] = append([]float32(nil), last...)
	}

	caches := make([]Cache, len(histories))
	for i, hist := range histories {
		cache, err := NewKVCache(cfg, cfg.ContextLen)
		if err != nil {
			t.Fatalf("cache: %v", err)
		}
		if _, err := m.Prefill(cache, hist, 0); err != nil {
			t.Fatalf("Prefill(%d): %v", i, err)
		}
		caches[i] = cache
	}
	got, err := m.DecodeStep(caches, step[:len(histories)])
	if err != nil {
		t.Fatalf("DecodeStep: %v", err)
	}

	for i := range histories {
		row := got[i*cfg.VocabSize : (i+1)*cfg.VocabSize]
		var worst float64
		for j := range row {
			if d := math.Abs(float64(row[j] - want[i][j])); d > worst {
				worst = d
			}
		}
		if worst > 1e-3 {
			t.Errorf("sequence %d: batched decode differs from Forward by up to %v", i, worst)
		}
	}
}

// TestQwen3CachedPrefillMatchesAFreshOne covers the fourth and last path that normalises.
//
// When a whole prompt is served from the shared prefix cache, the last position's query is
// recomputed by attentionCached and rotated, without re-doing the keys -- they were normalised
// when they were first written. That is a different code path from any above, and a missing norm
// there would make a reused prompt answer differently from the same prompt seen for the first
// time, which is the one behaviour a prefix cache must not have.
//
// TestLogitsForCachedMatchesAFreshPrefill covers the same contract for attention bias, on a qwen2
// fixture, for the same reason: each architecture adds a step to this path.
func TestQwen3CachedPrefillMatchesAFreshOne(t *testing.T) {
	path := gguftest.Model{Arch: "qwen3", Layers: 2, Hidden: 32, Heads: 4, FFN: 64,
		QKNorm: true, NoEOS: true, ContextLen: 128}.Write(t, t.TempDir())
	m, err := NewWithOptions(path, Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	cfg := m.Config()
	const blockSize = 4
	pool, err := NewBlockPool(cfg, BlockPoolOptions{BlockSize: blockSize, Blocks: 64})
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	pool.EnablePrefixCache()

	prompt := make([]int32, 2*blockSize)
	for i := range prompt {
		prompt[i] = int32(1 + i%5)
	}

	fresh, err := m.PrefillWithPrefix(pool.New(), prompt, 0)
	if err != nil {
		t.Fatalf("first PrefillWithPrefix: %v", err)
	}
	fresh = append([]float32(nil), fresh...)

	second := pool.New()
	cached, err := m.PrefillWithPrefix(second, prompt, 0)
	if err != nil {
		t.Fatalf("second PrefillWithPrefix: %v", err)
	}
	cached = append([]float32(nil), cached...)
	if got := second.ReusedPositions(); got != len(prompt) {
		t.Fatalf("the second pass reused %d positions of %d; the cached path did not run",
			got, len(prompt))
	}

	var worst float64
	for i := range fresh {
		if d := math.Abs(float64(fresh[i] - cached[i])); d > worst {
			worst = d
		}
	}
	if worst > 1e-3 {
		t.Errorf("a reused prompt differs from a fresh one by up to %v", worst)
	}
}
