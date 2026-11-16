package llama

import (
	"math"
	"os"
	"testing"
)

// These tests need a real model for the same reason the engine tests do: the
// property under test is that restructuring the projections does not change the
// numbers, and a synthetic model's values are too uniform to expose a reordering
// that only breaks numerically on real weights.
//
// Set MODEL to a GGUF file to run them.

func loadReal(t *testing.T) *Model {
	t.Helper()
	p := os.Getenv("MODEL")
	if p == "" {
		t.Skip("set MODEL to a GGUF file")
	}
	m, err := New(p)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return m
}

// TestBatchMatchesSequential is the contract for Prefill: batching is a
// restructuring, not an approximation. The last position's logits must match
// what the per-token path produces for the same prefix.
func TestBatchMatchesSequential(t *testing.T) {
	m := loadReal(t)
	cfg := m.Config()

	tokens := []int32{1, 9038, 2501, 263, 931}

	// Sequential: one Forward per token, keeping the last row.
	seqCache, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatal(err)
	}
	var seqLast []float32
	for pos, tok := range tokens {
		last, err := m.Forward(seqCache, int(tok), pos)
		if err != nil {
			t.Fatal(err)
		}
		if pos == len(tokens)-1 {
			seqLast = append([]float32(nil), last...)
		}
	}

	// Batched: one call for the whole prompt.
	batchCache, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatal(err)
	}
	batchLast, err := m.Prefill(batchCache, tokens, 0)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	if len(batchLast) != cfg.VocabSize {
		t.Fatalf("prefill returned %d logits, want %d", len(batchLast), cfg.VocabSize)
	}
	if got, want := batchCache.Len(), len(tokens); got != want {
		t.Errorf("cache len = %d, want %d", got, want)
	}

	var maxDiff float64
	for i := range seqLast {
		d := math.Abs(float64(seqLast[i] - batchLast[i]))
		maxDiff = math.Max(maxDiff, d)
	}
	t.Logf("max logit difference: %.6f", maxDiff)
	if maxDiff > 1e-3 {
		t.Errorf("batched logits differ from sequential by %.6f", maxDiff)
	}
}

// TestBatchMatchesSequentialTokens checks the other half of the contract: the
// KV cache the batched pass wrote must be the same one the per-token pass would
// have written, so decoding after a prefill continues identically.
func TestBatchMatchesSequentialTokens(t *testing.T) {
	m := loadReal(t)
	cfg := m.Config()

	prompt := []int32{1, 9038, 2501, 263, 931}
	const steps = 4

	// Reference: prefill one token at a time, then decode greedily.
	seqCache, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatal(err)
	}
	var logits []float32
	for pos, tok := range prompt {
		if logits, err = m.Forward(seqCache, int(tok), pos); err != nil {
			t.Fatal(err)
		}
	}
	pos := len(prompt)
	var want []int32
	for range steps {
		tok := Greedy{}.Next(logits)
		want = append(want, tok)
		if logits, err = m.Forward(seqCache, int(tok), pos); err != nil {
			t.Fatal(err)
		}
		pos++
	}

	// Batched prefill, then the same greedy loop.
	batchCache, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatal(err)
	}
	logits, err = m.Prefill(batchCache, prompt, 0)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	pos = len(prompt)
	var got []int32
	for range steps {
		tok := Greedy{}.Next(logits)
		got = append(got, tok)
		if logits, err = m.Forward(batchCache, int(tok), pos); err != nil {
			t.Fatal(err)
		}
		pos++
	}

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token %d: batched prefill gave %d, sequential gave %d (%v vs %v)",
				i, got[i], want[i], got, want)
		}
	}
}

func TestPrefillRejectsBadInput(t *testing.T) {
	m := loadReal(t)
	cfg := m.Config()

	cache, _ := NewKVCache(cfg, cfg.ContextLen)
	if _, err := m.Prefill(cache, nil, 0); err == nil {
		t.Error("Prefill accepted an empty batch")
	}
	if _, err := m.Prefill(cache, []int32{int32(cfg.VocabSize)}, 0); err == nil {
		t.Error("Prefill accepted a token past the vocabulary")
	}
	if _, err := m.Prefill(cache, []int32{1, 2}, 0); err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	// The cache now holds two positions; a call claiming to start elsewhere must
	// be refused rather than silently overwriting.
	if _, err := m.Prefill(cache, []int32{3}, 5); err == nil {
		t.Error("Prefill accepted a start position ahead of the cache")
	}
	// Batches past the cache capacity are refused.
	if _, err := m.Prefill(cache, make([]int32, cfg.ContextLen), 0); err == nil {
		t.Error("Prefill accepted an oversized batch")
	}
}

func TestPrefillGrowsScratch(t *testing.T) {
	// A second, larger batch must work: the scratch buffer is grown on demand
	// rather than pinned to the first batch's size.
	m := loadReal(t)
	cfg := m.Config()

	small, _ := NewKVCache(cfg, cfg.ContextLen)
	if _, err := m.Prefill(small, []int32{1}, 0); err != nil {
		t.Fatalf("Prefill(1): %v", err)
	}
	big, _ := NewKVCache(cfg, cfg.ContextLen)
	n := min(8, cfg.ContextLen)
	ids := make([]int32, n)
	for i := range ids {
		ids[i] = int32(9000 + i)
	}
	logits, err := m.Prefill(big, ids, 0)
	if err != nil {
		t.Fatalf("Prefill(%d): %v", n, err)
	}
	if len(logits) != cfg.VocabSize {
		t.Errorf("returned %d logits, want %d", len(logits), cfg.VocabSize)
	}
	// Every value must be finite.
	for i, v := range logits {
		if v != v {
			t.Fatalf("logit[%d] is NaN", i)
		}
	}
}

// TestPrefillTinyModel runs the batched path on the synthetic model, so a
// regression is caught even when no checkpoint is configured.
func TestPrefillTinyModel(t *testing.T) {
	c := defaultTiny()
	path := buildTinyWeights(t, c)
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	tokens := []int32{0, 1, 2, 3}

	seqCache, err := NewKVCache(cfg, c.ctx)
	if err != nil {
		t.Fatal(err)
	}
	var want []float32
	for pos, tok := range tokens {
		last, err := m.Forward(seqCache, int(tok), pos)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
		if pos == len(tokens)-1 {
			want = append([]float32(nil), last...)
		}
	}

	batchCache, err := NewKVCache(cfg, c.ctx)
	if err != nil {
		t.Fatal(err)
	}
	got, err := m.Prefill(batchCache, tokens, 0)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	if got, want := batchCache.Len(), len(tokens); got != want {
		t.Errorf("cache len = %d, want %d", got, want)
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Errorf("logit[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

// TestPrefillStopsAtContextLimit checks that a batch longer than the remaining
// context is the caller's problem: Prefill refuses rather than truncating
// silently, because a truncated prefill would produce a plausible-looking
// continuation from the wrong prefix.
func TestPrefillStopsAtContextLimit(t *testing.T) {
	c := defaultTiny()
	path := buildTinyWeights(t, c)
	m, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	cfg := m.Config()
	cache, err := NewKVCache(cfg, 4)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Prefill(cache, make([]int32, 5), 0); err == nil {
		t.Error("Prefill accepted a batch larger than the cache")
	}
	if _, err := m.Prefill(cache, []int32{0, 1, 2, 3}, 0); err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	// Exactly full: a further position must be refused.
	if _, err := m.Prefill(cache, []int32{0}, 4); err == nil {
		t.Error("Prefill accepted a position past the cache")
	}
}

// benchPrompt returns a model and a prompt long enough for prefill to dominate.
func benchPrompt(b *testing.B) (*Model, []int32) {
	b.Helper()
	m := loadRealB(b)
	cfg := m.Config()
	n := min(64, cfg.ContextLen/2)
	if n < 2 {
		b.Skipf("model context %d is too small to benchmark prefill", cfg.ContextLen)
	}
	ids := make([]int32, n)
	for i := range ids {
		ids[i] = int32(100 + i)
	}
	return m, ids
}

func loadRealB(b *testing.B) *Model {
	b.Helper()
	p := os.Getenv("MODEL")
	if p == "" {
		b.Skip("set MODEL to a GGUF file")
	}
	m, err := New(p)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	return m
}

// BenchmarkPrefillSequential is the per-token baseline the batched path is
// measured against: one Forward call per prompt token.
func BenchmarkPrefillSequential(b *testing.B) {
	m, ids := benchPrompt(b)
	cfg := m.Config()
	cache, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for range b.N {
		cache.Reset()
		for pos, tok := range ids {
			if _, err := m.Forward(cache, int(tok), pos); err != nil {
				b.Fatal(err)
			}
		}
	}
}

// BenchmarkPrefillBatched is the batched path over the same prompt.
func BenchmarkPrefillBatched(b *testing.B) {
	m, ids := benchPrompt(b)
	cfg := m.Config()
	cache, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for range b.N {
		cache.Reset()
		if _, err := m.Prefill(cache, ids, 0); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkDecode measures one token against an already-filled cache, which is
// the cost every generated token pays and the batched prefill does not change.
func BenchmarkDecode(b *testing.B) {
	m, ids := benchPrompt(b)
	cfg := m.Config()
	cache, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		b.Fatal(err)
	}
	if _, err := m.Prefill(cache, ids, 0); err != nil {
		b.Fatal(err)
	}
	pos := len(ids)
	b.ResetTimer()
	for range b.N {
		if _, err := m.Forward(cache, int(ids[0]), pos); err != nil {
			b.Fatal(err)
		}
		pos++
		if pos >= cache.Cap() {
			cache.Reset()
			if _, err := m.Prefill(cache, ids, 0); err != nil {
				b.Fatal(err)
			}
			pos = len(ids)
		}
	}
}
