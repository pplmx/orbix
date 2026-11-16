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

// TestParallelMatchesSerialWeights is the end-to-end form of the Runner
// contract: a model built with several workers must produce exactly the logits a
// serial one does. The kernel tests pin the individual products; this pins the
// forward pass, where a buffer shared between two projections would show up as a
// wrong value only under concurrency.
func TestParallelMatchesSerialWeights(t *testing.T) {
	path := os.Getenv("MODEL")
	if path == "" {
		t.Skip("set MODEL to a GGUF file")
	}
	serial, err := NewWithOptions(path, Options{Workers: 1})
	if err != nil {
		t.Fatalf("NewWithOptions(serial): %v", err)
	}
	parallel, err := NewWithOptions(path, Options{Workers: 8})
	if err != nil {
		t.Fatalf("NewWithOptions(parallel): %v", err)
	}

	c := serial.Config()
	prompt := []int32{1, 9038, 2501, 263, 931}

	seqCache, err := NewKVCache(c, c.ContextLen)
	if err != nil {
		t.Fatal(err)
	}
	for pos, tok := range prompt {
		if _, err := serial.Forward(seqCache, int(tok), pos); err != nil {
			t.Fatal(err)
		}
	}
	want := append([]float32(nil), serial.Logits()...)

	batchCache, err := NewKVCache(c, c.ContextLen)
	if err != nil {
		t.Fatal(err)
	}
	got, err := parallel.Prefill(batchCache, prompt, 0)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("logit[%d] = %v with 8 workers, %v serial", i, got[i], want[i])
		}
	}

	// Decode a few tokens on both and compare each step, so the single-token
	// path is covered too and not just the batched one.
	pos := len(prompt)
	for range 3 {
		tok := Greedy{}.Next(want)
		wantNext, err := serial.Forward(seqCache, int(tok), pos)
		if err != nil {
			t.Fatal(err)
		}
		want = append([]float32(nil), wantNext...)

		tok2 := Greedy{}.Next(got)
		if tok2 != tok {
			t.Fatalf("token at %d: parallel model chose %d, serial chose %d", pos, tok2, tok)
		}
		gotNext, err := parallel.Forward(batchCache, int(tok), pos)
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if gotNext[i] != want[i] {
				t.Fatalf("decode logit[%d] = %v with 8 workers, %v serial", i, gotNext[i], want[i])
			}
		}
		got = append([]float32(nil), gotNext...)
		pos++
	}
}

func TestNewWithOptionsZeroWorkersUsesGOMAXPROCS(t *testing.T) {
	loadReal(t) // skips without a model
	m2, err := NewWithOptions(os.Getenv("MODEL"), Options{})
	if err != nil {
		t.Fatalf("NewWithOptions: %v", err)
	}
	// Both defaults must agree, which means the CLI path and an explicit zero
	// are the same configuration.
	if got, want := m2.run.Workers(), workers(0); got != want {
		t.Errorf("default workers = %d, want %d", got, want)
	}
}

// TestDecodeStepMatchesForward is the contract for DecodeStep: advancing several
// sequences together must produce exactly the logits each would get alone. The
// sequences deliberately have different cache lengths, because a batched decode
// that used one shared position would look correct only while every sequence
// happened to be at the same place.
func TestDecodeStepMatchesForward(t *testing.T) {
	m := loadReal(t)
	cfg := m.Config()

	// Three sequences with different histories, so their cache lengths differ.
	histories := [][]int32{
		{1, 9038},
		{1, 9038, 2501, 263},
		{1, 29871},
	}
	tokens := []int32{931, 3000, 13}

	// Reference: one Forward per sequence.
	want := make([][]float32, len(histories))
	seqCaches := make([]*KVCache, len(histories))
	for i, hist := range histories {
		cache, err := NewKVCache(cfg, cfg.ContextLen)
		if err != nil {
			t.Fatal(err)
		}
		seqCaches[i] = cache
		if _, err := m.Prefill(cache, hist, 0); err != nil {
			t.Fatalf("Prefill(%d): %v", i, err)
		}
		last, err := m.Forward(cache, int(tokens[i]), cache.Len())
		if err != nil {
			t.Fatalf("Forward(%d): %v", i, err)
		}
		want[i] = append([]float32(nil), last...)
	}

	// Batched: one DecodeStep for all three.
	batchCaches := make([]*KVCache, len(histories))
	for i, hist := range histories {
		cache, err := NewKVCache(cfg, cfg.ContextLen)
		if err != nil {
			t.Fatal(err)
		}
		batchCaches[i] = cache
		if _, err := m.Prefill(cache, hist, 0); err != nil {
			t.Fatalf("Prefill(%d): %v", i, err)
		}
	}
	got, err := m.DecodeStep(batchCaches, tokens)
	if err != nil {
		t.Fatalf("DecodeStep: %v", err)
	}
	if len(got) != len(histories)*cfg.VocabSize {
		t.Fatalf("DecodeStep returned %d values, want %d", len(got), len(histories)*cfg.VocabSize)
	}
	for i := range histories {
		row := got[i*cfg.VocabSize : (i+1)*cfg.VocabSize]
		for j := range want[i] {
			if row[j] != want[i][j] {
				t.Fatalf("sequence %d logit[%d] = %v, want %v", i, j, row[j], want[i][j])
			}
		}
		if got, want := batchCaches[i].Len(), seqCaches[i].Len(); got != want {
			t.Errorf("sequence %d cache len = %d, want %d", i, got, want)
		}
	}
}

// TestDecodeStepContinuesLikeForward runs several greedy steps through both paths
// and requires the generated tokens to stay identical, so the caches written by a
// batched decode keep matching what a sequential one would hold.
func TestDecodeStepContinuesLikeForward(t *testing.T) {
	m := loadReal(t)
	cfg := m.Config()
	prompts := [][]int32{
		{1, 9038, 2501},
		{1, 31373, 612},
		{1, 29871, 13, 29906},
	}
	const steps = 5

	// Sequential reference.
	seqCaches := make([]*KVCache, len(prompts))
	seqTokens := make([][]int32, len(prompts))
	for i, p := range prompts {
		cache, err := NewKVCache(cfg, cfg.ContextLen)
		if err != nil {
			t.Fatal(err)
		}
		seqCaches[i] = cache
		logits, err := m.Prefill(cache, p, 0)
		if err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		logits = append([]float32(nil), logits...)
		for range steps {
			tok := Greedy{}.Next(logits)
			seqTokens[i] = append(seqTokens[i], tok)
			next, err := m.Forward(cache, int(tok), cache.Len())
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			logits = append([]float32(nil), next...)
		}
	}

	// Batched.
	batchCaches := make([]*KVCache, len(prompts))
	pending := make([]float32, len(prompts)*cfg.VocabSize)
	batchTokens := make([][]int32, len(prompts))
	for i, p := range prompts {
		cache, err := NewKVCache(cfg, cfg.ContextLen)
		if err != nil {
			t.Fatal(err)
		}
		batchCaches[i] = cache
		logits, err := m.Prefill(cache, p, 0)
		if err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		copy(pending[i*cfg.VocabSize:], logits)
	}
	next := make([]int32, len(prompts))
	for range steps {
		for i := range prompts {
			next[i] = Greedy{}.Next(pending[i*cfg.VocabSize : (i+1)*cfg.VocabSize])
			batchTokens[i] = append(batchTokens[i], next[i])
		}
		out, err := m.DecodeStep(batchCaches, next)
		if err != nil {
			t.Fatalf("DecodeStep: %v", err)
		}
		copy(pending, out)
	}

	for i := range prompts {
		for j := range seqTokens[i] {
			if batchTokens[i][j] != seqTokens[i][j] {
				t.Fatalf("sequence %d token %d: batched %d, sequential %d (%v vs %v)",
					i, j, batchTokens[i][j], seqTokens[i][j], batchTokens[i], seqTokens[i])
			}
		}
	}
}

func TestDecodeStepRejectsBadInput(t *testing.T) {
	m := loadReal(t)
	cfg := m.Config()

	if _, err := m.DecodeStep(nil, nil); err == nil {
		t.Error("DecodeStep accepted an empty batch")
	}
	cache, _ := NewKVCache(cfg, cfg.ContextLen)
	if _, err := m.DecodeStep([]*KVCache{cache}, nil); err == nil {
		t.Error("DecodeStep accepted mismatched token and cache counts")
	}
	if _, err := m.DecodeStep([]*KVCache{cache}, []int32{int32(cfg.VocabSize)}); err == nil {
		t.Error("DecodeStep accepted a token past the vocabulary")
	}
	if _, err := m.DecodeStep([]*KVCache{nil}, []int32{1}); err == nil {
		t.Error("DecodeStep accepted a nil cache")
	}
	// A full cache cannot take another position.
	full, _ := NewKVCache(cfg, 2)
	if _, err := m.Prefill(full, []int32{1, 2}, 0); err != nil {
		t.Fatal(err)
	}
	if _, err := m.DecodeStep([]*KVCache{full}, []int32{3}); err == nil {
		t.Error("DecodeStep accepted a full cache")
	}
}

// TestDecodeStepGrowsScratch checks that a later, larger batch works: the scratch
// grows on demand rather than staying pinned to the first batch's size.
func TestDecodeStepGrowsScratch(t *testing.T) {
	m := loadReal(t)
	cfg := m.Config()

	one, _ := NewKVCache(cfg, cfg.ContextLen)
	if _, err := m.Prefill(one, []int32{1, 9038}, 0); err != nil {
		t.Fatal(err)
	}
	if _, err := m.DecodeStep([]*KVCache{one}, []int32{2501}); err != nil {
		t.Fatalf("DecodeStep(1): %v", err)
	}

	n := min(6, cfg.ContextLen-2)
	caches := make([]*KVCache, n)
	tokens := make([]int32, n)
	for i := range caches {
		caches[i], _ = NewKVCache(cfg, cfg.ContextLen)
		if _, err := m.Prefill(caches[i], []int32{1, 9038}, 0); err != nil {
			t.Fatal(err)
		}
		tokens[i] = int32(2501 + i)
	}
	out, err := m.DecodeStep(caches, tokens)
	if err != nil {
		t.Fatalf("DecodeStep(%d): %v", n, err)
	}
	if len(out) != n*cfg.VocabSize {
		t.Errorf("returned %d values, want %d", len(out), n*cfg.VocabSize)
	}
	for i, v := range out {
		if v != v {
			t.Fatalf("logit[%d] is NaN", i)
		}
	}
}

// TestDecodeStepTinyModel covers the batched path on the synthetic model, so a
// regression is caught without a checkpoint configured.
func TestDecodeStepTinyModel(t *testing.T) {
	c := defaultTiny()
	path := buildTinyWeights(t, c)
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	prompts := [][]int32{{0, 1}, {0, 1, 2}}
	tokens := []int32{3, 4}

	want := make([][]float32, len(prompts))
	seqCaches := make([]*KVCache, len(prompts))
	for i, p := range prompts {
		cache, err := NewKVCache(cfg, c.ctx)
		if err != nil {
			t.Fatal(err)
		}
		seqCaches[i] = cache
		if _, err := m.Prefill(cache, p, 0); err != nil {
			t.Fatal(err)
		}
		last, err := m.Forward(cache, int(tokens[i]), cache.Len())
		if err != nil {
			t.Fatal(err)
		}
		want[i] = append([]float32(nil), last...)
	}

	batchCaches := make([]*KVCache, len(prompts))
	for i, p := range prompts {
		cache, _ := NewKVCache(cfg, c.ctx)
		batchCaches[i] = cache
		if _, err := m.Prefill(cache, p, 0); err != nil {
			t.Fatal(err)
		}
	}
	got, err := m.DecodeStep(batchCaches, tokens)
	if err != nil {
		t.Fatalf("DecodeStep: %v", err)
	}
	for i := range want {
		row := got[i*cfg.VocabSize : (i+1)*cfg.VocabSize]
		for j := range want[i] {
			if row[j] != want[i][j] {
				t.Errorf("sequence %d logit[%d] = %v, want %v", i, j, row[j], want[i][j])
			}
		}
	}
}
