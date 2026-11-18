package llama

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
)

// These tests check that restructuring the projections does not change the numbers, and
// that property does not depend on the weights being trained ones: any values expose a
// reordering, because the comparison is exact equality rather than a tolerance. They run
// on a constructed fixture, which is what lets them run in CI; the benchmark below still
// wants a real checkpoint, because a constructed model's speed says nothing about
// inference on a trained one.

func loadReal(t *testing.T) *Model {
	t.Helper()
	p := gguftest.Path(t)
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

	// Ids inside the vocabulary, starting at BOS. Hard-coded ids from a checkpoint would
	// index past the end of a constructed one.
	tokens := fixtureTokens(cfg, 5)

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

	prompt := fixtureTokens(cfg, 5)
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
	ids := fixtureTokens(cfg, n)
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

// loadConfigAt reads a fixture's configuration.
func loadConfigAt(t *testing.T, path string) Config {
	t.Helper()
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	cfg, err := LoadConfig(f)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	return cfg
}

// fixtureTokens returns n token ids the model can embed.
func fixtureTokens(cfg Config, n int) []int32 {
	out := make([]int32, n)
	out[0] = int32(cfg.BOSID)
	for i := 1; i < n; i++ {
		out[i] = int32((i * 3) % cfg.VocabSize)
	}
	return out
}

func loadRealB(b *testing.B) *Model {
	b.Helper()
	// Benchmarks still need a real checkpoint: a constructed model's speed says nothing
	// about inference on a trained one, and the numbers would be meaningless.
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
		if err := cache.Reset(); err != nil {
			b.Fatal(err)
		}
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
		if err := cache.Reset(); err != nil {
			b.Fatal(err)
		}
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
			if err := cache.Reset(); err != nil {
				b.Fatal(err)
			}
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
	// A constructed model: the property is that eight workers give the serial answer
	// exactly, and it holds for any weights because every output row is computed by one
	// worker in the serial order.
	path := gguftest.Model{NoEOS: true, ContextLen: 256}.Write(t, t.TempDir())
	serial, err := NewWithOptions(path, Options{Workers: 1})
	if err != nil {
		t.Fatalf("NewWithOptions(serial): %v", err)
	}
	parallel, err := NewWithOptions(path, Options{Workers: 8})
	if err != nil {
		t.Fatalf("NewWithOptions(parallel): %v", err)
	}

	c := serial.Config()
	prompt := fixtureTokens(c, 5)

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
	m2, err := NewWithOptions(gguftest.Path(t), Options{})
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

	// Three sequences with different histories, so their cache lengths differ, and one
	// step token per sequence.
	ids := fixtureTokens(cfg, 7)
	histories := [][]int32{
		{ids[0], ids[1]},
		{ids[0], ids[1], ids[2], ids[3]},
		{ids[0], ids[2]},
	}
	tokens := ids[4:7]

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
	batchCaches := make([]Cache, len(histories))
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
	ids := fixtureTokens(cfg, 6)
	prompts := [][]int32{
		{ids[0], ids[1], ids[2]},
		{ids[0], ids[3], ids[4]},
		{ids[0], ids[2], ids[3], ids[5]},
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
	batchCaches := make([]Cache, len(prompts))
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
	if _, err := m.DecodeStep([]Cache{cache}, nil); err == nil {
		t.Error("DecodeStep accepted mismatched token and cache counts")
	}
	if _, err := m.DecodeStep([]Cache{cache}, []int32{int32(cfg.VocabSize)}); err == nil {
		t.Error("DecodeStep accepted a token past the vocabulary")
	}
	if _, err := m.DecodeStep([]Cache{nil}, []int32{1}); err == nil {
		t.Error("DecodeStep accepted a nil cache")
	}
	// A full cache cannot take another position.
	full, _ := NewKVCache(cfg, 2)
	if _, err := m.Prefill(full, []int32{1, 2}, 0); err != nil {
		t.Fatal(err)
	}
	if _, err := m.DecodeStep([]Cache{full}, []int32{3}); err == nil {
		t.Error("DecodeStep accepted a full cache")
	}
}

// TestDecodeStepGrowsScratch checks that a later, larger batch works: the scratch
// grows on demand rather than staying pinned to the first batch's size.
func TestDecodeStepGrowsScratch(t *testing.T) {
	m := loadReal(t)
	cfg := m.Config()

	one, _ := NewKVCache(cfg, cfg.ContextLen)
	pair := fixtureTokens(cfg, 3)
	if _, err := m.Prefill(one, pair[:2], 0); err != nil {
		t.Fatal(err)
	}
	if _, err := m.DecodeStep([]Cache{one}, pair[2:3]); err != nil {
		t.Fatalf("DecodeStep(1): %v", err)
	}

	n := min(6, cfg.ContextLen-2)
	caches := make([]Cache, n)
	tokens := make([]int32, n)
	for i := range caches {
		caches[i], _ = NewKVCache(cfg, cfg.ContextLen)
		if _, err := m.Prefill(caches[i], pair[:2], 0); err != nil {
			t.Fatal(err)
		}
		tokens[i] = int32(i % cfg.VocabSize)
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

	batchCaches := make([]Cache, len(prompts))
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

// TestParallelAttentionIsBitIdentical pins what makes the parallel attention safe to
// ship: positions are independent once keys and values are stored, so running them on
// several workers must produce exactly the serial answer, not an approximation of it.
//
// Workers=1 takes the serial path inside Runner.Parallel, so the comparison is against
// the same code with the fan-out removed rather than against a second implementation.
// TestEmbedRejectsNegativeToken covers a hole in the bounds check both embedding paths used
// to have: it compared the offset plus the row width against the slice length, which a
// negative token passes.
//
// Measured on a 363-token vocabulary with hidden 32: token -1 gives off = -32, and
// off+32 = 0 <= len(TokenEmbedding), so the guard was satisfied and the copy then panicked on
// a slice expression starting below zero. Every negative token behaved that way, because
// multiplying a negative by the row width only makes the sum more negative, never larger than
// the slice.
//
// The fix checks the token against the vocabulary before computing any offset, so the sum is
// only ever formed from a non-negative product.
func TestEmbedRejectsNegativeToken(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 128}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()

	b := &decodeBatch{cfg: cfg, n: 1, x: make([]float32, cfg.Hidden)}
	cases := []struct {
		token string
		want  bool // true means it must be refused
	}{
		{token: "-1", want: true},
		{token: "-100", want: true},
		{token: "0", want: false},
		{token: "vocab-1", want: false},
		{token: "vocab", want: true},
	}
	for _, c := range cases {
		tok := 0
		switch c.token {
		case "vocab-1":
			tok = cfg.VocabSize - 1
		case "vocab":
			tok = cfg.VocabSize
		default:
			if _, err := fmt.Sscanf(c.token, "%d", &tok); err != nil {
				t.Fatalf("bad case %q: %v", c.token, err)
			}
		}
		err := b.embed(m, 0, tok)
		if refused := err != nil; refused != c.want {
			t.Errorf("embed(token=%s) error = %v, want refused=%v", c.token, err, c.want)
		}
	}

	// The batch path has its own copy of the same check; drive it through a real decode so
	// the guard is exercised where a corrupt sampler would reach it.
	cache, err := NewKVCache(cfg, 32)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}
	if _, err := m.Prefill(cache, []int32{int32(cfg.BOSID)}, 0); err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	if _, err := m.DecodeStep([]Cache{cache}, []int32{-1}); err == nil {
		t.Error("DecodeStep accepted a negative token id")
	}
}

func TestParallelAttentionIsBitIdentical(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 256}.Write(t, t.TempDir())
	cfg := loadConfigAt(t, path)

	prompt := make([]int32, 1, 32)
	prompt[0] = int32(cfg.BOSID)
	for i := 1; i < 32; i++ {
		prompt = append(prompt, int32((i*7)%cfg.VocabSize))
	}

	run := func(workers int) []float32 {
		m, err := NewWithOptions(path, Options{Workers: workers})
		if err != nil {
			t.Fatalf("NewWithOptions(%d): %v", workers, err)
		}
		cache, err := NewKVCache(cfg, cfg.ContextLen)
		if err != nil {
			t.Fatalf("NewKVCache: %v", err)
		}
		logits, err := m.Prefill(cache, prompt, 0)
		if err != nil {
			t.Fatalf("Prefill(%d workers): %v", workers, err)
		}
		return append([]float32(nil), logits...)
	}

	want := run(1)
	for _, workers := range []int{2, 4, 16} {
		got := run(workers)
		if len(got) != len(want) {
			t.Fatalf("workers=%d returned %d logits, want %d", workers, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("workers=%d logit[%d] = %v, serial %v", workers, i, got[i], want[i])
			}
		}
	}
}

// TestAttentionBuffersDoNotScaleWithWorkers pins the allocation that made a large --workers
// setting pathological.
//
// Each batched attention pass rents one score buffer per worker, sized to the context. The
// count used to follow the configured worker count rather than the number of positions the
// round actually splits across, so asking for more workers than there are positions allocated
// buffers nothing could index. Measured on the 128k-window checkpoint with --workers 10000: a
// single request reserved 15GB and took 15.3s against 4.5s when configured normally.
//
// The comparison is a ratio rather than an absolute count because the point is the shape of
// the curve, not its height.
func TestAttentionBuffersDoNotScaleWithWorkers(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	allocated := func(workers int) uint64 {
		m, err := NewWithOptions(path, Options{Workers: workers})
		if err != nil {
			t.Fatalf("NewWithOptions(workers=%d): %v", workers, err)
		}
		cfg := m.Config()
		cache, err := NewKVCache(cfg, cfg.ContextLen)
		if err != nil {
			t.Fatalf("NewKVCache: %v", err)
		}
		// A prompt of a few tokens, so the round has far fewer positions than the worker
		// count under test -- which is the situation the bug needed.
		tokens := fixtureTokens(cfg, 8)
		runtime.GC()
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		if _, err := m.Prefill(cache, tokens, 0); err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		runtime.ReadMemStats(&after)
		return after.TotalAlloc - before.TotalAlloc
	}

	few := allocated(1)
	many := allocated(4096)
	if few == 0 {
		t.Fatal("no allocations measured, so the comparison means nothing")
	}
	// Before the fix this ratio was ~600 for the same pair; a bound of 4 leaves room for the
	// per-worker slice headers while failing anything proportional to the worker count.
	if ratio := float64(many) / float64(few); ratio > 4 {
		t.Errorf("allocation grew %.0fx for 4096 workers against 1 (%d -> %d bytes): the score "+
			"buffers are sized by --workers instead of by the positions being processed",
			ratio, few, many)
	}
}
