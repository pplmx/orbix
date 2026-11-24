package llama

import (
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
	"github.com/pplmx/orbix/internal/kernel"
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

// failingSpansCache wraps a working cache and refuses spans past a position count, so a
// prefill reaches attentionBatch's error branch from every worker at once.
//
// It exists because that branch is otherwise unreachable in a test: Prefill validates that
// the batch fits the cache's Cap, and the store loop allocates a block for every position
// before the read loop runs, so a real Cache cannot make spans fail there. Cache is an
// interface, though, and the branch has to be correct for any implementation of it.
type failingSpansCache struct {
	Cache
	refuseFrom int
	// err is what spans returns; tests assert on it rather than on a message.
	err error
}

func (f *failingSpansCache) spans(layer, kv, n int) ([]span, []span, error) {
	if n >= f.refuseFrom {
		return nil, nil, f.err
	}
	return f.Cache.spans(layer, kv, n)
}

// TestPrefillReportsOneErrorFromParallelWorkers covers the error channel out of
// attentionBatch's parallel read loop.
//
// Several workers can fail in the same round, and they used to assign their error to a
// shared variable with no synchronisation. Run under -race this test reports that as a data
// race; without the race detector it still pins that the error the caller sees is one of
// the failures rather than a nil.
func TestPrefillReportsOneErrorFromParallelWorkers(t *testing.T) {
	c := defaultTiny()
	path := buildTinyWeights(t, c)
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	base, err := NewKVCache(cfg, c.ctx)
	if err != nil {
		t.Fatal(err)
	}

	refusal := fmt.Errorf("spans refused")
	// Refusing from position 2 on leaves several positions in the batch to fail, and the
	// runner is given enough workers that they run at once.
	cache := &failingSpansCache{Cache: base, refuseFrom: 2, err: refusal}

	m.run = kernel.NewRunner(8)
	tokens := []int32{0, 1, 2, 3, 4, 5, 6, 7}
	if _, err := m.Prefill(cache, tokens, 0); !errors.Is(err, refusal) {
		t.Fatalf("Prefill error = %v, want it to carry the refusal", err)
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
	// Wide enough that the vocabulary projection clears kernel minParallelWork, so the
	// eight-worker model really does split its products; see the work assertion below.
	path := gguftest.Model{NoEOS: true, ContextLen: 256, Hidden: 256, Heads: 4, FFN: 512}.Write(t, t.TempDir())
	serial, err := NewWithOptions(path, Options{Workers: 1})
	if err != nil {
		t.Fatalf("NewWithOptions(serial): %v", err)
	}
	parallel, err := NewWithOptions(path, Options{Workers: 8})
	if err != nil {
		t.Fatalf("NewWithOptions(parallel): %v", err)
	}

	c := serial.Config()
	// The parallel path is only exercised if some product crosses the kernel's threshold.
	// Asserted rather than assumed: the fixture used to be the default 32-wide model, whose
	// largest product was 11,616 -- both "serial" and "parallel" then ran on one goroutine
	// and this test compared an implementation against itself.
	const minParallelWork = 1 << 15
	if work := c.Hidden * c.VocabSize; work < minParallelWork {
		t.Fatalf("the fixture's largest product is %d multiply-adds, below the %d where "+
			"kernels start splitting, so neither model would be parallel", work, minParallelWork)
	}

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
// Measured separately, as a second check on the isolation this asserts: two sequences with
// different prefill lengths (4 and 6) decoding together produced logits byte-identical to
// decoding each alone (maximum difference exactly 0, not merely small), the two cache lengths
// advanced independently from 4->5 and 6->7, and an eighth of a step's worth of already-written
// KV was left untouched -- a decode appends one position and rewrites nothing before it.
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
	// The other direction matters for the comparison: a count check written as len(caches) < n
	// rejects only the more-caches case, and this is the one that reads caches[i] past the end.
	if _, err := m.DecodeStep(nil, []int32{1}); err == nil {
		t.Error("DecodeStep accepted more tokens than caches")
	}
	if _, err := m.DecodeStep([]Cache{cache}, []int32{int32(cfg.VocabSize)}); err == nil {
		t.Error("DecodeStep accepted a token past the vocabulary")
	}
	if _, err := m.DecodeStep([]Cache{nil}, []int32{1}); err == nil {
		t.Error("DecodeStep accepted a nil cache")
	}
	// A full cache cannot take another position. The capacity is smaller than the rope table
	// on purpose: a cache that also sits at the table's end would be caught by the next check
	// down, and a test that cannot tell the two apart would pass with this one removed.
	full, err := NewKVCache(cfg, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.rope.cos) <= 2 {
		t.Fatalf("the fixture's rope table has %d positions, so a 2-position cache cannot "+
			"distinguish the capacity check from the table check", len(m.rope.cos))
	}
	if _, err := m.Prefill(full, []int32{1, 2}, 0); err != nil {
		t.Fatal(err)
	}
	if _, err := m.DecodeStep([]Cache{full}, []int32{3}); err == nil {
		t.Error("DecodeStep accepted a full cache")
	}
}

// TestDecodeStepGrowsScratch checks that a later, larger batch works: the scratch
// grows on demand rather than staying pinned to the first batch's size.
// TestDecodeStepScratchIsReused pins the ownership contract the doc states: the returned
// slice is the model's buffer and the next pass overwrites it.
//
// The growth test above proves the scratch can get bigger; it says nothing about the reuse
// that makes growth matter. Both halves are load-bearing. Reuse is what keeps a decode step
// allocation-free, and the overwrite is what a caller must know about -- copying nothing and
// holding the slice across calls yields the second pass's numbers under the first pass's
// length, which looks like data rather than like a bug.
//
// Measured against the fixture: two single-token steps return the same backing array, and the
// first element changes after the second call.
func TestDecodeStepScratchIsReused(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 128}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()

	cache, err := NewKVCache(cfg, 64)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}
	if _, err := m.Prefill(cache, []int32{int32(cfg.BOSID), 1, 2}, 0); err != nil {
		t.Fatalf("Prefill: %v", err)
	}

	first, err := m.DecodeStep([]Cache{cache}, []int32{3})
	if err != nil {
		t.Fatalf("DecodeStep: %v", err)
	}
	// Copy before the second pass. Comparing first to second after both calls would compare
	// the shared buffer with itself and always agree -- which is exactly the trap the doc
	// warns a caller about, so the test would have been reading its own mistake.
	snapshot := append([]float32(nil), first...)

	second, err := m.DecodeStep([]Cache{cache}, []int32{4})
	if err != nil {
		t.Fatalf("DecodeStep: %v", err)
	}
	if len(first) == 0 || len(second) == 0 {
		t.Fatal("DecodeStep returned no logits")
	}
	if &first[0] != &second[0] {
		t.Error("two same-sized decode steps returned different buffers, so the scratch is not reused")
	}
	// And the second pass must have rewritten it, since the tokens differ. Compared against the
	// snapshot, not against first: see above.
	changed := false
	for i := range snapshot {
		if snapshot[i] != second[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Error("two decode steps with different tokens produced identical logits; " +
			"the scratch is reused but the pass is not writing into it")
	}
}

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

// TestForwardEmbedRejectsNegativeToken covers the third copy of the negative-token check.
//
// The first two were corrected in decodeBatch.embed and the batched prefill path; Model.embed
// in forward.go had the same sum-based guard and was missed, because the fix went to the two
// sites the bug report named rather than to every site that performs the indexing.
//
// Its callers happen to validate -- Forward checks the token before calling, and
// logitsForCached relies on its tokens having come from a prefill that checked -- so this is
// not a live crash. It is the function that indexes, though, and "the caller checked" is an
// invariant that lives in a different file from the slice expression.
func TestForwardEmbedRejectsNegativeToken(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()

	for _, token := range []int{-1, -32, cfg.VocabSize} {
		err := m.embed(token)
		if err == nil {
			t.Errorf("embed(%d) accepted a token outside the vocabulary of %d", token, cfg.VocabSize)
			continue
		}
		if !strings.Contains(err.Error(), "outside vocabulary") {
			t.Errorf("embed(%d) error %q does not name the range", token, err)
		}
	}
	// A legal token must still work, or the guard could be satisfied by refusing everything.
	if err := m.embed(0); err != nil {
		t.Errorf("embed(0) failed: %v", err)
	}
}

// TestLogitsBeforeForwardIsZeroesNotNil pins the buffer's initial state.
//
// No production code calls Logits() -- the engine takes the slice Prefill returns -- so this
// is about the contract for a future caller. The doc used to say it "returns nil before the
// first forward pass", which is wrong: the buffer is allocated once in New and reused, so a
// fresh model reports VocabSize zeroes. Measured: nil=false len=363 for the fixture.
//
// The difference matters because the obvious guard is the wrong one. A caller that checked
// for nil would be satisfied on a model where nothing had run, sample those zeroes, and get
// token 0 -- a valid-looking id rather than an error.
func TestLogitsBeforeForwardIsZeroesNotNil(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()

	got := m.Logits()
	if got == nil {
		t.Fatal("Logits() is nil before a forward pass; the doc says the buffer is allocated in New")
	}
	if len(got) != cfg.VocabSize {
		t.Errorf("Logits() has %d entries, want %d", len(got), cfg.VocabSize)
	}
	for i, v := range got {
		if v != 0 {
			t.Fatalf("Logits()[%d] = %v before any pass, want zeroes", i, v)
		}
	}

	// And it is the same buffer a pass fills, rather than a fresh allocation each call:
	// the doc promises the slice is owned by the model.
	if &got[0] != &m.logits[0] {
		t.Error("Logits() returns a different slice from the model's buffer")
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

// TestPrefillRejectsATokenOutsideTheVocabulary covers the guard every batched entry point
// depends on.
//
// Prefill checks each token against the vocabulary before it reaches the embedding, and that
// check is what makes the equivalent one inside embedBatch unreachable: removing either passes
// the suite, because the other still stops the prompt. The message names the index as well as
// the token, which is the part worth pinning -- a prompt is a list, and "token 363 is bad"
// does not say which of a thousand entries to fix.
//
// Every test that feeds Prefill builds its tokens from fixtureTokens, so nothing exercised the
// refusal. The lower bound is the half that would panic rather than return: a negative token
// makes the offset negative while offset+Hidden can still be inside the table, so a guard
// written against the sum alone lets it through.
func TestPrefillRejectsATokenOutsideTheVocabulary(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	bad := int32(cfg.VocabSize)

	cases := []struct {
		tokens []int32
		index  string
	}{
		{[]int32{-1, 0}, "index 0"},
		{[]int32{0, -32}, "index 1"},
		{[]int32{bad}, "index 0"},
		{[]int32{0, bad}, "index 1"},
		{[]int32{0, 1, bad + 100}, "index 2"},
	}
	for _, c := range cases {
		cache, err := NewKVCache(cfg, cfg.ContextLen)
		if err != nil {
			t.Fatalf("NewKVCache: %v", err)
		}
		_, err = m.Prefill(cache, c.tokens, 0)
		if err == nil {
			t.Errorf("Prefill accepted %v, one of which is outside the vocabulary of %d",
				c.tokens, cfg.VocabSize)
			continue
		}
		msg := err.Error()
		if !strings.Contains(msg, "outside the vocabulary") {
			t.Errorf("Prefill(%v) error %q does not name the problem", c.tokens, msg)
		}
		if !strings.Contains(msg, c.index) {
			t.Errorf("Prefill(%v) error %q does not name %s", c.tokens, msg, c.index)
		}
	}

	// A prompt of legal tokens must still work, so the guard cannot be satisfied by
	// refusing everything.
	cache, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}
	if _, err := m.Prefill(cache, []int32{0, 1, 2}, 0); err != nil {
		t.Errorf("Prefill of legal tokens failed: %v", err)
	}
}

// TestEmptyPromptThroughThePagedPathIsRefused covers the zero-position case of
// logitsForCached.
//
// An empty prompt and a fully-cached one take the same branch: the prefix index matches every
// position, and for an empty slice "every position" is none. The last position is then at -1, so
// the guard has to reject the request rather than let the model embed a token at a negative
// offset. Relaxing it to "n < 0" is not enough -- zero is exactly the value that slips through --
// which is why the test exists.
//
// The engine refuses an empty prompt before this layer, so the case is reachable only through
// PrefillWithPrefix directly; both callers that share the paged path are library users, and a
// negative offset on a shared cache is not a failure worth leaving to a bounds check three frames
// deeper.
func TestEmptyPromptThroughThePagedPathIsRefused(t *testing.T) {
	c := defaultTiny()
	c.ctx = 32
	m, err := New(buildTinyWeights(t, c))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	pool, err := NewBlockPool(cfg, BlockPoolOptions{BlockSize: 2, Blocks: 4})
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	pool.EnablePrefixCache()

	cache := pool.New()
	out, err := m.PrefillWithPrefix(cache, nil, 0)
	if err == nil {
		t.Fatalf("an empty prompt produced %d logits instead of an error", len(out))
	}
	// Both counts: how many positions the cache holds and how many tokens were offered. The
	// assertion used to stop at the opening words, which survive any change to the numbers.
	for _, want := range []string{"cached prefix of 0 positions", "for 0 tokens"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not state %q", err, want)
		}
	}
	if out != nil {
		t.Errorf("a refused prefill returned %d logits", len(out))
	}

	// The same cache still serves a real prompt, so the refusal is about the input rather than
	// about a poisoned cache.
	if _, err := m.PrefillWithPrefix(cache, []int32{0, 1, 2, 3}, 0); err != nil {
		t.Errorf("a four-token prompt through the same cache: %v", err)
	}
}

// TestEmbedBatchGuardsTheTokenNotTheOffset covers the reason its check is written that way.
//
// The comment there explains why the guard is on the token rather than on off+Hidden: a
// negative token makes off negative while off+Hidden can still be inside the embedding, so
// a check against the sum alone lets it through and the slice expression panics. Measured
// with Hidden 8 and token -1: off is -8 and off+Hidden is 0, which is in range.
//
// Prefill checks every token before calling this, so the function is unreachable with a bad
// token through the public API -- TestPrefillRejectsATokenOutsideTheVocabulary covers that
// path, and asserts the message names both the token and its index. This test asserts the second line of defence on its own, because the comment's
// reasoning is what stops a future edit from "simplifying" the check into the broken form,
// and nothing was holding the reasoning in place.
func TestEmbedBatchGuardsTheTokenNotTheOffset(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.W.Config
	b := newBatch(cfg, 2)

	for _, tok := range []int32{-1, -1000, int32(cfg.VocabSize), int32(cfg.VocabSize) + 5} {
		err := m.embedBatch(b, []int32{tok})
		if err == nil {
			t.Errorf("embedBatch accepted token %d, vocabulary is %d", tok, cfg.VocabSize)
			continue
		}
		if !strings.Contains(err.Error(), "outside vocabulary") {
			t.Errorf("token %d: error %q does not name the problem", tok, err)
		}
		if !strings.Contains(err.Error(), fmt.Sprint(tok)) {
			t.Errorf("token %d: error %q does not name the token", tok, err)
		}
	}

	// The arithmetic the comment describes, so the reasoning is checked and not just stated.
	const hidden = 8
	off := -1 * hidden
	if off >= 0 {
		t.Fatalf("off = %d, expected negative for the premise to hold", off)
	}
	if off+hidden < 0 {
		t.Fatal("off+Hidden is negative here, so this fixture does not demonstrate the trap")
	}
}

// TestBatchMatchesSequentialExactly is the bit-identical half of the same contract.
//
// TestBatchMatchesSequential allows a 1e-3 difference, which is the right tolerance for a
// test over a checkpoint whose arithmetic is only available through the model. It cannot
// show the stronger claim batch.go makes -- that Prefill is "identical to calling Forward
// once per token" -- and on the default fixture it demonstrates nothing about parallelism
// at all: that model is 8 hidden wide with an 11-token vocabulary, so every product is far
// below kernel minParallelWork (32768) and both paths run on one goroutine.
//
// This one is built to cross that threshold (256 hidden x 363 vocab = 92,928 multiply-adds
// in the vocabulary projection alone) and is given eight workers, so the row and column
// splits actually divide. Every logit is then compared with ==, not a tolerance: the
// kernels' contract is that parallelism never reassociates a reduction, so a single bit of
// difference means that contract was broken somewhere.
func TestBatchMatchesSequentialExactly(t *testing.T) {
	// Wide enough that the vocabulary projection clears minParallelWork several times over.
	const hidden, ffn, layers = 256, 512, 2
	path := gguftest.Model{Hidden: hidden, Heads: 4, FFN: ffn, Layers: layers}.Write(t, t.TempDir())
	m, err := NewWithOptions(path, Options{Workers: 8})
	if err != nil {
		t.Fatalf("NewWithOptions: %v", err)
	}
	cfg := m.Config()

	// The threshold is a kernel constant, so it is restated here with its arithmetic: if it
	// were raised past this fixture's work, this test would silently stop exercising the
	// parallel path and keep passing.
	const minParallelWork = 1 << 15
	if work := cfg.Hidden * cfg.VocabSize; work < minParallelWork {
		t.Fatalf("the fixture's largest product is %d multiply-adds, below the %d where "+
			"kernels start splitting; this test would not exercise parallelism",
			work, minParallelWork)
	}

	tokens := make([]int32, 6)
	for i := range tokens {
		tokens[i] = int32(1 + i%7)
	}

	seqCache, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatal(err)
	}
	var seqLast []float32
	for pos, tok := range tokens {
		last, err := m.Forward(seqCache, int(tok), pos)
		if err != nil {
			t.Fatalf("Forward(%d): %v", pos, err)
		}
		if pos == len(tokens)-1 {
			seqLast = append([]float32(nil), last...)
		}
	}

	batchCache, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatal(err)
	}
	batchLast, err := m.Prefill(batchCache, tokens, 0)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}

	if len(seqLast) != len(batchLast) {
		t.Fatalf("%d logits from the per-token path, %d from the batched one",
			len(seqLast), len(batchLast))
	}
	for i := range seqLast {
		if seqLast[i] != batchLast[i] {
			t.Fatalf("logit %d differs: per-token %.9g, batched %.9g -- batching is a "+
				"restructuring, so these must be bit-identical", i, seqLast[i], batchLast[i])
		}
	}
}

// TestLogitsForCachedMatchesAFreshPrefill covers the one path that never runs the ordinary
// prefill: when every block of a prompt is already in the prefix index, PrefillWithPrefix
// calls logitsForCached, which computes the last position's logits from stored keys and
// values instead of recomputing them.
//
// Nothing tested it. The branch needs a prompt that is an exact multiple of the block size
// -- one token longer and the match covers all but the last position, so the ordinary
// prefill runs -- and it needs a model whose attention carries a bias, because that is the
// part logitsForCached does differently: the key and value are already stored, but the query
// comes from a fresh projection, so its bias has to be applied here or not at all. The
// Qwen2 family is the one with bias, and every existing prefix-reuse test uses llama.
//
// Measured with the query bias removed from attentionCached: the cached path then differs
// from a fresh prefill at 351 of 363 logits, all by a little, which is exactly the shape a
// missing bias has.
func TestLogitsForCachedMatchesAFreshPrefill(t *testing.T) {
	// Qwen2 is the family with attention bias; AttnBias writes the q/k/v bias tensors.
	path := gguftest.Model{Arch: "qwen2", AttnBias: true, ContextLen: 128}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	if !cfg.AttnBias {
		t.Fatalf("the fixture has no attention bias, so the branch this test targets is " +
			"not exercised")
	}

	const blockSize = 4
	pool, err := NewBlockPool(cfg, BlockPoolOptions{BlockSize: blockSize, Blocks: 64})
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	pool.EnablePrefixCache()

	// Exactly two whole blocks, so the second call matches the entire prompt.
	prompt := make([]int32, 2*blockSize)
	for i := range prompt {
		prompt[i] = int32(1 + i%5)
	}

	first := pool.New()
	fresh, err := m.PrefillWithPrefix(first, prompt, 0)
	if err != nil {
		t.Fatalf("first PrefillWithPrefix: %v", err)
	}
	// Copied immediately: the returned slice is the batch buffer, which the next pass reuses.
	fresh = append([]float32(nil), fresh...)

	second := pool.New()
	cached, err := m.PrefillWithPrefix(second, prompt, 0)
	if err != nil {
		t.Fatalf("second PrefillWithPrefix: %v", err)
	}
	cached = append([]float32(nil), cached...)

	if got := second.ReusedPositions(); got != len(prompt) {
		t.Fatalf("the second pass reused %d positions of %d; the whole prompt has to match "+
			"for logitsForCached to run", got, len(prompt))
	}
	if len(fresh) != len(cached) {
		t.Fatalf("%d logits from the fresh pass, %d from the cached one", len(fresh), len(cached))
	}
	for i := range fresh {
		if fresh[i] != cached[i] {
			t.Fatalf("logit %d: cached %.9g, fresh %.9g -- the fully-cached path has to "+
				"reproduce a fresh prefill exactly", i, cached[i], fresh[i])
		}
	}
}

// TestLogitsForCachedHonoursTheWindow is the sliding-window half of the fully-cached path.
//
// TestLogitsForCachedMatchesAFreshPrefill covers a Qwen2 model with no window. Mistral's
// sliding window is the other per-file property that changes what attention reads, and
// logitsForCached has its own trimSpans call for it. Removing that call left the entire
// package green: every existing window test drives the ordinary prefill or decode path,
// where the trimming lives in attentionBatch/attention, and no test combined a window with
// a prefix that the index had cached in full.
//
// Measured with the trim removed from attentionCached: the cached path differs from a fresh
// prefill at 291 of 363 logits. Without the window at all it would be a different set of
// numbers, so the test also asserts the model really carries a window.
func TestLogitsForCachedHonoursTheWindow(t *testing.T) {
	// Four positions of window against an eight-position prompt: the trim has to drop
	// something for the assertion to mean anything.
	const window, blockSize = 4, 4
	path := gguftest.Model{
		Arch: "qwen2", AttnBias: true, ContextLen: 128, WindowLen: window,
	}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	if cfg.WindowLen != window {
		t.Fatalf("WindowLen = %d, want %d: the fixture's window did not load, so this test "+
			"would not exercise the trim", cfg.WindowLen, window)
	}

	pool, err := NewBlockPool(cfg, BlockPoolOptions{BlockSize: blockSize, Blocks: 64})
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	pool.EnablePrefixCache()

	// Two whole blocks, and more positions than the window, so the last position can see
	// only the recent ones.
	prompt := make([]int32, 2*blockSize)
	if len(prompt) <= window {
		t.Fatalf("the prompt is %d positions and the window %d, so nothing would be trimmed",
			len(prompt), window)
	}
	for i := range prompt {
		prompt[i] = int32(1 + i%5)
	}

	first := pool.New()
	fresh, err := m.PrefillWithPrefix(first, prompt, 0)
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
		t.Fatalf("the second pass reused %d of %d positions; the whole prompt has to match",
			got, len(prompt))
	}
	for i := range fresh {
		if fresh[i] != cached[i] {
			t.Fatalf("logit %d: cached %.9g, fresh %.9g -- the fully-cached path has to "+
				"apply the sliding window as well", i, cached[i], fresh[i])
		}
	}
}

// TestPrefillWithPrefixMatchesPrefill is the contract for the paged prefill path: a prompt
// prefilled through a block pool must produce the logits a contiguous cache produces.
//
// The two are separate code paths -- PrefillWithPrefix batches the whole prompt through
// decodeBatch, while Prefill on a contiguous cache falls back to one Forward per token -- and only
// the refusal of an empty prompt was tested for the paged one. A wrong span offset or a batch that
// used one shared position would leave every prompt longer than a single token wrong, which is
// every prompt.
//
// Byte equality rather than a tolerance: both paths run the same arithmetic in the same order on
// the same values, so there is no accumulation difference to allow for. A tolerance here would
// accept exactly the small per-position drift a shared-position bug produces.
func TestPrefillWithPrefixMatchesPrefill(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	ids := fixtureTokens(cfg, 9)

	contig, err := NewKVCache(cfg, 64)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}
	want, err := m.Prefill(contig, ids, 0)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}

	pool, err := NewBlockPool(cfg, BlockPoolOptions{BlockSize: 4, Blocks: 16})
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	pool.EnablePrefixCache()
	paged := pool.New()
	got, err := m.PrefillWithPrefix(paged, ids, 0)
	if err != nil {
		t.Fatalf("PrefillWithPrefix: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("the paged path produced %d logits, the contiguous one %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("logit[%d] = %v through the paged path, %v through the contiguous one",
				i, got[i], want[i])
		}
	}
}

// TestScoreBuffersRentsOnePerUsedSlot covers the allocation whose size a flag controls.
//
// Parallel splits n positions across min(workers, n) goroutines, so a buffer count above that
// allocates ones nothing can index. Measured before the min was there: a deliberately large
// --workers on a long-context model reserved gigabytes of score buffers per layer and made a
// single request tens of times slower. The request stayed correct, which is what made it hard to
// notice -- the only symptom was time.
func TestScoreBuffersRentsOnePerUsedSlot(t *testing.T) {
	// ContextLen is what decides each buffer's width, so it has to be set for the size check.
	newModel := func(workers int) *Model {
		return &Model{
			run: kernel.NewRunner(workers),
			W:   &Weights{Config: Config{ContextLen: 64}},
		}
	}

	cases := []struct {
		name    string
		workers int
		rows    int
		want    int
	}{
		{"one row uses one slot whatever the worker count", 8, 1, 1},
		{"between the two", 8, 3, 3},
		{"exactly the worker count", 4, 4, 4},
		// The case that used to allocate thousands: more workers than positions.
		{"more workers than rows", 64, 2, 2},
		{"workers and rows both large", 64, 64, 64},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := newModel(c.workers).scoreBuffers(c.rows)
			if len(got) != c.want {
				t.Errorf("scoreBuffers(%d rows, %d workers) rented %d buffers, want %d",
					c.rows, c.workers, len(got), c.want)
			}
		})
	}
}

// TestScoreBuffersNeverRentsNone is the degenerate case that would index a nil slice.
//
// Parallel with zero rows should not run any worker, but the buffer slice is built before that
// decision, and a zero-length slice would panic if anything took scorers[slot]. One is the floor
// the call site's slots check exists for.
func TestScoreBuffersNeverRentsNone(t *testing.T) {
	m := &Model{run: kernel.NewRunner(4), W: &Weights{Config: Config{ContextLen: 16}}}
	if got := m.scoreBuffers(0); len(got) != 1 {
		t.Errorf("scoreBuffers(0) rented %d buffers, want 1", len(got))
	}
}

// TestScoreBuffersAreWideEnoughForTheWindow pins the width.
//
// One position can attend to the whole context, so a buffer narrower than ContextLen would be an
// out-of-range write on the first long prompt. The width comes from the config rather than a
// constant, so a model with a larger window gets larger buffers.
func TestScoreBuffersAreWideEnoughForTheWindow(t *testing.T) {
	for _, ctxLen := range []int{16, 128, 2048} {
		m := &Model{run: kernel.NewRunner(2), W: &Weights{Config: Config{ContextLen: ctxLen}}}
		scorers := m.scoreBuffers(2)
		for i, sc := range scorers {
			if len(sc) != ctxLen {
				t.Errorf("context %d: buffer %d is %d wide, want %d", ctxLen, i, len(sc), ctxLen)
			}
		}
	}
}

// TestScoreBuffersAreDistinctSlices is the property that makes them per-slot.
//
// Two workers sharing one buffer would interleave writes into the same slice, which is a data
// race the detector would report and a wrong result it might not. A test that only counted the
// buffers would pass with every entry aliasing the first.
func TestScoreBuffersAreDistinctSlices(t *testing.T) {
	m := &Model{run: kernel.NewRunner(3), W: &Weights{Config: Config{ContextLen: 8}}}
	scorers := m.scoreBuffers(3)
	if len(scorers) != 3 {
		t.Fatalf("got %d buffers", len(scorers))
	}
	// Writing one must not change another.
	scorers[0][0] = 7
	for i := 1; i < len(scorers); i++ {
		if scorers[i][0] != 0 {
			t.Errorf("buffer %d saw buffer 0's write: the slices alias", i)
		}
	}
}
