package llama

import (
	"math"
	"testing"

	"github.com/pplmx/orbix/internal/gguftest"
)

// The fixture defaults to KVHeads == Heads, which is a group size of one. Every downstream
// test in the repository therefore ran the ordinary multi-head path and never the grouped
// one, even though grouped-query attention is what the models this engine targets actually
// use: Qwen2.5-0.5B has 14 query heads over 2 KV heads.
//
// The mapper is the part that goes wrong quietly. A wrong `h / group` still produces logits
// of the right shape, still reads valid memory, and simply attends with the wrong head's
// keys -- text that reads as slightly off rather than as broken.

// TestMultiHeadGroupedAttention checks a model with a real group size runs and is finite.
func TestMultiHeadGroupedAttention(t *testing.T) {
	// 8 query heads over 2 KV heads: group of 4.
	path := gguftest.Model{
		Heads: 8, KVHeads: 2, NoEOS: true, ContextLen: 128,
	}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	if cfg.Heads != 8 || cfg.KVHeads != 2 {
		t.Fatalf("loaded %d query heads over %d KV heads, want 8 and 2", cfg.Heads, cfg.KVHeads)
	}

	cache, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}
	logits, err := m.Prefill(cache, fixtureTokens(cfg, 6), 0)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logit[%d] is %v", i, v)
		}
	}
}

// TestGroupedAttentionSharesKeyValueHeads is the property that separates grouped attention
// from plain multi-head: one KV head serves a whole group of query heads, so zeroing that
// head's projection must change the output.
//
// It also checks the cache is allocated for KV heads rather than query heads. Those differ
// by the group size, and allocating for query heads would still work -- just with unused
// space -- so nothing else would notice.
func TestGroupedAttentionSharesKeyValueHeads(t *testing.T) {
	path := gguftest.Model{
		Heads: 8, KVHeads: 2, NoEOS: true, ContextLen: 128,
	}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()

	layer := &m.W.Layers[0]
	// A projection is [outFeatures, hidden] row-major, and K and V have one row per KV
	// head where Q has one per query head. With a group of four the two differ by a factor
	// of four, so an implementation that sized the projections from Heads would be four
	// times too big -- and would still run, just attending with heads that do not exist.
	kvOut := cfg.KVHeads * cfg.HeadDim
	if len(layer.wk) != kvOut*cfg.Hidden {
		t.Errorf("wk holds %d weights, want %d for %d KV heads of %d dims",
			len(layer.wk), kvOut*cfg.Hidden, cfg.KVHeads, cfg.HeadDim)
	}
	qOut := cfg.Heads * cfg.HeadDim
	if len(layer.wq) != qOut*cfg.Hidden {
		t.Errorf("wq holds %d weights, want %d for %d query heads",
			len(layer.wq), qOut*cfg.Hidden, cfg.Heads)
	}

	run := func() []float32 {
		cache, err := NewKVCache(cfg, cfg.ContextLen)
		if err != nil {
			t.Fatalf("NewKVCache: %v", err)
		}
		logits, err := m.Prefill(cache, fixtureTokens(cfg, 4), 0)
		if err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		return append([]float32(nil), logits...)
	}

	before := run()
	// Zero the first KV head's weights. It serves a quarter of the query heads, so the
	// logits must move; a forward pass that ignored the grouping would not care.
	for row := 0; row < cfg.HeadDim; row++ {
		clear(layer.wk[row*cfg.Hidden : (row+1)*cfg.Hidden])
	}
	after := run()

	for i := range before {
		if before[i] != after[i] {
			return
		}
	}
	t.Error("zeroing one KV head's projection left the logits unchanged")
}

// TestDecodePathAgreesWithPrefillUnderGrouping covers the other half of the mapper.
//
// The two tests above go through Prefill, which is batch.go's grouped path. A decode step does
// not: Model.Forward calls m.attention in forward.go, and that function has its own `kv := h /
// group`. Only one of the two was ever exercised, so a wrong mapper in the decode half would
// have left the whole package green -- measured: replacing forward.go's `h / group` with `h %
// KVHeads` changes all 363 logits and no existing test noticed.
//
// Both halves now have to agree with each other bit for bit. That is the property that matters
// to a served request, where the prefill fills the cache and every generated token goes through
// the other path: a disagreement here is a model that answers differently depending on how much
// of the prompt was batched.
func TestDecodePathAgreesWithPrefillUnderGrouping(t *testing.T) {
	path := gguftest.Model{
		Heads: 8, KVHeads: 2, NoEOS: true, ContextLen: 128,
	}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	if cfg.Heads != 8 || cfg.KVHeads != 2 {
		t.Fatalf("loaded %d query heads over %d KV heads, so this test is not exercising a "+
			"group of four", cfg.Heads, cfg.KVHeads)
	}
	tokens := fixtureTokens(cfg, 5)

	batchedCache, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}
	batched, err := m.Prefill(batchedCache, tokens, 0)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}

	stepCache, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}
	var stepped []float32
	for pos, tok := range tokens {
		stepped, err = m.Forward(stepCache, int(tok), pos)
		if err != nil {
			t.Fatalf("Forward at %d: %v", pos, err)
		}
	}

	if len(stepped) != len(batched) {
		t.Fatalf("Forward produced %d logits, Prefill produced %d", len(stepped), len(batched))
	}
	for i := range batched {
		if math.Float32bits(batched[i]) != math.Float32bits(stepped[i]) {
			t.Fatalf("logit %d: Prefill %v (0x%08x), Forward %v (0x%08x) -- the two paths map "+
				"query heads to KV heads differently", i,
				batched[i], math.Float32bits(batched[i]), stepped[i], math.Float32bits(stepped[i]))
		}
	}
}

// TestCachedPrefillAgreesUnderGrouping covers the paged-cache path's mapper.
//
// attentionCached in attention.go has its own `kv := h / group`, and it is the mapper a served
// request with --blocks actually uses: PrefillWithPrefix routes each position through it. The
// test that reaches it, TestLogitsForCachedMatchesAFreshPrefill, builds a default fixture, whose
// KVHeads is Heads -- a group of one. Under a group of one h/group and h%KVHeads agree on
// everything, so a wrong mapper there is invisible: measured, replacing it with `h % KVHeads`
// left the whole package green.
//
// The comparison is against the unbatchd path on the same model, which is what a cache has to
// be transparent to.
func TestCachedPrefillAgreesUnderGrouping(t *testing.T) {
	path := gguftest.Model{
		Heads: 8, KVHeads: 2, NoEOS: true, ContextLen: 128,
	}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	if group := cfg.Heads / cfg.KVHeads; group != 4 {
		t.Fatalf("this fixture has a group of %d, so the two mappers agree by accident", group)
	}

	const blockSize = 4
	pool, err := NewBlockPool(cfg, BlockPoolOptions{BlockSize: blockSize, Blocks: 64})
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	pool.EnablePrefixCache()

	// Exactly two whole blocks, so the second call matches the entire prompt and runs through
	// attentionCached rather than the ordinary prefill.
	prompt := make([]int32, 2*blockSize)
	for i := range prompt {
		prompt[i] = int32(3 + i)
	}

	first := pool.New()
	if _, err := m.PrefillWithPrefix(first, prompt, 0); err != nil {
		t.Fatalf("first PrefillWithPrefix: %v", err)
	}
	second := pool.New()
	cached, err := m.PrefillWithPrefix(second, prompt, 0)
	if err != nil {
		t.Fatalf("second PrefillWithPrefix: %v", err)
	}
	cached = append([]float32(nil), cached...)
	// The whole prompt has to match for logitsForCached to run; without this the test would be
	// comparing an ordinary prefill with itself and the mapper would never be exercised.
	if got := second.ReusedPositions(); got != len(prompt) {
		t.Fatalf("the second pass reused %d positions of %d, so attentionCached did not run",
			got, len(prompt))
	}

	// The same prompt with no paged cache at all.
	plainCache, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}
	fresh, err := m.Prefill(plainCache, prompt, 0)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}

	if len(cached) != len(fresh) {
		t.Fatalf("cached path produced %d logits, fresh produced %d", len(cached), len(fresh))
	}
	for i := range fresh {
		if math.Float32bits(cached[i]) != math.Float32bits(fresh[i]) {
			t.Fatalf("logit %d: cached %v (0x%08x), fresh %v (0x%08x) -- the paged path maps "+
				"query heads to KV heads differently", i,
				cached[i], math.Float32bits(cached[i]), fresh[i], math.Float32bits(fresh[i]))
		}
	}
}

// TestDecodeStepAgreesUnderGrouping covers the fourth mapper, the one in decodeBatch.attention.
//
// DecodeStep takes a different function from Forward: it builds a decodeBatch and calls its own
// attention per block. TestDecodeStepMatchesForward compares the two, but on the default fixture
// whose KVHeads is Heads -- a group of one, where h/group and h%KVHeads agree on every head. So
// the same wrong-mapper injection that left the package green in the other two places leaves it
// green here as well.
//
// Compared against Forward one sequence at a time, which is what DecodeStep promises to equal.
func TestDecodeStepAgreesUnderGrouping(t *testing.T) {
	path := gguftest.Model{
		Heads: 8, KVHeads: 2, NoEOS: true, ContextLen: 128,
	}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	if group := cfg.Heads / cfg.KVHeads; group != 4 {
		t.Fatalf("this fixture has a group of %d, so the two mappers agree by accident", group)
	}

	ids := fixtureTokens(cfg, 7)
	histories := [][]int32{
		{ids[0], ids[1]},
		{ids[0], ids[1], ids[2], ids[3]},
		{ids[0], ids[2]},
	}
	tokens := ids[4:7]

	// Reference: one Forward per sequence.
	want := make([][]float32, len(histories))
	for i, hist := range histories {
		cache, err := NewKVCache(cfg, cfg.ContextLen)
		if err != nil {
			t.Fatalf("NewKVCache: %v", err)
		}
		if _, err := m.Prefill(cache, hist, 0); err != nil {
			t.Fatalf("Prefill %d: %v", i, err)
		}
		logits, err := m.Forward(cache, int(tokens[i]), len(hist))
		if err != nil {
			t.Fatalf("Forward %d: %v", i, err)
		}
		want[i] = append([]float32(nil), logits...)
	}

	// The same three in one step.
	batchCaches := make([]Cache, len(histories))
	for i, hist := range histories {
		cache, err := NewKVCache(cfg, cfg.ContextLen)
		if err != nil {
			t.Fatalf("NewKVCache: %v", err)
		}
		if _, err := m.Prefill(cache, hist, 0); err != nil {
			t.Fatalf("Prefill %d: %v", i, err)
		}
		batchCaches[i] = cache
	}
	got, err := m.DecodeStep(batchCaches, tokens)
	if err != nil {
		t.Fatalf("DecodeStep: %v", err)
	}

	for i := range histories {
		row := got[i*cfg.VocabSize : (i+1)*cfg.VocabSize]
		for j := range want[i] {
			if math.Float32bits(row[j]) != math.Float32bits(want[i][j]) {
				t.Fatalf("sequence %d logit %d: DecodeStep %v (0x%08x), Forward %v (0x%08x)",
					i, j, row[j], math.Float32bits(row[j]), want[i][j],
					math.Float32bits(want[i][j]))
			}
		}
	}
}

// TestPagedCacheMatchesContiguousBeyondTheWindow covers the two caches on a prompt far longer
// than the window, which is a different question from the one the existing window tests ask.
//
// TestLogitsForCachedHonoursTheWindow uses a window of 4 against a prompt of 8, so it checks that
// the trim happens at all. This one runs a window of 32 against 9x that many positions, so the
// trim lands across many blocks and the span arithmetic -- which block holds row
// total-keep, where a trimmed span starts -- has room to be wrong.
//
// Both caches are compared bit for bit. They are separate code paths over the same weights, and a
// model served with --blocks and one served without produce the same answer only if those paths
// agree.
func TestPagedCacheMatchesContiguousBeyondTheWindow(t *testing.T) {
	const (
		window    = 32
		blockSize = 16
	)
	path := gguftest.Model{
		ContextLen: 512, WindowLen: window, NoEOS: true,
	}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	if cfg.WindowLen != window {
		t.Fatalf("WindowLen = %d, want %d: the fixture's window did not load", cfg.WindowLen, window)
	}

	// Nine times the window, so several blocks sit entirely outside it.
	prompt := make([]int32, 9*window)
	for i := range prompt {
		prompt[i] = int32(1 + i%7)
	}

	contiguous, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}
	want, err := m.Prefill(contiguous, prompt, 0)
	if err != nil {
		t.Fatalf("Prefill contiguous: %v", err)
	}

	pool, err := NewBlockPool(cfg, BlockPoolOptions{BlockSize: blockSize, Blocks: 64})
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	paged := pool.New()
	got, err := m.Prefill(paged, prompt, 0)
	if err != nil {
		t.Fatalf("Prefill paged: %v", err)
	}

	if len(got) != len(want) {
		t.Fatalf("paged produced %d logits, contiguous produced %d", len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
			t.Fatalf("logit %d: contiguous %v (0x%08x), paged %v (0x%08x) -- a prompt of %d "+
				"positions against a window of %d has to agree whichever cache holds it", i,
				want[i], math.Float32bits(want[i]), got[i], math.Float32bits(got[i]),
				len(prompt), window)
		}
	}

	// This comparison cannot see a wrong window, and the note is here rather than the assertion
	// because the fixture cannot support one: both caches run the same attentionBatch, so doubling
	// the trim there changes both sides equally, and a windowless fixture produces the same logits
	// for two different distant prefixes anyway -- measured, WindowLen 0 and 32 both give 0
	// differing logits for the same pair. The trim's own coverage comes from the decode path, where
	// injecting the error does turn the package red.
}
