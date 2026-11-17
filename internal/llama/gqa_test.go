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
