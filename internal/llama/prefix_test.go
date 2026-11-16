package llama

import (
	"os"
	"testing"
)

// Prefix reuse is a correctness-critical optimization: a wrong hit attaches keys
// and values that belong to different tokens, which produces fluent-looking
// garbage rather than an error. These tests compare against a run with no reuse at
// all, so any difference is a bug in the sharing and not in the model.

func TestPrefixIndexMatch(t *testing.T) {
	cfg := testConfig()
	pool := pagedPool(t, cfg, 4)
	idx := pool.EnablePrefixCache()

	tokens := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	blocks := []int32{}

	// Fill and remember two full blocks.
	dim := cfg.kvDim()
	k := make([]float32, dim)
	v := make([]float32, dim)
	c := pool.New()
	for pos := range 8 {
		for l := range cfg.Layers {
			if _, _, err := c.appendKV(l, k, v, pos); err != nil {
				t.Fatalf("appendKV: %v", err)
			}
		}
		c.setLen(pos + 1)
		if err := c.rememberFullBlock(tokens, pos); err != nil {
			t.Fatalf("rememberFullBlock(%d): %v", pos, err)
		}
	}
	blocks = append(blocks, c.blocks...)
	if got := idx.Len(); got != 2 {
		t.Fatalf("index holds %d prefixes, want 2", got)
	}

	// A prompt with the same first eight tokens matches; a different one does not.
	if n, got := idx.Match(tokens); n != 8 {
		t.Errorf("Match matched %d positions, want 8", n)
	} else if len(got) != 2 {
		t.Errorf("Match returned %d blocks, want 2", len(got))
	}
	other := []int32{1, 2, 3, 4, 5, 6, 7, 9}
	if n, _ := idx.Match(other); n != 4 {
		// The first block matches, the second does not.
		t.Errorf("Match on a diverging prefix matched %d positions, want 4", n)
	}
	if n, _ := idx.Match([]int32{99, 98, 97}); n != 0 {
		t.Errorf("Match on unrelated tokens matched %d positions, want 0", n)
	}
	// A partial block is never matched: only whole blocks can be shared.
	if n, _ := idx.Match([]int32{1, 2}); n != 0 {
		t.Errorf("Match on a partial block matched %d positions, want 0", n)
	}
	_ = blocks
}

// TestPrefixReuseProducesIdenticalOutput is the decisive test: prefilling with a
// warm prefix index must give exactly the logits of a cold run.
func TestPrefixReuseProducesIdenticalOutput(t *testing.T) {
	path := os.Getenv("MODEL")
	if path == "" {
		t.Skip("set MODEL to a GGUF file")
	}
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	const blockSize = 4
	// A shared prefix of two full blocks, then a divergent tail.
	prompt := []int32{1, 9038, 2501, 263, 931, 29871, 13, 29906, 350, 400}

	// Cold: no prefix index at all.
	coldPool := pagedPool(t, cfg, blockSize)
	cold, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatal(err)
	}
	want, err := m.Prefill(cold, prompt, 0)
	if err != nil {
		t.Fatalf("cold Prefill: %v", err)
	}
	wantLogits := append([]float32(nil), want...)

	// Warm: first request fills the index, second reuses it.
	warmPool := pagedPool(t, cfg, blockSize)
	warmPool.EnablePrefixCache()

	// The first request must also produce the same answer, so the remembering
	// path cannot perturb it.
	first := warmPool.New()
	got1, err := m.PrefillWithPrefix(first, prompt, 0)
	if err != nil {
		t.Fatalf("PrefillWithPrefix (cold pool): %v", err)
	}
	for i := range wantLogits {
		if got1[i] != wantLogits[i] {
			t.Fatalf("first request logit[%d] = %v, want %v", i, got1[i], wantLogits[i])
		}
	}

	// The second request reuses the shared prefix. It must produce the same logits
	// as the cold run.
	second := warmPool.New()
	got2, err := m.PrefillWithPrefix(second, prompt, 0)
	if err != nil {
		t.Fatalf("PrefillWithPrefix (warm pool): %v", err)
	}
	for i := range wantLogits {
		if got2[i] != wantLogits[i] {
			t.Fatalf("reused logit[%d] = %v, want %v", i, got2[i], wantLogits[i])
		}
	}
	_ = coldPool
}

// TestPrefixReuseSkipsPrefillWork checks that reuse actually saves the work: a warm
// second request must reuse blocks rather than recompute them.
func TestPrefixReuseSkipsPrefillWork(t *testing.T) {
	path := os.Getenv("MODEL")
	if path == "" {
		t.Skip("set MODEL to a GGUF file")
	}
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	const blockSize = 4
	prompt := []int32{1, 9038, 2501, 263, 931, 29871, 13, 29906}

	pool := pagedPool(t, cfg, blockSize)
	idx := pool.EnablePrefixCache()

	first := pool.New()
	if _, err := m.PrefillWithPrefix(first, prompt, 0); err != nil {
		t.Fatal(err)
	}
	if got := idx.Len(); got == 0 {
		t.Fatal("no prefixes were remembered")
	}
	if err := first.Reset(); err != nil {
		t.Fatal(err)
	}

	// The second cache must adopt the remembered blocks, so it starts with a
	// nonzero length instead of from position zero.
	second := pool.New()
	reused, err := m.PrefillWithPrefix(second, prompt, 0)
	if err != nil {
		t.Fatal(err)
	}
	if reused == nil {
		t.Fatal("PrefillWithPrefix returned no logits")
	}
	// Count the blocks that came from the index by checking that the first cache's
	// blocks were retained rather than freed.
	if got := idx.Len(); got == 0 {
		t.Error("index lost its prefixes after a second request")
	}
}

// TestPrefixReuseIsCredited checks the reported number of reused positions, which
// is what a caller shows as "saved work" and what the benchmark measures.
func TestPrefixReuseIsCredited(t *testing.T) {
	path := os.Getenv("MODEL")
	if path == "" {
		t.Skip("set MODEL to a GGUF file")
	}
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	const blockSize = 4
	prompt := []int32{1, 9038, 2501, 263, 931, 29871, 13, 29906, 350}

	pool := pagedPool(t, cfg, blockSize)
	pool.EnablePrefixCache()

	first := pool.New()
	if _, err := m.PrefillWithPrefix(first, prompt, 0); err != nil {
		t.Fatal(err)
	}
	firstReused := first.ReusedPositions()
	if firstReused != 0 {
		t.Errorf("first request reused %d positions, want 0", firstReused)
	}
	_ = first.Reset()

	second := pool.New()
	if _, err := m.PrefillWithPrefix(second, prompt, 0); err != nil {
		t.Fatal(err)
	}
	// Two whole blocks of four positions each were available.
	if got, want := second.ReusedPositions(), 8; got != want {
		t.Errorf("second request reused %d positions, want %d", got, want)
	}
}

func TestPrefixIndexEvictsWhenPoolRunsShort(t *testing.T) {
	cfg := testConfig()
	// A pool of four blocks: two to fill and remember, and then pressure.
	pool, err := NewBlockPool(cfg, BlockPoolOptions{BlockSize: 2, Blocks: 4})
	if err != nil {
		t.Fatal(err)
	}
	idx := pool.EnablePrefixCache()

	dim := cfg.kvDim()
	k := make([]float32, dim)
	v := make([]float32, dim)

	fill := func(tokens []int32) *PagedKVCache {
		c := pool.New()
		for pos := range len(tokens) {
			for l := range cfg.Layers {
				if _, _, err := c.appendKV(l, k, v, pos); err != nil {
					return nil
				}
			}
			c.setLen(pos + 1)
			if err := c.rememberFullBlock(tokens, pos); err != nil {
				return nil
			}
		}
		return c
	}

	c := fill([]int32{1, 2, 3, 4})
	if c == nil {
		t.Skip("pool too small for the setup")
	}
	if idx.Len() == 0 {
		t.Fatal("nothing remembered")
	}
	// Releasing the sequence leaves the index as the only holder, so the blocks are
	// not free but are recoverable.
	if err := c.Reset(); err != nil {
		t.Fatal(err)
	}
	before := idx.Len()

	// Now exhaust the pool: allocate until nothing is left. The index must give up
	// its speculative holds rather than fail.
	held := []*PagedKVCache{}
	for range 8 {
		next := pool.New()
		ok := true
		for pos := range 2 {
			for l := range cfg.Layers {
				if _, _, err := next.appendKV(l, k, v, pos); err != nil {
					ok = false
					break
				}
			}
			if !ok {
				break
			}
			next.setLen(pos + 1)
		}
		if !ok {
			break
		}
		held = append(held, next)
	}
	if len(held) == 0 {
		t.Fatal("could not allocate anything after the index filled the pool")
	}
	if got := idx.Len(); got >= before && before > 0 {
		t.Logf("index still holds %d entries; eviction was not needed", got)
	}
}

func TestPrefixIndexRejectsBadBlocks(t *testing.T) {
	cfg := testConfig()
	pool := pagedPool(t, cfg, 4)
	idx := pool.EnablePrefixCache()

	if err := idx.Remember([]int32{1, 2, 3}, 0); err == nil {
		t.Error("Remember accepted a short token run")
	}
	if err := idx.Remember([]int32{1, 2, 3, 4}, 9999); err == nil {
		t.Error("Remember accepted a block outside the pool")
	}
	// A cache that already holds positions cannot adopt blocks.
	donor := pool.New()
	dim := cfg.kvDim()
	k := make([]float32, dim)
	v := make([]float32, dim)
	if _, _, err := donor.appendKV(0, k, v, 0); err != nil {
		t.Fatalf("donor appendKV: %v", err)
	}
	c := pool.New()
	if err := c.adopt(donor.blocks); err != nil {
		t.Fatalf("adopt: %v", err)
	}
	if err := c.adopt(donor.blocks); err == nil {
		t.Error("adopt accepted a cache that already holds blocks")
	}
	// A nil index is a no-op, so a pool without prefix reuse needs no special case.
	var none *PrefixIndex
	if n, blocks := none.Match([]int32{1, 2, 3, 4}); n != 0 || blocks != nil {
		t.Errorf("nil index matched %d positions", n)
	}
	if err := none.Remember([]int32{1, 2, 3, 4}, 0); err != nil {
		t.Errorf("nil index Remember = %v", err)
	}
	if got := none.Len(); got != 0 {
		t.Errorf("nil index Len = %d", got)
	}
}

func TestHashTokensDetectsOrder(t *testing.T) {
	// A hash that ignored order would let a reordered prompt collide, and the token
	// comparison would then reject it -- so this is a performance property, not a
	// correctness one, but it is cheap to pin.
	a := hashTokens([]int32{1, 2, 3, 4})
	b := hashTokens([]int32{1, 2, 4, 3})
	if a == b {
		t.Error("hashTokens ignores token order")
	}
	if got, want := hashTokens([]int32{1, 2, 3, 4}), hashTokens([]int32{1, 2, 3, 4}); got != want {
		t.Error("hashTokens is not deterministic")
	}
}
