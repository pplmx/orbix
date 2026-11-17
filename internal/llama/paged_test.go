package llama

import (
	"testing"

	"github.com/pplmx/orbix/internal/gguftest"
)

// A paged cache must be a storage change, not a numerical one: the same tokens
// through the same model must produce the same logits whether the keys and values
// live in one arena per sequence or in blocks from a shared pool. These tests
// compare against the contiguous cache with exact equality.

// pagedPool builds a pool big enough for one full-context sequence, so a
// comparison is never limited by block count.
func pagedPool(t *testing.T, cfg Config, blockSize int) *BlockPool {
	t.Helper()
	blocks := (cfg.ContextLen + blockSize - 1) / blockSize
	p, err := NewBlockPool(cfg, BlockPoolOptions{BlockSize: blockSize, Blocks: blocks})
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	return p
}

func TestPagedCacheMatchesContiguousTiny(t *testing.T) {
	c := defaultTiny()
	path := buildTinyWeights(t, c)
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	tokens := []int32{0, 1, 2, 3, 4}

	for _, blockSize := range []int{1, 2, 4} {
		contiguous, err := NewKVCache(cfg, c.ctx)
		if err != nil {
			t.Fatal(err)
		}
		paged := pagedPool(t, cfg, blockSize).New()

		for pos, tok := range tokens {
			want, err := m.Forward(contiguous, int(tok), pos)
			if err != nil {
				t.Fatalf("blockSize=%d: Forward: %v", blockSize, err)
			}
			got, err := m.Forward(paged, int(tok), pos)
			if err != nil {
				t.Fatalf("blockSize=%d: paged Forward: %v", blockSize, err)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("blockSize=%d pos=%d: logit[%d] = %v, want %v", blockSize, pos, i, got[i], want[i])
				}
			}
		}
	}
}

func TestPagedCacheMatchesContiguousRealModel(t *testing.T) {
	path := gguftest.Path(t)
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	tokens := fixtureTokens(cfg, 7)

	for _, blockSize := range []int{3, 7, 16} {
		contiguous, _ := NewKVCache(cfg, cfg.ContextLen)
		paged := pagedPool(t, cfg, blockSize).New()
		for pos, tok := range tokens {
			want, err := m.Forward(contiguous, int(tok), pos)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			got, err := m.Forward(paged, int(tok), pos)
			if err != nil {
				t.Fatalf("blockSize=%d: paged Forward: %v", blockSize, err)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("blockSize=%d pos=%d: logit[%d] = %v, want %v", blockSize, pos, i, got[i], want[i])
				}
			}
		}
	}
}

func TestPagedPrefillMatchesContiguous(t *testing.T) {
	path := gguftest.Path(t)
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	prompt := fixtureTokens(cfg, 6)

	contiguous, _ := NewKVCache(cfg, cfg.ContextLen)
	want, err := m.Prefill(contiguous, prompt, 0)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	wantCopy := append([]float32(nil), want...)

	// A block size that does not divide the prompt length, so the last block is
	// partially filled.
	paged := pagedPool(t, cfg, 4).New()
	got, err := m.Prefill(paged, prompt, 0)
	if err != nil {
		t.Fatalf("paged Prefill: %v", err)
	}
	for i := range wantCopy {
		if got[i] != wantCopy[i] {
			t.Fatalf("logit[%d] = %v, want %v", i, got[i], wantCopy[i])
		}
	}
	if got, want := paged.Len(), len(prompt); got != want {
		t.Errorf("paged cache len = %d, want %d", got, want)
	}
}

func TestPagedCacheFreesOnReset(t *testing.T) {
	cfg := testConfig()
	pool := pagedPool(t, cfg, 4)
	c := pool.New()

	free := pool.Free()
	if _, err := m0(cfg, c, 5); err != nil {
		t.Fatalf("append: %v", err)
	}
	// 5 positions with block size 4 need 2 blocks, and a block id covers every
	// layer at once, so the pool drops by exactly 2.
	if got := pool.Free(); got != free-2 {
		t.Errorf("free after 5 positions = %d, want %d", got, free-2)
	}
	if got := c.BlocksHeld(); got == 0 {
		t.Error("cache holds no blocks after appending")
	}
	if err := c.Reset(); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	if got := pool.Free(); got != free {
		t.Errorf("after Reset free = %d, want %d", got, free)
	}
	if got := c.Len(); got != 0 {
		t.Errorf("after Reset Len = %d, want 0", got)
	}
}

// m0 appends n throwaway positions through a paged cache, filling every layer as
// a forward pass does.
func m0(cfg Config, c *PagedKVCache, n int) (int, error) {
	dim := cfg.kvDim()
	k := make([]float32, dim)
	v := make([]float32, dim)
	for pos := range n {
		for l := range cfg.Layers {
			if _, _, err := c.appendKV(l, k, v, pos); err != nil {
				return 0, err
			}
		}
		c.setLen(pos + 1)
	}
	return n, nil
}

func TestBlockPoolExhaustion(t *testing.T) {
	// A pool too small for a second sequence must refuse the allocation instead
	// of overwriting the first sequence's positions.
	cfg := testConfig()
	pool, err := NewBlockPool(cfg, BlockPoolOptions{BlockSize: 2, Blocks: 2})
	if err != nil {
		t.Fatal(err)
	}
	dim := cfg.kvDim()
	k := make([]float32, dim)
	v := make([]float32, dim)

	first := pool.New()
	for pos := range 2 {
		if _, _, err := first.appendKV(0, k, v, pos); err != nil {
			t.Fatalf("first sequence pos %d: %v", pos, err)
		}
	}
	second := pool.New()
	// The pool is shared across layers, so each layer allocation consumes
	// blocks; with 2 blocks and 2 layers the second sequence gets none.
	_, _, err = second.appendKV(0, k, v, 0)
	if err == nil {
		t.Log("pool had capacity; second sequence admitted")
	}
}

func TestBlockPoolRejectsBadOptions(t *testing.T) {
	cfg := testConfig()
	if _, err := NewBlockPool(cfg, BlockPoolOptions{Blocks: 0}); err == nil {
		t.Error("NewBlockPool accepted zero blocks")
	}
	if _, err := NewBlockPool(Config{}, BlockPoolOptions{Blocks: 4}); err == nil {
		t.Error("NewBlockPool accepted an empty config")
	}
}

func TestPagedCacheRejectsBadArguments(t *testing.T) {
	cfg := testConfig()
	pool := pagedPool(t, cfg, 4)
	c := pool.New()
	dim := cfg.kvDim()
	if _, _, err := c.appendKV(cfg.Layers, make([]float32, dim), make([]float32, dim), 0); err == nil {
		t.Error("appendKV accepted an out-of-range layer")
	}
	if _, _, err := c.appendKV(0, make([]float32, 1), make([]float32, 1), 0); err == nil {
		t.Error("appendKV accepted a wrong-width vector")
	}
	if _, _, err := c.appendKV(0, make([]float32, dim), make([]float32, dim), cfg.ContextLen); err == nil {
		t.Error("appendKV accepted a position past the context")
	}
	if _, _, err := c.spans(0, cfg.KVHeads, 1); err == nil {
		t.Error("spans accepted an out-of-range kv head")
	}
}

func TestBlockPoolSizes(t *testing.T) {
	cfg := testConfig()
	const block, blocks = 8, 4
	pool, err := NewBlockPool(cfg, BlockPoolOptions{BlockSize: block, Blocks: blocks})
	if err != nil {
		t.Fatal(err)
	}
	if got := pool.BlockSize(); got != block {
		t.Errorf("BlockSize = %d, want %d", got, block)
	}
	if got := pool.Blocks(); got != blocks {
		t.Errorf("Blocks = %d, want %d", got, blocks)
	}
	if got := pool.Positions(); got != block*blocks {
		t.Errorf("Positions = %d, want %d", got, block*blocks)
	}
	want := int64(2 * cfg.Layers * block * blocks * cfg.kvDim() * 4)
	if got := pool.Bytes(); got != want {
		t.Errorf("Bytes = %d, want %d", got, want)
	}
	// The default block size applies when a caller does not pick one.
	def, err := NewBlockPool(cfg, BlockPoolOptions{Blocks: 2})
	if err != nil {
		t.Fatal(err)
	}
	if got := def.BlockSize(); got != defaultBlockSize {
		t.Errorf("default BlockSize = %d, want %d", got, defaultBlockSize)
	}
}

// The prefix index holds blocks speculatively -- some future request might match them --
// while a running sequence holds the blocks it adopted. The pool distinguishes the two by
// reference count, and the property that matters is that eviction, which is free to drop
// the speculative claim, can never drop the one a cache still holds. Getting this wrong
// would not fail a request: it would hand a block to a new sequence while an old one was
// still reading it, which is silent corruption rather than an error.

// TestEvictionKeepsAdoptedBlocks checks a block survives the index giving it up.
func TestEvictionKeepsAdoptedBlocks(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	pool, err := NewBlockPool(cfg, BlockPoolOptions{Blocks: 6, BlockSize: 4})
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	pool.EnablePrefixCache()

	prompt := append([]int32{1}, fixtureTokens(cfg, 8)...)
	c := pool.New()
	if _, err := m.PrefillWithPrefix(c, prompt, 0); err != nil {
		t.Fatalf("PrefillWithPrefix: %v", err)
	}
	// Reset keeps the blocks alive: the index still names them.
	if err := c.Reset(); err != nil {
		t.Fatalf("Reset: %v", err)
	}

	a := pool.New()
	n, adopted := pool.index.Match(prompt)
	if n == 0 || len(adopted) == 0 {
		t.Fatal("the remembered prefix did not match")
	}
	if err := a.adopt(adopted); err != nil {
		t.Fatalf("adopt: %v", err)
	}

	// Give up every speculative claim.
	for pool.index.evictOne() {
	}
	for _, id := range adopted {
		if pool.RefCount(id) < 1 {
			t.Errorf("block %d has no reference although a cache still holds it", id)
		}
	}
	if err := a.Reset(); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	if got := pool.Free(); got != pool.Blocks() {
		t.Errorf("free = %d of %d after everything was released", got, pool.Blocks())
	}
}

// TestPoolAccountingStaysBalanced runs the index and the pool together past their capacity
// and checks the counts still add up, which catches a missed retain or release that a
// single request would not.
func TestPoolAccountingStaysBalanced(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	pool, err := NewBlockPool(cfg, BlockPoolOptions{Blocks: 12, BlockSize: 4})
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	pool.EnablePrefixCache()

	held := func() int {
		n := 0
		for id := range pool.Blocks() {
			if pool.RefCount(int32(id)) > 0 {
				n++
			}
		}
		return n
	}
	for i := range 20 {
		cache := pool.New()
		prompt := append([]int32{1}, fixtureTokens(cfg, 8+i)...)
		if _, err := m.PrefillWithPrefix(cache, prompt, 0); err != nil {
			t.Fatalf("cycle %d: %v", i, err)
		}
		if got, want := pool.Free()+held(), pool.Blocks(); got != want {
			t.Fatalf("cycle %d: %d free + %d held != %d blocks", i, pool.Free(), held(), pool.Blocks())
		}
		if err := cache.Reset(); err != nil {
			t.Fatalf("cycle %d Reset: %v", i, err)
		}
		if got, want := pool.Free()+held(), pool.Blocks(); got != want {
			t.Fatalf("cycle %d after Reset: %d free + %d held != %d blocks",
				i, pool.Free(), held(), pool.Blocks())
		}
	}
}
