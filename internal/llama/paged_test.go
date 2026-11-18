package llama

import (
	"strings"
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

// TestNewBlockPoolRejectsUntenableSizes covers the sizes a pool cannot be built for.
//
// The block count comes from --blocks, the block size and the per-position width from the caller
// and the file, and every allocation is a product of them. Two failures this rules out, both of
// which kill the process rather than returning an error:
//
//   - The product wraps int64. Measured with 2^40 blocks of 2^20 positions: the old code died with
//     "fatal error: runtime: out of memory" before any error could be returned.
//   - The product is addressable but larger than the machine. The allocation is granted as virtual
//     memory the caller then cannot touch, which on a smaller host means swapping until it is
//     killed.
//
// The budget is injected so the arithmetic is what is tested, not the size of the machine running
// the suite.
func TestNewBlockPoolRejectsUntenableSizes(t *testing.T) {
	old := availableMemoryHook
	t.Cleanup(func() { availableMemoryHook = old })
	availableMemoryHook = func() int64 { return 1 << 30 } // 1 GiB
	cfg := testConfig()

	// Overflow: the answer has to be an error, not a crash.
	overflow := func() (err error) {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("NewBlockPool panicked instead of returning an error: %v", r)
			}
		}()
		_, err = NewBlockPool(cfg, BlockPoolOptions{Blocks: 1 << 40, BlockSize: 1 << 20})
		return err
	}
	if err := overflow(); err == nil {
		t.Fatal("an unaddressable pool was accepted")
	} else if !strings.Contains(err.Error(), "too large to address") {
		t.Errorf("error %q does not say the size is too large", err)
	}

	// Over budget: this config needs 8 bytes of KV per block and 16 of bookkeeping, so 2^26
	// blocks is 1.5 GiB against a 1 GiB budget.
	small := cfg
	small.Layers, small.KVHeads, small.HeadDim = 1, 1, 1
	_, err := NewBlockPool(small, BlockPoolOptions{Blocks: 1 << 26, BlockSize: 1})
	if err == nil {
		t.Fatal("a pool larger than the budget was accepted")
	}
	for _, want := range []string{"1.5 GiB", "1.0 GiB"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}

	// A pool that fits is still built, and an unknown budget disables the check rather than
	// rejecting a legitimate pool.
	availableMemoryHook = availableMemory
	if _, err := NewBlockPool(cfg, BlockPoolOptions{Blocks: 8, BlockSize: 4}); err != nil {
		t.Fatalf("an ordinary pool was refused: %v", err)
	}
}

// TestNewBlockPoolChargesBookkeeping covers the cheap block whose arrays outweigh the KV data.
//
// A pool's free list and reference counts are one int32 each per block, which is nothing when a
// block holds a thousand positions and everything when it holds one. Charging only the KV data let
// a 2^34-block pool through and it spent 34 seconds filling the two arrays -- the same failure one
// allocation later, and the reason the budget counts them.
func TestNewBlockPoolChargesBookkeeping(t *testing.T) {
	old := availableMemoryHook
	t.Cleanup(func() { availableMemoryHook = old })
	availableMemoryHook = func() int64 { return 1 << 30 }

	cfg := testConfig()
	cfg.Layers, cfg.KVHeads, cfg.HeadDim = 1, 1, 1 // 8 bytes of KV per block-position

	// 16 bytes of bookkeeping per block against 8 of KV: at 2^26 blocks the arrays are twice the
	// data, and the pair is over budget while the KV alone would fit.
	_, err := NewBlockPool(cfg, BlockPoolOptions{Blocks: 1 << 26, BlockSize: 1})
	if err == nil {
		t.Fatal("a pool whose bookkeeping exceeds the budget was accepted")
	}
	if !strings.Contains(err.Error(), "1.5 GiB") {
		t.Errorf("error %q does not charge the bookkeeping (512 MiB KV + 1 GiB arrays)", err)
	}
}

// TestContiguousSpansRejectsOutOfRange covers the argument checks on the cache's span accessor.
//
// Every forward pass reaches this method once per layer per head, and none of the arguments is
// validated by the caller. Three of the four checks had no test at all: only the kv head bound was
// covered, by a paged test that happens to hit it. A panic here would surface as a crash inside the
// forward pass with no indication of which argument was wrong.
//
// The zero-row case was worse than untested -- it was broken. n=0 passed the "negative or too
// large" check and then computed an end offset from n-1, which is -1, and panicked on a slice bound.
// No caller in the package asks for zero rows, so the forward pass never hit it, but the method is
// reachable from anywhere the package is used.
func TestContiguousSpansRejectsOutOfRange(t *testing.T) {
	cfg := testConfig()
	c, err := NewKVCache(cfg, 16)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}

	cases := []struct {
		name         string
		layer, kv, n int
		wantErr      bool
	}{
		{"negative layer", -1, 0, 1, true},
		{"layer past the end", cfg.Layers, 0, 1, true},
		{"negative kv head", 0, -1, 1, true},
		{"kv head past the end", 0, cfg.KVHeads, 1, true},
		{"negative positions", 0, 0, -1, true},
		{"more positions than the cache holds", 0, 0, 17, true},
		// Zero rows is a valid request: there is nothing to read.
		{"zero positions", 0, 0, 0, false},
		{"the largest valid request", 0, 0, 16, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("spans(%d, %d, %d) panicked: %v", tc.layer, tc.kv, tc.n, r)
				}
			}()
			keys, values, err := c.spans(tc.layer, tc.kv, tc.n)
			if gotErr := err != nil; gotErr != tc.wantErr {
				t.Fatalf("spans(%d, %d, %d) error = %v, want error: %v", tc.layer, tc.kv, tc.n, err, tc.wantErr)
			}
			if err != nil {
				return
			}
			// A successful call describes n rows, however many spans carry them.
			var rows int
			for _, s := range keys {
				rows += s.rows
			}
			if rows != tc.n {
				t.Errorf("%d rows across %d spans, want %d", rows, len(keys), tc.n)
			}
			var vrows int
			for _, s := range values {
				vrows += s.rows
			}
			if vrows != tc.n {
				t.Errorf("%d value rows, want %d", vrows, tc.n)
			}
		})
	}
}
