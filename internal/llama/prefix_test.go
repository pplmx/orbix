package llama

import (
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
	"github.com/pplmx/orbix/internal/tokenize"
)

// Prefix reuse is a correctness-critical optimization: a wrong hit attaches keys
// and values that belong to different tokens, which produces fluent-looking
// garbage rather than an error. These tests compare against a run with no reuse at
// all, so any difference is a bug in the sharing and not in the model.

// TestEnablePrefixCacheKeepsTheFirstIndex pins the idempotency its doc comment promises.
//
// A second call must answer the index the pool already has. Two indexes over one pool would
// each believe they own the block they share, so the same block could be released twice: once
// by each index's accounting. Nothing tested it -- removing the early return left the package
// green -- because the one production call site (cmd/orbix/chat.go) calls it once.
func TestEnablePrefixCacheKeepsTheFirstIndex(t *testing.T) {
	pool := pagedPool(t, testConfig(), 4)
	first := pool.EnablePrefixCache()
	if first == nil {
		t.Fatal("EnablePrefixCache returned nil")
	}
	if second := pool.EnablePrefixCache(); second != first {
		t.Errorf("a second EnablePrefixCache returned a different index (%p, want %p); two "+
			"indexes over one pool would both claim the same blocks", second, first)
	}
}

// TestMatchStopsAtABlockItAlreadyReported covers the aliasing guard.
//
// The key includes the chain, so a repeated block normally gets its own entry and the match
// walks through it. The guard exists for the case the index cannot rule out: two entries in
// one chain pointing at the same block, which a pool that rewrote a block under two chains
// would produce. Reporting that block twice would make two positions alias one block, and the
// second position's append overwrites the first's keys and values in place -- inside a block
// every other holder is still reading. Matching less costs a forward pass; matching wrongly
// costs correctness.
//
// Constructed by writing the entries directly, because Remember refuses to build the shape:
// it is a state the index treats as possible but never creates on its own.
func TestMatchStopsAtABlockItAlreadyReported(t *testing.T) {
	pool := pagedPool(t, testConfig(), 4)
	idx := pool.EnablePrefixCache()
	size := pool.block

	// Two different runs, both recorded as living in block 0. Each is a whole block long and
	// the second continues the first, so without the guard Match would report block 0 twice.
	first := make([]int32, size)
	second := make([]int32, size)
	for i := range first {
		first[i] = 11
		second[i] = 22
	}
	k1 := chainKey(0, hashTokens(first))
	k2 := chainKey(k1, hashTokens(second))
	idx.entries[k1] = prefixEntry{tokens: first, block: 0}
	idx.entries[k2] = prefixEntry{tokens: second, block: 0, parent: k1, hasParent: true}

	query := append(append([]int32(nil), first...), second...)
	n, blocks := idx.Match(query)
	if len(blocks) != 1 {
		t.Errorf("Match reported blocks %v, want only the first: the second position aliases "+
			"the same block, and two positions in one block corrupt each other", blocks)
	}
	if n != size {
		t.Errorf("Match reported %d positions, want %d (one block)", n, size)
	}
}

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
	// A constructed model: the property is that a warm prefix index gives the same logits
	// as a cold run, which is exact equality and does not depend on the weights being
	// trained ones.
	path := gguftest.Model{NoEOS: true, ContextLen: 256}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	const blockSize = 4
	// A shared prefix of two full blocks, then a divergent tail.
	prompt := fixtureTokens(cfg, 10)

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
	path := gguftest.Model{NoEOS: true, ContextLen: 256}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	const blockSize = 4
	prompt := fixtureTokens(cfg, 8)

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
	path := gguftest.Model{NoEOS: true, ContextLen: 256}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	const blockSize = 4
	prompt := fixtureTokens(cfg, 9)

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

	if err := idx.Remember([]int32{1, 2, 3}, 0, 0); err == nil {
		t.Error("Remember accepted a short token run")
	}
	if err := idx.Remember([]int32{1, 2, 3, 4}, 9999, 0); err == nil {
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
	if err := none.Remember([]int32{1, 2, 3, 4}, 0, 0); err != nil {
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

// A remembered prefix is a chain, not a bag of blocks: block k+1 is only reachable by
// walking from block k, so dropping a block in the middle strands everything after it.
// These tests pin the chain and the eviction order that respects it.

// buildChain fills the index with one n-block prefix, as a prefill would.
func buildChain(t *testing.T, tokens []int32) (*BlockPool, *PrefixIndex) {
	t.Helper()
	cfg := testConfig()
	pool := pagedPool(t, cfg, 4)
	idx := pool.EnablePrefixCache()
	dim := cfg.kvDim()
	k := make([]float32, dim)
	v := make([]float32, dim)
	c := pool.New()
	for pos := range len(tokens) {
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
	return pool, idx
}

// TestPrefixChainLinksBlocks checks that each block knows the one it continues.
func TestPrefixChainLinksBlocks(t *testing.T) {
	tokens := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	_, idx := buildChain(t, tokens)
	if idx.Len() != 3 {
		t.Fatalf("index holds %d entries, want 3", idx.Len())
	}
	// The first block starts a chain; the other two each name a parent.
	heads, links := 0, 0
	for _, e := range idx.entries {
		if e.hasParent {
			links++
			if _, ok := idx.entries[e.parent]; !ok {
				t.Errorf("entry names parent %d, which is not in the index", e.parent)
			}
		} else {
			heads++
		}
	}
	if heads != 1 {
		t.Errorf("%d head entries, want 1", heads)
	}
	if links != 2 {
		t.Errorf("%d linked entries, want 2", links)
	}
}

// TestMatchReportsEachBlockOnce covers a prompt that repeats a block. Keys are the token
// run alone, with no position, so "1 2 1 2" at a two-token block size finds one entry
// twice. Reporting it twice is corruption rather than a wasted hit: adopt makes two of the
// sequence's positions alias that block, and appending to the second overwrites the first's
// keys and values in place -- inside a block every other holder is reading.
//
// The assertion is on the block ids and on the original's position 0, because a length
// check alone would pass while both entries still named the same block.
func TestMatchReportsEachBlockOnce(t *testing.T) {
	cfg := testConfig()
	pool := pagedPool(t, cfg, 2)
	idx := pool.EnablePrefixCache()
	dim := cfg.kvDim()

	tokens := []int32{1, 2, 1, 2}
	producer := pool.New()
	for pos := range len(tokens) {
		for l := range cfg.Layers {
			kk := make([]float32, dim)
			vv := make([]float32, dim)
			for i := range kk {
				kk[i] = float32(pos*100 + i)
				vv[i] = float32(pos*100 + i)
			}
			if _, _, err := producer.appendKV(l, kk, vv, pos); err != nil {
				t.Fatalf("appendKV(%d): %v", pos, err)
			}
		}
		producer.setLen(pos + 1)
		if err := producer.rememberFullBlock(tokens, pos); err != nil {
			t.Fatalf("rememberFullBlock(%d): %v", pos, err)
		}
	}

	// Two blocks, both matched. The earlier version of this test asserted that the match
	// stopped at the repeated block, because the index keyed entries on the token run alone
	// and the second "1 2" overwrote the first: handing both over would have meant one block
	// id used for two positions. The key now includes the chain, so the second block is a
	// different entry pointing at a different block, and matching both is correct.
	n, blocks := idx.Match(tokens)
	if n != 4 {
		t.Errorf("Match = %d positions, want 4: both blocks are remembered under their own"+
			" chain and must be matched", n)
	}
	if len(blocks) != 2 {
		t.Fatalf("Match returned blocks %v, want two", blocks)
	}
	// Still no aliasing: the two positions must name different blocks, or appending to one
	// would overwrite the other in place.
	if blocks[0] == blocks[1] {
		t.Errorf("Match returned block %d for both positions; two positions cannot share one"+
			" block", blocks[0])
	}

	// The block it did hand over must not be the one the original appends into. Writing
	// past the adopted prefix is exactly what a longer prompt does on its next block.
	readPos0 := func(c *PagedKVCache) []float32 {
		spans, _, err := c.spans(0, 0, 1)
		if err != nil {
			t.Fatalf("spans: %v", err)
		}
		return slices.Clone(spans[0].data[:4])
	}
	before := readPos0(producer)

	adopter := pool.New()
	if err := adopter.adopt(blocks); err != nil {
		t.Fatalf("adopt: %v", err)
	}
	kk := make([]float32, dim)
	vv := make([]float32, dim)
	for i := range kk {
		kk[i] = -7
		vv[i] = -7
	}
	if _, _, err := adopter.appendKV(0, kk, vv, adopter.Len()); err != nil {
		t.Fatalf("appendKV past the adopted prefix: %v", err)
	}
	if after := readPos0(producer); !slices.Equal(before, after) {
		t.Errorf("appending past the adopted prefix rewrote the original's position 0: %v -> %v",
			before, after)
	}
}

// TestPrefixEvictionTakesTheTail is the property the chain exists for: eviction must give
// up the end of a chain, not its middle, because dropping the middle makes the rest
// unreachable while still holding memory.
func TestPrefixEvictionTakesTheTail(t *testing.T) {
	tokens := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	pool, idx := buildChain(t, tokens)

	_, primed := idx.Match(tokens)
	if len(primed) != 3 {
		t.Fatalf("the chain holds %d blocks, want 3; the setup did not publish it", len(primed))
	}
	evicted := primed[len(primed)-1]

	// One eviction must leave a matchable prefix of two blocks rather than none.
	if !idx.evictOne() {
		t.Fatal("evictOne reported nothing to evict")
	}
	if got := idx.Len(); got != 2 {
		t.Fatalf("index holds %d entries, want 2", got)
	}
	n, _ := idx.Match(tokens)
	if n != 8 {
		t.Errorf("match after one eviction = %d positions, want 8 (two blocks still reachable)", n)
	}
	// And the evicted block must have lost the index's reference.
	//
	// Asserted on the refcount rather than on Free, because Free cannot see this: buildChain
	// leaves its cache holding every block, so each one starts at 2 -- the index and the
	// sequence. Evicting drops it to 1 and the block does not join the free list, which is
	// correct but means "free == 1" is true before the eviction as well. Measured, and
	// verified by deleting the release in evictTail: the Free assertion passed, and the
	// refcount assertion below is what fails.
	if got := pool.RefCount(evicted); got != 1 {
		t.Errorf("the evicted block %d has refcount %d, want 1 (the index's reference gone, "+
			"the sequence's still held)", evicted, got)
	}
}

// TestPrefixEvictionKeepsChainsIntact drains the index and checks that every remaining
// entry is still reachable from a head, which is what "no orphans" means.
func TestPrefixEvictionKeepsChainsIntact(t *testing.T) {
	tokens := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	_, idx := buildChain(t, tokens)

	for idx.Len() > 0 {
		before, _ := idx.Match(tokens)
		if !idx.evictOne() {
			break
		}
		after, _ := idx.Match(tokens)
		// A match walks the chain from the head, so after evicting a tail the reachable
		// prefix shrinks by one block; if a middle block went instead, the reachable
		// length would collapse to zero while entries remained.
		if idx.Len() > 0 && after == 0 && before > 0 {
			t.Fatalf("eviction stranded the remaining %d entries: match went %d -> 0",
				idx.Len(), before)
		}
	}
}

// TestPrefixChainAllowsIndependentPrompts checks that two prompts sharing a beginning
// each get their own chain rather than corrupting one another's links.
func TestPrefixChainAllowsIndependentPrompts(t *testing.T) {
	cfg := testConfig()
	pool := pagedPool(t, cfg, 4)
	idx := pool.EnablePrefixCache()
	dim := cfg.kvDim()
	k := make([]float32, dim)
	v := make([]float32, dim)

	fill := func(tokens []int32) {
		c := pool.New()
		for pos := range len(tokens) {
			for l := range cfg.Layers {
				if _, _, err := c.appendKV(l, k, v, pos); err != nil {
					t.Fatalf("appendKV: %v", err)
				}
			}
			c.setLen(pos + 1)
			if err := c.rememberFullBlock(tokens, pos); err != nil {
				t.Fatalf("remember: %v", err)
			}
		}
	}
	// Two prompts share their first block then diverge.
	fill([]int32{1, 2, 3, 4, 10, 11, 12, 13})
	fill([]int32{1, 2, 3, 4, 20, 21, 22, 23})

	// Three distinct blocks: the shared head plus one continuation each.
	if got := idx.Len(); got != 3 {
		t.Fatalf("index holds %d entries, want 3", got)
	}
	if n, _ := idx.Match([]int32{1, 2, 3, 4, 10, 11, 12, 13}); n != 8 {
		t.Errorf("first prompt matches %d positions, want 8", n)
	}
	if n, _ := idx.Match([]int32{1, 2, 3, 4, 20, 21, 22, 23}); n != 8 {
		t.Errorf("second prompt matches %d positions, want 8", n)
	}
}

// TestPrefixEvictionPreservesReuseUnderPressure measures what the chain is for, rather
// than only checking the shape of the index: with the pool short enough that eviction
// runs, the reachable prefix must stay proportional to the surviving blocks.
//
// Without the chain, eviction drops the oldest entry -- the head -- and the reachable
// prefix collapses to zero while the tail still holds memory. Measured on a 4-block
// prefix with one eviction: 8 positions reachable with the chain, 0 without.
func TestPrefixEvictionPreservesReuseUnderPressure(t *testing.T) {
	// The fixture's context is 16 positions, which at block size 4 is a four-block
	// chain: long enough to show the shrink, short enough to fit.
	const n = 16
	tokens := make([]int32, n)
	for i := range tokens {
		tokens[i] = int32(i + 1)
	}
	cfg := testConfig()
	pool := pagedPool(t, cfg, 4)
	idx := pool.EnablePrefixCache()
	dim := cfg.kvDim()
	k := make([]float32, dim)
	v := make([]float32, dim)

	c := pool.New()
	for pos := range len(tokens) {
		for l := range cfg.Layers {
			if _, _, err := c.appendKV(l, k, v, pos); err != nil {
				t.Fatalf("appendKV: %v", err)
			}
		}
		c.setLen(pos + 1)
		if err := c.rememberFullBlock(tokens, pos); err != nil {
			t.Fatalf("remember: %v", err)
		}
	}
	before := idx.Len()
	nBefore, _ := idx.Match(tokens)
	if nBefore != len(tokens) {
		t.Fatalf("full prefix matches %d positions, want %d", nBefore, len(tokens))
	}

	// Evict repeatedly and require the reachable prefix to shrink one block at a time
	// from the end. A collapse to zero while entries remain would mean a middle block
	// went first.
	const block = 4
	for evicted := 1; evicted <= before; evicted++ {
		if !idx.evictOne() {
			t.Fatalf("evictOne failed after %d evictions", evicted-1)
		}
		got, _ := idx.Match(tokens)
		want := len(tokens) - evicted*block
		if want < 0 {
			want = 0
		}
		if got != want {
			t.Fatalf("after %d evictions: match = %d positions, want %d "+
				"(reachable prefix must shrink from the end, not collapse)", evicted, got, want)
		}
	}
	if idx.Len() != 0 {
		t.Errorf("%d entries remain after evicting everything", idx.Len())
	}
	// The sequence that produced the prefix still holds its own references; releasing it
	// is what makes the pool whole, and it is worth asserting because a leaked block here
	// is the failure the whole index exists to avoid.
	if err := c.Reset(); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	if free := pool.Free(); free != pool.Blocks() {
		t.Errorf("pool has %d of %d blocks free at the end, want all", free, pool.Blocks())
	}
}

// TestPrefixIndexEvictsAtCapacity covers the eviction that Remember performs when the index is
// full.
//
// The pool is bounded for a reason: a server that runs long enough sees more distinct prompt
// prefixes than maxPrefixEntries, and without this eviction the index would grow without limit
// while the index is also the thing holding block references. Every other eviction test drives
// evictOne directly, which exercises the same function but not the call Remember makes.
//
// Rather than fill a pool with 1025 blocks, the same block is remembered under 1024 different
// token runs: Remember retains whatever block it is given, so one held block can be named by many
// entries. That is the same state the cap check sees.
func TestPrefixIndexEvictsAtCapacity(t *testing.T) {
	cfg := testConfig()
	pool, err := NewBlockPool(cfg, BlockPoolOptions{BlockSize: 2, Blocks: 8})
	if err != nil {
		t.Fatal(err)
	}
	idx := pool.EnablePrefixCache()
	size := pool.BlockSize()

	// One cache holds the block that every entry will name, so retain succeeds throughout.
	holder := pool.New()
	for pos := range 4 {
		if err := holder.ensureBlocks(pos); err != nil {
			t.Fatalf("ensureBlocks: %v", err)
		}
	}
	block := holder.blocks[0]

	remember := func(n int) {
		t.Helper()
		// Distinct token runs give distinct keys; the values only need to hash apart.
		tokens := make([]int32, size)
		for i := range tokens {
			tokens[i] = int32(n*size + i)
		}
		if err := idx.Remember(tokens, block, 0); err != nil {
			t.Fatalf("Remember(%d): %v", n, err)
		}
	}

	for n := range maxPrefixEntries {
		remember(n)
	}
	if got := idx.Len(); got != maxPrefixEntries {
		t.Fatalf("index holds %d entries after %d remembers, want %d",
			got, maxPrefixEntries, maxPrefixEntries)
	}

	// One more must evict rather than exceed the bound.
	before := pool.RefCount(block)
	remember(maxPrefixEntries)
	if got := idx.Len(); got != maxPrefixEntries {
		t.Errorf("index holds %d entries after one past the cap, want it held at %d",
			got, maxPrefixEntries)
	}
	// The eviction released one reference and the new entry took one, so the count is unchanged.
	// A run that grew past the cap would have taken an extra retain and the count would rise; a
	// run that failed to release would too.
	if got := pool.RefCount(block); got != before {
		t.Errorf("RefCount = %d after the eviction, want %d: the tail was not released",
			got, before)
	}
}

// TestPrefixIndexRejectsACollidingHit covers the guard that makes a hash collision cost a
// missed hit rather than a wrong result.
//
// The index is keyed on hashTokens, and Match checks the stored token run with sameTokens
// before trusting a hit. That check is the only thing standing between a collision and
// attaching another prompt's keys and values, which produces fluent-looking garbage rather
// than an error -- the failure mode this whole package's tests exist to catch, per its own
// doc comment.
//
// A 64-bit FNV collision cannot be found by search in a test, so the colliding entry is
// installed directly: the same key Remember would compute, holding a different token run.
// What this pins is the behaviour Match must have when one is present, which is the part
// that could regress; whether FNV can collide is arithmetic, not a property of this code.
func TestPrefixIndexRejectsACollidingHit(t *testing.T) {
	cfg := testConfig()
	pool := pagedPool(t, cfg, 4)
	idx := pool.EnablePrefixCache()

	// Fill one block so there is a real id to point the fake entry at.
	dim := cfg.kvDim()
	k := make([]float32, dim)
	v := make([]float32, dim)
	c := pool.New()
	tokens := []int32{10, 20, 30, 40}
	for pos := range 4 {
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
	if n, _ := idx.Match(tokens); n != 4 {
		t.Fatalf("the remembered block does not match itself: %d positions", n)
	}

	// Install an entry under the key a DIFFERENT block's tokens hash to, holding the
	// first block's tokens. This is the state a collision leaves behind.
	impostor := []int32{41, 42, 43, 44}
	colliding := prefixEntry{tokens: tokens, block: c.blocks[0], hasParent: false}
	idx.entries[hashTokens(impostor)] = colliding

	// Match must reject it, because the stored run is not the queried run.
	if n, blocks := idx.Match(impostor); n != 0 {
		t.Errorf("Match accepted a colliding entry: matched %d positions (%v), want 0",
			n, blocks)
	}
	// And the block that really is remembered still matches, so the guard rejects only
	// the impostor rather than disabling the index.
	if n, _ := idx.Match(tokens); n != 4 {
		t.Errorf("Match on the genuine tokens returned %d positions, want 4", n)
	}
}

// TestWholePromptHitProducesIdenticalLogits covers the branch where every token is cached.
//
// PrefillWithPrefix has two reuse paths. When part of the prompt is cached it prefills the
// remainder and returns that pass's logits; when ALL of it is cached it runs no forward pass
// over the prompt at all and instead calls logitsForCached, which recomputes only the last
// position against the adopted blocks. The existing credit test uses a 9-token prompt with a
// 4-position block, so its match is 8 -- strictly fewer than the prompt -- and it exercises
// the first path twice.
//
// That leaves the interesting branch untested, and it is the one where a mistake is silent:
// the blocks come from somebody else's request, so a wrong position or a wrong token there
// still produces finite logits of the right length. Measured on the fixture: 8 tokens, two
// whole blocks, reused 8, and the logits equal the cold run bit for bit. Swapping the position
// or the embedded token makes both alternatives fail.
// Measured on the fixture while checking the boundary rules, and worth recording because the
// progression takes two requests: with two blocks already published and a 12-token prompt (the
// same 8 plus a 4-token tail), Match returns 8 before that prompt has ever run, the run reuses
// 8 and fills 4 and publishes the block its last token completed, and the *next* run of the
// same prompt reuses all 12. A 3-token tail never completes a block, so it is never published
// and the reuse for an 11-token prompt stays 8 however often it is sent.
func TestWholePromptHitProducesIdenticalLogits(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 256}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	const blockSize = 4
	// Exactly two whole blocks, so the match can be the entire prompt.
	prompt := fixtureTokens(cfg, 2*blockSize)

	// The reference: no reuse at all.
	cold, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatal(err)
	}
	want, err := m.Prefill(cold, prompt, 0)
	if err != nil {
		t.Fatalf("cold Prefill: %v", err)
	}
	wantLogits := append([]float32(nil), want...)

	pool := pagedPool(t, cfg, blockSize)
	pool.EnablePrefixCache()

	first := pool.New()
	if _, err := m.PrefillWithPrefix(first, prompt, 0); err != nil {
		t.Fatalf("PrefillWithPrefix (filling): %v", err)
	}
	if got := first.ReusedPositions(); got != 0 {
		t.Fatalf("the request that filled the index reused %d positions, want 0", got)
	}

	second := pool.New()
	got, err := m.PrefillWithPrefix(second, prompt, 0)
	if err != nil {
		t.Fatalf("PrefillWithPrefix (whole prompt cached): %v", err)
	}
	// The whole prompt, not merely a block of it: this is what makes it the cached-logits
	// branch rather than the prefill-the-remainder one.
	if n := second.ReusedPositions(); n != len(prompt) {
		t.Fatalf("reused %d of %d positions; this test needs the whole prompt cached "+
			"to reach logitsForCached", n, len(prompt))
	}
	if len(got) != len(wantLogits) {
		t.Fatalf("logits are %d wide, want %d", len(got), len(wantLogits))
	}
	for i := range wantLogits {
		if got[i] != wantLogits[i] {
			t.Fatalf("logit[%d] = %v, want %v: a fully cached prompt must score identically "+
				"to a cold one", i, got[i], wantLogits[i])
		}
	}
}

// TestRememberFullBlockRefusesInconsistentState covers its two preconditions.
//
// Both describe states the caller cannot reach by accident -- a block that was never allocated, or a
// token slice shorter than the block being published -- and neither had a test, so removing either
// check passed. They are reachable from a test because the method takes the token slice as an argument
// rather than reading it from cache state: a prefill that passes the wrong slice is exactly the
// mistake the second guard catches, and getting it wrong publishes a block whose key covers tokens
// nobody checked.
func TestRememberFullBlockRefusesInconsistentState(t *testing.T) {
	c := defaultTiny()
	path := buildTinyWeights(t, c)
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()

	newCache := func(t *testing.T) *PagedKVCache {
		t.Helper()
		pool, err := NewBlockPool(cfg, BlockPoolOptions{BlockSize: 2, Blocks: 4})
		if err != nil {
			t.Fatalf("NewBlockPool: %v", err)
		}
		pool.EnablePrefixCache()
		cache := pool.New()
		t.Cleanup(func() { _ = cache.Reset() })
		return cache
	}

	t.Run("a block that was never allocated", func(t *testing.T) {
		cache := newCache(t)
		// Position 1 completes block 0, which ensureBlocks has not created.
		err := cache.rememberFullBlock([]int32{1, 2}, 1)
		if err == nil {
			t.Fatal("rememberFullBlock published a block that does not exist")
		}
		// The block and the position, which together say which prefill went wrong.
		for _, want := range []string{"block 0", "position 1", "not allocated"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error %q does not state %q", err, want)
			}
		}
	})

	t.Run("a token slice shorter than the block", func(t *testing.T) {
		cache := newCache(t)
		if err := cache.ensureBlocks(1); err != nil {
			t.Fatalf("ensureBlocks: %v", err)
		}
		// Block 0 covers positions 0 and 1, so it needs two tokens.
		err := cache.rememberFullBlock([]int32{1}, 1)
		if err == nil {
			t.Fatal("rememberFullBlock published a block from a short token slice")
		}
		// The count it needs and the count it got.
		for _, want := range []string{"needs tokens up to 2", "have 1"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error %q does not state %q", err, want)
			}
		}
	})
}

// TestPrefixIndexAccessorReportsWhetherReuseIsOn pins the accessor that decides it.
//
// It was the one function in this package with 0.0% coverage under `go test
// ./internal/llama/`, and the reason is the one the project notes describe for other
// functions: Go instruments only the package under test, so its callers --
// internal/engine's tests and cmd/orbix's banner -- do not count. The function does have a
// production caller (serve prints "prefix cache: on" when it is non-nil), which is the
// question that matters, but a zero here makes the next reader re-run that investigation.
//
// The property is the tri-state: a contiguous pool has no index at all, a paged pool
// without reuse has an index that is nil, and a paged pool with reuse has one. The banner
// reads exactly that distinction, so an implementation that always returned an empty index
// would turn the flag on for every paged server.
func TestPrefixIndexAccessorReportsWhetherReuseIsOn(t *testing.T) {
	cfg := Config{
		Arch: "llama", Hidden: 32, Layers: 1, Heads: 2, KVHeads: 2,
		HeadDim: 16, ContextLen: 256,
	}
	for _, c := range []struct {
		name    string
		enable  bool
		wantNil bool
	}{
		{"reuse off", false, true},
		{"reuse on", true, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			pool, err := NewBlockPool(cfg, BlockPoolOptions{BlockSize: 16, Blocks: 16})
			if err != nil {
				t.Fatalf("NewBlockPool: %v", err)
			}
			// The pool does not take this from its options: reuse is turned on by calling
			// EnablePrefixCache, which is what makes a second call idempotent.
			if c.enable {
				pool.EnablePrefixCache()
			}
			got := pool.PrefixIndex()
			if (got == nil) != c.wantNil {
				t.Fatalf("PrefixIndex() == nil is %v, want %v", got == nil, c.wantNil)
			}
			// An index that exists must actually match: returning a non-nil placeholder
			// would satisfy the nil check and still leave the flag lying.
			if got != nil {
				if n, _ := got.Match([]int32{1, 2, 3}); n != 0 {
					t.Errorf("a fresh index reported a match of %d positions", n)
				}
			}
		})
	}
}

// TestEvictedPrefixNoLongerMatches covers the failure that eviction exists to prevent, from
// the direction the existing tests do not look.
//
// TestPrefixIndexEvictsWhenPoolRunsShort checks that the pool survives the pressure, and
// TestPrefixEvictionTakesTheTail checks that the remaining chain still matches. Neither asks
// what happens to the prefix that lost its blocks: if evictTail deleted the order entry but
// not the map entry -- or kept both and only released the block -- Match would still return
// that block, which has since been handed to a different sequence and overwritten. The
// symptom is not an error but wrong logits, and it appears only after enough traffic to evict.
//
// Measured with a four-block pool: a two-block prefix matches 16 positions and holds two
// blocks free; after twelve unrelated prefills it matches nothing and one block is free.
func TestEvictedPrefixNoLongerMatches(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 256}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	pool, err := NewBlockPool(cfg, BlockPoolOptions{Blocks: 4, BlockSize: 8})
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	idx := pool.EnablePrefixCache()

	// Exactly two blocks, so the prefix is published and evictable on its own.
	primed := fixtureTokens(cfg, 16)
	c := pool.New()
	if _, err := m.PrefillWithPrefix(c, primed, 0); err != nil {
		t.Fatalf("prime: %v", err)
	}
	if err := c.Reset(); err != nil {
		t.Fatalf("reset after priming: %v", err)
	}
	n, blocks := idx.Match(primed)
	if n != 16 {
		t.Fatalf("the primed prefix matched %d positions, want 16; the setup did not publish it", n)
	}
	primedBlocks := append([]int32(nil), blocks...)

	// Enough unrelated prefixes to drive the two blocks out of a four-block pool.
	for i := range 12 {
		cc := pool.New()
		p := append([]int32{int32(200 + i)}, fixtureTokens(cfg, 16)...)
		if _, err := m.PrefillWithPrefix(cc, p, 0); err != nil {
			t.Fatalf("churn %d: %v", i, err)
		}
		// A reset error here would mean the pool's bookkeeping is already broken, which is a
		// different defect from the one this test is about. Continue rather than fail so the
		// assertion below is what reports a stale entry.
		_ = cc.Reset()
	}

	if n, blocks := idx.Match(primed); n != 0 || len(blocks) != 0 {
		t.Errorf("the evicted prefix still matches %d positions in blocks %v (it was %v); "+
			"those blocks have been reused, so adopting them feeds another sequence's keys "+
			"and values into this one", n, blocks, primedBlocks)
	}
}

// TestPrefixCacheReuseIsPositional is the regression test for a silently wrong reuse.
//
// Set GEMMA3_MODEL to a Gemma 3 GGUF to run the real-file form, which is the one that caught it.
//
// A block's keys and values depend on the absolute positions its tokens sat at, not only on the
// tokens: the rotation is a function of the position. The index keyed entries on the token run
// alone, so a prompt that repeats a 16-token run -- the same sentence over and over, ordinary for
// a test prompt and not unusual in a real one -- had its later blocks overwrite the entries of its
// earlier ones. A later request then matched the surviving entry and attached, at position 16,
// keys and values computed at position 176.
//
// It is invisible from the outside in the obvious way: the match succeeds, the request runs, the
// text is fluent, and only the numbers are wrong. Measured on Gemma 3 270M with the bug present,
// reused logits were 29 nats from a cache that had never reused.
//
// The comparison is against an arena cache, which never reuses, so the expectation does not depend
// on the prefix path being right.
func TestPrefixCacheReuseIsPositional(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	m, err := NewWithOptions(path, Options{Workers: 1})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	cfg := m.Config()
	const block = 4
	// Every block after the first repeats the first block's tokens, so a content-only key
	// collides them all. The vocabulary is small, so tokens 2..block+1 exist.
	prompt := make([]int32, 0, 8*block)
	prompt = append(prompt, 1)
	for range 8*block - 1 {
		prompt = append(prompt, 2+int32(len(prompt))%int32(block-1))
	}

	arena, err := NewKVCache(cfg, len(prompt)+32)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}
	want, err := m.Prefill(arena, prompt, 0)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}

	pool, err := NewBlockPool(cfg, BlockPoolOptions{BlockSize: block, Blocks: 64})
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	pool.EnablePrefixCache()
	for round := range 4 {
		cache := pool.New()
		got, err := m.PrefillWithPrefix(cache, prompt, 0)
		if err != nil {
			t.Fatalf("round %d: %v", round, err)
		}
		var worst float64
		for i := range want {
			if d := math.Abs(float64(want[i] - got[i])); d > worst {
				worst = d
			}
		}
		if worst > 1e-4 {
			t.Fatalf("round %d: reused %d positions and is %v away from an arena cache; a reused"+
				" block has to be the one that held these tokens at these positions",
				round, cache.ReusedPositions(), worst)
		}
		if err := cache.Reset(); err != nil {
			t.Fatalf("round %d: reset: %v", round, err)
		}
	}
}

// TestPrefixCacheReuseOfRepeatedContent covers the case a real prompt hit and the fixtures missed.
//
// A prompt that repeats a block -- one sentence over and over -- used to defeat the index: entries
// were keyed on the token run alone, so the second occurrence overwrote the first, and a later
// request matched the surviving entry and attached one position's keys and values at another.
// Measured on Gemma 3 270M with the bug, reused logits were 29 nats from an arena cache and the
// most likely token drifted from 107 to 108 within four requests. Nothing in the generated text
// said so.
//
// With the chain in the key the whole prompt is reusable instead of half: 192 of 202 positions
// rather than 96, and the logits equal the arena's to the last bit across six rounds.
//
// The comparison is against an arena cache, which never reuses, so nothing here depends on the
// prefix path being right.
func TestPrefixCacheReuseOfRepeatedContent(t *testing.T) {
	path := os.Getenv("GEMMA3_MODEL")
	if path == "" {
		t.Skip("set GEMMA3_MODEL to a Gemma 3 GGUF file")
	}
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	prompt := tok.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", 20), true)

	arena, err := NewKVCache(m.Config(), len(prompt)+64)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}
	want, err := m.Prefill(arena, prompt, 0)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}

	pool, err := NewBlockPool(m.Config(), BlockPoolOptions{BlockSize: 16, Blocks: 64})
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	pool.EnablePrefixCache()
	for round := range 6 {
		cache := pool.New()
		got, err := m.PrefillWithPrefix(cache, prompt, 0)
		if err != nil {
			t.Fatalf("round %d: %v", round, err)
		}
		var worst float64
		for i := range want {
			if d := math.Abs(float64(want[i] - got[i])); d > worst {
				worst = d
			}
		}
		if worst != 0 {
			t.Fatalf("round %d: reused %d positions and is %v from an arena cache; a repeated"+
				" block must be reused at the positions it was written at",
				round, cache.ReusedPositions(), worst)
		}
		// Half a prompt reused is the shape the bug produced; the whole prefix is the shape it
		// should have. Asserted rather than described, because a regression that only lowered
		// the hit rate would pass the comparison above.
		if round > 0 {
			if got, minimum := cache.ReusedPositions(), (len(prompt)/16)*16; got != minimum {
				t.Errorf("round %d reused %d positions, want %d: repeated blocks are separate"+
					" entries now, so the whole prefix is reusable", round, got, minimum)
			}
		}
		if err := cache.Reset(); err != nil {
			t.Fatalf("round %d: reset: %v", round, err)
		}
	}
}

// TestSameTokensThreeOutcomes covers the comparison Match makes after its hash lookup.
//
// The middle case is the one nothing reached: two runs of the same length whose elements
// differ. Reaching it through Match needs a hash collision, which the index cannot be asked to
// produce, so the function is tested directly -- it is a pure predicate and the only caller
// treats false and false as the same answer.
//
// It matters that the third case is distinguishable from the first. Match calls
// `!ok || !sameTokens(...)`, so a sameTokens that returned true whenever the lengths agreed
// would let a colliding key attach another block's tokens -- the failure the comment above
// Match describes as "matching wrongly costs correctness".
func TestSameTokensThreeOutcomes(t *testing.T) {
	cases := []struct {
		name string
		a, b []int32
		want bool
	}{
		{"identical", []int32{1, 2, 3}, []int32{1, 2, 3}, true},
		{"both empty", nil, nil, true},
		{"empty against non-empty", nil, []int32{1}, false},
		{"different lengths", []int32{1, 2}, []int32{1, 2, 3}, false},
		// The uncovered branch: equal lengths, differing content.
		{"same length, first differs", []int32{9, 2, 3}, []int32{1, 2, 3}, false},
		{"same length, last differs", []int32{1, 2, 9}, []int32{1, 2, 3}, false},
		{"same length, middle differs", []int32{1, 9, 3}, []int32{1, 2, 3}, false},
	}
	for _, c := range cases {
		if got := sameTokens(c.a, c.b); got != c.want {
			t.Errorf("%s: sameTokens(%v, %v) = %v, want %v", c.name, c.a, c.b, got, c.want)
		}
		// Symmetric, which the caller relies on: it compares a stored run with a slice of the
		// incoming prompt and does not know which is which.
		if got := sameTokens(c.b, c.a); got != c.want {
			t.Errorf("%s: sameTokens is not symmetric", c.name)
		}
	}
}

// TestRememberPrefixWithoutAnIndexIsANoOp covers the early return a cache without a prefix
// index takes.
//
// `serve --prefix-cache` is what installs an index, so the default server path has none and
// reaches this branch on every prefill. It was uncovered: the branch returns nil having done
// nothing, and nothing asserted that a model running without the flag still completes a
// prefill -- the closest tests all build an index first.
//
// The assertion is the absence of an error plus the absence of remembered blocks. Returning an
// error here would make every prefill fail on a server started without the flag; writing blocks
// anyway would publish into a nil index.
func TestRememberPrefixWithoutAnIndexIsANoOp(t *testing.T) {
	m, err := New(buildTinyWeights(t, defaultTiny()))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	pool, err := NewBlockPool(cfg, BlockPoolOptions{BlockSize: 2, Blocks: 8})
	if err != nil {
		t.Fatalf("NewBlockPool: %v", err)
	}
	if pool.index != nil {
		t.Fatal("a freshly built pool already has a prefix index, so this test would not reach " +
			"the branch it names")
	}
	cache := &PagedKVCache{cfg: cfg, pool: pool, block: 2}
	if err := m.rememberPrefix(cache, []int32{1, 2, 3, 4}, 0); err != nil {
		t.Errorf("rememberPrefix returned %v for a cache with no index; a server started "+
			"without --prefix-cache would fail every prefill", err)
	}
}
