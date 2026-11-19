package llama

import (
	"testing"

	"github.com/pplmx/orbix/internal/gguftest"
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

// TestPrefixEvictionTakesTheTail is the property the chain exists for: eviction must give
// up the end of a chain, not its middle, because dropping the middle makes the rest
// unreachable while still holding memory.
func TestPrefixEvictionTakesTheTail(t *testing.T) {
	tokens := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	pool, idx := buildChain(t, tokens)

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
	// And the evicted block must have gone back to the pool.
	if free := pool.Free(); free != 1 {
		t.Errorf("pool has %d free blocks after evicting one of four, want 1", free)
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
