package llama

import "fmt"

// PrefixIndex remembers token prefixes whose keys and values are already in the
// pool, so a later request that starts with the same tokens can attach them
// instead of recomputing them.
//
// This is the reason to page a KV cache at all beyond memory accounting. A block
// holds one fixed run of positions, and for a given model those positions' keys and
// values depend only on the token prefix, never on which request produced them: two
// conversations with the same system prompt genuinely share the same KV state, not
// merely similar state. Sharing the block therefore costs one reference count and
// saves the whole prefill of the shared part -- the part that is usually the longest
// and most expensive, since it is re-sent verbatim on every turn.
//
// The index is keyed on the hash of a block's token prefix. A hash collision would
// attach the wrong keys and values, which produces fluent-looking garbage and is
// exactly the failure mode this project's tests exist to catch, so entries also
// store the token run and compare it. A collision then costs a rejected hit, not a
// wrong result.
//
// The index holds a reference on every block it remembers. That reference is what
// keeps the keys alive between the request that produced them and the request that
// reuses them, and it is what BlockPool.acquire gives up first when the pool runs
// short: a remembered prefix is a speculation about the future while a live
// sequence is not.
type PrefixIndex struct {
	pool *BlockPool

	// entries maps a block's token prefix to the block id holding it.
	entries map[uint64]prefixEntry

	// order records insertion order so eviction can drop the oldest entry. A
	// prefix index with no policy would either grow without bound or evict at
	// random; oldest-first matches the observation that a long conversation's early
	// prefixes are the ones most likely to be reused again.
	order []uint64
}

// prefixEntry is one remembered prefix.
type prefixEntry struct {
	tokens []int32
	block  int32
}

// maxPrefixEntries bounds the index.
//
// The index costs a hash map entry per remembered block, which is tiny next to the
// block itself, but an unbounded map is still a leak in a server that runs for
// weeks. The bound is generous: at the default 16 positions per block this
// remembers a 16384-token prefix, longer than any context this engine supports.
const maxPrefixEntries = 1024

// NewPrefixIndex returns an empty index over pool.
//
// Calling it more than once for one pool would give two indexes competing for the
// same blocks, so the pool keeps the one it was given.
func (p *BlockPool) EnablePrefixCache() *PrefixIndex {
	if p.index != nil {
		return p.index
	}
	p.index = &PrefixIndex{
		pool:    p,
		entries: make(map[uint64]prefixEntry),
	}
	return p.index
}

// PrefixIndex returns the pool's index, or nil when prefix reuse is off.
func (p *BlockPool) PrefixIndex() *PrefixIndex { return p.index }

// Match returns how many leading positions of tokens are already cached, and the
// block ids holding them.
//
// Only whole blocks are reported: the last partial block of a remembered prefix is
// not reusable, because a sequence that attached half a block would then append to
// it and corrupt the other holder's data. Reusing whole blocks only is what makes
// a shared block genuinely read-only until its single owner appends to it.
//
// The returned blocks are not retained: the caller must retain the ones it keeps
// before letting another request run, which PagedKVCache.adopt does.
func (x *PrefixIndex) Match(tokens []int32) (n int, blocks []int32) {
	if x == nil || len(tokens) == 0 {
		return 0, nil
	}
	size := x.pool.block
	blocks = make([]int32, 0, len(tokens)/size)
	for off := 0; off+size <= len(tokens); off += size {
		key := hashTokens(tokens[off : off+size])
		entry, ok := x.entries[key]
		if !ok || !sameTokens(entry.tokens, tokens[off:off+size]) {
			break
		}
		blocks = append(blocks, entry.block)
	}
	return len(blocks) * size, blocks
}

// Remember records that block id holds the blockSize tokens at off in tokens.
//
// A block is only remembered once it is full: a partial block will still be
// appended to, so its contents are not yet a reusable prefix.
func (x *PrefixIndex) Remember(tokens []int32, block int32) error {
	if x == nil {
		return nil
	}
	size := x.pool.block
	if len(tokens) != size {
		return fmt.Errorf("llama: prefix index needs %d tokens per entry, got %d", size, len(tokens))
	}
	if err := x.pool.retain(block); err != nil {
		return err
	}
	key := hashTokens(tokens)
	if old, ok := x.entries[key]; ok {
		// The same token run, or a collision. Either way the old entry is replaced,
		// so its reference has to go.
		if err := x.pool.release(old.block); err != nil {
			return err
		}
		x.entries[key] = prefixEntry{tokens: append([]int32(nil), tokens...), block: block}
		return nil
	}
	if len(x.entries) >= maxPrefixEntries {
		x.evictOldest()
	}
	x.entries[key] = prefixEntry{tokens: append([]int32(nil), tokens...), block: block}
	x.order = append(x.order, key)
	return nil
}

// evictOne gives up the oldest entry, reporting whether there was one.
func (x *PrefixIndex) evictOne() bool {
	if len(x.order) == 0 {
		return false
	}
	x.evictOldest()
	return true
}

// evictOldest drops the oldest entry and releases its reference.
func (x *PrefixIndex) evictOldest() {
	if len(x.order) == 0 {
		return
	}
	key := x.order[0]
	x.order = x.order[1:]
	entry, ok := x.entries[key]
	if !ok {
		return
	}
	delete(x.entries, key)
	_ = x.pool.release(entry.block)
}

// Len returns how many prefixes are remembered.
func (x *PrefixIndex) Len() int {
	if x == nil {
		return 0
	}
	return len(x.entries)
}

// hashTokens hashes a token run with FNV-1a.
//
// FNV is chosen because it is in the standard library, has no dependencies, and its
// collision behaviour is understood: a collision is possible and is handled by
// comparing the tokens, which is why the cost of a better hash is not worth
// paying here.
func hashTokens(tokens []int32) uint64 {
	const (
		offset = 14695981039346656037
		prime  = 1099511628211
	)
	h := uint64(offset)
	for _, tok := range tokens {
		// Mix the four bytes of the id, so a hash sees the token order and not just
		// the multiset.
		v := uint32(tok)
		for range 4 {
			h ^= uint64(v & 0xff)
			h *= prime
			v >>= 8
		}
	}
	return h
}

// sameTokens reports whether two token runs are equal.
func sameTokens(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
