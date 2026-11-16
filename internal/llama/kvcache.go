package llama

import "fmt"

// Cache is what a forward pass needs from a key/value cache.
//
// Both layouts satisfy it: KVCache keeps one contiguous arena per sequence and
// PagedKVCache draws fixed-size blocks from a pool shared by every sequence.
// Attention is written against this interface so that paging is a change to where
// the bytes live, not to the arithmetic that reads them.
//
// The interface is deliberately narrow -- append one position, ask for the rows to
// attend over, report the length -- because those three operations are the whole
// contract. Widening it would let a forward pass start depending on storage
// layout, which is what this abstraction exists to prevent.
type Cache interface {
	// Len is how many positions have been written.
	Len() int
	// Cap is the position limit the cache enforces.
	Cap() int
	// appendKV stores one position's key and value for a layer and returns the
	// slices to attend over.
	appendKV(layer int, k, v []float32, pos int) ([]float32, []float32, error)
	// spans returns the rows to attend over for one layer and kv head as one
	// span per contiguous run.
	spans(layer, kv, n int) (keys, values []span, err error)
	// setLen records how many positions are live once a pass completes.
	setLen(n int)
	// Reset drops every stored position and returns whatever storage the cache
	// holds. A contiguous cache keeps its arena; a paged one hands its blocks back
	// to the pool so another sequence can use them.
	Reset() error
}

// setLen records the live position count.
func (c *KVCache) setLen(n int) { c.len = n }

// setLen records the live position count.
func (c *PagedKVCache) setLen(n int) { c.len = n }

// appendKV stores one position's key and value for a layer, allocating the block
// that position falls in when it is the first to need it.
func (c *PagedKVCache) appendKV(layer int, k, v []float32, pos int) ([]float32, []float32, error) {
	if layer < 0 || layer >= c.cfg.Layers {
		return nil, nil, fmt.Errorf("llama: layer %d outside the cache (%d)", layer, c.cfg.Layers)
	}
	if pos < 0 || pos >= c.cfg.ContextLen {
		return nil, nil, fmt.Errorf("llama: position %d outside the context (%d)", pos, c.cfg.ContextLen)
	}
	dim := c.cfg.kvDim()
	if len(k) != dim || len(v) != dim {
		return nil, nil, fmt.Errorf("llama: kv width %d/%d, want %d", len(k), len(v), dim)
	}
	if err := c.ensureBlocks(pos); err != nil {
		return nil, nil, err
	}
	bid := c.blocks[pos/c.block]
	off := (pos % c.block) * dim
	bk, bv := c.pool.rows(layer, bid)
	copy(bk[off:off+dim], k)
	copy(bv[off:off+dim], v)
	return nil, nil, nil
}
