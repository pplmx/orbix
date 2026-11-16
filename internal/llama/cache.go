package llama

import "fmt"

// KVCache holds the attention keys and values for one sequence.
//
// This is the simplest thing that works: one contiguous block per layer, sized
// for the whole context. It makes the memory cost explicit -- a Llama-7B with a
// 4096-token window needs 1 GiB of cache -- which is the point of implementing
// it before adding paging. A paged cache, the layer above this one, allocates
// these blocks on demand and shares prefixes between sequences.
type KVCache struct {
	cfg Config

	// ctx is the number of positions the cache was sized for.
	ctx int

	// len is how many positions have been written.
	len int

	// k and v are [layer][ctx * kvDim], where kvDim is kvHeads*headDim.
	k [][]float32
	v [][]float32
}

// kvDim is the width of one position's key or value vector.
func (c Config) kvDim() int { return c.KVHeads * c.HeadDim }

// NewKVCache allocates a cache for nCtx positions.
//
// The allocation is all-or-nothing: the cache is a fixed-size arena, and running
// out of positions is a scheduling decision, not something attention can work
// around.
func NewKVCache(cfg Config, nCtx int) (*KVCache, error) {
	if nCtx <= 0 {
		return nil, fmt.Errorf("llama: cache size must be positive, got %d", nCtx)
	}
	if cfg.Layers <= 0 || cfg.KVHeads <= 0 || cfg.HeadDim <= 0 {
		return nil, fmt.Errorf("llama: cache needs a valid config, got %s", cfg)
	}
	c := &KVCache{cfg: cfg, ctx: nCtx, k: make([][]float32, cfg.Layers), v: make([][]float32, cfg.Layers)}
	// Grow: allocating every layer's block in one call would need a single
	// slice of layers*ctx*kvDim floats, which for a 7B model is past the point
	// where a single allocation reliably succeeds.
	for i := range c.k {
		c.k[i] = make([]float32, nCtx*cfg.kvDim())
		c.v[i] = make([]float32, nCtx*cfg.kvDim())
	}
	return c, nil
}

// Len returns the number of positions written.
func (c *KVCache) Len() int { return c.len }

// Cap returns the number of positions the cache can hold.
func (c *KVCache) Cap() int { return c.ctx }

// Reset drops every stored position. The memory is kept for reuse.
func (c *KVCache) Reset() { c.len = 0 }

// Bytes returns the size of the cache in bytes.
func (c *KVCache) Bytes() int64 {
	var n int64
	for i := range c.k {
		n += int64(len(c.k[i])) * 4
		n += int64(len(c.v[i])) * 4
	}
	return n
}

// appendKV stores one position's key and value for layer, returning the slices
// to attend over.
//
// The returned views alias the cache: attention reads them immediately and does
// not retain them, which is why no copy is made.
func (c *KVCache) appendKV(layer int, k, v []float32, pos int) ([]float32, []float32, error) {
	if layer < 0 || layer >= len(c.k) {
		return nil, nil, fmt.Errorf("llama: layer %d outside the cache (%d)", layer, len(c.k))
	}
	if pos < 0 || pos >= c.ctx {
		return nil, nil, fmt.Errorf("llama: position %d outside the cache (%d)", pos, c.ctx)
	}
	dim := c.cfg.kvDim()
	if len(k) != dim || len(v) != dim {
		return nil, nil, fmt.Errorf("llama: kv width %d/%d, want %d", len(k), len(v), dim)
	}
	off := pos * dim
	copy(c.k[layer][off:off+dim], k)
	copy(c.v[layer][off:off+dim], v)
	return c.k[layer][:off+dim], c.v[layer][:off+dim], nil
}
