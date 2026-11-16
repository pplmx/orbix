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

// Reset drops every stored position. The arena is kept for reuse, which is why
// resetting a contiguous cache costs nothing and returns no memory.
//
// It returns an error only to satisfy Cache: releasing an arena cannot fail.
func (c *KVCache) Reset() error {
	c.len = 0
	return nil
}

// Bytes returns the size of the cache in bytes.
func (c *KVCache) Bytes() int64 {
	var n int64
	for i := range c.k {
		n += int64(len(c.k[i])) * 4
		n += int64(len(c.v[i])) * 4
	}
	return n
}

// spans returns the key and value rows a query at pos attends over, for layer l
// and kv head kv.
//
// A contiguous cache covers every position in one run, so this is always a single
// span. It exists so attention can be written once against both cache layouts.
func (c *KVCache) spans(layer, kv, n int) (keys, values []span, err error) {
	if layer < 0 || layer >= len(c.k) {
		return nil, nil, fmt.Errorf("llama: layer %d outside the cache (%d)", layer, len(c.k))
	}
	if kv < 0 || kv >= c.cfg.KVHeads {
		return nil, nil, fmt.Errorf("llama: kv head %d outside the cache (%d)", kv, c.cfg.KVHeads)
	}
	dim := c.cfg.kvDim()
	head := c.cfg.HeadDim
	off := kv * head
	// n is checked against the arena, not against len: a forward pass writes its
	// position and then attends over it before len is advanced, so the row it
	// just wrote is live even though len has not counted it yet. Callers must not
	// ask for more than the arena holds.
	if n < 0 || n > c.ctx {
		return nil, nil, fmt.Errorf("llama: %d positions requested, cache holds %d", n, c.ctx)
	}
	end := off + (n-1)*dim + head
	keys = []span{{data: c.k[layer][off:end], stride: dim, rows: n, width: head}}
	values = []span{{data: c.v[layer][off:end], stride: dim, rows: n, width: head}}
	return keys, values, nil
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
