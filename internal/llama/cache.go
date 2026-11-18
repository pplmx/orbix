package llama

import (
	"fmt"
	"math"
)

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
	// Every value below comes from the file, and the products of them size the allocations, so the
	// arithmetic is done in int64 before anything is handed to make(). cfg.kvDim() multiplies two
	// ints and can overflow on its own, which would let an already-wrapped width through the
	// product check; computing it here from the parts keeps the guard honest.
	//
	// A wrapped negative length does not return an error -- make() panics on it -- so the failure
	// this prevents is a crash on a corrupt file rather than a message about one.
	kvDim := int64(cfg.KVHeads) * int64(cfg.HeadDim)
	if kvDim <= 0 {
		return nil, fmt.Errorf("llama: cache needs a valid config, got %s", cfg)
	}
	// The bound is on bytes rather than on elements: make() refuses a length whose size overflows,
	// and a float32 slice of 2^62 elements is four times that in bytes. Checking the element count
	// alone lets a length through that the runtime then panics on, which is the outcome this is
	// meant to replace.
	const valueSize = 4 // bytes per float32
	if int64(nCtx) > math.MaxInt64/kvDim/valueSize {
		// The byte count itself is not printed: it is the value that would overflow, so
		// formatting it would put a negative number in the message. The two factors are what a
		// reader needs to see what the file claimed, and the wording says what is wrong with it.
		return nil, fmt.Errorf("llama: a %d-position cache at %d values per position per layer"+
			" is too large to address", nCtx, kvDim)
	}
	// A header can declare a context length the machine cannot hold -- 2^30 positions for a small
	// model is hundreds of gigabytes -- and the allocation then either swaps the machine to death or
	// is killed partway, which reads as an out-of-memory crash rather than as a bad file. Checking
	// before allocating turns it into a message naming both the request and the budget.
	//
	// The comparison is against what the kernel says is available, and is skipped when that is
	// unknown: refusing a legitimate model because the estimate was wrong would be the worse
	// mistake of the two.
	if budget := availableMemoryHook(); budget > 0 {
		need := int64(nCtx) * kvDim * valueSize * 2 * int64(cfg.Layers) // keys and values, every layer
		if need > budget {
			return nil, fmt.Errorf("llama: a %d-position cache for this model needs %.1f GB"+
				" but only %.1f GB is available; the context length in the file is larger than"+
				" this machine can hold", nCtx, bytesToGB(need), bytesToGB(budget))
		}
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

// availableMemoryHook is availableMemory, indirected so a test can set a budget without needing a
// machine that is actually short of memory. A nil value falls back to the real reading, which is
// what every caller except a test sees.
var availableMemoryHook = availableMemory

// bytesToGB renders a byte count in gigabytes for an error message.
func bytesToGB(n int64) float64 { return float64(n) / (1 << 30) }

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
	// Zero rows is a request attention can carry out -- there is nothing to read -- but the end
	// offset below is computed from n-1, which is -1 here and would panic on a slice bound. The
	// callers in this package always ask for at least one row, so this is about the contract rather
	// than about a path in the forward pass; a panic from a public method is still a crash.
	if n == 0 {
		return []span{}, []span{}, nil
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
