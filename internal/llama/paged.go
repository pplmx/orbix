package llama

import "fmt"

// PagedKVCache stores one sequence's keys and values in fixed-size blocks taken
// from a shared pool.
//
// The contiguous KVCache allocates the whole context for every sequence the
// moment it is created, which is simple but charges a request for memory it will
// probably never use: a 4-token reply behind a 4096-token window still reserves
// 4096 positions. With many queued requests that reservation is the limit on how
// many can be admitted, not the weights.
//
// Paging inverts that: a sequence owns only the blocks its tokens actually
// occupied, and blocks come from one pool shared by every sequence. A cache of
// 64 blocks therefore serves one long request or sixteen short ones, whichever
// the workload needs.
//
// The pool is explicit rather than global so its size is a decision a caller
// makes once, and so a test can build a pool with the exact block count its
// arithmetic needs.
//
// A block id names one group of positions in every layer at once: each layer has
// its own key and value arrays, indexed by the same block id. A sequence that
// holds B blocks therefore holds B slots in every layer, and the pool's block
// count is a count of position slots rather than of per-layer allocations.
//
// Blocks are allocated on demand and never moved, so the pointers attention reads
// stay valid for a position's lifetime. A sequence that runs out of blocks gets
// an error rather than a silent truncation: admitting a request that cannot be
// finished is a scheduling decision, and the scheduler is the layer that can make
// it.
//
// Blocks are reference counted, because a block may legitimately be shared: for a
// given model, a run of positions' keys and values depend only on the token prefix
// that produced them, so two requests with the same prompt start hold the same
// state, not merely similar state. PrefixIndex exploits that (see prefix.go); the
// reference count is what keeps a shared block alive until its last holder is done,
// and what makes releasing it a decrement rather than a free.
type PagedKVCache struct {
	cfg   Config
	pool  *BlockPool
	block int // positions per block

	blocks []int32 // block ids holding this sequence's positions
	len    int     // positions written

	// reused is how many leading positions came from the prefix index instead of a
	// forward pass. It is reported so a caller can see what reuse bought.
	reused int
}

// BlockPool owns the key and value storage that paged caches draw from.
//
// It is shared by every sequence on a model, and is not safe for concurrent use:
// the engine drives one forward pass at a time, which is the same constraint the
// model's scratch buffers already impose.
type BlockPool struct {
	cfg   Config
	block int // positions per block
	total int // blocks in the pool

	// free lists block ids that no sequence holds. It is a stack: the most
	// recently released block is reused first, which keeps the working set
	// small when sequences finish in bursts.
	free []int32

	// refs is how many holders each block has. A block may be held by several
	// sequences at once, because a shared prompt prefix is genuinely the same
	// keys and values for every request that starts with it. The block returns
	// to the free list when the last holder releases it.
	refs []int32

	// index, when set, remembers which token prefix each complete block holds, so
	// a later request with the same prefix can attach the block instead of
	// recomputing it. It holds a reference on every block it remembers.
	index *PrefixIndex

	// k and v are [layer][block * blockSize * kvDim]. Blocks are addressed as
	// blockID * blockSize + offset within the block, so a block is contiguous
	// and attention can read a run of positions without chasing a table.
	k [][]float32
	v [][]float32
}

// BlockPoolOptions configures a BlockPool.
type BlockPoolOptions struct {
	// BlockSize is the number of positions one block holds. Zero means 16.
	// Larger blocks waste less on bookkeeping and more on the tail of each
	// sequence; smaller blocks page more finely.
	BlockSize int
	// Blocks is how many blocks the pool owns. It is the whole memory budget:
	// the pool's size is blocks * BlockSize positions per layer.
	Blocks int
}

// defaultBlockSize is the block size used when a caller does not pick one.
//
// 16 is a compromise: it is small enough that the average wasted tail is under
// eight positions per sequence, and large enough that a 4096-position context is
// 256 blocks rather than thousands, which keeps the free list and the per-layer
// block table cheap.
const defaultBlockSize = 16

// NewBlockPool allocates a pool of blocks.
func NewBlockPool(cfg Config, opts BlockPoolOptions) (*BlockPool, error) {
	block := opts.BlockSize
	if block <= 0 {
		block = defaultBlockSize
	}
	if opts.Blocks <= 0 {
		return nil, fmt.Errorf("llama: block pool needs a positive block count, got %d", opts.Blocks)
	}
	if cfg.Layers <= 0 || cfg.KVHeads <= 0 || cfg.HeadDim <= 0 {
		return nil, fmt.Errorf("llama: block pool needs a valid config, got %s", cfg)
	}

	p := &BlockPool{
		cfg:   cfg,
		block: block,
		total: opts.Blocks,
		free:  make([]int32, opts.Blocks),
		refs:  make([]int32, opts.Blocks),
		k:     make([][]float32, cfg.Layers),
		v:     make([][]float32, cfg.Layers),
	}
	// Fill the free list so the lowest block id is allocated first, which keeps
	// runs reproducible across processes.
	for i := range p.free {
		p.free[i] = int32(opts.Blocks - 1 - i)
	}
	perLayer := opts.Blocks * block * cfg.kvDim()
	for i := range p.k {
		p.k[i] = make([]float32, perLayer)
		p.v[i] = make([]float32, perLayer)
	}
	return p, nil
}

// BlockSize returns the number of positions one block holds.
func (p *BlockPool) BlockSize() int { return p.block }

// Blocks returns the total number of blocks the pool owns.
func (p *BlockPool) Blocks() int { return p.total }

// Free returns how many blocks are unallocated.
func (p *BlockPool) Free() int { return len(p.free) }

// Positions returns how many positions the pool can hold in total.
func (p *BlockPool) Positions() int { return p.total * p.block }

// Bytes returns the pool's size in bytes.
func (p *BlockPool) Bytes() int64 {
	var n int64
	for i := range p.k {
		n += int64(len(p.k[i])) * 4
		n += int64(len(p.v[i])) * 4
	}
	return n
}

// acquire takes a block off the free list with a single reference.
//
// When the free list is empty it first gives up remembered prefixes: a cached
// prefix is speculative (some future request might reuse it) while an allocation
// request is not, so the speculative claim loses. Only when nothing is left to
// give up does exhaustion surface as an error.
func (p *BlockPool) acquire() (int32, error) {
	for len(p.free) == 0 {
		if p.index == nil || !p.index.evictOne() {
			return 0, fmt.Errorf("llama: block pool exhausted (%d blocks of %d positions)", p.total, p.block)
		}
	}
	id := p.free[len(p.free)-1]
	p.free = p.free[:len(p.free)-1]
	p.refs[id] = 1
	return id, nil
}

// retain adds a holder to a block.
func (p *BlockPool) retain(id int32) error {
	if id < 0 || int(id) >= p.total {
		return fmt.Errorf("llama: block %d outside the pool (%d)", id, p.total)
	}
	if p.refs[id] == 0 {
		return fmt.Errorf("llama: retain of unreferenced block %d", id)
	}
	p.refs[id]++
	return nil
}

// release drops one holder, returning the block to the free list when none is left.
func (p *BlockPool) release(id int32) error {
	if id < 0 || int(id) >= p.total {
		return fmt.Errorf("llama: block %d outside the pool (%d)", id, p.total)
	}
	if p.refs[id] == 0 {
		return fmt.Errorf("llama: release of unreferenced block %d", id)
	}
	p.refs[id]--
	if p.refs[id] == 0 {
		p.free = append(p.free, id)
	}
	return nil
}

// RefCount returns how many holders a block has, for tests and diagnostics.
func (p *BlockPool) RefCount(id int32) int {
	if id < 0 || int(id) >= p.total {
		return 0
	}
	return int(p.refs[id])
}

// rows returns the k and v storage for one block in one layer, viewed as
// blockSize rows of kvDim.
func (p *BlockPool) rows(layer int, id int32) (k, v []float32) {
	off := int(id) * p.block * p.cfg.kvDim()
	end := off + p.block*p.cfg.kvDim()
	return p.k[layer][off:end], p.v[layer][off:end]
}

// New returns a page-backed cache for one sequence, drawing its blocks from pool.
func (p *BlockPool) New() *PagedKVCache {
	return &PagedKVCache{
		cfg:   p.cfg,
		pool:  p,
		block: p.block,
	}
}

// Len returns the number of positions written.
func (c *PagedKVCache) Len() int { return c.len }

// Cap returns the position limit this cache enforces.
//
// It is the model's context length, not the pool's remaining space: the pool is
// shared, so how many positions this sequence can actually still get depends on
// what the other sequences are holding. Running out of blocks therefore surfaces
// as an error from AppendKV, where the pool's state is known, rather than as a
// capacity the cache would have to guess at.
func (c *PagedKVCache) Cap() int { return c.cfg.ContextLen }

// ReusedPositions returns how many leading positions this cache got from a shared
// prefix rather than from a forward pass.
func (c *PagedKVCache) ReusedPositions() int { return c.reused }

// BlocksHeld returns the number of blocks this sequence currently holds.
func (c *PagedKVCache) BlocksHeld() int { return len(c.blocks) }

// Bytes returns the storage this sequence holds, in bytes.
func (c *PagedKVCache) Bytes() int64 {
	return int64(c.BlocksHeld()) * int64(c.block) * int64(c.cfg.kvDim()) * 2 * 4 * int64(c.cfg.Layers)
}

// Reset releases every block and forgets the positions.
func (c *PagedKVCache) Reset() error {
	for _, id := range c.blocks {
		if err := c.pool.release(id); err != nil {
			return err
		}
	}
	clear(c.blocks)
	c.blocks = c.blocks[:0]
	c.len = 0
	c.reused = 0
	return nil
}

// adopt attaches already-filled blocks to this cache, as a prefix hit does.
//
// The blocks must hold exactly the first positions of this sequence's prompt, in
// order. adopt retains each one, so the pool keeps them alive even if the request
// that produced them finishes first.
//
// Only whole blocks are adopted, and this cache must be empty; both follow from
// what a shared block is: a run of positions whose keys and values no holder may
// modify. A cache that adopted part of a block would append into it and corrupt the
// other holders.
func (c *PagedKVCache) adopt(blocks []int32) error {
	if len(c.blocks) != 0 || c.len != 0 {
		return fmt.Errorf("llama: adopt into a cache that already holds %d positions", c.len)
	}
	for _, id := range blocks {
		if err := c.pool.retain(id); err != nil {
			// Undo the retains done so far: a partial adopt would leave the cache
			// holding blocks with no positions recorded.
			for _, done := range c.blocks {
				_ = c.pool.release(done)
			}
			return err
		}
		c.blocks = append(c.blocks, id)
	}
	c.len = len(blocks) * c.block
	return nil
}

// rememberFullBlock publishes the block covering position pos to the prefix index,
// once that block is complete.
//
// A block is published when its last position is written, which is the moment its
// keys and values stop changing. Publishing earlier would hand out a block that its
// owner is still appending to.
func (c *PagedKVCache) rememberFullBlock(tokens []int32, pos int) error {
	pool := c.pool
	if pool.index == nil {
		return nil
	}
	if (pos+1)%c.block != 0 {
		return nil
	}
	idx := (pos + 1) / c.block
	if idx > len(c.blocks) {
		return fmt.Errorf("llama: block %d for position %d is not allocated", idx-1, pos)
	}
	start := (idx - 1) * c.block
	if start+c.block > len(tokens) {
		return fmt.Errorf("llama: block %d needs tokens up to %d, have %d", idx-1, start+c.block, len(tokens))
	}
	// The block continues the one before it in this same prefill, which is what the index
	// needs to know to evict a chain from its end rather than its middle. The parent's key
	// is the hash of the preceding block's tokens; idx-1 == 0 means this is the head.
	var parent uint64
	if idx-1 > 0 {
		parentStart := start - c.block
		parent = pool.index.Key(tokens[parentStart : parentStart+c.block])
	}
	// Remember takes the index's own reference. This cache already holds its
	// reference from the allocation, so the block now has two holders and stays
	// alive after the sequence finishes -- which is the whole point.
	return pool.index.Remember(tokens[start:start+c.block], c.blocks[idx-1], parent)
}

// ensureBlocks makes sure the sequence holds a block covering pos.
//
// One block id gives the sequence that position slot in every layer, so at most
// one allocation happens per new block and a position either has storage
// everywhere or nowhere.
func (c *PagedKVCache) ensureBlocks(pos int) error {
	need := pos/c.block + 1
	have := len(c.blocks)
	if need <= have {
		return nil
	}
	if need > have+1 {
		return fmt.Errorf("llama: cache needs block %d but holds %d", need-1, have)
	}
	id, err := c.pool.acquire()
	if err != nil {
		return err
	}
	c.blocks = append(c.blocks, id)
	return nil
}

// spans returns the key and value rows a query at pos attends over, for layer l
// and kv head kv, described as one span per block.
//
// The rows are handed back as spans rather than as one flat slice because a
// paged cache's runs are not contiguous. Copying them into a contiguous buffer
// once per token would cost exactly the memory traffic paging is meant to save,
// so attention walks the spans instead.
func (c *PagedKVCache) spans(layer, kv, n int) (keys, values []span, err error) {
	if layer < 0 || layer >= c.cfg.Layers {
		return nil, nil, fmt.Errorf("llama: layer %d outside the cache (%d)", layer, c.cfg.Layers)
	}
	if kv < 0 || kv >= c.cfg.KVHeads {
		return nil, nil, fmt.Errorf("llama: kv head %d outside the cache (%d)", kv, c.cfg.KVHeads)
	}
	dim := c.cfg.kvDim()
	head := c.cfg.HeadDim
	off := kv * head
	// n is bounded by the blocks actually allocated, not by len: a pass writes
	// its positions and attends over them before the length advances, which during
	// a prefill means attending over positions len has not counted yet.
	if n < 0 || n > c.cfg.ContextLen {
		return nil, nil, fmt.Errorf("llama: %d positions requested, context is %d", n, c.cfg.ContextLen)
	}
	if need := (n + c.block - 1) / c.block; need > len(c.blocks) {
		return nil, nil, fmt.Errorf("llama: %d positions need %d blocks, cache holds %d", n, need, len(c.blocks))
	}
	nblocks := (n + c.block - 1) / c.block
	keys = make([]span, 0, nblocks)
	values = make([]span, 0, nblocks)
	for b := range nblocks {
		bk, bv := c.pool.rows(layer, c.blocks[b])
		rows := min(c.block, n-b*c.block)
		// The last row ends at (rows-1)*dim + off + head, not off + rows*dim:
		// heads are interleaved within a position.
		end := off + (rows-1)*dim + head
		keys = append(keys, span{data: bk[off:end], stride: dim, rows: rows, width: head})
		values = append(values, span{data: bv[off:end], stride: dim, rows: rows, width: head})
	}
	return keys, values, nil
}
