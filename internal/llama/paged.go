package llama

import (
	"errors"
	"fmt"
	"math"
	"sync"
)

// ErrPoolExhausted is returned when a sequence needs a block and the pool has none left,
// even after giving up every cached prefix it was holding on speculation.
//
// It is a sentinel so callers can tell capacity from corruption: the request is well
// formed and the engine is working, there is simply no room right now. A server answering
// with 500 tells the client its request broke something; 503 says to come back later.
var ErrPoolExhausted = errors.New("llama: block pool exhausted")

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
//
// A PagedKVCache is owned by one sequence and is not safe for concurrent use: every
// field below is read and written without a lock, because the engine advances a
// sequence from the pump goroutine alone. The pool it draws from is a different
// matter -- that one is shared between sequences and guards its own counters -- so
// the absence of a mutex here says nothing about the pool.
//
// Len, BlocksHeld and ReusedPositions are subject to the same rule. They read the
// fields directly rather than taking a snapshot, so calling one from a second
// goroutine while the sequence runs is a data race even though each call looks like
// a plain getter. Verified with the race detector, which reports one.
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

	// mu guards free and refs. The engine mutates them from the pump goroutine, and
	// /metrics reads Free() from an HTTP handler on another goroutine, so the two must
	// not overlap: the race detector reports exactly that pair under load.
	mu sync.Mutex

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

// DefaultBlockSize is the block size NewBlockPool uses when BlockPoolOptions.BlockSize is zero.
//
// Exported because a caller that has to reason about block sizes before it builds the pool needs
// the same number: cmd/orbix's `chat` sizes a pool from the model's context, and it had 16
// written out locally while this constant also said 16 -- agreement by coincidence, which would
// have broken the moment either side changed. A caller deciding "how many blocks do I need" and
// the pool deciding "how big is a block" have to start from one value.
func DefaultBlockSize() int { return defaultBlockSize }

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

	// Every value that sizes the allocations comes from either the caller or the file, and the
	// products are computed in int64 before anything is handed to make(). Two ways this goes wrong
	// otherwise, both of which kill the process rather than returning an error: the multiplication
	// wraps to a negative length, which make() panics on, and a large-but-addressable length is
	// granted as virtual memory the caller then cannot touch.
	kvDim := int64(cfg.KVHeads) * int64(cfg.HeadDim)
	if kvDim <= 0 {
		return nil, fmt.Errorf("llama: block pool needs a valid config, got %s", cfg)
	}
	positions := int64(opts.Blocks) * int64(block)
	perLayer := positions * kvDim
	if int64(opts.Blocks) > math.MaxInt64/2/int64(block)/kvDim/4 {
		return nil, fmt.Errorf("llama: a pool of %d blocks of %d positions at %d values per"+
			" position is too large to address", opts.Blocks, block, kvDim)
	}
	// The cache holds keys and values for every layer, so the budget check uses the whole figure
	// rather than one layer's share. Skipped when the budget is unknown, for the same reason as in
	// NewKVCache: refusing a legitimate pool over a bad estimate is the worse mistake.
	const valueSize = 4
	// The block bookkeeping is charged too: the free list and the reference counts are each one
	// int32 per block, and for a pool whose blocks are small those arrays outweigh the KV data.
	// Leaving them out of the budget let a 2^34-block pool through and it spent 34 seconds filling
	// them, which is the same failure one allocation later.
	total := int64(cfg.Layers)*2*perLayer*valueSize + 2*8*int64(opts.Blocks)
	if budget := availableMemoryHook(); budget > 0 && total > budget {
		return nil, fmt.Errorf("llama: a pool of %d blocks of %d positions needs %.1f GiB"+
			" but only %.1f GiB is available", opts.Blocks, block,
			bytesToGiB(total), bytesToGiB(budget))
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

// Free returns how many blocks are unallocated. It is safe to call from another goroutine
// while the pump is running; /metrics does exactly that.
func (p *BlockPool) Free() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.free)
}

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
	p.mu.Lock()
	defer p.mu.Unlock()
	for len(p.free) == 0 {
		if p.index == nil || !p.index.evictOne() {
			return 0, fmt.Errorf("%w (%d blocks of %d positions)", ErrPoolExhausted, p.total, p.block)
		}
	}
	id := p.free[len(p.free)-1]
	p.free = p.free[:len(p.free)-1]
	p.refs[id] = 1
	return id, nil
}

// retain adds a holder to a block.
func (p *BlockPool) retain(id int32) error {
	p.mu.Lock()
	defer p.mu.Unlock()
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
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.releaseLocked(id)
}

// releaseLocked is release's body for a caller that already holds p.mu. acquire reaches it
// through PrefixIndex.evictOne, which gives up a block to make room -- calling release there
// would take the same non-reentrant mutex a second time and deadlock.
func (p *BlockPool) releaseLocked(id int32) error {
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
	p.mu.Lock()
	defer p.mu.Unlock()
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
//
// A block id may appear at most once. Two entries for one id would make two of this
// sequence's positions alias a block, and the append into the second would rewrite the
// first in place -- inside a block other holders are reading. PrefixIndex.Match is what
// normally guarantees this; the check here is because this function is the point where
// the read-only-shared-block invariant is enforced, so a caller cannot reintroduce the
// corruption by handing over a list the index did not produce.
func (c *PagedKVCache) adopt(blocks []int32) error {
	if len(c.blocks) != 0 || c.len != 0 {
		return fmt.Errorf("llama: adopt into a cache that already holds %d positions", c.len)
	}
	// Validate the whole list before taking a single reference: a repeat found part way
	// through would otherwise leave the blocks retained so far recorded in the cache, and
	// the error path below only undoes a failed retain, not a successful one.
	seen := make(map[int32]bool, len(blocks))
	for _, id := range blocks {
		if seen[id] {
			return fmt.Errorf("llama: adopt block %d twice; two positions would alias one block",
				id)
		}
		seen[id] = true
	}
	for _, id := range blocks {
		if err := c.pool.retain(id); err != nil {
			// Undo the retains done so far, and forget the blocks: a partial adopt would leave
			// the cache holding blocks with no positions recorded, and a cache that still lists
			// them would release somebody else's references when it is reset.
			for _, done := range c.blocks {
				_ = c.pool.release(done)
			}
			clear(c.blocks)
			c.blocks = c.blocks[:0]
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
	// needs to know to evict a chain from its end rather than its middle.
	//
	// The parent is the root of the whole chain up to here, not just the preceding block's
	// hash: the index keys a block by its chain, so that the same tokens at two offsets are
	// two entries. Walking the blocks already published for this prompt rebuilds the same
	// chain Match will walk.
	parent := chainRoot(c.block, tokens, idx-1)
	// Remember takes the index's own reference. This cache already holds its
	// reference from the allocation, so the block now has two holders and stays
	// alive after the sequence finishes -- which is the whole point.
	return pool.index.Remember(tokens[start:start+c.block], c.blocks[idx-1], parent)
}

// chainRoot returns the key the block at position end belongs under, given the chain that precedes
// it. It folds every block up to end, so the result identifies a position rather than a token run.
//
// It took the index as its first parameter until a round that noticed the parameter was never read;
// the function needs the block size and the tokens, not the index, and a parameter nothing reads is
// a claim about a dependency that does not exist.
func chainRoot(block int, tokens []int32, end int) uint64 {
	var parent uint64
	for i := range end {
		start := i * block
		parent = chainKey(parent, hashTokens(tokens[start:start+block]))
	}
	return parent
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
