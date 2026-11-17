package llama

import (
	"fmt"
	"slices"

	"github.com/pplmx/orbix/internal/kernel"
)

// batch holds the multi-row scratch used by Prefill.
//
// Prefill is where batching pays: a prompt of n tokens applies the same weights
// to n different activation rows. A per-token loop turns every projection into a
// matrix-vector product that reads the whole weight matrix; running the rows
// together turns them into one matrix-matrix product that reads it once and
// computes n results per pass over it. That is the difference between sharing
// time and sharing work.
type batch struct {
	n   int // positions in flight
	cfg Config

	x       []float32 // [n, hidden] residual stream
	xn      []float32 // [n, hidden] normalized input to a sublayer
	q       []float32 // [n, heads*headDim]
	k       []float32 // [n, kvHeads*headDim]
	v       []float32 // [n, kvHeads*headDim]
	attn    []float32 // [n, heads*headDim] attention output before wo
	attnOut []float32 // [n, hidden] after wo, also reused for the down projection
	gate    []float32 // [n, ffn]
	up      []float32 // [n, ffn]

	// MoE scratch. A batch routes each row independently, so an expert's operand is a
	// gathered subset of rows: expertProj holds that subset's up-projection and
	// expertFFN its hidden-width output before it is scattered back. Both are sized for
	// the whole batch because the worst case is every row choosing the same expert.
	expertProj []float32 // [n, ffn]
	expertFFN  []float32 // [n, hidden]

	scores []float32 // [ctx] one head's scores
	logits []float32 // [vocab] the last position only
}

// newBatch allocates scratch for up to n positions.
func newBatch(cfg Config, n int) *batch {
	qdim := cfg.Heads * cfg.HeadDim
	kvdim := cfg.kvDim()
	return &batch{
		n:          n,
		cfg:        cfg,
		x:          make([]float32, n*cfg.Hidden),
		xn:         make([]float32, n*cfg.Hidden),
		q:          make([]float32, n*qdim),
		k:          make([]float32, n*kvdim),
		v:          make([]float32, n*kvdim),
		attn:       make([]float32, n*qdim),
		attnOut:    make([]float32, n*cfg.Hidden),
		gate:       make([]float32, n*cfg.FFN),
		up:         make([]float32, n*cfg.FFN),
		expertProj: make([]float32, n*cfg.FFN),
		expertFFN:  make([]float32, n*cfg.Hidden),
		scores:     make([]float32, cfg.ContextLen),
		logits:     make([]float32, cfg.VocabSize),
	}
}

// PrefillWithPrefix runs Prefill, but first attaches whatever part of the prompt a
// shared prefix index already holds.
//
// This is the prefill path a server wants. A conversation's system prompt and its
// earlier turns are re-sent on every turn, and for a given model those positions'
// keys and values depend only on the token prefix: two requests with the same first
// n tokens have identical KV state for those n positions. When the cache is paged
// and the pool has a prefix index, the matching blocks can simply be attached, which
// skips both the projections and the attention for that whole prefix.
//
// Only whole blocks are reused, and only from position zero, so the shared part is a
// prefix by construction rather than a match found in the middle. That keeps the
// invariant attention depends on: a block is appended to by exactly one sequence.
//
// A cache that is not paged, or a pool without an index, takes the plain path, so a
// caller does not have to know which configuration it has.
func (m *Model) PrefillWithPrefix(cache Cache, tokens []int32, startPos int) ([]float32, error) {
	if startPos != 0 {
		return nil, fmt.Errorf("llama: PrefillWithPrefix needs to start at position 0, got %d", startPos)
	}
	paged, ok := cache.(*PagedKVCache)
	if !ok {
		return m.Prefill(cache, tokens, startPos)
	}
	idx := paged.pool.index
	if idx == nil {
		return m.Prefill(cache, tokens, startPos)
	}

	n, blocks := idx.Match(tokens)
	if n > 0 {
		if err := paged.adopt(blocks); err != nil {
			return nil, err
		}
		paged.reused = n
	}
	// Prefill the rest, continuing from the adopted length. The adopted blocks are
	// shared until the first append reaches them, which is why only whole blocks are
	// reused: the block being appended to is never one somebody else holds.
	if n == len(tokens) {
		// The whole prompt is cached; no forward pass is needed, but the caller still
		// needs the next-token logits. Run the last position through the model with
		// its prefix already in place.
		return m.logitsForCached(paged, tokens, n)
	}
	out, err := m.Prefill(paged, tokens[n:], n)
	if err != nil {
		return nil, err
	}
	// Publish the blocks this pass completed, so the next request with this prefix
	// can reuse them.
	if err := m.rememberPrefix(paged, tokens, n); err != nil {
		return nil, err
	}
	return out, nil
}

// rememberPrefix publishes the blocks a prefill just completed.
func (m *Model) rememberPrefix(cache *PagedKVCache, tokens []int32, from int) error {
	if cache.pool.index == nil {
		return nil
	}
	for pos := from; pos < len(tokens); pos++ {
		if err := cache.rememberFullBlock(tokens, pos); err != nil {
			return err
		}
	}
	return nil
}

// logitsForCached returns the logits of the last prompt position when the whole
// prompt was already cached.
//
// The cache holds every position including the last one, so nothing has to be
// appended: the model only needs that position's attention (as a query over the
// cache it is already part of) and its projections. Running Forward here would
// recompute and re-append the position, which would either double-count it or write
// into a block that other sequences share.
//
// This is the case a long shared system prompt hits on every turn, so it is the one
// worth getting right rather than a corner case.
func (m *Model) logitsForCached(cache *PagedKVCache, tokens []int32, n int) ([]float32, error) {
	if n <= 0 || n > len(tokens) {
		return nil, fmt.Errorf("llama: cached prefix of %d positions for %d tokens", n, len(tokens))
	}
	cfg := m.W.Config
	pos := n - 1
	if err := m.embed(int(tokens[pos])); err != nil {
		return nil, err
	}
	for l := range cfg.Layers {
		if err := m.attentionCached(cache, l, pos); err != nil {
			return nil, err
		}
		if err := m.feedForward(l); err != nil {
			return nil, err
		}
	}
	if err := m.finish(); err != nil {
		return nil, err
	}
	return m.logits, nil
}

// Prefill appends tokens to cache starting at startPos in one batched pass and
// returns the logits of the last position.
//
// Only the last position's logits are computed. Generation continues from that
// row and nothing consumes the intermediate ones, so projecting all n rows onto
// the vocabulary would be the single most expensive part of a prefill spent on
// values that are then discarded -- on stories15M that projection is about 40%
// of the pass and grows with the vocabulary.
//
// The result is identical to calling Forward once per token: the same arithmetic
// in the same order, only with the projections restructured to process several
// rows together. The returned slice is owned by the model and is overwritten by
// the next forward pass; callers that need it past that point must copy.
//
// A returned error means the cache was not advanced, apart from the positions
// written by the layer that failed; callers treat a batched pass as
// all-or-nothing and must not decode after an error.
func (m *Model) Prefill(cache Cache, tokens []int32, startPos int) ([]float32, error) {
	cfg := m.W.Config
	n := len(tokens)
	if n == 0 {
		return nil, fmt.Errorf("llama: empty prefill")
	}
	if startPos < 0 || startPos+n > cache.Cap() {
		return nil, fmt.Errorf("llama: prefill [%d,%d) outside the cache (%d)", startPos, startPos+n, cache.Cap())
	}
	if cache.Len() != startPos {
		return nil, fmt.Errorf("llama: cache holds %d positions, prefilling at %d", cache.Len(), startPos)
	}
	if startPos+n > len(m.rope.cos) {
		return nil, fmt.Errorf("llama: prefill ends at %d, past the rope table (%d)", startPos+n, len(m.rope.cos))
	}
	for i, tok := range tokens {
		if tok < 0 || int(tok) >= cfg.VocabSize {
			return nil, fmt.Errorf("llama: token %d at index %d outside the vocabulary (%d)", tok, i, cfg.VocabSize)
		}
	}

	if m.buf == nil || m.buf.n < n {
		m.buf = newBatch(cfg, n)
	}
	b := m.buf
	b.n = n

	if err := m.embedBatch(b, tokens); err != nil {
		return nil, err
	}
	for l := range cfg.Layers {
		if err := m.attentionBatch(b, cache, l, startPos); err != nil {
			return nil, err
		}
		if err := m.feedForwardBatch(b, cfg, l); err != nil {
			return nil, err
		}
	}
	cache.setLen(startPos + n)

	if err := m.finishBatch(b); err != nil {
		return nil, err
	}
	return b.logits, nil
}

// addBiasBatch applies the attention projections' biases to every row of a batch.
func addBiasBatch(b *batch, layer *layer, n int) error {
	if layer.qBias == nil && layer.kBias == nil && layer.vBias == nil {
		return nil
	}
	cfg := b.cfg
	if err := kernel.AddBias(b.q, layer.qBias, n, cfg.Heads*cfg.HeadDim); err != nil {
		return err
	}
	if err := kernel.AddBias(b.k, layer.kBias, n, cfg.kvDim()); err != nil {
		return err
	}
	return kernel.AddBias(b.v, layer.vBias, n, cfg.kvDim())
}

// normRows normalizes every row of the residual stream into b.xn.
//
// The rows are independent, so this is a plain loop over the per-row kernel
// rather than a batched variant: a multi-row norm measured slower than the loop
// it would replace, and a second implementation of the same arithmetic is a
// second thing to keep correct.
func (m *Model) normRows(b *batch, weight []float32) error {
	cfg := m.W.Config
	for r := range b.n {
		row := b.x[r*cfg.Hidden : (r+1)*cfg.Hidden]
		out := b.xn[r*cfg.Hidden : (r+1)*cfg.Hidden]
		if err := kernel.RMSNorm(out, row, weight, cfg.RMSNormEps); err != nil {
			return err
		}
	}
	return nil
}

// embedBatch gathers the embedding rows for every token.
func (m *Model) embedBatch(b *batch, tokens []int32) error {
	cfg := m.W.Config
	for i, tok := range tokens {
		off := int(tok) * cfg.Hidden
		if off+cfg.Hidden > len(m.W.TokenEmbedding) {
			return fmt.Errorf("llama: token %d embedding outside %d values", tok, len(m.W.TokenEmbedding))
		}
		copy(b.x[i*cfg.Hidden:(i+1)*cfg.Hidden], m.W.TokenEmbedding[off:off+cfg.Hidden])
	}
	return nil
}

// attentionBatch runs one block's attention for every position in the batch.
func (m *Model) attentionBatch(b *batch, cache Cache, l, startPos int) error {
	cfg := m.W.Config
	layer := &m.W.Layers[l]
	n := b.n
	qdim := cfg.Heads * cfg.HeadDim
	kvdim := cfg.kvDim()

	if err := m.normRows(b, layer.attnNorm); err != nil {
		return err
	}
	if err := m.run.MatMulTransposed(b.q, b.xn, layer.wq, n, qdim, cfg.Hidden); err != nil {
		return err
	}
	if err := m.run.MatMulTransposed(b.k, b.xn, layer.wk, n, kvdim, cfg.Hidden); err != nil {
		return err
	}
	if err := m.run.MatMulTransposed(b.v, b.xn, layer.wv, n, kvdim, cfg.Hidden); err != nil {
		return err
	}
	if err := addBiasBatch(b, layer, n); err != nil {
		return err
	}

	// Rotate every position with its own table entry, but do not store it yet:
	// a row may only attend to positions up to and including itself. Writing the
	// whole batch into the cache first would let every row see the future, which
	// is exactly the mask a decoder is supposed to enforce.
	for i := range n {
		pos := startPos + i
		for h := range cfg.Heads {
			off := i*qdim + h*cfg.HeadDim
			if err := m.rope.apply(b.q[off:off+cfg.HeadDim], pos); err != nil {
				return err
			}
		}
		for h := range cfg.KVHeads {
			off := i*kvdim + h*cfg.HeadDim
			if err := m.rope.apply(b.k[off:off+cfg.HeadDim], pos); err != nil {
				return err
			}
		}
	}

	scale := attentionScale(cfg)
	group := cfg.Heads / cfg.KVHeads

	// Store the whole batch first, in position order, then attend. Writing first is safe
	// because every row's read is bounded by its own position: spans is asked for m2 rows
	// and each row of the batch is stored at its own offset, so a row cannot see a later
	// one whatever order the writes happened in. The writes still go in position order
	// because a paged cache appends a block per position boundary and refuses to skip
	// one.
	for i := range n {
		pos := startPos + i
		kRow := b.k[i*kvdim : (i+1)*kvdim]
		vRow := b.v[i*kvdim : (i+1)*kvdim]
		if _, _, err := cache.appendKV(l, kRow, vRow, pos); err != nil {
			return err
		}
	}

	//
	// The positions are independent once the keys and values are stored, so they run in
	// parallel. Each worker needs its own score buffer: scores is per head and per
	// position, and sharing one would let two workers interleave writes into the same
	// slice. The buffers are rented per layer rather than allocated per position, which
	// is the difference between a few megabytes and a few gigabytes of garbage.
	scorers := make([][]float32, m.run.Workers())
	for i := range scorers {
		scorers[i] = make([]float32, cfg.ContextLen)
	}
	var attErr error
	m.run.Parallel(n, func(i, slot int) {
		sc := scorers[slot]
		pos := startPos + i
		m2 := pos + 1
		for h := range cfg.Heads {
			kv := h / group
			keys, values, err := cache.spans(l, kv, m2)
			if err != nil {
				attErr = err
				return
			}
			if cfg.WindowLen > 0 {
				keys = trimSpans(keys, cfg.WindowLen)
				values = trimSpans(values, cfg.WindowLen)
			}
			nWin := spanRows(keys)
			qOff := i*qdim + h*cfg.HeadDim
			qh := b.q[qOff : qOff+cfg.HeadDim]
			out := b.attn[i*qdim+h*cfg.HeadDim : i*qdim+(h+1)*cfg.HeadDim]
			if err := attend(qh, keys, values, nWin, sc, out, scale); err != nil {
				attErr = err
				return
			}
		}
	})
	if attErr != nil {
		return attErr
	}

	if err := m.run.MatMulTransposed(b.attnOut, b.attn, layer.wo, n, cfg.Hidden, qdim); err != nil {
		return err
	}
	return kernel.AddInPlace(b.x[:n*cfg.Hidden], b.attnOut[:n*cfg.Hidden])
}

// feedForwardBatch runs one block's feed-forward network for every row.
func (m *Model) feedForwardBatch(b *batch, cfg Config, l int) error {
	layer := &m.W.Layers[l]
	n := b.n

	if err := m.normRows(b, layer.ffnNorm); err != nil {
		return err
	}
	if cfg.Experts > 0 {
		return m.feedForwardMoERows(moeRowsFromBatch(b, cfg), layer, cfg)
	}
	if err := m.run.MatMulTransposed(b.gate, b.xn, layer.wGate, n, cfg.FFN, cfg.Hidden); err != nil {
		return err
	}
	if err := m.run.MatMulTransposed(b.up, b.xn, layer.wUp, n, cfg.FFN, cfg.Hidden); err != nil {
		return err
	}
	kernel.SiLU(b.gate[:n*cfg.FFN])
	if err := kernel.MulInPlace(b.up[:n*cfg.FFN], b.gate[:n*cfg.FFN]); err != nil {
		return err
	}
	if err := m.run.MatMulTransposed(b.attnOut, b.up, layer.wDown, n, cfg.Hidden, cfg.FFN); err != nil {
		return err
	}
	return kernel.AddInPlace(b.x[:n*cfg.Hidden], b.attnOut[:n*cfg.Hidden])
}

// finishBatch normalizes the last position and projects it onto the vocabulary.
//
// The other rows are deliberately dropped: they exist to build the KV cache, and
// their logits have no reader.
func (m *Model) finishBatch(b *batch) error {
	cfg := m.W.Config
	n := b.n
	last := b.x[(n-1)*cfg.Hidden : n*cfg.Hidden]
	// Reuse the first row of xn as scratch; every row's normalized form has been
	// consumed by the layers by this point.
	xn := b.xn[:cfg.Hidden]
	if err := kernel.RMSNorm(xn, last, m.W.OutputNorm, cfg.RMSNormEps); err != nil {
		return err
	}
	w := m.W.Output
	if w == nil {
		w = m.W.TokenEmbedding
	}
	return m.run.MatVec(b.logits, w, xn, cfg.VocabSize, cfg.Hidden)
}

// DecodeStep advances several sequences by one token each in a single pass.
//
// This is the batched form of Forward. Each request decodes one token at its own
// position, and the tokens, positions, and caches differ per sequence, but the
// projections are the same weights against one row per sequence. Running them as
// one matrix-matrix product reads each weight once for the whole batch instead of
// once per sequence, which is what turns concurrency into throughput rather than
// only into fairness.
//
// Only attention is per sequence: its causal mask, its cache, and its position
// are per request, and its cost is small next to the projections.
//
// The result is one logits row per request, in request order, and is
// bit-identical to calling Forward once per request: every row is computed by the
// same arithmetic in the same order. The returned slice is owned by the model and
// is overwritten by the next pass; callers must copy what they keep.
//
// A returned error leaves the caches in an unspecified but individually
// consistent state: a batch either advances every cache or reports an error, and
// the caller must not keep decoding after one.
func (m *Model) DecodeStep(caches []Cache, tokens []int32) ([]float32, error) {
	cfg := m.W.Config
	n := len(tokens)
	if n == 0 {
		return nil, fmt.Errorf("llama: empty decode batch")
	}
	if len(caches) != n {
		return nil, fmt.Errorf("llama: %d tokens but %d caches", n, len(caches))
	}
	for i, tok := range tokens {
		if tok < 0 || int(tok) >= cfg.VocabSize {
			return nil, fmt.Errorf("llama: token %d at index %d outside the vocabulary (%d)", tok, i, cfg.VocabSize)
		}
		c := caches[i]
		if c == nil {
			return nil, fmt.Errorf("llama: cache %d is nil", i)
		}
		// A decode step appends exactly one position at the cache's current end.
		if c.Len() >= c.Cap() {
			return nil, fmt.Errorf("llama: cache %d is full at %d positions", i, c.Len())
		}
		if c.Len() >= len(m.rope.cos) {
			return nil, fmt.Errorf("llama: cache %d at %d is past the rope table (%d)", i, c.Len(), len(m.rope.cos))
		}
	}

	if m.dbuf == nil || m.dbuf.n < n {
		m.dbuf = newDecodeBatch(cfg, n)
	}
	b := m.dbuf
	b.n = n

	for i, tok := range tokens {
		if err := b.embed(m, i, int(tok)); err != nil {
			return nil, err
		}
	}
	for l := range cfg.Layers {
		if err := b.attention(m, caches, l); err != nil {
			return nil, err
		}
		if err := b.feedForward(m, l); err != nil {
			return nil, err
		}
	}
	// Advance every cache only after the whole pass succeeded, matching Forward's
	// contract: a failure leaves each cache consistent with a shorter sequence.
	for _, c := range caches {
		c.setLen(c.Len() + 1)
	}

	if err := b.finish(m); err != nil {
		return nil, err
	}
	return b.logits, nil
}

// decodeBatch holds the per-batch scratch for DecodeStep.
//
// It is separate from batch because prefill and decode need different shapes: a
// prefill keeps a score vector sized to the context and projects only its last
// row, while a decode projects every row and touches each sequence's own cache.
type decodeBatch struct {
	n   int // sequences in flight
	cfg Config

	x       []float32 // [n, hidden] residual stream
	xn      []float32 // [n, hidden] normalized input to a sublayer
	q       []float32 // [n, heads*headDim]
	k       []float32 // [n, kvHeads*headDim]
	v       []float32 // [n, kvHeads*headDim]
	attn    []float32 // [n, heads*headDim] attention output before wo
	attnOut []float32 // [n, hidden] after wo, also reused for the down projection
	gate    []float32 // [n, ffn]
	up      []float32 // [n, ffn]

	expertProj []float32   // [n, ffn] MoE scratch, see batch
	expertFFN  []float32   // [n, hidden]
	scores     [][]float32 // per sequence, one head's logits over its own context
	logits     []float32   // [n, vocab] one row per sequence
}

// newDecodeBatch allocates scratch for n sequences.
//
// The score vectors are allocated per sequence because each attends over its own
// cache, and those lengths differ. They are sized to the context once and reused,
// which keeps a steady-state decode free of allocation.
func newDecodeBatch(cfg Config, n int) *decodeBatch {
	qdim := cfg.Heads * cfg.HeadDim
	kvdim := cfg.kvDim()
	b := &decodeBatch{
		n:          n,
		cfg:        cfg,
		x:          make([]float32, n*cfg.Hidden),
		xn:         make([]float32, n*cfg.Hidden),
		q:          make([]float32, n*qdim),
		k:          make([]float32, n*kvdim),
		v:          make([]float32, n*kvdim),
		attn:       make([]float32, n*qdim),
		attnOut:    make([]float32, n*cfg.Hidden),
		gate:       make([]float32, n*cfg.FFN),
		up:         make([]float32, n*cfg.FFN),
		expertProj: make([]float32, n*cfg.FFN),
		expertFFN:  make([]float32, n*cfg.Hidden),
		scores:     make([][]float32, n),
		logits:     make([]float32, n*cfg.VocabSize),
	}
	for i := range b.scores {
		b.scores[i] = make([]float32, cfg.ContextLen)
	}
	return b
}

// embed copies one sequence's token embedding into its row.
func (b *decodeBatch) embed(m *Model, row, token int) error {
	cfg := m.W.Config
	off := token * cfg.Hidden
	if off+cfg.Hidden > len(m.W.TokenEmbedding) {
		return fmt.Errorf("llama: token %d embedding outside %d values", token, len(m.W.TokenEmbedding))
	}
	copy(b.x[row*cfg.Hidden:(row+1)*cfg.Hidden], m.W.TokenEmbedding[off:off+cfg.Hidden])
	return nil
}

// attention runs one block's attention for every sequence in the batch.
func (b *decodeBatch) attention(m *Model, caches []Cache, l int) error {
	cfg := m.W.Config
	layer := &m.W.Layers[l]
	n := b.n
	qdim := cfg.Heads * cfg.HeadDim
	kvdim := cfg.kvDim()

	for r := range n {
		row := b.x[r*cfg.Hidden : (r+1)*cfg.Hidden]
		out := b.xn[r*cfg.Hidden : (r+1)*cfg.Hidden]
		if err := kernel.RMSNorm(out, row, layer.attnNorm, cfg.RMSNormEps); err != nil {
			return err
		}
	}
	// The projections are the batched part: [n, hidden] x [hidden, out].
	if err := m.run.MatMulTransposed(b.q, b.xn, layer.wq, n, qdim, cfg.Hidden); err != nil {
		return err
	}
	if err := m.run.MatMulTransposed(b.k, b.xn, layer.wk, n, kvdim, cfg.Hidden); err != nil {
		return err
	}
	if err := m.run.MatMulTransposed(b.v, b.xn, layer.wv, n, kvdim, cfg.Hidden); err != nil {
		return err
	}
	if err := addBiasDecode(b, layer, n); err != nil {
		return err
	}

	// Each sequence rotates at its own position and appends at its own cache end.
	scale := attentionScale(cfg)
	group := cfg.Heads / cfg.KVHeads

	for r := range n {
		cache := caches[r]
		pos := cache.Len()
		qRow := b.q[r*qdim : (r+1)*qdim]
		kRow := b.k[r*kvdim : (r+1)*kvdim]
		vRow := b.v[r*kvdim : (r+1)*kvdim]

		for h := range cfg.Heads {
			if err := m.rope.apply(qRow[h*cfg.HeadDim:(h+1)*cfg.HeadDim], pos); err != nil {
				return err
			}
		}
		for h := range cfg.KVHeads {
			if err := m.rope.apply(kRow[h*cfg.HeadDim:(h+1)*cfg.HeadDim], pos); err != nil {
				return err
			}
		}

		if _, _, err := cache.appendKV(l, kRow, vRow, pos); err != nil {
			return err
		}
		m2 := pos + 1
		for h := range cfg.Heads {
			kv := h / group
			keys, values, err := cache.spans(l, kv, m2)
			if err != nil {
				return err
			}
			if cfg.WindowLen > 0 {
				keys = trimSpans(keys, cfg.WindowLen)
				values = trimSpans(values, cfg.WindowLen)
			}
			nWin := spanRows(keys)
			qh := qRow[h*cfg.HeadDim : (h+1)*cfg.HeadDim]
			out := b.attn[r*qdim+h*cfg.HeadDim : r*qdim+(h+1)*cfg.HeadDim]
			if err := attend(qh, keys, values, nWin, b.scores[r], out, scale); err != nil {
				return err
			}
		}
	}

	if err := m.run.MatMulTransposed(b.attnOut, b.attn, layer.wo, n, cfg.Hidden, qdim); err != nil {
		return err
	}
	return kernel.AddInPlace(b.x[:n*cfg.Hidden], b.attnOut[:n*cfg.Hidden])
}

// addBiasDecode applies the attention projections' biases to every row of a decode
// batch.
func addBiasDecode(b *decodeBatch, layer *layer, n int) error {
	if layer.qBias == nil && layer.kBias == nil && layer.vBias == nil {
		return nil
	}
	cfg := b.cfg
	if err := kernel.AddBias(b.q, layer.qBias, n, cfg.Heads*cfg.HeadDim); err != nil {
		return err
	}
	if err := kernel.AddBias(b.k, layer.kBias, n, cfg.kvDim()); err != nil {
		return err
	}
	return kernel.AddBias(b.v, layer.vBias, n, cfg.kvDim())
}

// feedForward runs one block's feed-forward network for every row.
func (b *decodeBatch) feedForward(m *Model, l int) error {
	cfg := m.W.Config
	layer := &m.W.Layers[l]
	n := b.n

	for r := range n {
		row := b.x[r*cfg.Hidden : (r+1)*cfg.Hidden]
		out := b.xn[r*cfg.Hidden : (r+1)*cfg.Hidden]
		if err := kernel.RMSNorm(out, row, layer.ffnNorm, cfg.RMSNormEps); err != nil {
			return err
		}
	}
	if cfg.Experts > 0 {
		return m.feedForwardMoERows(moeRowsFromDecode(b, cfg), layer, cfg)
	}
	if err := m.run.MatMulTransposed(b.gate, b.xn, layer.wGate, n, cfg.FFN, cfg.Hidden); err != nil {
		return err
	}
	if err := m.run.MatMulTransposed(b.up, b.xn, layer.wUp, n, cfg.FFN, cfg.Hidden); err != nil {
		return err
	}
	kernel.SiLU(b.gate[:n*cfg.FFN])
	if err := kernel.MulInPlace(b.up[:n*cfg.FFN], b.gate[:n*cfg.FFN]); err != nil {
		return err
	}
	if err := m.run.MatMulTransposed(b.attnOut, b.up, layer.wDown, n, cfg.Hidden, cfg.FFN); err != nil {
		return err
	}
	return kernel.AddInPlace(b.x[:n*cfg.Hidden], b.attnOut[:n*cfg.Hidden])
}

// finish normalizes every row and projects the batch onto the vocabulary.
func (b *decodeBatch) finish(m *Model) error {
	cfg := m.W.Config
	n := b.n
	for r := range n {
		row := b.x[r*cfg.Hidden : (r+1)*cfg.Hidden]
		out := b.xn[r*cfg.Hidden : (r+1)*cfg.Hidden]
		if err := kernel.RMSNorm(out, row, m.W.OutputNorm, cfg.RMSNormEps); err != nil {
			return err
		}
	}
	w := m.W.Output
	if w == nil {
		w = m.W.TokenEmbedding
	}
	return m.run.MatMulTransposed(b.logits, b.xn, w, n, cfg.VocabSize, cfg.Hidden)
}

// feedForwardMoERows runs a mixture-of-experts block for every row.
//
// Routing is per token, so the rows split into groups by chosen expert. The natural
// implementation runs each row through its experts one at a time, which for a prefill
// means n times the matmul launches. Grouping by expert instead lets one matmul cover
// every row that chose that expert, with one launch per distinct expert rather than per
// row -- the same arithmetic, far fewer calls.
//
// A token whose weight is zero contributes nothing, so a row joins an expert's group
// only when its weight is non-zero. That also keeps the accumulation order independent
// of how many rows happened to round to zero.
func (m *Model) feedForwardMoERows(r moeRows, layer *layer, cfg Config) error {
	n, hidden, ffn := r.n, r.hidden, r.ffn

	// Route every row first, so the grouping below is a pure rearrangement. The router
	// is one row per expert, so scoring is cheap next to the experts it selects.
	var assigned []expertAssignment
	for row := range n {
		in := r.xn[row*hidden : (row+1)*hidden]
		if err := m.run.MatVec(m.routerLogits, layer.router, in, cfg.Experts, hidden); err != nil {
			return err
		}
		selected, weights := selectExperts(m.routerLogits, cfg.ExpertUsed)
		for i, e := range selected {
			if weights[i] == 0 {
				continue
			}
			assigned = append(assigned, expertAssignment{expert: e, row: row, weight: weights[i]})
		}
	}

	// Accumulate into a separate buffer: a row appears in several experts' outputs, and
	// adding straight into the residual stream would let a later expert see an earlier
	// one's contribution.
	clear(r.scatter[:n*hidden])

	// One pass per distinct expert, ascending, so the accumulation order -- and the
	// float rounding it implies -- does not depend on map iteration.
	for _, e := range distinctExperts(assigned) {
		var rows []int
		var weights []float32
		for _, a := range assigned {
			if a.expert == e {
				rows = append(rows, a.row)
				weights = append(weights, a.weight)
			}
		}
		if len(rows) == 0 {
			continue
		}

		// Gather the selected rows so one matmul can serve them; a scatter would need
		// the matmul to accept an index list.
		for i, row := range rows {
			copy(r.up[i*hidden:(i+1)*hidden], r.xn[row*hidden:(row+1)*hidden])
		}

		if err := m.run.MatMulTransposed(r.gate, r.up, layer.expertGate[e], len(rows), ffn, hidden); err != nil {
			return err
		}
		if err := m.run.MatMulTransposed(r.proj, r.up, layer.expertUp[e], len(rows), ffn, hidden); err != nil {
			return err
		}
		kernel.SiLU(r.gate[:len(rows)*ffn])
		if err := kernel.MulInPlace(r.proj[:len(rows)*ffn], r.gate[:len(rows)*ffn]); err != nil {
			return err
		}
		if err := m.run.MatMulTransposed(r.out, r.proj, layer.expertDown[e], len(rows), hidden, ffn); err != nil {
			return err
		}

		// Scatter the expert's output back to its source rows, scaled by the routing
		// weight.
		for i, row := range rows {
			src := r.out[i*hidden : (i+1)*hidden]
			dst := r.scatter[row*hidden : (row+1)*hidden]
			for j := range dst {
				dst[j] += weights[i] * src[j]
			}
		}
	}
	return kernel.AddInPlace(r.x[:n*hidden], r.scatter[:n*hidden])
}

// moeRows is the per-row state a mixture-of-experts block needs, which both batch
// layouts provide. Writing the MoE path once against this view is what keeps the
// prefill and decode paths from drifting apart: the arithmetic is identical and only the
// buffers differ.
type moeRows struct {
	n      int
	hidden int
	ffn    int

	x       []float32 // [n, hidden] residual stream
	xn      []float32 // [n, hidden] normalized input
	gate    []float32 // [n, ffn] scratch
	up      []float32 // [n, ffn] gathered rows, then up-projection
	proj    []float32 // [n, ffn] expert up-projection
	out     []float32 // [n, hidden] expert output, then the accumulated sum
	scatter []float32 // [n, hidden] the residual addition target
}

// moeRowsFromBatch adapts a prefill batch.
func moeRowsFromBatch(b *batch, cfg Config) moeRows {
	return moeRows{
		n: b.n, hidden: cfg.Hidden, ffn: cfg.FFN,
		x: b.x, xn: b.xn, gate: b.gate, up: b.up,
		proj: b.expertProj, out: b.expertFFN,
		scatter: b.attnOut,
	}
}

// moeRowsFromDecode adapts a decode batch.
func moeRowsFromDecode(b *decodeBatch, cfg Config) moeRows {
	return moeRows{
		n: b.n, hidden: cfg.Hidden, ffn: cfg.FFN,
		x: b.x, xn: b.xn, gate: b.gate, up: b.up,
		proj: b.expertProj, out: b.expertFFN,
		scatter: b.attnOut,
	}
}

// expertAssignment is one row's use of one expert.
type expertAssignment struct {
	expert int
	row    int
	weight float32
}

// distinctExperts returns the expert ids referenced by the assignments, ascending and
// without duplicates.
func distinctExperts(assigned []expertAssignment) []int {
	seen := make(map[int]bool, len(assigned))
	out := make([]int, 0, len(assigned))
	for _, a := range assigned {
		if seen[a.expert] {
			continue
		}
		seen[a.expert] = true
		out = append(out, a.expert)
	}
	slices.Sort(out)
	return out
}
