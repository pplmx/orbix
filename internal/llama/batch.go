package llama

import (
	"fmt"

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
	n int // positions in flight

	x       []float32 // [n, hidden] residual stream
	xn      []float32 // [n, hidden] normalized input to a sublayer
	q       []float32 // [n, heads*headDim]
	k       []float32 // [n, kvHeads*headDim]
	v       []float32 // [n, kvHeads*headDim]
	attn    []float32 // [n, heads*headDim] attention output before wo
	attnOut []float32 // [n, hidden] after wo, also reused for the down projection
	gate    []float32 // [n, ffn]
	up      []float32 // [n, ffn]
	scores  []float32 // [ctx] one head's scores
	logits  []float32 // [vocab] the last position only
}

// newBatch allocates scratch for up to n positions.
func newBatch(cfg Config, n int) *batch {
	qdim := cfg.Heads * cfg.HeadDim
	kvdim := cfg.kvDim()
	return &batch{
		n:       n,
		x:       make([]float32, n*cfg.Hidden),
		xn:      make([]float32, n*cfg.Hidden),
		q:       make([]float32, n*qdim),
		k:       make([]float32, n*kvdim),
		v:       make([]float32, n*kvdim),
		attn:    make([]float32, n*qdim),
		attnOut: make([]float32, n*cfg.Hidden),
		gate:    make([]float32, n*cfg.FFN),
		up:      make([]float32, n*cfg.FFN),
		scores:  make([]float32, cfg.ContextLen),
		logits:  make([]float32, cfg.VocabSize),
	}
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

	// Attention itself stays per position: the causal mask differs for each row,
	// and the score matrix is small compared with the projections above.
	for i := range n {
		pos := startPos + i
		// Store this row, then attend over everything up to it.
		kRow := b.k[i*kvdim : (i+1)*kvdim]
		vRow := b.v[i*kvdim : (i+1)*kvdim]
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
			qOff := i*qdim + h*cfg.HeadDim
			qh := b.q[qOff : qOff+cfg.HeadDim]
			out := b.attn[i*qdim+h*cfg.HeadDim : i*qdim+(h+1)*cfg.HeadDim]
			if err := attend(qh, keys, values, m2, b.scores, out, scale); err != nil {
				return err
			}
		}
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
	n int // sequences in flight

	x       []float32   // [n, hidden] residual stream
	xn      []float32   // [n, hidden] normalized input to a sublayer
	q       []float32   // [n, heads*headDim]
	k       []float32   // [n, kvHeads*headDim]
	v       []float32   // [n, kvHeads*headDim]
	attn    []float32   // [n, heads*headDim] attention output before wo
	attnOut []float32   // [n, hidden] after wo, also reused for the down projection
	gate    []float32   // [n, ffn]
	up      []float32   // [n, ffn]
	scores  [][]float32 // per sequence, one head's logits over its own context
	logits  []float32   // [n, vocab] one row per sequence
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
		n:       n,
		x:       make([]float32, n*cfg.Hidden),
		xn:      make([]float32, n*cfg.Hidden),
		q:       make([]float32, n*qdim),
		k:       make([]float32, n*kvdim),
		v:       make([]float32, n*kvdim),
		attn:    make([]float32, n*qdim),
		attnOut: make([]float32, n*cfg.Hidden),
		gate:    make([]float32, n*cfg.FFN),
		up:      make([]float32, n*cfg.FFN),
		scores:  make([][]float32, n),
		logits:  make([]float32, n*cfg.VocabSize),
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
			qh := qRow[h*cfg.HeadDim : (h+1)*cfg.HeadDim]
			out := b.attn[r*qdim+h*cfg.HeadDim : r*qdim+(h+1)*cfg.HeadDim]
			if err := attend(qh, keys, values, m2, b.scores[r], out, scale); err != nil {
				return err
			}
		}
	}

	if err := m.run.MatMulTransposed(b.attnOut, b.attn, layer.wo, n, cfg.Hidden, qdim); err != nil {
		return err
	}
	return kernel.AddInPlace(b.x[:n*cfg.Hidden], b.attnOut[:n*cfg.Hidden])
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
