package llama

import (
	"fmt"
	"math"

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
func (m *Model) Prefill(cache *KVCache, tokens []int32, startPos int) ([]float32, error) {
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
	cache.len = startPos + n

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
func (m *Model) attentionBatch(b *batch, cache *KVCache, l, startPos int) error {
	cfg := m.W.Config
	layer := &m.W.Layers[l]
	n := b.n
	qdim := cfg.Heads * cfg.HeadDim
	kvdim := cfg.kvDim()

	if err := m.normRows(b, layer.attnNorm); err != nil {
		return err
	}
	if err := kernel.MatMulTransposed(b.q, b.xn, layer.wq, n, qdim, cfg.Hidden); err != nil {
		return err
	}
	if err := kernel.MatMulTransposed(b.k, b.xn, layer.wk, n, kvdim, cfg.Hidden); err != nil {
		return err
	}
	if err := kernel.MatMulTransposed(b.v, b.xn, layer.wv, n, kvdim, cfg.Hidden); err != nil {
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

	scale := float32(1 / math.Sqrt(float64(cfg.HeadDim)))
	group := cfg.Heads / cfg.KVHeads

	// Attention itself stays per position: the causal mask differs for each row,
	// and the score matrix is small compared with the projections above.
	for i := range n {
		pos := startPos + i
		// Store this row, then attend over everything up to it.
		kRow := b.k[i*kvdim : (i+1)*kvdim]
		vRow := b.v[i*kvdim : (i+1)*kvdim]
		allK, allV, err := cache.appendKV(l, kRow, vRow, pos)
		if err != nil {
			return err
		}
		for h := range cfg.Heads {
			kv := h / group
			qOff := i*qdim + h*cfg.HeadDim
			qh := b.q[qOff : qOff+cfg.HeadDim]

			m2 := pos + 1
			scores := b.scores[:m2]
			for t := range m2 {
				off := t*kvdim + kv*cfg.HeadDim
				kh := allK[off : off+cfg.HeadDim]
				var dot float32
				for j := range qh {
					dot += qh[j] * kh[j]
				}
				scores[t] = dot * scale
			}
			kernel.Softmax(scores)

			out := b.attn[i*qdim+h*cfg.HeadDim : i*qdim+(h+1)*cfg.HeadDim]
			clear(out)
			for t := range m2 {
				w := scores[t]
				off := t*kvdim + kv*cfg.HeadDim
				vh := allV[off : off+cfg.HeadDim]
				for j := range out {
					out[j] += w * vh[j]
				}
			}
		}
	}

	if err := kernel.MatMulTransposed(b.attnOut, b.attn, layer.wo, n, cfg.Hidden, qdim); err != nil {
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
	if err := kernel.MatMulTransposed(b.gate, b.xn, layer.wGate, n, cfg.FFN, cfg.Hidden); err != nil {
		return err
	}
	if err := kernel.MatMulTransposed(b.up, b.xn, layer.wUp, n, cfg.FFN, cfg.Hidden); err != nil {
		return err
	}
	kernel.SiLU(b.gate[:n*cfg.FFN])
	if err := kernel.MulInPlace(b.up[:n*cfg.FFN], b.gate[:n*cfg.FFN]); err != nil {
		return err
	}
	if err := kernel.MatMulTransposed(b.attnOut, b.up, layer.wDown, n, cfg.Hidden, cfg.FFN); err != nil {
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
	return kernel.MatVec(b.logits, w, xn, cfg.VocabSize, cfg.Hidden)
}
