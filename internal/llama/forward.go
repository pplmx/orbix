package llama

import (
	"fmt"
	"math"

	"github.com/pplmx/orbix/internal/kernel"
)

// Model is a loaded Llama model ready to run a forward pass.
//
// It is not safe for concurrent use: the forward pass writes into scratch
// buffers held on the Model so that decoding a token does not allocate. A
// scheduler would give each in-flight sequence its own Model state, which is
// what the KV cache split already anticipates.
type Model struct {
	W    *Weights
	rope *rope

	// Scratch buffers, sized once. Reusing them across tokens is what keeps
	// per-token allocation at zero after the first call.
	x       []float32 // [hidden] residual stream
	xn      []float32 // [hidden] normalized input to a sublayer
	q       []float32 // [heads*headDim]
	k       []float32 // [kvHeads*headDim]
	v       []float32 // [kvHeads*headDim]
	attn    []float32 // [heads*headDim] attention output before wo
	attnOut []float32 // [hidden] after wo
	gate    []float32 // [ffn]
	up      []float32 // [ffn]
	scores  []float32 // [ctx] attention scores for one head
	logits  []float32 // [vocab]
}

// New loads a model from a GGUF file and prepares its scratch buffers.
func New(path string) (*Model, error) {
	w, err := LoadWeights(path)
	if err != nil {
		return nil, err
	}
	cfg := w.Config
	r, err := newRope(cfg, cfg.ContextLen)
	if err != nil {
		return nil, err
	}
	return &Model{
		W:       w,
		rope:    r,
		x:       make([]float32, cfg.Hidden),
		xn:      make([]float32, cfg.Hidden),
		q:       make([]float32, cfg.Heads*cfg.HeadDim),
		k:       make([]float32, cfg.kvDim()),
		v:       make([]float32, cfg.kvDim()),
		attn:    make([]float32, cfg.Heads*cfg.HeadDim),
		attnOut: make([]float32, cfg.Hidden),
		gate:    make([]float32, cfg.FFN),
		up:      make([]float32, cfg.FFN),
		scores:  make([]float32, cfg.ContextLen),
		logits:  make([]float32, cfg.VocabSize),
	}, nil
}

// Config returns the model's hyperparameters.
func (m *Model) Config() Config { return m.W.Config }

// Logits returns the logits from the most recent successful Forward call.
//
// The slice is owned by the model and is overwritten by the next forward pass,
// so a caller that needs values past that point must copy. It returns nil before
// the first forward pass.
func (m *Model) Logits() []float32 { return m.logits }

// Forward runs one token through the model at the given position and returns
// the logits over the vocabulary.
//
// The returned slice is owned by the model and is overwritten by the next call;
// callers that need to keep it must copy. cache must have exactly pos positions
// written, so the first token is pos 0.
//
// A returned error means the forward pass did not complete and the cache was not
// advanced, so the model is still consistent with the positions it had before.
//
// The block structure is the standard pre-norm transformer: each sublayer adds
// its output to the residual stream instead of replacing it.
//
//	x = embed(token)
//	for each layer:
//	    x += attention(rmsnorm(x))
//	    x += ffn(rmsnorm(x))
//	logits = output(rmsnorm(x))
func (m *Model) Forward(cache *KVCache, token int, pos int) ([]float32, error) {
	cfg := m.W.Config
	if token < 0 || token >= cfg.VocabSize {
		return nil, fmt.Errorf("llama: token %d outside the vocabulary (%d)", token, cfg.VocabSize)
	}
	if pos < 0 || pos >= cache.Cap() {
		return nil, fmt.Errorf("llama: position %d outside the cache (%d)", pos, cache.Cap())
	}
	if cache.Len() != pos {
		return nil, fmt.Errorf("llama: cache holds %d positions, forwarding at %d", cache.Len(), pos)
	}
	if pos >= len(m.rope.cos) {
		return nil, fmt.Errorf("llama: position %d outside the rope table (%d)", pos, len(m.rope.cos))
	}

	if err := m.embed(token); err != nil {
		return nil, err
	}
	for l := range cfg.Layers {
		if err := m.attention(cache, l, pos); err != nil {
			return nil, err
		}
		if err := m.feedForward(l); err != nil {
			return nil, err
		}
	}
	// The cache length advances only once the position is fully written, so a
	// failure part-way through leaves the cache consistently short.
	cache.len = pos + 1

	if err := m.finish(); err != nil {
		return nil, err
	}
	return m.logits, nil
}

// embed copies the token's embedding row into the residual stream.
func (m *Model) embed(token int) error {
	cfg := m.W.Config
	off := token * cfg.Hidden
	if off+cfg.Hidden > len(m.W.TokenEmbedding) {
		return fmt.Errorf("llama: token %d embedding outside %d values", token, len(m.W.TokenEmbedding))
	}
	copy(m.x, m.W.TokenEmbedding[off:off+cfg.Hidden])
	return nil
}

// attention runs one block's self-attention and adds it to the residual.
func (m *Model) attention(cache *KVCache, l int, pos int) error {
	cfg := m.W.Config
	layer := &m.W.Layers[l]

	if err := kernel.RMSNorm(m.xn, m.x, layer.attnNorm, cfg.RMSNormEps); err != nil {
		return err
	}
	// Three projections of the same normalized input. wq is [hidden, qDim] and
	// MatVec produces one output per row, which is exactly the layout.
	if err := kernel.MatVec(m.q, layer.wq, m.xn, cfg.Heads*cfg.HeadDim, cfg.Hidden); err != nil {
		return err
	}
	if err := kernel.MatVec(m.k, layer.wk, m.xn, cfg.kvDim(), cfg.Hidden); err != nil {
		return err
	}
	if err := kernel.MatVec(m.v, layer.wv, m.xn, cfg.kvDim(), cfg.Hidden); err != nil {
		return err
	}

	// Rotate each head's query and the corresponding key at this position.
	for h := range cfg.Heads {
		if err := m.rope.apply(m.q[h*cfg.HeadDim:(h+1)*cfg.HeadDim], pos); err != nil {
			return err
		}
	}
	for h := range cfg.KVHeads {
		if err := m.rope.apply(m.k[h*cfg.HeadDim:(h+1)*cfg.HeadDim], pos); err != nil {
			return err
		}
	}

	allK, allV, err := cache.appendKV(l, m.k, m.v, pos)
	if err != nil {
		return err
	}

	// Scale applied to the query rather than to every score: one multiply per
	// element instead of one per score.
	scale := float32(1 / math.Sqrt(float64(cfg.HeadDim)))
	group := cfg.Heads / cfg.KVHeads

	for h := range cfg.Heads {
		// Grouped-query attention: several query heads share one KV head. The
		// mapping is contiguous, so head h reads KV head h/group.
		kv := h / group
		qh := m.q[h*cfg.HeadDim : (h+1)*cfg.HeadDim]

		// Scores against every stored position, 0..pos inclusive.
		n := pos + 1
		scores := m.scores[:n]
		for t := range n {
			off := t*cfg.kvDim() + kv*cfg.HeadDim
			var dot float32
			kh := allK[off : off+cfg.HeadDim]
			for i := range qh {
				dot += qh[i] * kh[i]
			}
			scores[t] = dot * scale
		}
		kernel.Softmax(scores)

		// Weighted sum of the values, accumulated directly into this head's
		// slice of the attention output.
		out := m.attn[h*cfg.HeadDim : (h+1)*cfg.HeadDim]
		clear(out)
		for t := range n {
			w := scores[t]
			off := t*cfg.kvDim() + kv*cfg.HeadDim
			vh := allV[off : off+cfg.HeadDim]
			for i := range out {
				out[i] += w * vh[i]
			}
		}
	}

	if err := kernel.MatVec(m.attnOut, layer.wo, m.attn, cfg.Hidden, cfg.Heads*cfg.HeadDim); err != nil {
		return err
	}
	return kernel.AddInPlace(m.x, m.attnOut)
}

// feedForward runs one block's gated feed-forward network.
func (m *Model) feedForward(l int) error {
	cfg := m.W.Config
	layer := &m.W.Layers[l]

	if err := kernel.RMSNorm(m.xn, m.x, layer.ffnNorm, cfg.RMSNormEps); err != nil {
		return err
	}
	// Llama's FFN is gated: up(x) * silu(gate(x)), then down().
	if err := kernel.MatVec(m.gate, layer.wGate, m.xn, cfg.FFN, cfg.Hidden); err != nil {
		return err
	}
	if err := kernel.MatVec(m.up, layer.wUp, m.xn, cfg.FFN, cfg.Hidden); err != nil {
		return err
	}
	kernel.SiLU(m.gate)
	if err := kernel.MulInPlace(m.up, m.gate); err != nil {
		return err
	}
	// wDown is [ffn, hidden], so the output width is hidden.
	if err := kernel.MatVec(m.attnOut, layer.wDown, m.up, cfg.Hidden, cfg.FFN); err != nil {
		return err
	}
	return kernel.AddInPlace(m.x, m.attnOut)
}

// finish normalizes the residual stream and projects it onto the vocabulary.
func (m *Model) finish() error {
	cfg := m.W.Config
	if err := kernel.RMSNorm(m.xn, m.x, m.W.OutputNorm, cfg.RMSNormEps); err != nil {
		return err
	}
	// A tied model reuses the embedding matrix, which has the same shape as an
	// output projection: [hidden, vocab].
	w := m.W.Output
	if w == nil {
		w = m.W.TokenEmbedding
	}
	return kernel.MatVec(m.logits, w, m.xn, cfg.VocabSize, cfg.Hidden)
}
