package llama

import (
	"fmt"
	"runtime"

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

	// run splits the row-partitioned kernels across goroutines. Decoding a
	// token is a chain of matrix-vector products that are bounded by memory
	// bandwidth, not arithmetic, so the rows parallelize well. The split is
	// over independent rows, which keeps every result bit-identical to the
	// serial kernel.
	run *kernel.Runner

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

	// buf holds the multi-row scratch used by Prefill. It is allocated on first
	// use and grows to the largest batch seen, so a single-token run never pays
	// for it.
	buf *batch

	// dbuf holds the scratch DecodeStep uses to advance several sequences in one
	// pass. It is allocated on first batched decode and grows to the largest
	// batch seen, so a single-sequence run never pays for it.
	dbuf *decodeBatch
}

// Options controls how a Model runs. The zero value is the default: one worker
// per available CPU, which is what a standalone CLI wants.
type Options struct {
	// Workers is the number of goroutines the row-partitioned kernels may use.
	// 0 means runtime.GOMAXPROCS(0) and 1 means serial. A scheduler that already
	// has its own parallelism sets this lower, so the two do not oversubscribe
	// the machine fighting each other.
	Workers int
}

// New loads a model from a GGUF file and prepares its scratch buffers.
func New(path string) (*Model, error) {
	return NewWithOptions(path, Options{})
}

// NewWithOptions is New with explicit run options.
func NewWithOptions(path string, opts Options) (*Model, error) {
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
		run:     kernel.NewRunner(workers(opts.Workers)),
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

// workers resolves the configured fan-out, defaulting to every available CPU.
func workers(n int) int {
	if n <= 0 {
		return runtime.GOMAXPROCS(0)
	}
	return n
}

// Config returns the model's hyperparameters.
func (m *Model) Config() Config { return m.W.Config }

// Runner returns the kernel scheduler this model uses, for reporting the
// fan-out a run actually got.
func (m *Model) Runner() *kernel.Runner { return m.run }

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
func (m *Model) Forward(cache Cache, token int, pos int) ([]float32, error) {
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
	cache.setLen(pos + 1)

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
func (m *Model) attention(cache Cache, l int, pos int) error {
	cfg := m.W.Config
	layer := &m.W.Layers[l]

	if err := kernel.RMSNorm(m.xn, m.x, layer.attnNorm, cfg.RMSNormEps); err != nil {
		return err
	}
	// Three projections of the same normalized input. wq is [hidden, qDim] and
	// MatVec produces one output per row, which is exactly the layout.
	if err := m.run.MatVec(m.q, layer.wq, m.xn, cfg.Heads*cfg.HeadDim, cfg.Hidden); err != nil {
		return err
	}
	if err := m.run.MatVec(m.k, layer.wk, m.xn, cfg.kvDim(), cfg.Hidden); err != nil {
		return err
	}
	if err := m.run.MatVec(m.v, layer.wv, m.xn, cfg.kvDim(), cfg.Hidden); err != nil {
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

	if _, _, err := cache.appendKV(l, m.k, m.v, pos); err != nil {
		return err
	}

	scale := attentionScale(cfg)
	group := cfg.Heads / cfg.KVHeads
	n := pos + 1

	for h := range cfg.Heads {
		// Grouped-query attention: several query heads share one KV head. The
		// mapping is contiguous, so head h reads KV head h/group.
		kv := h / group
		keys, values, err := cache.spans(l, kv, n)
		if err != nil {
			return err
		}
		qh := m.q[h*cfg.HeadDim : (h+1)*cfg.HeadDim]
		out := m.attn[h*cfg.HeadDim : (h+1)*cfg.HeadDim]
		if err := attend(qh, keys, values, n, m.scores, out, scale); err != nil {
			return err
		}
	}

	if err := m.run.MatVec(m.attnOut, layer.wo, m.attn, cfg.Hidden, cfg.Heads*cfg.HeadDim); err != nil {
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
	if err := m.run.MatVec(m.gate, layer.wGate, m.xn, cfg.FFN, cfg.Hidden); err != nil {
		return err
	}
	if err := m.run.MatVec(m.up, layer.wUp, m.xn, cfg.FFN, cfg.Hidden); err != nil {
		return err
	}
	kernel.SiLU(m.gate)
	if err := kernel.MulInPlace(m.up, m.gate); err != nil {
		return err
	}
	// wDown is [ffn, hidden], so the output width is hidden.
	if err := m.run.MatVec(m.attnOut, layer.wDown, m.up, cfg.Hidden, cfg.FFN); err != nil {
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
	return m.run.MatVec(m.logits, w, m.xn, cfg.VocabSize, cfg.Hidden)
}
