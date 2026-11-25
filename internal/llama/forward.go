package llama

import (
	"fmt"
	"math"
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

	// run splits the partitionable kernels across goroutines. Decoding a token is a
	// chain of matrix-vector products bounded by memory bandwidth, not arithmetic, so
	// they parallelize well. The split is over independent output elements -- rows or
	// columns depending on the product -- which keeps every result bit-identical to the
	// serial kernel.
	run *kernel.Runner

	// Scratch buffers, sized once. Reusing them across tokens is what keeps the
	// per-token cost to the kernels' own allocations rather than to growing these.
	//
	// It is not zero, and this comment said it was. Measured on the two-layer fixture
	// (`testing.AllocsPerRun` over Forward at successive positions): 23 allocations per
	// token. Every one comes from a kernel (MatVec contributes about three per call, and
	// a layer makes six), because `Runner.splitRows` takes a closure -- the escape
	// analysis sends that closure to the heap on every call, at every size, including the
	// serial path. The buffers below do not allocate; the kernels they are handed to do.
	//
	// The comment is corrected rather than the code because removing those 23 would mean
	// changing the kernel's shape, and the measurement to justify that is not this test:
	// the machine's load average was above 13 throughout, so no timing claim could be made
	// and only the allocation count is trustworthy.
	x       []float32 // [hidden] residual stream
	xn      []float32 // [hidden] normalized input to a sublayer
	q       []float32 // [heads*headDim]
	k       []float32 // [kvHeads*headDim]
	v       []float32 // [kvHeads*headDim]
	attn    []float32 // [heads*headDim] attention output before wo
	attnOut []float32 // [hidden] after wo
	gate    []float32 // [ffn]
	up      []float32 // [ffn]

	// MoE scratch. routerLogits holds one score per expert before selection, and
	// expertOut accumulates the weighted sum of the chosen experts' outputs; both are
	// sized from the config and nil for a dense model.
	routerLogits []float32 // [experts]
	expertOut    []float32 // [hidden]
	scores       []float32 // [ctx] attention scores for one head
	logits       []float32 // [vocab]

	// layerTrace, when set, receives a copy of the residual stream after each block. It exists
	// for the comparison test against the numpy reference: that test needs to say which layer
	// began to disagree, and the alternative -- re-deriving the layer states from the final
	// logits -- is what made the sliding window take several rounds to find. Nil in every
	// production path, and the check below is a pointer comparison.
	layerTrace [][]float32

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
	// Workers is the number of goroutines the partitionable kernels may use.
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
		// Only a MoE model needs these, and their size comes from the config rather
		// than from a guess: a model with no experts gets nil and never allocates.
		routerLogits: make([]float32, max(cfg.Experts, 0)),
		expertOut:    make([]float32, cfg.Hidden),
		scores:       make([]float32, cfg.ContextLen),
		logits:       make([]float32, cfg.VocabSize),
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

// Logits returns the logits from the most recent successful forward pass.
//
// The slice is owned by the model and is overwritten by the next forward pass,
// so a caller that needs values past that point must copy.
//
// Before the first pass it holds VocabSize zeroes rather than nil, because the buffer is
// allocated once in New and reused. A caller cannot use "nil" to tell whether a pass has
// happened -- sampling those zeroes would return token 0, which is a valid-looking id --
// so anything that must not run before a forward pass should be ordered by its own state
// rather than by inspecting this slice.
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
		if m.layerTrace != nil && l < len(m.layerTrace) {
			copy(m.layerTrace[l], m.x)
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
	return m.embedRow(m.x, token)
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
	if err := m.addBias(layer, 1); err != nil {
		return err
	}

	// Normalise each head, then rotate it. The order matters: Qwen3 normalises the projections
	// as produced and rotates the result.
	if err := m.normQK(layer, m.q, m.k, cfg.Heads, cfg.KVHeads); err != nil {
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

	params := attentionParams(cfg, l)
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
		// A sliding-window model reads only the recent positions, which bounds
		// attention rather than storage: the cache still holds the whole context.
		if w := windowFor(cfg, l); w > 0 {
			keys = trimSpans(keys, w)
			values = trimSpans(values, w)
		}
		nWin := spanRows(keys)
		qh := m.q[h*cfg.HeadDim : (h+1)*cfg.HeadDim]
		out := m.attn[h*cfg.HeadDim : (h+1)*cfg.HeadDim]
		if err := attend(qh, keys, values, nWin, m.scores, out, params); err != nil {
			return err
		}
	}

	if err := m.run.MatVec(m.attnOut, layer.wo, m.attn, cfg.Hidden, cfg.Heads*cfg.HeadDim); err != nil {
		return err
	}
	if err := m.postNorm(m.attnOut, layer.attnPostNorm); err != nil {
		return err
	}
	return kernel.AddInPlace(m.x, m.attnOut)
}

// postNorm normalizes a sub-layer's output in place when the architecture asks for it.
//
// The usual pre-norm block hands the sub-layer output straight to the residual add; Gemma 2
// normalizes it first, which is the "sandwich" in the name. weight is nil on every architecture
// that does not, and the flag is what decides rather than the nil check -- a Gemma 2 file whose
// norm tensors were missing should be reported by the loader, not silently downgraded to a
// Llama-shaped block.
func (m *Model) postNorm(v, weight []float32) error {
	return m.postNormRows(v, weight, 1)
}

// postNormRows is postNorm for a batch, one normalization per row.
func (m *Model) postNormRows(v, weight []float32, rows int) error {
	if !m.W.Config.PostNorms {
		return nil
	}
	if len(weight) == 0 {
		return fmt.Errorf("llama: post-norm weight is missing on a model that normalizes sub-layer outputs")
	}
	h := m.W.Config.Hidden
	// Checked rather than assumed: a short buffer here is a shape bug somewhere upstream, and a
	// slice-bounds panic inside a forward pass names neither the layer nor the tensor.
	if len(v) < rows*h {
		return fmt.Errorf("llama: post-norm received %d values for %d rows of %d",
			len(v), rows, h)
	}
	for r := range rows {
		row := v[r*h : (r+1)*h]
		if err := kernel.RMSNorm(row, row, weight, m.W.Config.RMSNormEps); err != nil {
			return err
		}
	}
	return nil
}

// feedForward runs one block's gated feed-forward network.
func (m *Model) feedForward(l int) error {
	cfg := m.W.Config
	if cfg.Experts > 0 {
		return m.feedForwardMoE(l)
	}
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
	m.W.Config.activateGate(m.gate)
	if err := kernel.MulInPlace(m.up, m.gate); err != nil {
		return err
	}
	// wDown is [ffn, hidden], so the output width is hidden.
	if err := m.run.MatVec(m.attnOut, layer.wDown, m.up, cfg.Hidden, cfg.FFN); err != nil {
		return err
	}
	if err := m.postNorm(m.attnOut, layer.ffnPostNorm); err != nil {
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
	if err := m.run.MatVec(m.logits, m.outputWeight(), m.xn, cfg.VocabSize, cfg.Hidden); err != nil {
		return err
	}
	m.capLogits(m.logits)
	return nil
}

// feedForwardMoE runs one mixture-of-experts block.
//
// The block is a dense gated FFN per expert plus a router that decides which experts a
// token uses:
//
//	logits = router @ xn                    [experts]
//	top    = the ExpertUsed highest logits
//	weight = softmax over exactly those top logits
//	out    = sum over top of weight * expert(xn)
//
// Renormalising over the selected experts is what makes the weights sum to one:
// a softmax over every expert and then renormalising the kept ones gives the same
// result, because the discarded terms cancel, and computing it the direct way avoids
// exponentiating experts whose values are thrown away.
//
// Routing is per token and per layer, so there is no stable expert choice for a
// prompt: which experts run depends on the hidden state at that position.
func (m *Model) feedForwardMoE(l int) error {
	cfg := m.W.Config
	layer := &m.W.Layers[l]

	if err := kernel.RMSNorm(m.xn, m.x, layer.ffnNorm, cfg.RMSNormEps); err != nil {
		return err
	}

	// Score every expert. The router is small -- one row per expert -- so this is
	// cheap next to the experts it selects between.
	if err := m.run.MatVec(m.routerLogits, layer.router, m.xn, cfg.Experts, cfg.Hidden); err != nil {
		return err
	}
	selected, weights := selectExperts(m.routerLogits, cfg.ExpertUsed)

	// Accumulate the chosen experts' outputs. clear rather than overwrite, because
	// each expert adds in.
	clear(m.expertOut)
	for i, e := range selected {
		if err := m.run.MatVec(m.gate, layer.expertGate[e], m.xn, cfg.FFN, cfg.Hidden); err != nil {
			return err
		}
		if err := m.run.MatVec(m.up, layer.expertUp[e], m.xn, cfg.FFN, cfg.Hidden); err != nil {
			return err
		}
		m.W.Config.activateGate(m.gate)
		if err := kernel.MulInPlace(m.up, m.gate); err != nil {
			return err
		}
		// Reuse attnOut as this expert's hidden-width output, then scale it by the
		// routing weight and add it to the running sum.
		if err := m.run.MatVec(m.attnOut, layer.expertDown[e], m.up, cfg.Hidden, cfg.FFN); err != nil {
			return err
		}
		for j := range m.attnOut {
			m.expertOut[j] += weights[i] * m.attnOut[j]
		}
	}
	return kernel.AddInPlace(m.x, m.expertOut)
}

// selectExperts picks the top-k logits and turns them into softmax weights.
//
// The returned experts are in descending score order, which makes the choice
// deterministic: two experts that tie are separated by their id, so a model with equal
// router scores for two experts always runs the lower one first instead of picking
// whichever the scan happened to visit.
func selectExperts(logits []float32, k int) (selected []int, weights []float32) {
	if k <= 0 || len(logits) == 0 {
		return nil, nil
	}
	if k > len(logits) {
		k = len(logits)
	}
	selected = make([]int, 0, k)
	chosen := make([]bool, len(logits))
	for range k {
		best := -1
		for i, v := range logits {
			if chosen[i] {
				continue
			}
			// Strict greater-than keeps the earlier (lower-id) expert on a tie.
			if best < 0 || v > logits[best] {
				best = i
			}
		}
		chosen[best] = true
		selected = append(selected, best)
	}

	// Softmax over the selected logits only, subtracting the maximum first so a large
	// logit does not overflow the exponential.
	weights = make([]float32, len(selected))
	peak := logits[selected[0]]
	for _, e := range selected {
		peak = max(peak, logits[e])
	}
	var sum float32
	for i, e := range selected {
		w := exp32(logits[e] - peak)
		weights[i] = w
		sum += w
	}
	if sum > 0 {
		for i := range weights {
			weights[i] /= sum
		}
	}
	return selected, weights
}

// exp32 is float32 exp.
func exp32(x float32) float32 { return float32(math.Exp(float64(x))) }
