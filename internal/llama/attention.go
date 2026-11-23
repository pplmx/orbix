package llama

import (
	"fmt"
	"math"

	"github.com/pplmx/orbix/internal/kernel"
)

// span is a run of consecutive positions' key or value rows for one attention
// head, as stored by a cache.
//
// A contiguous cache holds every position of a head in one run, so it produces a
// single span. A paged cache holds them in several blocks, so it produces one
// span per block. Describing the storage this way lets attention read either
// layout with the same loop and no copy: flattening a paged cache into one
// contiguous buffer per token would cost exactly the traffic paging saves.
type span struct {
	// data starts at the first row's head slice, which is why it is already
	// offset by the head index.
	data []float32
	// stride is the float distance between consecutive rows. It is the full kv
	// width, not the head width, because a head's values are interleaved with
	// the other heads' values in the same position.
	stride int
	// rows is how many positions this span covers.
	rows int
	// width is one row's length, the head dimension.
	width int
}

// row returns position i's slice for this head. It aliases the cache.
func (s span) row(i int) []float32 {
	off := i * s.stride
	return s.data[off : off+s.width]
}

// attend computes one head's attention over n stored positions and writes the
// result into out.
//
// This is the inner loop every decoder layer spends its time in once the
// projections are done, and all three callers -- single-token decode, batched
// prefill, and batched decode -- need exactly this. Keeping it in one place is
// what makes paging a change to the caches rather than to three copies of the
// attention math.
//
// q is this position's query for the head; scores is scratch at least n long; out
// is this head's slice of the attention output, which attend overwrites. keys and
// values must together cover exactly n positions.
// attnParams is how one layer turns a query-key dot product into a score.
//
// It is a struct rather than two float arguments because the two are always set together and
// because a caller that threaded one of them through and forgot the other would produce a model
// that runs, generates text, and is silently wrong -- which is the same failure the architecture
// table exists to prevent.
type attnParams struct {
	// scale multiplies every dot product.
	scale float32
	// softcap, when non-zero, bounds the scaled score with tanh(s/softcap)*softcap.
	//
	// Gemma 2 uses this to keep attention logits inside a range the model was trained on: the
	// curve is near-linear for small scores and saturates towards +/- softcap, so a head that
	// would have been extremely confident is pulled back without disturbing the ordinary case.
	softcap float32
}

// apply turns one raw dot product into a score.
func (p attnParams) apply(dot float32) float32 {
	return kernel.SoftcapValue(dot*p.scale, p.softcap)
}

// attentionParams returns the layer's score parameters.
//
// The scale is 1/sqrt(head dim), where the head dim comes from the file's key_length rather than
// from Hidden/Heads -- Gemma 2's attention projections are not square, and the two numbers differ
// by twelve percent there.
func attentionParams(cfg Config, l int) attnParams {
	return attnParams{scale: float32(1 / math.Sqrt(float64(cfg.HeadDim))), softcap: cfg.AttnSoftcap}
}

// windowFor returns how many recent positions layer l may attend to, or 0 for all of them.
//
// A pattern of alternating local and global layers is what Gemma 2 does: every second block
// reads a bounded window and the ones between it read the whole context, which halves the work
// of the long-range blocks without losing any position from the model's reach. Reading the
// window from Config for every layer is what would be wrong there -- it would make every block
// local.
func windowFor(cfg Config, l int) int {
	if !cfg.SlidingPattern {
		return cfg.WindowLen
	}
	if l%2 == 0 {
		return cfg.WindowLen
	}
	return 0
}

func attend(q []float32, keys, values []span, n int, scores, out []float32, p attnParams) error {
	if n <= 0 {
		return fmt.Errorf("llama: attention over %d positions", n)
	}
	if len(scores) < n {
		return fmt.Errorf("llama: score buffer holds %d, need %d", len(scores), n)
	}
	if len(out) != len(q) {
		return fmt.Errorf("llama: attention output is %d wide, query is %d", len(out), len(q))
	}
	if got := spanRows(keys); got != n {
		return fmt.Errorf("llama: keys cover %d positions, want %d", got, n)
	}
	if got := spanRows(values); got != n {
		return fmt.Errorf("llama: values cover %d positions, want %d", got, n)
	}

	// Scores against every stored position, 0..n-1.
	t := 0
	for _, sp := range keys {
		for i := range sp.rows {
			kh := sp.row(i)
			var dot float32
			for j := range q {
				dot += q[j] * kh[j]
			}
			scores[t] = p.apply(dot)
			t++
		}
	}
	kernel.Softmax(scores[:n])

	// Weighted sum of the values, accumulated directly into out.
	clear(out)
	t = 0
	for _, sp := range values {
		for i := range sp.rows {
			w := scores[t]
			vh := sp.row(i)
			for j := range out {
				out[j] += w * vh[j]
			}
			t++
		}
	}
	return nil
}

// spanRows returns how many positions a span list covers.
func spanRows(spans []span) int {
	n := 0
	for _, s := range spans {
		n += s.rows
	}
	return n
}

// normQK normalises each head's query and key over the head width, in place.
//
// Qwen3 does this where Qwen2 puts a bias: the vectors are the same length, so the two
// architectures differ in arithmetic rather than in shape, and a file loaded with the wrong one
// runs cleanly and answers from the wrong numbers. A nil k slice normalises only the query, which
// is the cached-decode case -- the keys at those positions were normalised when they were written.
//
// The weight is one vector per layer, not per head: every head of a layer shares it, which is why
// the loop indexes heads but not the weight.
func (m *Model) normQK(layer *layer, q, k []float32, qHeads, kHeads int) error {
	if !m.W.Config.QKNorm {
		return nil
	}
	d, eps := m.W.Config.HeadDim, m.W.Config.RMSNormEps
	if err := m.normHeads(q, qHeads, layer.qNorm, d, eps); err != nil {
		return err
	}
	if k == nil {
		return nil
	}
	return m.normHeads(k, kHeads, layer.kNorm, d, eps)
}

// normHeads applies one RMSNorm per head across heads slices of width d.
func (m *Model) normHeads(v []float32, heads int, weight []float32, d int, eps float32) error {
	if len(weight) < d {
		return fmt.Errorf("llama: qk-norm weight holds %d values, need %d", len(weight), d)
	}
	if len(v) < heads*d {
		return fmt.Errorf("llama: qk-norm over %d heads of %d needs %d values, got %d",
			heads, d, heads*d, len(v))
	}
	for h := range heads {
		head := v[h*d : (h+1)*d]
		if err := kernel.RMSNorm(head, head, weight, eps); err != nil {
			return err
		}
	}
	return nil
}

// attentionCached runs one block's self-attention for a position whose keys and
// values are already in the cache.
//
// It differs from attention in one way: the position is not appended. The cache
// already holds it -- that is what "cached" means here -- and the block may be shared
// with other sequences, so writing to it would corrupt them.
func (m *Model) attentionCached(cache *PagedKVCache, l, pos int) error {
	cfg := m.W.Config
	layer := &m.W.Layers[l]

	if err := kernel.RMSNorm(m.xn, m.x, layer.attnNorm, cfg.RMSNormEps); err != nil {
		return err
	}
	if err := m.run.MatVec(m.q, layer.wq, m.xn, cfg.Heads*cfg.HeadDim, cfg.Hidden); err != nil {
		return err
	}
	// The query bias applies even though the key and value are already stored: the
	// projection produced q, and the bias is part of what q is.
	if err := kernel.AddBias(m.q, layer.qBias, 1, cfg.Heads*cfg.HeadDim); err != nil {
		return err
	}
	// Only the query is needed: the key and value at this position are stored, and their
	// normalisation happened when they were written.
	if err := m.normQK(layer, m.q, nil, cfg.Heads, 0); err != nil {
		return err
	}
	for h := range cfg.Heads {
		if err := m.rope.apply(m.q[h*cfg.HeadDim:(h+1)*cfg.HeadDim], pos); err != nil {
			return err
		}
	}

	params := attentionParams(cfg, l)
	group := cfg.Heads / cfg.KVHeads
	n := pos + 1
	for h := range cfg.Heads {
		kv := h / group
		keys, values, err := cache.spans(l, kv, n)
		if err != nil {
			return err
		}
		// A sliding-window model reads only the recent positions, which is a bound on
		// attention rather than on storage.
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

// addBias applies the attention projections' biases after the three projections.
//
// It is one call rather than three because the three vectors always travel
// together: a model has all of them or none, which the config records, and keeping
// the nil check in one place is what keeps the three forward paths from each growing
// their own copy of it.
func (m *Model) addBias(layer *layer, rows int) error {
	cfg := m.W.Config
	if layer.qBias == nil && layer.kBias == nil && layer.vBias == nil {
		return nil
	}
	if err := kernel.AddBias(m.q, layer.qBias, rows, cfg.Heads*cfg.HeadDim); err != nil {
		return err
	}
	if err := kernel.AddBias(m.k, layer.kBias, rows, cfg.kvDim()); err != nil {
		return err
	}
	return kernel.AddBias(m.v, layer.vBias, rows, cfg.kvDim())
}

// trimSpans drops all but the last keep rows from a span list.
//
// A sliding-window model attends over only the most recent WindowLen positions, but the
// cache still holds them all: the window bounds attention, not storage. Trimming here
// instead of in the caches keeps the policy in one place -- a contiguous cache and a
// paged one both lose rows the same way, and neither needs to know a window exists.
//
// The returned spans alias the originals. A partially-trimmed span keeps its row stride
// and starts at the first surviving row, so one span may be split at a block boundary
// without copying anything.
func trimSpans(spans []span, keep int) []span {
	total := spanRows(spans)
	if keep >= total {
		return spans
	}
	if keep <= 0 {
		return nil
	}
	drop := total - keep
	out := make([]span, 0, len(spans))
	for _, s := range spans {
		if drop >= s.rows {
			drop -= s.rows
			continue
		}
		if drop > 0 {
			s.data = s.data[drop*s.stride:]
			s.rows -= drop
			drop = 0
		}
		out = append(out, s)
	}
	return out
}
