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
func attend(q []float32, keys, values []span, n int, scores, out []float32, scale float32) error {
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
			scores[t] = dot * scale
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

// attentionScale is the 1/sqrt(headDim) factor applied to every score.
//
// It is applied to the query-side product rather than to each score separately in
// the fused form, but the arithmetic is identical: one multiply per score.
func attentionScale(cfg Config) float32 {
	return float32(1 / math.Sqrt(float64(cfg.HeadDim)))
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
	// Only the query is needed: the key and value at this position are stored.
	for h := range cfg.Heads {
		if err := m.rope.apply(m.q[h*cfg.HeadDim:(h+1)*cfg.HeadDim], pos); err != nil {
			return err
		}
	}

	scale := attentionScale(cfg)
	group := cfg.Heads / cfg.KVHeads
	n := pos + 1
	for h := range cfg.Heads {
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
