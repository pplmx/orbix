package llama

import (
	"fmt"
	"math"
)

// rope is the rotary position embedding as Llama models use it.
//
// Dimensions rotate in adjacent pairs (2i, 2i+1), the layout from the original
// RoPE paper and ggml's GGML_ROPE_TYPE_NORMAL. The alternative -- pairing i with
// i+n_dims/2, which ggml calls NEOX and Qwen-family models use -- differs on
// every dimension past the first pair. The distinction is not cosmetic: rotating
// the wrong way leaves the model fluent but measurably off, which showed up here
// as log-probabilities diverging from llama.cpp by several nats while the text
// still read as English.
type rope struct {
	// dims is the number of head dimensions the rotation covers. Dimensions
	// past it are copied through untouched, which is what ggml's trailing
	// "fill the remain channels" loop does.
	dims int

	// headDim is the full head width.
	headDim int

	// cos and sin are the precomputed tables, indexed by position and then by
	// dimension pair.
	cos [][]float32
	sin [][]float32
}

// newRope precomputes the rotation tables for positions [0, nCtx).
//
// Precomputing costs one allocation and turns the per-token cost into two
// lookups. GGUF stores the base and scale, so the only inputs are the config and
// the cache length.
func newRope(cfg Config, nCtx int) (*rope, error) {
	if cfg.RopeDim%2 != 0 {
		return nil, fmt.Errorf("llama: rope dim %d is not even", cfg.RopeDim)
	}
	if nCtx <= 0 {
		return nil, fmt.Errorf("llama: rope needs a positive context length, got %d", nCtx)
	}

	nPairs := cfg.RopeDim / 2
	// theta_scale is freq_base^(-2/n_dims): each successive dimension pair
	// rotates at a geometrically lower frequency. This is the single constant
	// that defines the whole embedding.
	thetaScale := math.Pow(float64(cfg.RopeBase), -2.0/float64(cfg.RopeDim))

	r := &rope{
		dims:    cfg.RopeDim,
		headDim: cfg.HeadDim,
		cos:     make([][]float32, nCtx),
		sin:     make([][]float32, nCtx),
	}

	for pos := range nCtx {
		r.cos[pos] = make([]float32, nPairs)
		r.sin[pos] = make([]float32, nPairs)
		// theta starts at the position itself and is multiplied down the
		// dimension pairs, matching ggml_rope_cache_init's running product
		// rather than recomputing a power per index.
		theta := float64(pos)
		for k := range nPairs {
			// rope_yarn with ext_factor == 0 reduces to freq_scale * theta.
			// YaRN's interpolation and magnitude correction are not modelled
			// here; a model that needs them would set rope.scaling.type, which
			// LoadConfig rejects rather than silently ignoring.
			t := float64(cfg.RopeScale) * theta
			r.cos[pos][k] = float32(math.Cos(t))
			r.sin[pos][k] = float32(math.Sin(t))
			theta *= thetaScale
		}
	}
	return r, nil
}

// apply rotates one vector of headDim values at the given position, in place.
//
// x is a single head: dims [0, n_dims) are rotated as adjacent pairs (2i, 2i+1),
// and dims [n_dims, headDim) are left alone.
func (r *rope) apply(x []float32, pos int) error {
	if len(x) < r.headDim {
		return fmt.Errorf("llama: rope needs %d values, got %d", r.headDim, len(x))
	}
	if pos < 0 || pos >= len(r.cos) {
		return fmt.Errorf("llama: rope position %d outside the table (%d)", pos, len(r.cos))
	}
	cos := r.cos[pos]
	sin := r.sin[pos]
	for k := range r.dims / 2 {
		i0, i1 := 2*k, 2*k+1
		x0 := x[i0]
		x1 := x[i1]
		x[i0] = x0*cos[k] - x1*sin[k]
		x[i1] = x0*sin[k] + x1*cos[k]
	}
	return nil
}
