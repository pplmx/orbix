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
	// layout is how dimensions pair up. Llama pairs adjacent dimensions; Qwen
	// pairs i with i+dims/2. The two agree on the first pair and diverge after it,
	// which is why a wrong layout still produces fluent text.
	layout RopeLayout

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
	return newRopeAt(cfg, nCtx, cfg.RopeBase)
}

// newRopeAt is newRope for one frequency base, which is how a file that rotates its local and
// global layers at different rates gets two tables from one config.
//
// Everything but the base is shared: the same layout, dimension count and head width, so the two
// tables are interchangeable everywhere a rotation is applied. Only the angle differs, and it
// differs for every dimension past the first pair -- base 10000 and base 1e6 agree on pair 0
// (cos 0.5403) and disagree from pair 1 (0.9950 against 0.9995), which is why a model served at
// the wrong one stays fluent.
func newRopeAt(cfg Config, nCtx int, base float32) (*rope, error) {
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
	thetaScale := math.Pow(float64(base), -2.0/float64(cfg.RopeDim))

	r := &rope{
		layout:  cfg.RopeLayout,
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
			// rope_yarn with ext_factor == 0 reduces to freq_scale * theta, where theta
			// is the position stepped down by thetaScale once per dimension.
			//
			// A withFactor model divides this dimension's theta by its frequency factor.
			// The division applies to the current dimension only: the running theta is
			// left alone, so a slow dimension does not slow every later one too. Keeping
			// the running form rather than recomputing a power is also what keeps the
			// float rounding identical for models with no factors at all.
			dimTheta := theta
			if len(cfg.RopeFreqs) > k {
				dimTheta /= float64(cfg.RopeFreqs[k])
			}
			t := float64(cfg.RopeScale) * dimTheta
			r.cos[pos][k] = float32(math.Cos(t))
			r.sin[pos][k] = float32(math.Sin(t))
			theta *= thetaScale
		}
	}
	return r, nil
}

// newLocalRope builds the second table a file asks for when its windowed layers rotate at a
// different frequency from its full-attention ones, or returns nil when one table covers both.
//
// Nil rather than a duplicate table is the common case and the one worth not paying for: a model
// whose two bases agree has one rotation, and every architecture but Gemma 3 leaves RopeBaseSWA
// equal to RopeBase -- Gemma 2 included, because llama.cpp copies the global base into Gemma 2's
// local one before reading the key, so a Gemma 2 file that omits it asks for a single rate.
func newLocalRope(cfg Config, nCtx int) (*rope, error) {
	if cfg.RopeBaseSWA == cfg.RopeBase {
		return nil, nil
	}
	return newRopeAt(cfg, nCtx, cfg.RopeBaseSWA)
}

// ropeFor picks the rotation one layer uses.
//
// The predicate is windowFor, which is the same one attention uses to decide whether a layer reads
// a bounded prefix -- and that is not a coincidence of this package's layout: llama.cpp makes the
// same call, returning rope_freq_base_train_swa when hparams.is_swa(il) and the global base
// otherwise, with is_swa set from the sliding-window pattern alone. Tying the two to one predicate
// is what keeps a layer's window and its rotation from disagreeing.
//
// A layer with no window is never the local one, so a file that declares a second base but no
// window at all rotates every layer at the global base -- which is what llama.cpp does too, since
// its swa_type stays NONE and is_swa is false everywhere.
func (m *Model) ropeFor(l int) *rope {
	if m.ropeLocal != nil && windowFor(m.W.Config, l) > 0 {
		return m.ropeLocal
	}
	return m.rope
}

// apply rotates one vector of headDim values at the given position, in place.
//
// x is a single head: dims [0, n_dims) are rotated according to the model's
// layout, and dims [n_dims, headDim) are left alone.
func (r *rope) apply(x []float32, pos int) error {
	if len(x) < r.headDim {
		return fmt.Errorf("llama: rope needs %d values, got %d", r.headDim, len(x))
	}
	if pos < 0 || pos >= len(r.cos) {
		return fmt.Errorf("llama: rope position %d outside the table (%d)", pos, len(r.cos))
	}
	cos := r.cos[pos]
	sin := r.sin[pos]
	half := r.dims / 2
	for k := range half {
		// The two layouts pair different dimensions, and both indices move: Normal
		// rotates the adjacent pair (2k, 2k+1), while NEOX rotates k with half+k, so
		// its first index is k and not 2k. The frequency is the k-th one either way,
		// which is what keeps this a single table and a single loop.
		i0, i1 := 2*k, 2*k+1
		if r.layout == RopeNeox {
			i0, i1 = k, half+k
		}
		x0 := x[i0]
		x1 := x[i1]
		x[i0] = x0*cos[k] - x1*sin[k]
		x[i1] = x0*sin[k] + x1*cos[k]
	}
	return nil
}
