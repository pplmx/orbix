package llama

import (
	"fmt"
	"math"
)

// Sampler picks the next token from a distribution of logits.
//
// It is an interface because the choice of sampling is a policy decision that
// belongs above the model, and because tests want a deterministic sampler that
// is not the one a user runs.
type Sampler interface {
	// Next returns a token index for the given logits.
	Next(logits []float32) int32
}

// Greedy always takes the most likely token.
//
// It is the default for `orbix run`: reproducible output makes a wrong forward
// pass obvious, while a sampled one can hide it behind plausible text.
type Greedy struct{}

// Next returns the argmax of logits.
func (Greedy) Next(logits []float32) int32 {
	best := 0
	for i, v := range logits {
		// NaN never compares greater, so a NaN logit cannot win; that is the
		// behaviour we want, since a NaN elsewhere in the row would otherwise
		// silently become the chosen token at index 0 on ties.
		if v > logits[best] {
			best = i
		}
	}
	return int32(best)
}

// Temperature samples from the logits after dividing by temp.
//
// temp <= 0 is treated as greedy, which is what every sampler API does and what
// a user means by "no randomness". Top-k limits the candidate set before
// sampling; k <= 0 disables the limit.
type Temperature struct {
	Temp float32
	TopK int
	// NextFloat returns a value in [0, 1). It is a field rather than a call to
	// math/rand so a seed and a test can control it without global state.
	NextFloat func() float64
}

// Next samples one token.
func (s Temperature) Next(logits []float32) int32 {
	if s.Temp <= 0 || s.NextFloat == nil {
		return Greedy{}.Next(logits)
	}

	// Top-k first: keeping only the k largest logits means the tail, whose
	// probabilities are negligible but numerous, cannot collectively dominate.
	candidates := make([]int, len(logits))
	for i := range candidates {
		candidates[i] = i
	}
	if s.TopK > 0 && s.TopK < len(logits) {
		sortTopK(candidates, logits, s.TopK)
		candidates = candidates[:s.TopK]
	}

	// Softmax over a scaled copy, then walk the cumulative distribution.
	scaled := make([]float32, len(candidates))
	maxLogit := float32(math.Inf(-1))
	for _, idx := range candidates {
		if logits[idx] > maxLogit {
			maxLogit = logits[idx]
		}
	}
	var sum float64
	for i, idx := range candidates {
		e := math.Exp(float64((logits[idx] - maxLogit) / s.Temp))
		scaled[i] = float32(e)
		sum += e
	}
	r := s.NextFloat()
	var acc float64
	for i, idx := range candidates {
		acc += float64(scaled[i]) / sum
		if r < acc {
			return int32(idx)
		}
	}
	// Floating-point accumulation can leave acc a hair below 1; the last
	// candidate is the correct fallback.
	return int32(candidates[len(candidates)-1])
}

// sortTopK partially sorts idx so the first k entries are the k highest logits.
//
// A full sort is simpler than a heap and the candidate arrays are vocabulary
// sized, so the cost is one sort per token rather than per layer.
func sortTopK(idx []int, logits []float32, k int) {
	// Insertion into the first k slots: O(n*k), which for a 32k vocabulary and
	// a typical k of 40 is ~1.3M comparisons -- comparable to a sort but with
	// no allocation and an early exit for the common ordered case.
	for i := range k {
		best := i
		for j := i + 1; j < len(idx); j++ {
			if logits[idx[j]] > logits[idx[best]] {
				best = j
			}
		}
		idx[i], idx[best] = idx[best], idx[i]
	}
}

// GenerateOptions controls Generate.
type GenerateOptions struct {
	// MaxTokens caps the number of generated tokens, not counting the prompt.
	MaxTokens int
	// StopTokens end generation when produced. The end-of-sequence id is added
	// automatically by Generate.
	StopTokens []int32
	// OnToken, when set, is called for each generated token before it is
	// appended. Returning an error stops generation and is reported.
	OnToken func(token int32) error
}

// Generate runs the model over prompt and then autoregressively for up to
// MaxTokens tokens, returning the whole sequence including the prompt.
//
// The loop is the decode path in miniature: prefill every prompt token, then one
// forward pass per generated token, each attending over everything before it.
func (m *Model) Generate(cache *KVCache, prompt []int32, sampler Sampler, opts GenerateOptions) ([]int32, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("llama: empty prompt")
	}
	if sampler == nil {
		sampler = Greedy{}
	}
	stop := make(map[int32]bool, len(opts.StopTokens)+1)
	for _, t := range opts.StopTokens {
		stop[t] = true
	}
	if eos := m.W.Config.EOSID; eos >= 0 {
		stop[int32(eos)] = true
	}

	out := make([]int32, 0, len(prompt)+opts.MaxTokens)
	out = append(out, prompt...)

	var logits []float32
	for pos, tok := range prompt {
		next, err := m.Forward(cache, int(tok), pos)
		if err != nil {
			return nil, fmt.Errorf("llama: prefill token %d at %d: %w", tok, pos, err)
		}
		// Only the last prompt token's logits are used to start generating.
		if pos == len(prompt)-1 {
			logits = append([]float32(nil), next...)
		}
	}

	pos := len(prompt)
	for range opts.MaxTokens {
		if pos >= cache.Cap() {
			// Running out of context is a normal stopping condition, not an
			// error: the caller asked for more tokens than the window holds.
			return out, nil
		}
		tok := sampler.Next(logits)
		if stop[tok] {
			return out, nil
		}
		out = append(out, tok)
		if opts.OnToken != nil {
			if err := opts.OnToken(tok); err != nil {
				return out, err
			}
		}
		next, err := m.Forward(cache, int(tok), pos)
		if err != nil {
			return nil, fmt.Errorf("llama: decode token %d at %d: %w", tok, pos, err)
		}
		// Forward returns a buffer owned by the model, so the sampler must run
		// before the next call overwrites it. Copy when the loop needs it past
		// that point.
		logits = next
		pos++
	}
	return out, nil
}
