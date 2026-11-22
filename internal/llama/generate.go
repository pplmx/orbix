package llama

import (
	"fmt"
	"math"
	"slices"
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
//
// A NaN anywhere but index 0 cannot win: the comparison v > logits[best] is false against
// NaN, so the scan never moves onto one. Index 0 is the exception, and it is the seed
// rather than a comparison: best starts at 0 and only ever moves on a strict greater-than,
// so a NaN at 0 stays chosen when every other value fails to beat it. Measured: {NaN, 5, 3}
// returns 0, {5, NaN, 3} and {5, 3, NaN} both return 0 too, and {NaN, NaN} returns 0.
//
// A logit row like that cannot come from a loaded model -- the load-time finite check
// refuses a NaN weight, and the comment there explains what a single one would do -- so
// this is the cost of the comparison rule rather than a case to defend against. It is
// pinned by TestGreedyAndNaN, whose subtest names say which positions are covered.
func (Greedy) Next(logits []float32) int32 {
	best := 0
	for i, v := range logits {
		if v > logits[best] {
			best = i
		}
	}
	return int32(best)
}

// Temperature samples from the logits after dividing by temp.
//
// temp <= 0 is treated as greedy, which is what every sampler API does and what
// a user means by "no randomness".
//
// Two limits narrow the candidate set before the draw, and both are off at their
// zero value:
//
//   - TopK keeps the k most likely tokens. The tail is negligible token by token
//     and numerous, so it can dominate in aggregate.
//   - TopP is nucleus sampling: keep the smallest set of most likely tokens whose
//     probabilities sum to at least p. It adapts where TopK does not -- a peaked
//     distribution keeps one or two candidates, a flat one keeps many.
//
// TopK applies first when both are set, which is the order the common APIs use:
// the nucleus is computed over the k survivors rather than the whole vocabulary.
type Temperature struct {
	Temp float32
	TopK int
	// TopP is the nucleus threshold. The nucleus is computed when it is strictly between 0
	// and 1; zero and negative disable it, and so does a value at or above 1, because
	// keeping every candidate is what happens without the limit. Both ends are therefore
	// valid settings rather than only the open interval, which is what the callers document
	// -- the CLI's --top-p calls 0 the value that disables the cutoff, and the server
	// accepts the whole of [0, 1]. TestTopPOffWhenDisabled pins that 0, -1, 1 and 2 all
	// leave the distribution alone.
	TopP float32
	// NextFloat returns a value in [0, 1). It is a field rather than a call to
	// math/rand so a seed and a test can control it without global state.
	NextFloat func() float64
}

// Next samples one token.
//
// logits must be non-empty; it is one row of a forward pass and therefore has VocabSize
// entries. An empty slice panics at the fallback below rather than returning a token,
// which is deliberate: there is no token to name, and 0 would be a plausible-looking id for
// a sampler that was handed nothing. Greedy returns 0 on an empty slice instead, because its
// loop simply does not run -- the inconsistency is older than this comment.
//
// A model cannot reach here with an empty row: LoadConfig refuses a zero-width vocabulary
// because this is what an empty row would panic in.
func (s Temperature) Next(logits []float32) int32 {
	// A temperature that is not a finite positive number cannot mean anything here, and each
	// way of being wrong produces silently wrong text rather than an error.
	//
	// NaN fails "s.Temp <= 0", so it reached the softmax: exp(NaN) made sum NaN, every
	// `r < acc` evaluated false, and the loop fell through to its last-candidate fallback.
	// Measured on logits {1,5,3,0}, Temp NaN returned 3 -- the least likely of the four --
	// while Temp 0 and Temp 1 both returned the argmax.
	//
	// +Inf divides every logit difference to 0, so exp is 1 for every candidate and the draw
	// becomes uniform over the vocabulary. Measured on the same logits it returned 2. That
	// discards the distribution without saying so.
	//
	// Both are treated as greedy, which is the answer the <=0 branch already gives and the
	// only sensible reading of "no usable temperature".
	if !(s.Temp > 0) || math.IsInf(float64(s.Temp), 1) || s.NextFloat == nil {
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
	// Nucleus sampling needs the candidates in descending probability order to find
	// where the running total crosses p, and sortTopK leaves the survivors in no
	// particular order within the slice. Sorting here is cheap: the slice is already
	// narrowed by TopK in the common case.
	if s.TopP > 0 && s.TopP < 1 {
		slices.SortFunc(candidates, func(a, b int) int {
			switch {
			case logits[a] > logits[b]:
				return -1
			case logits[a] < logits[b]:
				return 1
			}
			// Equal logits keep index order, so the choice between them is stable.
			return a - b
		})
		candidates = nucleus(candidates, logits, s.Temp, s.TopP)
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

// nucleus trims candidates to the smallest leading run whose probability mass reaches p.
//
// candidates must be sorted by descending logit, which is what makes the running total a
// nucleus rather than an arbitrary subset. The run always keeps at least one token: a
// threshold no single token reaches would otherwise leave nothing to draw from, and an
// empty sampler is not a distribution.
//
// The probabilities are computed the same way Next computes them -- over the scaled logits
// with the max subtracted -- so the nucleus and the draw agree about what the mass is. Doing
// it twice is a few hundred exponentials; sharing it would mean returning a parallel array
// from a helper whose job is to trim a slice.
func nucleus(candidates []int, logits []float32, temp, p float32) []int {
	if len(candidates) <= 1 {
		return candidates
	}
	maxLogit := logits[candidates[0]]
	var sum float64
	exps := make([]float64, len(candidates))
	for i, idx := range candidates {
		e := math.Exp(float64((logits[idx] - maxLogit) / temp))
		exps[i] = e
		sum += e
	}
	if sum == 0 {
		// Unreachable, and kept as a guard on the division below. The caller sorts candidates by
		// descending logit, so the first term is exp(0) = 1 and the sum is at least that -- measured
		// with NaN logits, -Inf logits and a zero temperature, all of which give NaN rather than zero.
		// The comment here used to say a NaN logit poisoned the max into a zero sum, which it does
		// not; the branch is cheaper to keep than to prove to the compiler's satisfaction.
		return candidates[:1]
	}
	var acc float64
	for i := range candidates {
		acc += exps[i] / sum
		if acc >= float64(p) {
			return candidates[:i+1]
		}
	}
	return candidates
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
func (m *Model) Generate(cache Cache, prompt []int32, sampler Sampler, opts GenerateOptions) ([]int32, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("llama: empty prompt")
	}
	// A negative budget is not a small budget: the generation loop simply never runs and the
	// caller gets an empty continuation with a nil error, which reads as "the model produced
	// nothing" rather than "the flag was wrong". The engine rejects this already; Generate
	// is reachable without it.
	if opts.MaxTokens < 0 {
		return nil, fmt.Errorf("llama: negative max tokens (%d)", opts.MaxTokens)
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

	// The prompt is prefilled in one batched pass. Nothing consumes the
	// intermediate rows: only the last position's logits start generation, and
	// the KV cache is what the decode loop needs from the rest.
	prefilled, err := m.Prefill(cache, prompt, 0)
	if err != nil {
		return nil, fmt.Errorf("llama: prefill %d prompt tokens: %w", len(prompt), err)
	}
	logits := append([]float32(nil), prefilled...)

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
