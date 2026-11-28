package llama

import (
	"math"
	"slices"
)

// Alternative is one of the most likely tokens at a position, with its log probability.
//
// It mirrors what the OpenAI schema calls a top logprob: the token and how likely the model
// thought it was, so a caller can see how close the choice was to a different one.
type Alternative struct {
	// ID is the token index.
	ID int32
	// Logprob is the natural logarithm of this token's probability under the same
	// distribution TokenLogprobs reports for the sampled token.
	Logprob float32
}

// MaxTopLogprobs is the largest number of alternatives TokenLogprobs will return.
//
// OpenAI's chat endpoint caps top_logprobs at 20, and a longer list would be a longer
// response than any client asked for. The bound is here rather than at the HTTP layer
// because it is a property of what a logprob list is useful for: past a couple of dozen
// candidates the numbers are all below -10 and say nothing about the choice.
const MaxTopLogprobs = 20

// TokenLogprobs describes the distribution one token was drawn from.
//
// logits is the forward pass's output row for the position the token was sampled at -- the
// same row the sampler saw -- and chosen is the token that was emitted. temp is the request's
// temperature, applied the way the sampler applies it so the reported probabilities are the
// ones the draw was made from.
//
// Both returned values are the model's own: the chosen token's log probability and the most
// likely alternatives, computed from a log-softmax over the whole vocabulary. That is
// deliberately not the truncated distribution a TopK or TopP sampler draws from. Reporting the
// truncated one would give a token logprob of 0 whenever top_k is 1, which says "this was
// certain" about a position where the model was merely narrowed by a setting -- the number a
// caller reads is meant to describe how surprising the token was to the model, not how the
// request was configured.
//
// A temperature that is not a finite positive number is treated as 1. Greedy sampling
// divides by nothing, so the raw distribution is the honest one to describe; dividing by zero
// would make every logprob NaN and dividing by an infinite temperature would make them all
// equal.
//
// top bounds the alternatives returned; values at or below zero compute none, and values above
// MaxTopLogprobs are cut to it. The alternatives are the highest logits, which for a fixed
// positive temperature is the same order as the highest probabilities.
//
// The cost is two passes over the vocabulary plus a bounded selection, and it runs only for
// tokens a caller asked to have described -- the sampling path itself is untouched.
func TokenLogprobs(logits []float32, chosen int32, temp float32, top int) (float32, []Alternative) {
	if len(logits) == 0 {
		return 0, nil
	}
	scale := temp
	if !(scale > 0) || math.IsInf(float64(scale), 0) {
		// Covers zero, negative, NaN and +Inf in one condition: none of them is a divisor.
		scale = 1
	}

	// log-softmax, with the maximum subtracted first so exp cannot overflow. Two passes
	// rather than an online one because the second is a plain range over a slice, which is
	// what makes this readable in a repository whose point is being readable.
	maxScaled := float32(math.Inf(-1))
	for _, l := range logits {
		if v := l / scale; v > maxScaled {
			maxScaled = v
		}
	}
	if math.IsInf(float64(maxScaled), -1) {
		// Every logit was -Inf, or the row was entirely NaN. There is no distribution to
		// describe, and a log probability of NaN would poison whatever a caller does with it.
		return 0, nil
	}
	var sum float64
	for _, l := range logits {
		sum += math.Exp(float64(l/scale - maxScaled))
	}
	logSumExp := float64(maxScaled) + math.Log(sum)

	chosenLogprob := float32(0)
	if chosen >= 0 && int(chosen) < len(logits) {
		chosenLogprob = float32(float64(logits[chosen]/scale) - logSumExp)
	}

	if top <= 0 {
		return chosenLogprob, nil
	}
	top = min(top, MaxTopLogprobs)
	return chosenLogprob, topAlternatives(logits, scale, logSumExp, top)
}

// topAlternatives returns the top highest-logit tokens with their log probabilities, in
// descending order.
//
// It keeps a bounded, sorted slice rather than sorting the vocabulary: top is at most
// MaxTopLogprobs, so a full sort would spend most of its work on candidates nobody reads.
func topAlternatives(logits []float32, scale float32, logSumExp float64, top int) []Alternative {
	best := make([]Alternative, 0, top)
	// worst is the logit of the last entry, the cutoff a new candidate has to beat to enter.
	worst := float32(math.Inf(-1))
	for i, l := range logits {
		if len(best) == top && l <= worst {
			// The common case by a wide margin: most of a 32k vocabulary loses here without
			// touching the slice.
			continue
		}
		a := Alternative{ID: int32(i), Logprob: float32(float64(l/scale) - logSumExp)}
		pos, _ := slices.BinarySearchFunc(best, l, func(existing Alternative, want float32) int {
			// Descending by logit, so the slice stays ordered for the next search.
			switch other := logits[existing.ID]; {
			case other > want:
				return -1
			case other < want:
				return 1
			}
			// Equal logits keep token order, so the output is deterministic.
			switch {
			case existing.ID < int32(i):
				return -1
			case existing.ID > int32(i):
				return 1
			}
			return 0
		})
		best = slices.Insert(best, pos, a)
		if len(best) > top {
			best = best[:top]
		}
		worst = logits[best[len(best)-1].ID]
	}
	return best
}
