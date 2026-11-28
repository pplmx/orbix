package engine

import (
	"math/rand/v2"

	"github.com/pplmx/orbix/internal/llama"
)

// This file declares the sampling types callers of this package use.
//
// They are this package's own rather than re-exported aliases of llama's, and that is what lets
// internal/server stop importing internal/llama. The server decides which sampler a request
// gets, builds per-choice samplers, wraps them in logit_bias and reports log probabilities --
// scheduling and protocol concerns, all of them -- yet it had to reach into the model package
// for the vocabulary to express them. That put the HTTP layer's dependency on the cache's
// package into four files.
//
// The declarations are structural, not aliases, which matters more than it sounds:
//
//   - Sampler is one method. llama.Greedy, llama.Temperature and llama.Biased all satisfy this
//     interface without any change to llama, so a caller holding a llama sampler still passes
//     it, and this package still hands it to the model.
//   - Alternative and TokenBias are plain structs with the same fields, so conversion is a
//     copy the compiler makes at the boundary. They are duplicated on purpose: an alias would
//     make llama's field names part of this package's API, and the point of the exercise is
//     that this package decides what it promises.
//
// The cost is that a field added to llama.Alternative would not appear here. That failure is
// loud where it matters -- TokenLogprob converts field by field, so the compiler is what
// notices -- and silent nowhere else, because nothing reads the llama type through this one.

// Sampler picks the next token from one row of logits.
//
// Nil means greedy: Engine.Submit substitutes llama.Greedy, because a request that does not
// choose a sampler wants the reproducible one, and a caller that wants sampling has to say so.
type Sampler interface {
	// Next returns a token index for the given logits.
	Next(logits []float32) int32
}

// Alternative is one of the most likely tokens at a position, with its log probability.
//
// The list a response reports is in descending probability order, which is the order
// llama.TokenLogprobs produces and what the OpenAI schema's top_logprobs object implies: a
// client reading the first entry as the top prediction is reading it correctly.
type Alternative struct {
	// ID is the token index.
	ID int32
	// Logprob is the natural logarithm of this token's probability under the same distribution
	// the chosen token's log probability describes.
	Logprob float32
}

// TokenBias shifts one token's logit before sampling.
//
// It is OpenAI's logit_bias entry: the value is added to the logit, and the documented extremes
// of -100 and 100 act as a ban or an exclusive selection rather than as a preference. That is
// why the server honours the field instead of ignoring it -- a client that bans a token is
// relying on it not appearing.
type TokenBias struct {
	ID   int32
	Bias float32
}

// NewBiased returns a sampler that applies biases before delegating to inner.
//
// The work is done by llama.Biased, which modifies the logits row in place. That is worth
// stating because it is the reason this is a constructor rather than a type of its own: the
// row belongs to the engine and is not read again by the sampling path, so a copy per generated
// token would cost the whole vocabulary to avoid an aliasing that does not happen.
//
// Copying llama.TokenBias field by field is the price of the boundary, and it is paid once per
// request rather than once per token.
//
// An empty bias list is not special-cased. A caller with nothing to bias should not wrap at all
// -- internal/server already tests len(logitBias) > 0 before looping -- and returning inner here
// would be a branch nothing reaches, which is the kind of dead path this repository has removed
// elsewhere rather than kept for symmetry.
func NewBiased(bias []TokenBias, inner Sampler) Sampler {
	converted := make([]llama.TokenBias, len(bias))
	for i, b := range bias {
		converted[i] = llama.TokenBias{ID: b.ID, Bias: b.Bias}
	}
	return llama.Biased{Bias: converted, Inner: inner}
}

// Greedy returns the sampler that always takes the most likely token.
//
// It is the default a request gets when it names no sampler: reproducible output makes a wrong
// forward pass obvious, where a sampled one can hide it behind plausible text.
func Greedy() Sampler { return llama.Greedy{} }

// NewTemperature returns a sampler that divides the logits by temp and draws from the result.
//
// nextFloat is the uniform draw source. It is a parameter rather than global state so a seed and
// a test can control it; the caller owns the generator because it has no other use for one.
// topK and topP narrow the candidate set first, and zero disables each.
func NewTemperature(temp float32, topK int, topP float32, nextFloat func() float64) Sampler {
	return llama.Temperature{Temp: temp, TopK: topK, TopP: topP, NextFloat: nextFloat}
}

// TokenLogprob describes one generated token's probability and the alternatives near it.
//
// chosen is the log probability of the token that was actually sampled, and top holds the most
// likely tokens in descending order, including the chosen one where it appears. The values come
// from a log-softmax over the whole vocabulary at temp, not from the truncated set topK or topP
// left behind: reporting the sampled token's probability inside a set of one would claim
// certainty the model never had.
func TokenLogprob(logits []float32, chosen int32, temp float32, top int) (float32, []Alternative) {
	lp, alts := llama.TokenLogprobs(logits, chosen, temp, top)
	if len(alts) == 0 {
		return lp, nil
	}
	out := make([]Alternative, len(alts))
	for i, a := range alts {
		out[i] = Alternative{ID: a.ID, Logprob: a.Logprob}
	}
	return lp, out
}

// NewSamplerRand returns the pseudo-random generator a seed stands for.
//
// One seed has to mean one stream, and that only holds if every entry point builds the generator
// the same way. It did not: the server used PCG while `orbix run` used math/rand's older
// NewSource, so `--seed 42` and `seed: 42` over HTTP produced different replies for the same
// prompt -- a seed that is reproducible only within the tool that was told about it. Measured
// before the fix: the same three seeds disagreed through the CLI and through a running server,
// and after it all three agreed.
//
// The generator is PCG rather than NewSource because math/rand/v2 is the current package and
// NewSource is retained for compatibility with the sequences it already produced. The second
// word is derived from the seed rather than fixed so that consecutive seeds -- the server passes
// seed+i to choice i -- do not start streams that are adjacent in the generator's state space.
func NewSamplerRand(seed int64) *rand.Rand {
	return rand.New(rand.NewPCG(uint64(seed), uint64(seed>>1)))
}
