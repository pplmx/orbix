package server

import (
	"math/rand/v2"

	"github.com/pplmx/orbix/internal/llama"
)

// greedySampler always takes the most likely token.
//
// It is a local type rather than llama.Greedy so the server's default cannot be
// changed by an edit to the sampler package: a local engine whose output stops
// being reproducible is a debugging trap, and the README promises greedy as the
// default.
type greedySampler struct{}

// Next returns the argmax of logits.
func (greedySampler) Next(logits []float32) int32 { return llama.Greedy{}.Next(logits) }

// temperatureSampler samples from the logits after dividing by temp.
//
// llama.Temperature needs a NextFloat func, which is a field rather than global
// state so a seed and a test can control the draw. The generator lives here
// because a server has no other use for it.
type temperatureSampler struct {
	temp float32
	topK int
	topP float32
	src  *rand.Rand
}

// Next samples one token.
func (t temperatureSampler) Next(logits []float32) int32 {
	return llama.Temperature{
		Temp:      t.temp,
		TopK:      t.topK,
		TopP:      t.topP,
		NextFloat: t.src.Float64,
	}.Next(logits)
}

// newRand returns a deterministic generator from a seed.
//
// Seeding per request from the clock is what makes two requests with the same
// prompt differ when the caller asked for sampling; a shared generator would also
// work, but it would make one request's output depend on how many requests happened
// to run before it. The engine advances sequences one round at a time, so a shared
// generator's draw order would depend on scheduling.
func newRand(seed int64) *rand.Rand {
	return rand.New(rand.NewPCG(uint64(seed), uint64(seed>>1)))
}
