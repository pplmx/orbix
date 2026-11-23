package llama

import (
	"math"
	"testing"
)

// greedySampler picks the highest logit, which is what the temperature-0 path ends at.
type greedySampler struct{}

func (greedySampler) Next(logits []float32) int32 {
	best := int32(0)
	for i, v := range logits {
		if v > logits[best] {
			best = int32(i)
		}
	}
	return best
}

// TestBiasedAppliesBeforeTheInnerSampler pins the order TokenBias depends on.
//
// logit_bias reaches the engine as a wrapper around whatever sampler the request asked for, so the
// biases are added to the logits and the inner sampler then chooses. Applying them afterwards would
// be too late -- the token would already be picked -- and that is the whole point of the feature:
// a bias of -100 on a token makes it not be chosen, which is a different claim from "chosen less
// often".
//
// Nothing in this package tested TokenBias or Biased. The server tests exercise logit_bias end to
// end, but coverage is per package: those runs call into here without counting toward it, so a
// broken Biased would have failed nothing in this package and failed the server's tests only if
// they happened to assert the banned token was absent.
func TestBiasedAppliesBeforeTheInnerSampler(t *testing.T) {
	logits := make([]float32, 8)
	logits[3] = 5 // the inner sampler's own preference
	logits[6] = 1

	// Without a bias the inner sampler's answer stands, so the fixture is not already decided by
	// the bias.
	if got := (Biased{Inner: greedySampler{}}).Next(append([]float32(nil), logits...)); got != 3 {
		t.Fatalf("unbiased pick = %d, want 3; the fixture does not set up the case", got)
	}

	// A bias large enough to flip the choice has to flip it.
	flip := Biased{Bias: []TokenBias{{ID: 6, Bias: 10}}, Inner: greedySampler{}}
	if got := flip.Next(append([]float32(nil), logits...)); got != 6 {
		t.Errorf("pick with +10 on token 6 = %d, want 6: the bias was not applied before the"+
			" inner sampler chose", got)
	}

	// The ban case: -100 on the leader leaves the next-best token.
	ban := Biased{Bias: []TokenBias{{ID: 3, Bias: -100}}, Inner: greedySampler{}}
	if got := ban.Next(append([]float32(nil), logits...)); got == 3 {
		t.Error("pick with -100 on token 3 chose it anyway; a ban that does not ban is the" +
			" failure the documented -100 and 100 exist to avoid")
	}

	// The logits are modified in place: the engine hands over a row it owns, and copying a whole
	// vocabulary per token to avoid that would cost more than the arithmetic.
	inPlace := append([]float32(nil), logits...)
	Biased{Bias: []TokenBias{{ID: 1, Bias: 2.5}}, Inner: greedySampler{}}.Next(inPlace)
	if inPlace[1] != 2.5 {
		t.Errorf("logit 1 = %v after a 2.5 bias, want 2.5: the bias was not written in place",
			inPlace[1])
	}
}

// TestBiasedIgnoresIdsOutsideTheVocabulary covers the guard that keeps a caller's bad id from
// panicking the forward pass.
//
// A bias id arrives from a request body, so it is a number a caller chose. Indexing with it would
// panic on a negative or an id at or past the vocabulary size, and the panic would happen inside an
// engine goroutine where the HTTP handler cannot recover it.
func TestBiasedIgnoresIdsOutsideTheVocabulary(t *testing.T) {
	// Token 1 starts lowest, so only the in-range bias can make it the pick.
	logits := []float32{1, 0.5, 3}
	bias := Biased{
		Bias: []TokenBias{
			{ID: -1, Bias: 100},
			{ID: 3, Bias: 100},
			{ID: -1000, Bias: 100},
			{ID: 1, Bias: 4},
		},
		Inner: greedySampler{},
	}
	got := bias.Next(logits)
	if got != 1 {
		t.Errorf("pick = %d, want 1: only the in-range bias should have applied", got)
	}
	if logits[1] != 4.5 {
		t.Errorf("logit 1 = %v after a 4 bias, want 4.5; the in-range bias did not apply",
			logits[1])
	}
	for i, v := range logits {
		if i == 1 {
			continue
		}
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("logit %d became %v; an out-of-range id must be ignored, not applied", i, v)
		}
	}
}
