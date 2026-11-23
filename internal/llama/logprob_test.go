package llama

import (
	"math"
	"testing"
)

// TestTokenLogprobsSumsToOne checks the strongest invariant available: the log probabilities
// this reports have to describe a probability distribution, so the exponentials over the whole
// vocabulary have to sum to one.
//
// With top set at the vocabulary width every token comes back, which turns the alternatives
// list into the distribution itself and makes the check exact rather than sampled. A logprobs
// implementation that forgot the log-sum-exp normaliser, or that scaled the logits without
// renormalising, passes a "chosen is the highest" check and fails this one.
func TestTokenLogprobsSumsToOne(t *testing.T) {
	cases := []struct {
		name   string
		logits []float32
		temp   float32
	}{
		{"uniform", []float32{1, 1, 1, 1}, 1},
		{"peaked", []float32{5, 1, 0, -3}, 1},
		{"wide spread", []float32{20, -20, 0, 3.5}, 1},
		{"hot temperature", []float32{5, 1, 0, -3}, 4},
		{"cold temperature", []float32{5, 1, 0, -3}, 0.25},
		{"greedy uses temp one", []float32{5, 1, 0, -3}, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, all := TokenLogprobs(c.logits, 0, c.temp, len(c.logits))
			if len(all) != len(c.logits) {
				t.Fatalf("got %d alternatives, want one per token (%d)", len(all), len(c.logits))
			}
			var sum float64
			for _, a := range all {
				sum += math.Exp(float64(a.Logprob))
			}
			if math.Abs(sum-1) > 1e-4 {
				t.Errorf("sum of exp(logprob) = %v, want 1", sum)
			}
			// Every token appears exactly once, or a probability would be missing from the
			// sum and it could not come back to 1.
			seen := make(map[int32]bool, len(all))
			for _, a := range all {
				if seen[a.ID] {
					t.Errorf("token %d appears twice in the alternatives", a.ID)
				}
				seen[a.ID] = true
			}
		})
	}
}

// TestTokenLogprobsOfUniformVocabulary pins an exact value.
//
// A uniform vocabulary has one right answer: every token has probability 1/n, so every log
// probability is -ln(n). An independent arithmetic check is what catches a normaliser that is
// off by a constant, which every relative comparison would miss.
func TestTokenLogprobsOfUniformVocabulary(t *testing.T) {
	const n = 8
	logits := make([]float32, n)
	for i := range logits {
		logits[i] = 2.5 // any constant: a uniform distribution is uniform whatever the value
	}
	want := float32(-math.Log(n))
	for _, chosen := range []int32{0, 3, n - 1} {
		got, alts := TokenLogprobs(logits, chosen, 1, n)
		if math.Abs(float64(got-want)) > 1e-6 {
			t.Errorf("chosen=%d: logprob = %v, want %v", chosen, got, want)
		}
		for _, a := range alts {
			if math.Abs(float64(a.Logprob-want)) > 1e-6 {
				t.Errorf("token %d: logprob = %v, want %v", a.ID, a.Logprob, want)
			}
		}
	}
}

// TestTokenLogprobsTemperatureFlattens is the property that makes the temperature argument
// worth having: a higher temperature has to make the distribution flatter, so the most likely
// token becomes less certain and the gap to the alternatives narrows.
//
// Getting this backwards, or ignoring the temperature entirely, still produces a valid
// distribution -- which is why the sum-to-one test above cannot see it.
func TestTokenLogprobsTemperatureFlattens(t *testing.T) {
	logits := []float32{4, 0, -4}
	// The argmax and the least likely token, at each temperature.
	coldBest, _ := TokenLogprobs(logits, 0, 0.5, 0)
	coldWorst, _ := TokenLogprobs(logits, int32(len(logits)-1), 0.5, 0)
	hotBest, _ := TokenLogprobs(logits, 0, 2, 0)
	hotWorst, _ := TokenLogprobs(logits, int32(len(logits)-1), 2, 0)

	if !(hotBest < coldBest) {
		t.Errorf("logprob of the argmax at temp 2 = %v, at temp 0.5 = %v; a hotter "+
			"temperature must make it less certain", hotBest, coldBest)
	}
	// The gap between the best and the worst has to shrink too, not only the absolute value.
	if coldGap, hotGap := coldBest-coldWorst, hotBest-hotWorst; !(hotGap < coldGap) {
		t.Errorf("gap to the least likely token: temp 2 gave %v, temp 0.5 gave %v", hotGap, coldGap)
	}
}

// TestTokenLogprobsTopBound covers the limits on the alternatives list.
//
// top=0 asks for none, which is what a request that wants only the chosen token's logprob
// sends; it must still compute that value rather than return early. A top above
// MaxTopLogprobs is cut to it, because the response would otherwise be longer than any
// client asked for.
func TestTokenLogprobsTopBound(t *testing.T) {
	logits := make([]float32, 100)
	for i := range logits {
		logits[i] = float32(i) / 100
	}

	if got, alts := TokenLogprobs(logits, 50, 1, 0); got == 0 || alts != nil {
		t.Errorf("top=0: got logprob %v and %d alternatives, want the logprob and none", got, len(alts))
	}

	_, alts := TokenLogprobs(logits, 50, 1, 1000)
	if len(alts) != MaxTopLogprobs {
		t.Errorf("top=1000 over a 100-token vocabulary returned %d alternatives, want %d",
			len(alts), MaxTopLogprobs)
	}

	_, five := TokenLogprobs(logits, 50, 1, 5)
	if len(five) != 5 {
		t.Fatalf("top=5 returned %d alternatives", len(five))
	}
	// Descending, and the highest logit really is first: the vocabulary is ordered, so the
	// answer is knowable without recomputing the distribution.
	for i := 1; i < len(five); i++ {
		if five[i-1].Logprob <= five[i].Logprob {
			t.Errorf("alternatives out of order at %d: %v then %v",
				i, five[i-1].Logprob, five[i].Logprob)
		}
	}
	if five[0].ID != 99 {
		t.Errorf("the most likely token should be the last of an ascending vocabulary, got %d", five[0].ID)
	}
}

// TestTokenLogprobsChosenMatchesItsAlternative checks that the two numbers come from one
// distribution.
//
// A caller reads the chosen token's logprob and expects to find the same token in the
// alternatives with the same value. Computing them from different normalisers would leave both
// individually plausible and mutually inconsistent.
func TestTokenLogprobsChosenMatchesItsAlternative(t *testing.T) {
	logits := []float32{3, 2.5, 1, 0, -1}
	chosen, alts := TokenLogprobs(logits, 1, 1, 5)
	for _, a := range alts {
		if a.ID != 1 {
			continue
		}
		if a.Logprob != chosen {
			t.Errorf("chosen logprob = %v, the alternatives list says %v", chosen, a.Logprob)
		}
		return
	}
	t.Fatalf("the chosen token is not in its own alternatives list: %v", alts)
}

// TestTokenLogprobsDegenerateInput covers the rows that describe no distribution.
//
// These cannot come from a loaded model -- the load-time finite check refuses a NaN weight --
// so this is about the function being total rather than about a case to defend against. A NaN
// log probability is worse than a zero one: it propagates into whatever a caller sums or
// compares, and JSON cannot even encode it.
func TestTokenLogprobsDegenerateInput(t *testing.T) {
	nan := float32(math.NaN())
	inf := float32(math.Inf(1))

	cases := []struct {
		name   string
		logits []float32
		chosen int32
	}{
		{"empty", nil, 0},
		{"all negative infinity", []float32{float32(math.Inf(-1)), float32(math.Inf(-1))}, 0},
		{"all NaN", []float32{nan, nan}, 0},
		{"chosen out of range", []float32{1, 2}, 99},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			lp, alts := TokenLogprobs(c.logits, c.chosen, 1, 4)
			if math.IsNaN(float64(lp)) {
				t.Errorf("chosen logprob is NaN, which no response can carry")
			}
			for _, a := range alts {
				if math.IsNaN(float64(a.Logprob)) {
					t.Errorf("token %d has a NaN logprob", a.ID)
				}
			}
		})
	}

	// An infinite temperature would divide every difference to zero and report a uniform
	// distribution, which is a claim about the model rather than about the request. It falls
	// back to 1 like the other unusable values.
	_, atInf := TokenLogprobs([]float32{5, 1, 0}, 0, inf, 3)
	_, atOne := TokenLogprobs([]float32{5, 1, 0}, 0, 1, 3)
	if len(atInf) != len(atOne) || atInf[0].Logprob != atOne[0].Logprob {
		t.Errorf("+Inf temperature gave %v, want the same as temperature 1", atInf)
	}
}
