package llama

import (
	"math"
	"testing"
)

// TestNucleusTrimsToTheMassCoveringSet checks top-p against hand-computed probabilities.
//
// The logits are chosen so that a temperature-1 softmax gives exactly 0.5, 0.3, 0.15 and
// 0.05 -- they are the natural logs of those numbers -- which makes the cumulative sums
// 0.5, 0.8, 0.95 and 1.0. A nucleus threshold then has an unambiguous answer at each point,
// so the test can assert the exact candidate count rather than a statistical property.
func TestNucleusTrimsToTheMassCoveringSet(t *testing.T) {
	logits := []float32{
		float32(math.Log(0.5)),
		float32(math.Log(0.3)),
		float32(math.Log(0.15)),
		float32(math.Log(0.05)),
	}
	// The thresholds sit strictly between the cumulative sums rather than on them. Landing
	// exactly on a boundary puts the answer at the mercy of float32 rounding: the logits are
	// float32 and their exponentials drift by about 3e-9, enough to decide a comparison that
	// was meant to be exact.
	cases := []struct {
		p        float32
		wantKept int
	}{
		{0.3, 1}, // the first token alone holds 0.5, which covers 0.3
		{0.7, 2}, // 0.5 is short of 0.7; 0.8 is not
		{0.9, 3}, // 0.8 is short of 0.9; 0.95 is not
		{0.99, 4},
		{1.0, 4}, // at or above 1 the limit is off, so every candidate stays
		{0.1, 1}, // no single token reaches 0.1; the run still keeps one
	}
	for _, c := range cases {
		// A sampler whose draw lands at 0 would always pick the first candidate, so
		// assert on the candidate set through the observed choice: with NextFloat at 0.99
		// the choice is the last candidate the nucleus kept.
		s := Temperature{Temp: 1, TopP: c.p, NextFloat: func() float64 { return 0.99 }}
		got := s.Next(logits)
		// The kept set is the leading run; the draw at 0.99 lands in its last member.
		if got != int32(c.wantKept-1) {
			t.Errorf("top_p=%v: chose token %d, want the last of %d kept", c.p, got, c.wantKept)
		}
	}
}

// TestNucleusKeepsAtLeastOne covers the degenerate threshold.
//
// A p small enough that no single token's probability reaches it would leave the candidate
// set empty, and an empty set is not a distribution -- the sampler would have nothing to
// return. Keeping the single most likely token is the useful reading of "as narrow as
// possible".
func TestNucleusKeepsAtLeastOne(t *testing.T) {
	// Probabilities are 1/3, 1/3, 1/3; no single one reaches 0.1.
	logits := []float32{0, 0, 0}
	s := Temperature{Temp: 1, TopP: 0.1, NextFloat: func() float64 { return 0.0 }}
	if got := s.Next(logits); got != 0 {
		t.Errorf("chose token %d, want 0", got)
	}
}

// TestNucleusRespectsTopKFirst covers the interaction between the two limits.
//
// When both are set, top-k narrows the set and the nucleus is computed over the survivors.
// That order matters: a token outside the k best can never be kept however much mass the
// nucleus threshold would otherwise give it.
func TestNucleusRespectsTopKFirst(t *testing.T) {
	// Four equal logits, so the top-k cut keeps the first two by index order and each of
	// those holds half of the *surviving* mass.
	logits := []float32{0, 0, 0, 0}
	s := Temperature{Temp: 1, TopK: 2, TopP: 0.6, NextFloat: func() float64 { return 0.99 }}
	got := s.Next(logits)
	if got >= 2 {
		t.Errorf("chose token %d, which top-k excluded", got)
	}
}

// TestTopPOffWhenDisabled covers the values that mean "no nucleus".
//
// Zero is the unset value, and a threshold of 1 keeps everything by definition. Both have to
// leave the distribution alone, or a caller that never heard of top-p gets a different reply
// than before it was added.
func TestTopPOffWhenDisabled(t *testing.T) {
	logits := []float32{2.0, 1.0, 0.5, 0.1}
	for _, p := range []float32{0, -1, 1, 2} {
		// With the limit off, a draw of 0.99 reaches the tail.
		off := Temperature{Temp: 1, TopP: p, NextFloat: func() float64 { return 0.99 }}
		// Compare against a sampler with no top-p field set at all.
		plain := Temperature{Temp: 1, NextFloat: func() float64 { return 0.99 }}
		if got, want := off.Next(logits), plain.Next(logits); got != want {
			t.Errorf("top_p=%v chose %d, want the same as no limit (%d)", p, got, want)
		}
	}
}
