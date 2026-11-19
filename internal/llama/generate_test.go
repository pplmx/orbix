package llama

import (
	"math"
	"slices"
	"testing"

	"github.com/pplmx/orbix/internal/gguftest"
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

// TestNucleusSortsUnsortedCandidates pins the sort that the tests above cannot see.
//
// Every other nucleus test passes logits in descending order, which is also the order
// candidates starts in, so removing the TopP sort leaves them green: nucleus takes its
// max from candidates[0], and with sorted input that happens to be the true maximum. A
// real logits vector has no such order. This test shuffles the same four probabilities, so
// the answer must be identical to the sorted case and a missing sort changes it.
func TestNucleusSortsUnsortedCandidates(t *testing.T) {
	// Same probabilities as TestNucleusTrimsToTheMassCoveringSet, permuted: the 0.5 token
	// is now at index 1, the 0.05 one at index 3.
	logits := []float32{
		float32(math.Log(0.15)),
		float32(math.Log(0.5)),
		float32(math.Log(0.3)),
		float32(math.Log(0.05)),
	}
	// p=0.6 is chosen to separate the two orders rather than merely to pass. Sorted, the
	// cumulative sums are 0.5, 0.8 and the run stops at index 2. Unsorted they are 0.15,
	// 0.65 and it stops at index 1 -- the same position but a different token. A threshold
	// like 0.7 does not distinguish them, which is how the first version of this test
	// passed with the sort removed.
	s := Temperature{Temp: 1, TopP: 0.6, NextFloat: func() float64 { return 0.99 }}
	if got := s.Next(logits); got != 2 {
		t.Errorf("chose token %d, want 2: the nucleus must be found by probability, "+
			"not by the order the logits happen to arrive in", got)
	}
}

// TestGenerateOptionHandling covers the four branches in Generate that no test reached.
//
// Every existing caller passes a non-nil Greedy and sets neither StopTokens nor OnToken, and the
// tiny fixture defines no end-of-sequence id, so the nil-sampler default, both sides of the stop
// set, and the callback were all unexecuted. The negative budget is here too: the engine rejects
// it before calling, but Generate is reachable without the engine.
func TestGenerateOptionHandling(t *testing.T) {
	newModel := func(t *testing.T) *Model {
		t.Helper()
		// NoEOS false so the fixture carries a real end-of-sequence id; greedy keeps every run
		// deterministic.
		path := gguftest.Model{ContextLen: 64}.Write(t, t.TempDir())
		m, err := New(path)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		return m
	}
	run := func(t *testing.T, m *Model, sampler Sampler, opts GenerateOptions) []int32 {
		t.Helper()
		cache, err := NewKVCache(m.Config(), m.Config().ContextLen)
		if err != nil {
			t.Fatalf("NewKVCache: %v", err)
		}
		out, err := m.Generate(cache, []int32{0, 1}, sampler, opts)
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		return out
	}
	eos := int32(newModel(t).Config().EOSID)
	if eos < 0 {
		t.Fatal("the fixture defines no end-of-sequence id")
	}

	t.Run("nil sampler means greedy", func(t *testing.T) {
		m := newModel(t)
		withNil := run(t, m, nil, GenerateOptions{MaxTokens: 6})
		withGreedy := run(t, newModel(t), Greedy{}, GenerateOptions{MaxTokens: 6})
		if !slices.Equal(withNil, withGreedy) {
			t.Errorf("nil sampler gave %v, Greedy gave %v", withNil, withGreedy)
		}
	})

	t.Run("a stop token ends generation", func(t *testing.T) {
		m := newModel(t)
		// Learn what the model emits, then ask it to stop on that token.
		free := run(t, newModel(t), Greedy{}, GenerateOptions{MaxTokens: 6})
		if len(free) <= 2 {
			t.Skipf("the fixture generated only the prompt (%v)", free)
		}
		target := free[2]
		stopped := run(t, m, Greedy{}, GenerateOptions{MaxTokens: 6, StopTokens: []int32{target}})
		// The prompt is two tokens, so a stop on the first generated token leaves exactly those.
		if len(stopped) >= len(free) {
			t.Errorf("stopping on %d produced %v, not shorter than %v", target, stopped, free)
		}
		if !slices.Equal(stopped, free[:len(stopped)]) {
			t.Errorf("stopped output %v is not a prefix of %v", stopped, free)
		}
	})

	t.Run("the end-of-sequence id is stopped on without being passed", func(t *testing.T) {
		// A sampler that always answers EOS, so the implicit stop entry is what ends the run.
		// Greedy cannot be used here: on random weights it settles on one token and never emits
		// EOS, which is why the first version of this subtest measured nothing.
		m := newModel(t)
		out := run(t, m, fixedSampler{tok: eos}, GenerateOptions{MaxTokens: 32})
		// Prompt only: the first sampled token was EOS, and EOS is never appended.
		if len(out) != 2 {
			t.Errorf("output = %v, want just the 2 prompt tokens: EOS %d should stop the run", out, eos)
		}
		// The same sampler on a token that is not EOS must keep going, or the test would pass
		// for a run that ignores its sampler.
		other := int32(0)
		if other == eos {
			other = 1
		}
		out2 := run(t, newModel(t), fixedSampler{tok: other}, GenerateOptions{MaxTokens: 4})
		if len(out2) != 6 {
			t.Errorf("output with a non-EOS sampler = %v, want the prompt plus 4", out2)
		}
	})

	t.Run("OnToken sees every generated token", func(t *testing.T) {
		m := newModel(t)
		var seen []int32
		out := run(t, m, Greedy{}, GenerateOptions{
			MaxTokens: 4,
			OnToken:   func(tok int32) error { seen = append(seen, tok); return nil },
		})
		if !slices.Equal(seen, out[2:]) {
			t.Errorf("OnToken saw %v, the continuation was %v", seen, out[2:])
		}
		if len(seen) == 0 {
			t.Error("OnToken was never called")
		}
	})

	t.Run("a negative budget is an error", func(t *testing.T) {
		m := newModel(t)
		cache, err := NewKVCache(m.Config(), m.Config().ContextLen)
		if err != nil {
			t.Fatalf("NewKVCache: %v", err)
		}
		if _, err := m.Generate(cache, []int32{0}, Greedy{}, GenerateOptions{MaxTokens: -1}); err == nil {
			t.Error("Generate accepted a negative budget")
		}
	})
}

// fixedSampler always answers the same token, so a test can put a specific id where Generate's
// stop set looks at it.
type fixedSampler struct{ tok int32 }

func (s fixedSampler) Next([]float32) int32 { return s.tok }
