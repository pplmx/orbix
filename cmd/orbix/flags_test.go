package main

import (
	"math"
	"testing"
)

// TestSamplingFlagsAreSharedByRunAndChat guards the reason registerSamplingFlags exists.
//
// run and chat each registered --temp, --top-k and --top-p by hand, with wording that had to stay
// identical and validation that had to stay equivalent. Nothing failed while they agreed, which is
// exactly what makes hand-maintained duplication dangerous: the drift only shows up as a behaviour
// difference between two commands nobody thought to compare.
//
// The check is that one validation refuses the same values for both commands and that the message
// differs only by the prefix, which is the property the shared helper has to keep.
func TestSamplingFlagsAreSharedByRunAndChat(t *testing.T) {
	cases := []struct {
		name string
		set  func(*samplingFlags)
		want string
	}{
		{"negative temp", func(s *samplingFlags) { s.temp = -1 },
			"--temp must be a finite, non-negative number (0 means greedy)"},
		{"NaN temp", func(s *samplingFlags) { s.temp = math.NaN() },
			"--temp must be a finite, non-negative number (0 means greedy)"},
		{"infinite temp", func(s *samplingFlags) { s.temp = math.Inf(1) },
			"--temp must be a finite, non-negative number (0 means greedy)"},
		{"negative top-k", func(s *samplingFlags) { s.topK = -1 },
			"--top-k must not be negative (0 disables it)"},
		{"top-p above one", func(s *samplingFlags) { s.topP = 2 },
			"--top-p must be between 0 and 1 (0 disables it)"},
		{"NaN top-p", func(s *samplingFlags) { s.topP = math.NaN() },
			"--top-p must be between 0 and 1 (0 disables it)"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			for _, cmd := range []string{"run", "chat"} {
				var s samplingFlags
				c.set(&s)
				err := s.validate(cmd)
				if err == nil {
					t.Fatalf("%s accepted the value", cmd)
				}
				if want := cmd + ": " + c.want; err.Error() != want {
					t.Errorf("%s: err = %q, want %q", cmd, err, want)
				}
			}
		})
	}

	// A valid combination is accepted with either prefix, or the refusals above would be
	// satisfied by a function that rejected everything.
	for _, cmd := range []string{"run", "chat"} {
		if err := (samplingFlags{temp: 0.7, topK: 40, topP: 0.9}).validate(cmd); err != nil {
			t.Errorf("%s refused valid sampling settings: %v", cmd, err)
		}
		// Zero is meaningful for all three: greedy, cutoff off, cutoff off.
		if err := (samplingFlags{}).validate(cmd); err != nil {
			t.Errorf("%s refused the zero value: %v", cmd, err)
		}
	}
}
