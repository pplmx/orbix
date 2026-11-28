package main

import (
	"math"
	"strings"
	"testing"
)

// TestValidateRunFlagsRefusesNonFinite covers the values that pass every range comparison.
//
// The comment on the check records what a NaN temperature did: it reached the sampler, made sum
// NaN, failed every `r < acc`, and returned the last candidate -- the least likely token in the
// set. A range check alone cannot catch it, because NaN compares false with everything.
func TestValidateRunFlagsRefusesNonFinite(t *testing.T) {
	cases := []struct {
		name  string
		v     runFlags
		wants string
	}{
		{"NaN temperature", runFlags{sampling: samplingFlags{temp: math.NaN()}}, "--temp"},
		{"infinite temperature", runFlags{sampling: samplingFlags{temp: math.Inf(1)}}, "--temp"},
		{"NaN top-p", runFlags{sampling: samplingFlags{topP: math.NaN()}}, "--top-p"},
		{"infinite top-p", runFlags{sampling: samplingFlags{topP: math.Inf(-1)}}, "--top-p"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := validateRunFlags(c.v)
			if err == nil {
				t.Fatalf("%s was accepted", c.name)
			}
			// The flag, not the value: a caller needs to know which one to change. The
			// message names it with its double dash, which is what they typed.
			if !strings.Contains(err.Error(), c.wants) {
				t.Errorf("the refusal does not name %s: %v", c.wants, err)
			}
		})
	}
}

// TestValidateRunFlagsRefusesNegativeValues covers the flags the code compares against "> 0".
//
// A negative value silently meant the same as zero -- greedy decoding, and a top-k never
// applied -- which is the opposite of what "turn sampling down" reads as.
func TestValidateRunFlagsRefusesNegativeValues(t *testing.T) {
	cases := []struct {
		name  string
		v     runFlags
		wants string
	}{
		{"negative temperature", runFlags{sampling: samplingFlags{temp: -0.5}}, "--temp"},
		{"negative top-k", runFlags{sampling: samplingFlags{topK: -1}}, "--top-k"},
		{"negative top-p", runFlags{sampling: samplingFlags{topP: -0.1}}, "--top-p"},
		{"negative workers", runFlags{workers: -2}, "--workers"},
		{"negative max-tokens", runFlags{maxTokens: -5}, "--max-tokens"},
		{"negative ctx", runFlags{contextLen: -100}, "--ctx"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := validateRunFlags(c.v)
			if err == nil {
				t.Fatalf("%s was accepted", c.name)
			}
			if !strings.Contains(err.Error(), c.wants) {
				t.Errorf("the refusal does not name %s: %v", c.wants, err)
			}
		})
	}
}

// TestValidateRunFlagsRefusesATopPAboveOne is the one value that is neither negative nor
// non-finite.
//
// Zero and one both mean "no limit" to the sampler and the flag documents 0 as the value that
// disables the cutoff, so both are accepted. Above one is not a wider cutoff but a typo, and
// clamping it would apply a cutoff the caller did not ask for.
func TestValidateRunFlagsRefusesATopPAboveOne(t *testing.T) {
	err := validateRunFlags(runFlags{sampling: samplingFlags{topP: 1.5}})
	if err == nil {
		t.Fatal("--top-p 1.5 was accepted")
	}
	if !strings.Contains(err.Error(), "--top-p") {
		t.Errorf("the refusal does not name the flag: %v", err)
	}
}

// TestValidateRunFlagsAcceptsTheOrdinaryValues is the other half: a normal invocation must pass.
//
// Without it every check could refuse everything and the tests above would still be green.
func TestValidateRunFlagsAcceptsTheOrdinaryValues(t *testing.T) {
	cases := []struct {
		name string
		v    runFlags
	}{
		{"greedy, no limits", runFlags{}},
		{"sampling on", runFlags{sampling: samplingFlags{temp: 0.8, topK: 40, topP: 0.95}}},
		{"top-p at both ends", runFlags{sampling: samplingFlags{topP: 1}}},
		{"workers set", runFlags{workers: 4}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if err := validateRunFlags(c.v); err != nil {
				t.Errorf("ordinary flags were refused: %v", err)
			}
		})
	}
}
