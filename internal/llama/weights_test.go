package llama

import (
	"math"
	"strings"
	"testing"
)

// TestCheckFiniteRejectsNonFiniteWeights covers the load-time guard.
//
// A single NaN weight is not a local defect. It flows through the layer norms into every logit,
// and the model then answers with token 0 repeatedly -- valid ids that spell nothing -- while
// exiting 0. Nothing downstream reports it, so the user concludes the model is bad rather than the
// file. This guard turns that into a message naming the tensor.
//
// Found by writing a NaN into stories15M's output_norm.weight: all 32,000 logits came back NaN and
// the run printed a line of replacement characters with a zero exit code.
func TestCheckFiniteRejectsNonFiniteWeights(t *testing.T) {
	ok := map[string][]float32{"good": {1, 2, 3}}
	if err := checkFinite(ok); err != nil {
		t.Errorf("checkFinite on finite weights: %v", err)
	}

	cases := []struct {
		name  string
		value float32
		want  string
	}{
		{"NaN", float32(math.NaN()), "NaN"},
		{"+Inf", float32(math.Inf(1)), "+Inf"},
		{"-Inf", float32(math.Inf(-1)), "-Inf"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tensors := map[string][]float32{
				"aaa.weight": {1, 2},
				"zzz.weight": {1, c.value, 3},
			}
			err := checkFinite(tensors)
			if err == nil {
				t.Fatalf("checkFinite accepted a %s weight", c.name)
			}
			if !strings.Contains(err.Error(), "zzz.weight") {
				t.Errorf("error %q does not name the tensor", err)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("error %q does not report the value", err)
			}
		})
	}
}

// TestCheckFiniteNamesTheFirstTensorInOrder pins the determinism of the message.
//
// Two corrupt tensors have to produce the same error on every load. Iterating a map directly would
// name whichever came first, so a user comparing two runs would see different tensors and reasonably
// conclude the corruption was somewhere else.
func TestCheckFiniteNamesTheFirstTensorInOrder(t *testing.T) {
	nan := float32(math.NaN())
	tensors := map[string][]float32{
		"m.weight": {nan},
		"a.weight": {nan},
		"z.weight": {nan},
	}
	for range 20 {
		err := checkFinite(tensors)
		if err == nil {
			t.Fatal("no error")
		}
		if !strings.Contains(err.Error(), `"a.weight"`) {
			t.Fatalf("error %q does not name the alphabetically first tensor", err)
		}
	}
}
