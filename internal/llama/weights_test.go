package llama

import (
	"math"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguftest"
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

// TestQuantizedNaNAScaleIsRejectedAtLoad closes the gap between the decoder and the check.
//
// checkFinite was tested with hand-built maps and the decoder was tested on its own, but
// nothing joined them: a GGUF file whose quantized tensor carries a NaN scale is the shape a
// corrupt or truncated download actually has, and the path from "16 bytes decode to NaN" to
// "load fails naming the tensor" had no test end to end.
//
// The values matter. A Q4_0 block's first two bytes are an F16 scale, and 0x7E00 is the NaN
// pattern the format allows; the nibbles are zero, so every weight in the block would be
// 0 * NaN = NaN. Decoding succeeds, which is the point: the decoder's job is to read what the
// file says, and the load is where a bad file has to be caught.
//
// Three tensors rather than one, because they take different routes to the check: an
// attention projection, the embedding table, and a normalization vector. A guard that only
// covered the square matrices would pass the first and miss the others.
func TestQuantizedNaNAScaleIsRejectedAtLoad(t *testing.T) {
	for _, name := range []string{
		"blk.0.attn_q.weight",
		"token_embd.weight",
		"output_norm.weight",
	} {
		t.Run(name, func(t *testing.T) {
			path := gguftest.Model{
				NoEOS: true, ContextLen: 64, NaNScaleTensor: name,
			}.Write(t, t.TempDir())

			_, err := New(path)
			if err == nil {
				t.Fatal("a NaN scale loaded without error, so generation would run on NaN weights")
			}
			// The message has to name the tensor and the position: "the file is corrupt" alone
			// leaves a user with a 4 GB file and no way to tell which part of it went wrong.
			for _, want := range []string{name, "NaN", "non-finite"} {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not mention %q", err, want)
				}
			}
		})
	}
}
