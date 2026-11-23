package llama

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// realReference is one checkpoint's expected logits and per-layer residuals, produced by the numpy
// reference rather than by this engine.
type realReference struct {
	Ids []int32 `json:"ids"`
	// Logits and their ids: the whole vocabulary is 262144 wide, which is 5 MiB of JSON for a
	// comparison whose informative part is the top of the distribution. Keeping the top few and
	// their indices checks the same thing at 1/140th the size.
	Logits       []float64 `json:"logits"`
	LogitIndices []int32   `json:"logit_indices"`
	// Layers holds chosen blocks rather than every one, for the same reason: the first, middle
	// and last are enough to say where a divergence starts, and the full set would double the
	// file again.
	Layers       [][]float64 `json:"layers,omitempty"`
	LayerIndices []int32     `json:"layer_indices,omitempty"`
}

// TestMatchesNumPyReferenceOnRealWeights runs the same comparison on a checkpoint.
//
// The fixtures in internal/gguftest carry small pseudo-random values, so two implementations that
// agree on them agree on the shape of the arithmetic and little else. Real weights exercise the
// numerical range the fixtures cannot: activations three orders of magnitude larger, norms whose
// weights were trained rather than set to one, and a vocabulary large enough that the output
// projection dominates the error budget.
//
// Gated on GEMMA3_REFERENCE, a JSON file written by the numpy reference for the checkpoint named in
// GEMMA3_MODEL. See docs/models.md for how to produce it.
func TestMatchesNumPyReferenceOnRealWeights(t *testing.T) {
	model := os.Getenv("GEMMA3_MODEL")
	expectedName := os.Getenv("GEMMA3_REFERENCE")
	if model == "" || expectedName == "" {
		t.Skip("set GEMMA3_MODEL and GEMMA3_REFERENCE to compare against a real checkpoint")
	}
	// The reference is a file name under testdata/reference, not a path: the fixture it belongs to
	// is part of the repository, and a path would let a run compare against something outside it.
	expectedPath := filepath.Join("testdata", "reference", expectedName)
	raw, err := os.ReadFile(expectedPath)
	if err != nil {
		t.Fatalf("read %s: %v", expectedPath, err)
	}
	var expected realReference
	if err := json.Unmarshal(raw, &expected); err != nil {
		t.Fatalf("parse %s: %v", expectedPath, err)
	}
	if len(expected.Ids) == 0 {
		t.Fatalf("%s holds no prompt", expectedPath)
	}

	m, err := New(model)
	if err != nil {
		t.Fatalf("New(%s): %v", model, err)
	}
	cfg := m.Config()
	cache, err := NewKVCache(cfg, len(expected.Ids)+1)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}

	trace := make([][]float32, cfg.Layers)
	for l := range trace {
		trace[l] = make([]float32, cfg.Hidden)
	}
	var logits []float32
	for pos, id := range expected.Ids {
		if pos == len(expected.Ids)-1 && len(expected.Layers) > 0 {
			m.layerTrace = trace
		}
		logits, err = m.Forward(cache, int(id), pos)
		m.layerTrace = nil
		if err != nil {
			t.Fatalf("Forward at %d: %v", pos, err)
		}
	}
	if len(expected.Logits) != len(expected.LogitIndices) {
		t.Fatalf("the reference has %d logits and %d indices for them",
			len(expected.Logits), len(expected.LogitIndices))
	}
	for _, idx := range expected.LogitIndices {
		if int(idx) >= len(logits) {
			t.Fatalf("the reference names logit %d, the engine produces %d", idx, len(logits))
		}
	}

	// A checkpoint's logits span tens of units, so an absolute tolerance that fits a fixture is
	// meaningless here. What is compared is the argmax and the logits' relative distance: the two
	// implementations accumulate in different precisions, and the question is whether they agree
	// on the distribution, not whether they agree to a fixed number of decimal places.
	// The top of the distribution, by index, so a mismatch names the token rather than a
	// position in a reduced file.
	for k, idx := range expected.LogitIndices {
		got, want := float64(logits[idx]), expected.Logits[k]
		denom := math.Max(math.Abs(want), 1)
		if rel := math.Abs(got-want) / denom; rel > 1e-3 {
			t.Errorf("logit %d: engine %.6f, reference %.6f (relative %.2e)",
				idx, got, want, rel)
		}
	}
	if len(expected.LogitIndices) > 0 {
		got := argmaxFloat32(logits)
		if want := int(expected.LogitIndices[0]); got != want {
			t.Errorf("argmax %d, reference says %d", got, want)
		}
	}

	if len(expected.Layers) > 0 {
		selected := make([][]float32, len(expected.Layers))
		for k, li := range expected.LayerIndices {
			if k < len(selected) && int(li) < len(trace) {
				selected[k] = trace[li]
			}
		}
		reportFirstDivergentLayer(t, selected, expected.Layers)
	}
}

func argmaxFloat32(v []float32) int {
	b := 0
	for i, x := range v {
		if x > v[b] {
			b = i
		}
	}
	return b
}
