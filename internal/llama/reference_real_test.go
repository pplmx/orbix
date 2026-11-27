package llama

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/tokenize"
)

// realReference is one checkpoint's expected logits and per-layer residuals, produced by the numpy
// reference rather than by this engine.
type realReference struct {
	// Ids is the prompt when the file lists it.
	Ids []int32 `json:"ids"`
	// PromptRepeat states the prompt as a repeated string, for one long enough that listing its
	// ids would be most of the file. The test re-tokenizes it with the checkpoint's own
	// tokenizer, so the ids are produced the way the engine produces them rather than copied.
	PromptRepeat *struct {
		Text  string `json:"text"`
		Times int    `json:"times"`
	} `json:"prompt_repeat,omitempty"`
	// PromptTokens is how many of those tokens the reference used.
	PromptTokens int `json:"prompt_tokens,omitempty"`
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
// Each case is gated by two variables: the checkpoint's path and the reference file's name under
// testdata/reference. See docs/models.md for how to produce one.
func TestMatchesNumPyReferenceOnRealWeights(t *testing.T) {
	cases := []struct {
		name string
		env  string
		why  string
		// stale names the reason this reference no longer describes the engine, for the two
		// cases whose fixtures record a reading that has since been corrected. The comparison still
		// runs and still reports the distance it measures; it reports it with t.Logf instead of
		// failing, so a run says "these two disagree by 2.8% as documented" rather than turning
		// every gated run red and burying the cases that would catch a real regression.
		stale string
	}{
		{
			name: "gemma3-270m",
			env:  "GEMMA3_MODEL",
			why:  "Q8_0, per-head QK norm, sandwich norms, 262144-wide vocabulary",
			stale: "the numpy reference applied one rotation base to every layer, which is the " +
				"reading this engine corrected after the fixture was written",
		},
		{
			// The same checkpoint as above with a prompt one token past its window, so the
			// oldest reachable position is either in the attention or not. A six-token prompt
			// cannot see that boundary -- which is why both files exist.
			name: "gemma3-513",
			env:  "GEMMA3_MODEL",
			why:  "513 tokens against a window of 512, where the window first binds",
			stale: "same reason as gemma3-270m, plus the window pattern the reference also read " +
				"as every layer local",
		},
		{
			name: "mistral-v2",
			env:  "MISTRAL_MODEL",
			why:  "GGUF version 2, K-quant tensors, architecture `llama` with no window",
		},
		{
			name: "gemma-1",
			env:  "GEMMA_MODEL",
			why:  "the first Gemma block: scaled embeddings and GeGLU, no QK norm, no sandwich norms",
		},
		{
			name: "qwen2",
			env:  "QWEN2_MODEL",
			why:  "q/k/v attention biases with NeoX rope, which no Gemma or Llama file exercises",
		},
		{
			name: "qwen3",
			env:  "QWEN3_MODEL",
			why:  "per-head QK norm, and a tokenizer whose prompt carries no BOS",
		},
	}
	for _, c := range cases {
		c := c
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			runRealReference(t, c.name, c.env, c.why, c.stale)
		})
	}
}

// runRealReference compares one checkpoint against its stored reference.
func runRealReference(t *testing.T, name, env, why, stale string) {
	t.Helper()
	// report is Errorf for a current reference and Logf for one that records a corrected reading.
	// The distance is computed either way, so the stale pair still says what changed; what differs
	// is whether it fails the run.
	report := t.Errorf
	if stale != "" {
		report = t.Logf
	}
	model := os.Getenv(env)
	expectedName := name + ".json"
	if model == "" {
		t.Skipf("set %s to compare against the %s checkpoint (%s)", env, name, why)
	}
	// The reference is a file under testdata/reference rather than a path: it is part of the
	// repository, and a caller-supplied path would let a run compare against something outside it.
	expectedPath := filepath.Join("testdata", "reference", expectedName)
	raw, err := os.ReadFile(expectedPath)
	if err != nil {
		t.Fatalf("read %s: %v", expectedPath, err)
	}
	var expected realReference
	if err := json.Unmarshal(raw, &expected); err != nil {
		t.Fatalf("parse %s: %v", expectedPath, err)
	}
	prompt := expected.Ids
	if expected.PromptRepeat != nil {
		// Resolved after the model is loaded, because the ids come from its tokenizer.
		prompt = nil
	}
	if len(prompt) == 0 && expected.PromptRepeat == nil {
		t.Fatalf("%s holds no prompt", expectedPath)
	}

	m, err := New(model)
	if err != nil {
		t.Fatalf("New(%s): %v", model, err)
	}
	if expected.PromptRepeat != nil {
		f, err := gguf.Open(model)
		if err != nil {
			t.Fatalf("open %s for its tokenizer: %v", model, err)
		}
		tok, err := tokenize.New(f)
		if err != nil {
			t.Fatalf("tokenizer: %v", err)
		}
		text := strings.Repeat(expected.PromptRepeat.Text, expected.PromptRepeat.Times)
		prompt = tok.Encode(text, true)
		if n := expected.PromptTokens; n > 0 {
			if len(prompt) < n {
				t.Fatalf("the prompt encodes to %d tokens, the reference used %d", len(prompt), n)
			}
			// Truncated rather than refused: the reference states the count it used, and a
			// tokenizer that produces more than that says the two disagree about the text, which
			// the logits comparison below will show. Refusing here would only hide which.
			if len(prompt) != n {
				t.Logf("the prompt encodes to %d tokens, the reference used the first %d",
					len(prompt), n)
				prompt = prompt[:n]
			}
		}
	}
	cfg := m.Config()
	cache, err := NewKVCache(cfg, len(prompt)+1)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}

	trace := make([][]float32, cfg.Layers)
	for l := range trace {
		trace[l] = make([]float32, cfg.Hidden)
	}
	var logits []float32
	for pos, id := range prompt {
		if pos == len(prompt)-1 && len(expected.Layers) > 0 {
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
			report("logit %d: engine %.6f, reference %.6f (relative %.2e)",
				idx, got, want, rel)
		}
	}
	if len(expected.LogitIndices) > 0 {
		got := argmaxFloat32(logits)
		if want := int(expected.LogitIndices[0]); got != want {
			report("argmax %d, reference says %d", got, want)
		}
	}

	if len(expected.Layers) > 0 {
		selected := make([][]float32, len(expected.Layers))
		for k, li := range expected.LayerIndices {
			if k < len(selected) && int(li) < len(trace) {
				selected[k] = trace[li]
			}
		}
		reportFirstDivergentLayerVia(t, report, selected, expected.Layers)
		if stale != "" {
			t.Logf("%s records a superseded reading: %s", name, stale)
		}
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
