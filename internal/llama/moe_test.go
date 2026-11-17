package llama

import (
	"math"
	"testing"

	"github.com/pplmx/orbix/internal/gguftest"
)

// The mixture-of-experts forward pass had no test at all: the fixture could build an MoE
// model, but nothing ran inference on it, so the whole routing path -- scoring, selecting,
// gathering the chosen rows, scattering the results -- was covered only by whatever a
// hand-written check happened to touch. These are constructed-model tests, so they run in CI.

// TestSelectExperts checks the router's contract.
func TestSelectExperts(t *testing.T) {
	logits := []float32{0.5, 3.0, -1.0, 2.0, 0.1}

	selected, weights := selectExperts(logits, 2)
	if len(selected) != 2 || len(weights) != 2 {
		t.Fatalf("selected %d experts with %d weights, want 2 and 2", len(selected), len(weights))
	}
	// The two largest logits are at 1 and 3.
	if selected[0] != 1 || selected[1] != 3 {
		t.Errorf("selected %v, want [1 3] (the two largest logits)", selected)
	}
	// Weights are a softmax over the selected logits, so they sum to one.
	var sum float64
	for _, w := range weights {
		if w < 0 || w > 1 {
			t.Errorf("weight %v is outside [0,1]", w)
		}
		sum += float64(w)
	}
	if math.Abs(sum-1) > 1e-5 {
		t.Errorf("weights sum to %v, want 1", sum)
	}
	// A larger logit gets the larger weight.
	if weights[0] <= weights[1] {
		t.Errorf("weights %v do not follow the logit order", weights)
	}

	// Asking for more experts than exist returns all of them rather than failing.
	all, allW := selectExperts(logits, 99)
	if len(all) != len(logits) || len(allW) != len(logits) {
		t.Errorf("k larger than the expert count returned %d, want %d", len(all), len(logits))
	}
	// k of zero and an empty router select nothing.
	if s, w := selectExperts(logits, 0); s != nil || w != nil {
		t.Errorf("k=0 returned %v, %v; want nil, nil", s, w)
	}
	if s, w := selectExperts(nil, 2); s != nil || w != nil {
		t.Errorf("an empty router returned %v, %v; want nil, nil", s, w)
	}
}

// TestSelectExpertsTieGoesToLowerID pins the tie-break, which decides which expert a
// perfectly balanced router picks. An unstable choice would make the same prompt route
// differently between runs.
func TestSelectExpertsTieGoesToLowerID(t *testing.T) {
	selected, _ := selectExperts([]float32{2, 2, 2}, 1)
	if len(selected) != 1 || selected[0] != 0 {
		t.Errorf("selected %v on a three-way tie, want [0]", selected)
	}
}

// TestMoEPrefillProducesFiniteLogits runs inference on a model whose layers are mixtures.
func TestMoEPrefillProducesFiniteLogits(t *testing.T) {
	path := gguftest.Model{
		Experts:    4,
		ExpertUsed: 2,
		NoEOS:      true,
		ContextLen: 128,
	}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	if cfg.Experts != 4 || cfg.ExpertUsed != 2 {
		t.Fatalf("fixture loaded %d experts using %d, want 4 and 2", cfg.Experts, cfg.ExpertUsed)
	}
	if m.W.Layers[0].router == nil {
		t.Fatal("an MoE layer has no router")
	}

	cache, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}
	logits, err := m.Prefill(cache, fixtureTokens(cfg, 5), 0)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	if len(logits) != cfg.VocabSize {
		t.Fatalf("returned %d logits, want %d", len(logits), cfg.VocabSize)
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logit[%d] is %v", i, v)
		}
	}
}

// TestMoENeedsRouting checks the experts are actually consulted: zeroing every expert weight
// must change the output, which a forward pass that fell through to a dense path would not.
func TestMoENeedsRouting(t *testing.T) {
	path := gguftest.Model{
		Experts:    4,
		ExpertUsed: 2,
		NoEOS:      true,
		ContextLen: 128,
	}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()

	run := func() []float32 {
		cache, err := NewKVCache(cfg, cfg.ContextLen)
		if err != nil {
			t.Fatalf("NewKVCache: %v", err)
		}
		logits, err := m.Prefill(cache, fixtureTokens(cfg, 4), 0)
		if err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		return append([]float32(nil), logits...)
	}

	before := run()
	for l := range m.W.Layers {
		layer := &m.W.Layers[l]
		for e := range layer.expertGate {
			clear(layer.expertGate[e])
			clear(layer.expertUp[e])
			clear(layer.expertDown[e])
		}
	}
	after := run()

	for i := range before {
		if before[i] != after[i] {
			return
		}
	}
	t.Error("zeroing the expert weights left the logits unchanged, so the forward pass is " +
		"not running the experts")
}
