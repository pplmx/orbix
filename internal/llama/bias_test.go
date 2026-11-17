package llama

import (
	"testing"

	"github.com/pplmx/orbix/internal/gguftest"
)

// TestAttentionBiasChangesLogits checks the forward pass reads the q, k, and v bias vectors.
//
// The bias is the difference between the Qwen2 and Llama attention shapes, and a forward
// pass that loads the vectors and then ignores them is the hardest kind of wrong to notice:
// the model still produces fluent text, from slightly different numbers, so nothing fails
// and nothing looks broken. Comparing the logits with and without the vectors is the direct
// check.
//
// A constructed model, so this runs in CI. The Qwen2 test that covered this before was gated
// on QWEN2_MODEL and ran only when someone had the file -- which is how the fixture's
// AttnBias field ended up unused while the coverage gap it was added for stayed open.
func TestAttentionBiasChangesLogits(t *testing.T) {
	path := gguftest.Model{
		Arch:       "qwen2",
		AttnBias:   true,
		NoEOS:      true,
		ContextLen: 128,
	}.Write(t, t.TempDir())

	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	if !cfg.AttnBias {
		t.Fatal("the fixture did not load as a model with attention bias")
	}
	for i := range m.W.Layers {
		if len(m.W.Layers[i].qBias) != cfg.Heads*cfg.HeadDim {
			t.Fatalf("layer %d q bias is %d wide, want %d",
				i, len(m.W.Layers[i].qBias), cfg.Heads*cfg.HeadDim)
		}
	}

	run := func() []float32 {
		cache, err := NewKVCache(cfg, cfg.ContextLen)
		if err != nil {
			t.Fatalf("NewKVCache: %v", err)
		}
		logits, err := m.Prefill(cache, fixtureTokens(cfg, 6), 0)
		if err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		return append([]float32(nil), logits...)
	}

	withBias := run()
	for i := range m.W.Layers {
		clear(m.W.Layers[i].qBias)
		clear(m.W.Layers[i].kBias)
		clear(m.W.Layers[i].vBias)
	}
	withoutBias := run()

	if len(withBias) != len(withoutBias) {
		t.Fatalf("logit counts differ: %d and %d", len(withBias), len(withoutBias))
	}
	for i := range withBias {
		if withBias[i] != withoutBias[i] {
			return // found the difference, which is what this asserts
		}
	}
	t.Error("zeroing the attention biases did not change the logits, so the forward " +
		"pass is not reading them")
}

// TestAttentionBiasAbsentForLlama checks the other half: a model with no bias tensors loads
// without them, so the flag follows the file rather than the arch name.
func TestAttentionBiasAbsentForLlama(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 128}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if m.Config().AttnBias {
		t.Error("a Llama fixture loaded with attention bias")
	}
	for i := range m.W.Layers {
		if len(m.W.Layers[i].qBias) != 0 {
			t.Errorf("layer %d has a q bias although the file declares none", i)
		}
	}
}
