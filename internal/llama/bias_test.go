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

// TestDecodeStepReadsAttentionBias covers the bias on the batched decode path.
//
// TestAttentionBiasChangesLogits covers Forward, which serves one sequence per call. A
// served request does not go through Forward: engine.Pump batches every active sequence
// into one DecodeStep, and that path has its own bias call. The two are separate code, and
// the coverage said so -- addBiasDecode sat at 25% with every checkpoint set, which is the
// nil-bias early return and nothing else. Neutering its three calls left the whole suite
// green, so nothing was checking the biases a Qwen2 model gets in a batch of one or more.
//
// Both cache pairs are prefilled while the biases are in place, and the biases are cleared
// only after the second prefill. That ordering is the whole test: prefill has its own bias
// call, so a pair prefilled without the vectors would differ in its keys and values and the
// logits would differ whatever DecodeStep did -- which is what an earlier version of this
// test asserted, and it passed with DecodeStep's bias call removed.
func TestDecodeStepReadsAttentionBias(t *testing.T) {
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

	// Two sequences, so the decode step runs the per-row bias add rather than a single row.
	ids := fixtureTokens(cfg, 6)
	tokens := []int32{ids[4], ids[5]}
	prefilled := func() []Cache {
		caches := make([]Cache, len(tokens))
		for i := range tokens {
			cache, err := NewKVCache(cfg, cfg.ContextLen)
			if err != nil {
				t.Fatalf("NewKVCache: %v", err)
			}
			caches[i] = cache
			if _, err := m.Prefill(cache, ids[:4], 0); err != nil {
				t.Fatalf("Prefill: %v", err)
			}
		}
		return caches
	}

	withBiasCaches := prefilled()
	withoutBiasCaches := prefilled()

	withBias, err := m.DecodeStep(withBiasCaches, tokens)
	if err != nil {
		t.Fatalf("DecodeStep: %v", err)
	}
	withBias = append([]float32(nil), withBias...)

	for i := range m.W.Layers {
		clear(m.W.Layers[i].qBias)
		clear(m.W.Layers[i].kBias)
		clear(m.W.Layers[i].vBias)
	}
	withoutBias, err := m.DecodeStep(withoutBiasCaches, tokens)
	if err != nil {
		t.Fatalf("DecodeStep without bias: %v", err)
	}

	if len(withBias) != len(withoutBias) {
		t.Fatalf("logit counts differ: %d and %d", len(withBias), len(withoutBias))
	}
	for i := range withBias {
		if withBias[i] != withoutBias[i] {
			return // found the difference, which is what this asserts
		}
	}
	t.Error("zeroing the attention biases did not change the batched decode logits, so " +
		"DecodeStep is not reading them")
}

// TestForwardReadsAttentionBias covers the bias on the single-token path.
//
// Forward is what Generate decodes with, and it has a bias call of its own apart from
// Prefill's and DecodeStep's. TestAttentionBiasChangesLogits exercises Prefill, so a
// forward pass that loaded the vectors and then skipped them in Forward would still have
// looked covered: the coverage showed addBias reached, from the DecodeStep test's prefill
// neighbours rather than from this call. Neutering addBias left the suite green.
//
// As in the DecodeStep test, both caches are prefilled with the biases in place and the
// vectors are cleared between the two Forward calls, so prefill cannot be what the two
// runs differ in.
func TestForwardReadsAttentionBias(t *testing.T) {
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

	ids := fixtureTokens(cfg, 5)
	next, pos := int(ids[4]), 4
	cacheWith := func() *KVCache {
		cache, err := NewKVCache(cfg, cfg.ContextLen)
		if err != nil {
			t.Fatalf("NewKVCache: %v", err)
		}
		if _, err := m.Prefill(cache, ids[:4], 0); err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		return cache
	}
	cacheWithBias, cacheWithoutBias := cacheWith(), cacheWith()

	withBias, err := m.Forward(cacheWithBias, next, pos)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	withBias = append([]float32(nil), withBias...)

	for i := range m.W.Layers {
		clear(m.W.Layers[i].qBias)
		clear(m.W.Layers[i].kBias)
		clear(m.W.Layers[i].vBias)
	}
	withoutBias, err := m.Forward(cacheWithoutBias, next, pos)
	if err != nil {
		t.Fatalf("Forward without bias: %v", err)
	}

	if len(withBias) != len(withoutBias) {
		t.Fatalf("logit counts differ: %d and %d", len(withBias), len(withoutBias))
	}
	for i := range withBias {
		if withBias[i] != withoutBias[i] {
			return // found the difference, which is what this asserts
		}
	}
	t.Error("zeroing the attention biases did not change Forward's logits, so the " +
		"single-token path is not reading them")
}
