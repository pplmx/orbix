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

// TestPrefillAppliesAttentionBiasToEveryRow covers the row count addBiasBatch is given.
//
// The two existing bias tests compare a biased run against a zeroed one, which is satisfied as
// long as *some* row reads the bias. Passing 1 instead of n applies it to the first row only and
// every existing test stayed green -- measured, with both the full suite and the bias tests alone.
//
// The rows are distinguished by comparing a multi-position prefill against the same positions run
// one at a time. Prefill writes each row from the same weights and the same bias, so the last
// position's logits have to match whether the earlier tokens arrived in one batched pass or in
// several; if the batch applies the bias to row 0 only, the second and later positions feed
// un-biased keys and values into the attention of the position that is compared.
func TestPrefillAppliesAttentionBiasToEveryRow(t *testing.T) {
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
	tokens := fixtureTokens(cfg, 6)

	// One pass over all six positions.
	batched, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatal(err)
	}
	want, err := m.Prefill(batched, tokens, 0)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	wantLast := append([]float32(nil), want...)

	// The same six positions, accumulated one at a time. Each single-token pass has n == 1, so a
	// row-count bug in the batched path cannot hide here.
	stepwise, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatal(err)
	}
	for i := range tokens {
		got, err := m.Prefill(stepwise, tokens[i:i+1], i)
		if err != nil {
			t.Fatalf("Prefill at %d: %v", i, err)
		}
		if i == len(tokens)-1 {
			// Exact equality, not a tolerance: both paths run the same arithmetic on the same
			// inputs, so the only way they differ is a row that did not get its bias.
			if len(got) != len(wantLast) {
				t.Fatalf("logit counts differ: %d and %d", len(got), len(wantLast))
			}
			for j := range wantLast {
				if got[j] != wantLast[j] {
					t.Fatalf("logit %d differs between one batched pass (%v) and six single "+
						"passes (%v); the batched prefill applied the attention bias to fewer "+
						"rows than it processed", j, wantLast[j], got[j])
				}
			}
		}
	}
}

// TestDecodeStepAppliesAttentionBiasToEverySequence covers the row count on the decode path.
//
// TestDecodeStepReadsAttentionBias compares a biased decode against one whose vectors were
// zeroed with two sequences in flight, which reads as a per-row assertion but is not: applying
// the bias to row 0 only still makes the two runs differ. Narrowing n to 1 in addBiasDecode left
// the whole suite green -- measured -- for all three of q, k and v.
//
// What distinguishes the rows is decoding one sequence at a time and comparing each against its
// own position in a batch. A single-sequence step has a row count of one by construction, so the
// batched call is the only place the count can be wrong, and the two have to agree exactly.
func TestDecodeStepAppliesAttentionBiasToEverySequence(t *testing.T) {
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

	ids := fixtureTokens(cfg, 8)
	const prefix = 4
	tokens := []int32{ids[4], ids[5], ids[6]}

	// Three sequences, each prefilled to the same prefix, then decoded together.
	batched := make([]Cache, len(tokens))
	for i := range batched {
		cache, err := NewKVCache(cfg, cfg.ContextLen)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := m.Prefill(cache, ids[:prefix], 0); err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		batched[i] = cache
	}
	together, err := m.DecodeStep(batched, tokens)
	if err != nil {
		t.Fatalf("DecodeStep: %v", err)
	}
	// The model's logits buffer is sized to the largest batch it has ever decoded, so only the
	// first n rows belong to this call. Rows are VocabSize wide.
	rowLen := cfg.VocabSize
	if len(together) < rowLen*len(tokens) {
		t.Fatalf("DecodeStep returned %d logits for %d sequences of %d",
			len(together), len(tokens), rowLen)
	}
	togetherCopy := append([]float32(nil), together[:rowLen*len(tokens)]...)

	// The same three, decoded one at a time from equal caches.
	for i := range tokens {
		cache, err := NewKVCache(cfg, cfg.ContextLen)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := m.Prefill(cache, ids[:prefix], 0); err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		alone, err := m.DecodeStep([]Cache{cache}, tokens[i:i+1])
		if err != nil {
			t.Fatalf("DecodeStep alone: %v", err)
		}
		if len(alone) < rowLen {
			t.Fatalf("a single-sequence step returned %d logits, a row is %d", len(alone), rowLen)
		}
		// The buffer is the model's and the next call overwrites it, so the comparison has to
		// finish before that happens. togetherCopy was taken before the loop for the same reason.
		alone = append([]float32(nil), alone[:rowLen]...)
		for j := range alone {
			if got := togetherCopy[i*rowLen+j]; got != alone[j] {
				t.Fatalf("sequence %d logit %d: batched %v, alone %v; the batched decode "+
					"applied the attention bias to fewer rows than sequences", i, j, got, alone[j])
			}
		}
	}
}
