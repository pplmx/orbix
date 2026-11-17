package llama

import (
	"math"
	"os"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
)

// The routing math is the part of a mixture-of-experts layer that is easy to get subtly
// wrong, and wrong routing produces output of the right shape with the wrong values --
// the failure mode that shows up as slightly-off text rather than an error. These tests
// check the selection and weighing rules directly, before any model is involved.

func TestSelectExpertsPicksTopK(t *testing.T) {
	logits := []float32{0.1, 5.0, -2.0, 3.0, 0.0}
	selected, _ := selectExperts(logits, 2)
	if len(selected) != 2 {
		t.Fatalf("selected %d experts, want 2", len(selected))
	}
	// Highest scores are expert 1 (5.0) and expert 3 (3.0), in that order.
	if selected[0] != 1 || selected[1] != 3 {
		t.Errorf("selected %v, want [1 3] (descending score order)", selected)
	}
}

func TestSelectExpertsWeightsSumToOne(t *testing.T) {
	// The weights have to be a distribution over the selected experts: a caller scales
	// each expert's output by its weight, so weights that do not sum to one change the
	// scale of the whole layer.
	for _, logits := range [][]float32{
		{1.0, 2.0, 3.0},
		{-1.0, -2.0, -3.0},
		{0, 0, 0},
		{100, -100, 0},
	} {
		_, weights := selectExperts(logits, 2)
		var sum float32
		for _, w := range weights {
			sum += w
		}
		if math.Abs(float64(sum)-1) > 1e-5 {
			t.Errorf("logits %v: weights %v sum to %v, want 1", logits, weights, sum)
		}
	}
}

// TestSelectExpertsMatchesFullSoftmax pins the equivalence the implementation relies on:
// softmax over the selected logits equals a softmax over all logits restricted to the
// same experts and renormalised.
func TestSelectExpertsMatchesFullSoftmax(t *testing.T) {
	logits := []float32{2.0, 1.0, 0.5, -1.0}
	selected, weights := selectExperts(logits, 2)

	// Full softmax over all four, then keep and renormalise the selected two.
	peak := logits[0]
	for _, v := range logits {
		peak = max(peak, v)
	}
	full := make([]float64, len(logits))
	var total float64
	for i, v := range logits {
		full[i] = math.Exp(float64(v - peak))
		total += full[i]
	}
	var keptSum float64
	for _, e := range selected {
		keptSum += full[e] / total
	}
	for i, e := range selected {
		want := (full[e] / total) / keptSum
		if math.Abs(float64(weights[i])-want) > 1e-5 {
			t.Errorf("expert %d weight = %v, want %v from the full-softmax route", e, weights[i], want)
		}
	}
}

// TestSelectExpertsBreaksTiesByID keeps selection deterministic: equal router scores
// must not depend on which expert the scan happened to visit first.
func TestSelectExpertsBreaksTiesByID(t *testing.T) {
	logits := []float32{1.0, 1.0, 1.0, 1.0}
	selected, _ := selectExperts(logits, 2)
	if selected[0] != 0 || selected[1] != 1 {
		t.Errorf("tied logits selected %v, want [0 1] (lowest ids)", selected)
	}
}

func TestSelectExpertsEdgeCases(t *testing.T) {
	if s, w := selectExperts(nil, 2); s != nil || w != nil {
		t.Errorf("empty logits selected %v %v", s, w)
	}
	if s, w := selectExperts([]float32{1, 2}, 0); s != nil || w != nil {
		t.Errorf("k=0 selected %v %v", s, w)
	}
	// k larger than the expert count must clamp rather than index out of range.
	s, w := selectExperts([]float32{1, 2}, 10)
	if len(s) != 2 || len(w) != 2 {
		t.Errorf("k>experts selected %v %v, want both", s, w)
	}
}

// TestSplitExperts checks the stacked-tensor split, which is where a wrong stride would
// hand each expert a shifted copy of its neighbour's weights.
func TestSplitExperts(t *testing.T) {
	// Three experts of two elements each.
	stacked := []float32{0, 1, 2, 3, 4, 5}
	got := splitExperts(stacked, 3)
	if len(got) != 3 {
		t.Fatalf("split into %d experts, want 3", len(got))
	}
	for e, want := range [][]float32{{0, 1}, {2, 3}, {4, 5}} {
		if len(got[e]) != len(want) {
			t.Fatalf("expert %d has %d values, want %d", e, len(got[e]), len(want))
		}
		for i := range want {
			if got[e][i] != want[i] {
				t.Errorf("expert %d value %d = %v, want %v", e, i, got[e][i], want[i])
			}
		}
	}
	// The slices must alias the original rather than copy it: the weights are hundreds
	// of megabytes and a copy per expert would double the model's footprint.
	got[1][0] = 99
	if stacked[2] != 99 {
		t.Error("splitExperts copied instead of aliasing the stacked tensor")
	}
}

// TestSplitExpertsRejectsMismatch checks that a stacked tensor whose size does not divide
// evenly is reported rather than silently truncated into wrong-shaped experts.
func TestSplitExpertsRejectsMismatch(t *testing.T) {
	if got := splitExperts([]float32{1, 2, 3}, 2); got != nil {
		t.Errorf("split of a mismatched tensor returned %v, want nil", got)
	}
}

// TestMoEWithOneExpertMatchesDense is the decisive correctness check: a mixture of one
// expert with a weight of 1 is a dense FFN, so the two paths must produce identical
// logits. Any error in the routing, the gather/scatter, or the weighting shows up as a
// difference here, without needing a checkpoint.
//
// The fixtures differ only in how the FFN weights are named and shaped: the MoE file
// stacks the same numbers in a third dimension with a router that always picks expert 0.
func TestMoEWithOneExpertMatchesDense(t *testing.T) {
	dense := defaultTiny()
	densePath := buildTinyWeights(t, dense)

	// Build the MoE fixture by hand so the expert weights are the dense ones, and the
	// router gives expert 0 such a large score that selection always takes it.
	moe := defaultTiny()
	moe.experts = 1
	moe.used = 1

	dir := moe.tensors()
	var data []byte
	offset := 0
	for i, ti := range dir {
		dir[i] = ti
		dir[i].Offset = uint64(offset)
		dir[i].Type = gguf.F32
		n := int64(1)
		for _, d := range ti.Dims {
			n *= int64(d)
		}
		vals := make([]float32, n)
		if ti.Name == "blk.0.ffn_gate_inp.weight" || ti.Name == "blk.1.ffn_gate_inp.weight" {
			// A single expert: any score selects it, and the softmax over one element
			// is 1.
			vals[0] = 1
		} else {
			for j := range vals {
				vals[j] = float32((j%17)-8) / 64
			}
		}
		data = appendFloat32s(data, vals)
		offset += int(n) * 4
	}
	moePath := t.TempDir() + "/moe.gguf"
	if err := os.WriteFile(moePath, buildFile(moe.kvs(), dir, data), 0o644); err != nil {
		t.Fatal(err)
	}

	run := func(path string) []float32 {
		m, err := New(path)
		if err != nil {
			t.Fatalf("New(%s): %v", path, err)
		}
		cache, err := NewKVCache(m.Config(), m.Config().ContextLen)
		if err != nil {
			t.Fatalf("NewKVCache: %v", err)
		}
		// One position is enough: the FFN runs on every position, so a single forward
		// pass exercises it.
		logits, err := m.Forward(cache, 3, 0)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
		return logits
	}

	denseLogits := run(densePath)
	moeLogits := run(moePath)
	if len(denseLogits) != len(moeLogits) {
		t.Fatalf("logit counts differ: %d vs %d", len(denseLogits), len(moeLogits))
	}
	// The tensors are identical values in a different layout, so the arithmetic is the
	// same operations in the same order up to the projection width; a tolerance covers
	// the fact that the MoE path multiplies by a weight that is exactly 1.
	for i := range denseLogits {
		if math.Abs(float64(denseLogits[i]-moeLogits[i])) > 1e-5 {
			t.Fatalf("logit[%d] = %v for MoE, %v for dense; one expert with weight 1 must match",
				i, moeLogits[i], denseLogits[i])
		}
	}
}

// TestMoERoutingChangesOutput checks the other direction: with two experts holding
// different weights and a router that can pick either, the output must depend on which
// one is chosen. Without this, a forward pass that ignored the router entirely would
// still pass the equivalence test above.
func TestMoERoutingChangesOutput(t *testing.T) {
	c := defaultTiny()
	c.experts = 2
	c.used = 1

	build := func(routerFirst float32) string {
		dir := c.tensors()
		var data []byte
		offset := 0
		for i, ti := range dir {
			dir[i] = ti
			dir[i].Offset = uint64(offset)
			dir[i].Type = gguf.F32
			n := int64(1)
			for _, d := range ti.Dims {
				n *= int64(d)
			}
			vals := make([]float32, n)
			switch ti.Name {
			case "blk.0.ffn_gate_inp.weight", "blk.1.ffn_gate_inp.weight":
				// Two experts, one row each: the first score decides the choice.
				vals[0] = routerFirst
				vals[1] = -routerFirst
			default:
				// Expert 0 and expert 1 sit in different thirds of the stacked tensor
				// and get different values, so the chosen one is visible in the output.
				for j := range vals {
					if len(ti.Dims) == 3 {
						per := n / int64(ti.Dims[2])
						if int64(j) >= per {
							vals[j] = float32((j%7)-3) / 32
							continue
						}
					}
					vals[j] = float32((j%17)-8) / 64
				}
			}
			data = appendFloat32s(data, vals)
			offset += int(n) * 4
		}
		path := t.TempDir() + "/moe2.gguf"
		if err := os.WriteFile(path, buildFile(c.kvs(), dir, data), 0o644); err != nil {
			t.Fatal(err)
		}
		return path
	}

	run := func(path string) []float32 {
		m, err := New(path)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		cache, _ := NewKVCache(m.Config(), m.Config().ContextLen)
		logits, err := m.Forward(cache, 3, 0)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
		return logits
	}

	a := run(build(1))
	b := run(build(-1))
	same := true
	for i := range a {
		if a[i] != b[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("reversing the router scores did not change the output; routing is ignored")
	}
}

// TestMoEPrefillProducesFiniteLogits checks a batched prefill through an MoE layer, which
// the per-token Forward path above does not exercise: the batched route gathers rows from
// several positions into one matmul and scatters the results back, so an indexing error
// there shows up as NaN or a silently wrong value rather than a crash.
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
