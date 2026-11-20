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

// TestMoEReachesTheExpertsOnARealModel runs the grouped-expert path on an actual checkpoint
// rather than a constructed one.
//
// The fixture tests above check the router's arithmetic and that a constructed MoE model
// produces finite logits, but a constructed model's tensors are all the same shape and its
// routing may land on one expert everywhere. This one loads a real mixture-of-experts file, so
// the shapes, the stacked expert tensors, and the per-expert split all come from something
// that was written by a converter rather than by hand.
//
// Gated on MIXTRAL_MODEL because the file is not in the repository. It is the only check
// here that the MoE path runs end to end on weights nobody on this project produced.
func TestMoEReachesTheExpertsOnARealModel(t *testing.T) {
	path := os.Getenv("MIXTRAL_MODEL")
	if path == "" {
		t.Skip("set MIXTRAL_MODEL to a mixture-of-experts GGUF file")
	}
	m, err := New(path)
	if err != nil {
		t.Fatalf("New(%s): %v", path, err)
	}
	cfg := m.Config()
	if cfg.Experts <= 0 {
		t.Fatalf("%s loaded with %d experts; it is not a mixture-of-experts file", path, cfg.Experts)
	}
	if m.W.Layers[0].router == nil {
		t.Fatal("an expert model loaded without a router")
	}
	for l := range m.W.Layers {
		layer := &m.W.Layers[l]
		if got, want := len(layer.expertGate), cfg.Experts; got != want {
			t.Fatalf("layer %d has %d expert gate tensors, want %d", l, got, want)
		}
	}

	run := func() []float32 {
		cache, err := NewKVCache(cfg, min(cfg.ContextLen, 256))
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
			return // the experts are consulted, which is what this asserts
		}
	}
	t.Error("zeroing every expert left the logits unchanged, so this file's experts are " +
		"not being run")
}

// TestMoEDecodeMatchesPrefill checks the second adapter over the expert path.
//
// The expert feed-forward is reached through two shims: moeRowsFromBatch for a prefill, where a
// row is a prompt position, and moeRowsFromDecode for a decode step, where a row is a sequence.
// Both hand the same moeRows struct to feedForwardMoERows, so the two differ only in which batch
// they read -- and that is exactly the kind of difference that survives a finite-logit check while
// being wrong.
//
// This compares the logits a decode step produces against a full prefill of the same tokens. A
// shim that copied the wrong field would route the wrong row to the experts, and routing is
// data-dependent, so the difference shows up in the numbers rather than as an error.
//
// Nothing covered the decode shim before: the four MoE tests all go through Prefill, so
// moeRowsFromDecode sat at 0% while being on the path of every generated token.
func TestMoEDecodeMatchesPrefill(t *testing.T) {
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
	tokens := fixtureTokens(cfg, 4)

	// Reference: prefill the whole prompt in one pass.
	whole, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}
	wantLogits, err := m.Prefill(whole, tokens, 0)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	want := append([]float32(nil), wantLogits...)

	// Same cache, built by prefilling all but the last token and then decoding it through
	// DecodeStep. That is the call that reads moeRowsFromDecode; Forward has its own single-sequence
	// expert path and would not exercise the shim this test is about.
	split, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}
	if _, err := m.Prefill(split, tokens[:len(tokens)-1], 0); err != nil {
		t.Fatalf("Prefill prefix: %v", err)
	}
	flat, err := m.DecodeStep([]Cache{split}, []int32{tokens[len(tokens)-1]})
	if err != nil {
		t.Fatalf("DecodeStep: %v", err)
	}
	if len(flat) != len(want) {
		t.Fatalf("DecodeStep returned %d values for one request, want %d", len(flat), len(want))
	}
	gotLogits := flat
	if len(gotLogits) != len(want) {
		t.Fatalf("%d logits from decode, %d from prefill", len(gotLogits), len(want))
	}

	var maxDiff float64
	for i := range want {
		if d := math.Abs(float64(want[i] - gotLogits[i])); d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("max logit difference between decode and prefill: %.6f", maxDiff)
	// The two paths perform the same arithmetic over the same inputs, so they should agree to
	// within float32 accumulation order rather than merely approximately.
	if maxDiff > 1e-3 {
		t.Errorf("decode and prefill logits differ by %.6f", maxDiff)
	}
}

// TestParamsCountsExpertWeights covers the weight count for a mixture-of-experts model.
//
// A MoE layer has no wGate/wUp/wDown: it has one router and a stack of per-expert matrices, so a
// count that sums only the dense fields reports a few percent of the true figure for a model whose
// entire size is in the experts. `orbix run --stats` and `orbix bench` print this number, which makes
// it a claim about the model rather than an internal counter.
//
// Measured on the four-expert fixture before the fix: 31,584 reported against 49,152 expert weights
// held, so the dense fields were being counted and the experts were not.
func TestParamsCountsExpertWeights(t *testing.T) {
	path := gguftest.Model{Experts: 4, ExpertUsed: 2, NoEOS: true, ContextLen: 128}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got := m.W.Params()

	// Sum the weights the model actually holds, independently of how Params walks them.
	var want int64
	want += int64(len(m.W.TokenEmbedding) + len(m.W.OutputNorm) + len(m.W.Output))
	for i := range m.W.Layers {
		l := m.W.Layers[i]
		want += int64(len(l.attnNorm) + len(l.wq) + len(l.wk) + len(l.wv) + len(l.wo))
		want += int64(len(l.qBias) + len(l.kBias) + len(l.vBias))
		want += int64(len(l.ffnNorm))
		want += int64(len(l.router))
		for _, e := range l.expertGate {
			want += int64(len(e))
		}
		for _, e := range l.expertUp {
			want += int64(len(e))
		}
		for _, e := range l.expertDown {
			want += int64(len(e))
		}
	}
	if got != want {
		t.Errorf("Params() = %d, want %d", got, want)
	}

	// The experts have to be a large share of it, which is what makes a count that omits them
	// wrong rather than merely imprecise.
	var expertWeights int64
	for i := range m.W.Layers {
		for _, e := range m.W.Layers[i].expertGate {
			expertWeights += int64(len(e))
		}
		for _, e := range m.W.Layers[i].expertUp {
			expertWeights += int64(len(e))
		}
		for _, e := range m.W.Layers[i].expertDown {
			expertWeights += int64(len(e))
		}
	}
	if expertWeights == 0 {
		t.Fatal("fixture holds no expert weights, so this test checks nothing")
	}
	if share := float64(expertWeights) / float64(got); share < 0.3 {
		t.Errorf("experts are %.1f%% of the weights, so this fixture does not exercise the omission",
			share*100)
	}
}

// TestSplitExpertsDegenerateInputs covers the guard in front of the division.
//
// `per := len(stacked) / experts` is undefined for experts == 0, and the guard is what turns
// that into a nil return rather than a panic. No caller can reach it today -- LoadWeights only
// splits when Config.Experts > 0 -- so widening the guard from `experts <= 0` to `experts < 0`
// passed the llama and engine suites. It is pinned anyway because the function is exported to the
// rest of the package and a panic from a division is a worse failure than a nil result the caller
// already has to handle for the mismatched-size case.
func TestSplitExpertsDegenerateInputs(t *testing.T) {
	cases := []struct {
		name     string
		stacked  []float32
		experts  int
		wantNone bool
	}{
		{"no experts", []float32{1, 2, 3, 4}, 0, true},
		{"negative experts", []float32{1, 2, 3, 4}, -1, true},
		{"no weights", nil, 2, true},
		{"empty weights", []float32{}, 2, true},
		{"one element one expert", []float32{7}, 1, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			// A panic here is the failure mode the guard exists to prevent, so it is caught
			// rather than left to fail the whole run.
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("splitExperts(%v, %d) panicked: %v", c.stacked, c.experts, r)
				}
			}()
			got := splitExperts(c.stacked, c.experts)
			if c.wantNone {
				if got != nil {
					t.Errorf("splitExperts(%v, %d) = %v, want nil", c.stacked, c.experts, got)
				}
				return
			}
			if len(got) != c.experts {
				t.Fatalf("split into %d parts, want %d", len(got), c.experts)
			}
			if len(got[0]) != len(c.stacked) {
				t.Errorf("one expert has %d values, want %d", len(got[0]), len(c.stacked))
			}
		})
	}
}

// TestParamsMatchesTheTensorElements checks Params against something other than a second copy
// of itself.
//
// TestParamsCountsExpertWeights recomputes the expected figure by walking the same fields in the
// same order, so the two agree on what to leave out. Dropping the k/v bias terms from Params
// passed that test, because the independent sum drops them too.
//
// The check here is arithmetic on the file: every element of every tensor the checkpoint holds.
// A bias vector, an expert stack, and a tied output projection all count once, which is the
// definition the stats line is making a claim about. The fixture's shapes are known from its
// config, so the expected total is derivable without asking the loader what it loaded.
func TestParamsMatchesTheTensorElements(t *testing.T) {
	// A fixture with attention bias, so the k/v vectors are non-empty and their omission is
	// arithmetically visible rather than hidden behind zero-length slices.
	path := gguftest.Model{
		Arch: "qwen2", AttnBias: true, NoEOS: true, ContextLen: 64,
	}.Write(t, t.TempDir())
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()

	// Recompute from the config the file declares, not from the loaded slices: a tensor that
	// was silently dropped during loading would be missing from both the model and the sum
	// above, but only this version notices.
	h := int64(cfg.Hidden)
	ffn := int64(cfg.FFN)
	vocab := int64(cfg.VocabSize)
	qdim := int64(cfg.Heads * cfg.HeadDim)
	kvdim := int64(cfg.KVHeads * cfg.HeadDim)
	var want int64
	want += vocab * h // token_embd
	want += h         // output_norm
	want += h * vocab // output (not tied in the fixture)
	for range cfg.Layers {
		want += h         // attn_norm
		want += h * qdim  // attn_q
		want += h * kvdim // attn_k
		want += h * kvdim // attn_v
		want += qdim * h  // attn_output
		want += h         // ffn_norm
		want += h * ffn   // ffn_gate
		want += h * ffn   // ffn_up
		want += ffn * h   // ffn_down
		if cfg.AttnBias {
			want += qdim  // attn_q.bias
			want += kvdim // attn_k.bias
			want += kvdim // attn_v.bias
		}
	}
	if got := m.W.Params(); got != want {
		t.Errorf("Params = %d, want %d (the element count the declared shapes add up to)", got, want)
	}

	// The bias terms must be a real part of the difference, or the test would not notice them
	// going missing again.
	if !cfg.AttnBias {
		t.Fatal("the fixture has no attention bias, so this test cannot see one omitted")
	}
	if want-2*kvdim-qdim <= 0 {
		t.Fatal("the bias terms are not a positive part of the total")
	}
}

// TestSelectExpertsLeadingNaNAreNotSkipped pins the same comparison-rule cost Greedy has.
//
// The search seeds best = -1 and takes the first candidate unconditionally, so expert 0
// wins even when its logit is NaN: every later comparison is `v > logits[0]`, which is
// false against a NaN. Measured on {NaN, 3, 2} with k=2, the selection is [0 1] and both
// weights are NaN. A NaN anywhere else is skipped normally ({1, NaN, 2} selects [2 0]),
// because there the NaN is the one being compared and the comparison is false.
//
// This is the rule Greedy documents and TestGreedyNeverSelectsNaN pins -- the same shape,
// reached by a different search. It is not a second bug, and it is unreachable in a served
// model: the load-time finite check refuses a NaN weight, which is what makes the router
// logits finite. The behaviour is pinned here so a future change to the seed value has to
// think about both searches rather than only the sampler.
func TestSelectExpertsLeadingNaNAreNotSkipped(t *testing.T) {
	nan := float32(math.NaN())

	t.Run("leading NaN wins its slot", func(t *testing.T) {
		sel, w := selectExperts([]float32{nan, 3, 2}, 2)
		if len(sel) != 2 || sel[0] != 0 {
			t.Fatalf("selectExperts({NaN,3,2}, 2) = %v, want 0 selected first", sel)
		}
		for i, v := range w {
			if !math.IsNaN(float64(v)) {
				t.Errorf("weight %d = %v, want NaN: the selected set holds a NaN logit", i, v)
			}
		}
	})

	t.Run("a NaN elsewhere is skipped", func(t *testing.T) {
		sel, w := selectExperts([]float32{1, nan, 2}, 2)
		if len(sel) != 2 || sel[0] != 2 || sel[1] != 0 {
			t.Fatalf("selectExperts({1,NaN,2}, 2) = %v, want [2 0]", sel)
		}
		for i, v := range w {
			if math.IsNaN(float64(v)) {
				t.Errorf("weight %d is NaN, but no selected logit was NaN", i)
			}
		}
	})
}
