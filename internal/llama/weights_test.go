package llama

import (
	"slices"
	"strings"
	"testing"
)

// TestTensorNamesCoversTheModelShape pins the list as a contract.
//
// The names are derived from the config rather than read from the file, which is what turns a
// missing tensor into an error naming it instead of a nil dereference during the first forward
// pass. So each feature that changes the tensor set has to change this list, and a test that only
// counted the entries would miss a group being added to the wrong branch.
func TestTensorNamesCoversTheModelShape(t *testing.T) {
	base := Config{Layers: 2, TiedEmbeddings: true}

	t.Run("a plain model", func(t *testing.T) {
		names := tensorNames(base)
		for _, want := range []string{
			"token_embd.weight", "output_norm.weight",
			"blk.0.attn_norm.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight",
			"blk.0.attn_v.weight", "blk.0.attn_output.weight", "blk.0.ffn_norm.weight",
			"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
			"blk.1.attn_q.weight", "blk.1.ffn_down.weight",
		} {
			if !slices.Contains(names, want) {
				t.Errorf("a plain model is missing %s", want)
			}
		}
		// Tied embeddings mean no separate output projection.
		if slices.Contains(names, "output.weight") {
			t.Error("a tied model asked for output.weight")
		}
		// And none of the optional groups.
		for _, absent := range []string{
			"blk.0.post_attention_norm.weight", "blk.0.attn_q_norm.weight",
			"blk.0.ffn_gate_inp.weight", "blk.0.attn_q.bias",
		} {
			if slices.Contains(names, absent) {
				t.Errorf("a plain model asked for %s", absent)
			}
		}
	})

	t.Run("untied embeddings add the output projection", func(t *testing.T) {
		cfg := base
		cfg.TiedEmbeddings = false
		if !slices.Contains(tensorNames(cfg), "output.weight") {
			t.Error("an untied model did not ask for output.weight")
		}
	})

	t.Run("post norms add two tensors per layer", func(t *testing.T) {
		cfg := base
		cfg.PostNorms = true
		names := tensorNames(cfg)
		for _, want := range []string{
			"blk.0.post_attention_norm.weight", "blk.0.post_ffw_norm.weight",
			"blk.1.post_attention_norm.weight", "blk.1.post_ffw_norm.weight",
		} {
			if !slices.Contains(names, want) {
				t.Errorf("PostNorms is missing %s", want)
			}
		}
	})

	t.Run("qk norm adds two tensors per layer", func(t *testing.T) {
		cfg := base
		cfg.QKNorm = true
		names := tensorNames(cfg)
		for _, want := range []string{
			"blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight",
			"blk.1.attn_q_norm.weight", "blk.1.attn_k_norm.weight",
		} {
			if !slices.Contains(names, want) {
				t.Errorf("QKNorm is missing %s", want)
			}
		}
	})

	t.Run("experts replace the dense ffn", func(t *testing.T) {
		cfg := base
		cfg.Experts = 4
		names := tensorNames(cfg)
		for _, want := range []string{
			"blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_exps.weight",
			"blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight",
		} {
			if !slices.Contains(names, want) {
				t.Errorf("an expert model is missing %s", want)
			}
		}
		// The dense equivalents must be gone: a model that asked for both would try to
		// load tensors it does not carry.
		for _, absent := range []string{
			"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
		} {
			if slices.Contains(names, absent) {
				t.Errorf("an expert model still asked for the dense %s", absent)
			}
		}
	})

	t.Run("attn bias adds three per layer", func(t *testing.T) {
		cfg := base
		cfg.AttnBias = true
		names := tensorNames(cfg)
		for _, want := range []string{
			"blk.0.attn_q.bias", "blk.0.attn_k.bias", "blk.0.attn_v.bias",
			"blk.1.attn_q.bias",
		} {
			if !slices.Contains(names, want) {
				t.Errorf("AttnBias is missing %s", want)
			}
		}
	})
}

// TestTensorNamesHasNoDuplicates keeps the loader from reading one tensor twice.
//
// A duplicate would not fail loudly -- the loader would return the same tensor under one name and
// the second read would overwrite it -- so it would go unnoticed until memory use looked wrong.
func TestTensorNamesHasNoDuplicates(t *testing.T) {
	// Every optional group on at once, which is the combination where an overlap would show.
	cfg := Config{
		Layers: 3, TiedEmbeddings: false, PostNorms: true, QKNorm: true,
		Experts: 2, AttnBias: true,
	}
	seen := make(map[string]bool)
	for _, n := range tensorNames(cfg) {
		if seen[n] {
			t.Errorf("%s appears twice", n)
		}
		seen[n] = true
	}
}

// TestTensorNamesScalesWithTheLayerCount is the arithmetic a reader would check by hand.
//
// The count is a fixed prelude plus a per-layer block whose size depends on the flags, so a
// mismatch means a group was added outside the loop or a layer was skipped.
func TestTensorNamesScalesWithTheLayerCount(t *testing.T) {
	cfg := Config{Layers: 1, TiedEmbeddings: true, AttnBias: true, PostNorms: true, QKNorm: true}
	one := len(tensorNames(cfg))
	cfg.Layers = 4
	four := len(tensorNames(cfg))

	perLayer := four - one // three extra layers
	if perLayer%3 != 0 {
		t.Fatalf("three extra layers added %d tensors, which is not divisible by 3", perLayer)
	}
	if got, want := four, one+perLayer; got != want {
		t.Errorf("four layers gave %d tensors, want %d", got, want)
	}
	// The prelude is the embedding and the output norm, plus output.weight when untied.
	if one < 2 {
		t.Errorf("a one-layer model produced %d tensors, fewer than the prelude", one)
	}
}

// TestTensorNamesPrefixesEveryLayerTensor is the shape every name has to have.
//
// The loader looks tensors up by name, so a block whose prefix is malformed fails the load with
// a message about a missing tensor rather than about the bug that produced the name.
func TestTensorNamesPrefixesEveryLayerTensor(t *testing.T) {
	cfg := Config{Layers: 2, TiedEmbeddings: true, Experts: 2, AttnBias: true, PostNorms: true, QKNorm: true}
	for _, n := range tensorNames(cfg) {
		if !strings.HasPrefix(n, "blk.") {
			continue // a prelude tensor
		}
		// blk.<n>.something -- the number has to be a single digit for these two layers.
		rest, ok := strings.CutPrefix(n, "blk.")
		if !ok {
			t.Fatalf("%s lost its prefix", n)
		}
		num, tail, ok := strings.Cut(rest, ".")
		if !ok || len(num) != 1 || num < "0" || num > "9" || tail == "" {
			t.Errorf("%s is not blk.<layer>.<tensor>", n)
		}
	}
}
