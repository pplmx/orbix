package llama

import (
	"math"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguftest"
	"github.com/pplmx/orbix/internal/kernel"
)

// TestGemmaConfigRecordsItsDifferences pins what the architecture table says about Gemma, and
// in particular the two derived values a checkpoint cannot state directly.
//
// The flags are not decoration: each one changes a number in the forward pass, and a model
// loaded with the wrong ones generates fluent text with the wrong distribution. Asserting them
// here means a change to the table fails a test rather than a generation quality report.
func TestGemmaConfigRecordsItsDifferences(t *testing.T) {
	path := gguftest.Model{Arch: "gemma", Layers: 2, Hidden: 32, Heads: 4, FFN: 64}.Write(t, t.TempDir())
	m, err := NewWithOptions(path, Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	cfg := m.Config()

	if cfg.Arch != "gemma" {
		t.Errorf("Arch = %q", cfg.Arch)
	}
	if !cfg.NormOnes {
		t.Error("NormOnes is false; Gemma's checkpoint stores its norms one off")
	}
	if cfg.FFNAct != ffnGELU {
		t.Errorf("FFNAct = %v, want GELU", cfg.FFNAct)
	}
	// sqrt(hidden), computed here rather than copied from the table.
	if want := float32(math.Sqrt(32)); math.Abs(float64(cfg.EmbedScale-want)) > 1e-6 {
		t.Errorf("EmbedScale = %v, want sqrt(32) = %v", cfg.EmbedScale, want)
	}
	// The other architectures must be untouched by the new fields.
	if cfg.RopeLayout != RopeNormal || cfg.AttnBias {
		t.Errorf("gemma rope/bias = %v/%v, want the normal rotation without bias",
			cfg.RopeLayout, cfg.AttnBias)
	}
}

// TestGemmaSummaryNamesItsDifferences checks that the three numbers a reader cannot see are
// printed.
//
// The summary is what an operator has when a Gemma file generates something odd, and "gemma: 18
// layers ..." alone does not distinguish a correct load from one that skipped the norm offset.
func TestGemmaSummaryNamesItsDifferences(t *testing.T) {
	path := gguftest.Model{Arch: "gemma", Layers: 2, Hidden: 32, Heads: 4, FFN: 64}.Write(t, t.TempDir())
	m, err := NewWithOptions(path, Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	got := m.Config().String()
	for _, want := range []string{"norms stored as w-1", "embeddings scaled by", "gelu gate"} {
		if !strings.Contains(got, want) {
			t.Errorf("summary %q does not mention %q", got, want)
		}
	}

	// A Llama model must not gain any of them: the additions describe differences, so a model
	// without them should read exactly as it did before.
	llamaPath := gguftest.Model{Layers: 2, Hidden: 32, Heads: 4, FFN: 64}.Write(t, t.TempDir())
	lm, err := NewWithOptions(llamaPath, Options{})
	if err != nil {
		t.Fatalf("load llama: %v", err)
	}
	plain := lm.Config().String()
	for _, unwanted := range []string{"w-1", "embeddings scaled", "gelu"} {
		if strings.Contains(plain, unwanted) {
			t.Errorf("llama summary %q mentions %q", plain, unwanted)
		}
	}
}

// TestGemmaNormWeightsAreBroughtToIdentity pins the load-time conversion.
//
// Gemma saves (w-1) so that random initialisation starts at the identity; the loader adds the 1
// back. The fixture writes the identity the way each architecture stores it, so both loads must
// end with norm weights of exactly 1 -- an assertion about the conversion rather than about the
// number the file held. Without the conversion a Gemma model loads with every norm doubled,
// which still generates text, so nothing else would notice.
func TestGemmaNormWeightsAreBroughtToIdentity(t *testing.T) {
	load := func(arch string) *Model {
		t.Helper()
		path := gguftest.Model{Arch: arch, Layers: 2, Hidden: 32, Heads: 4, FFN: 64}.Write(t, t.TempDir())
		m, err := NewWithOptions(path, Options{})
		if err != nil {
			t.Fatalf("%s: load: %v", arch, err)
		}
		return m
	}

	for _, arch := range []string{"llama", "gemma"} {
		m := load(arch)
		check := func(what string, w []float32) {
			t.Helper()
			if len(w) == 0 {
				t.Fatalf("%s: %s is empty", arch, what)
			}
			for i, v := range w {
				if v != 1 {
					t.Fatalf("%s: %s[%d] = %v, want the identity 1", arch, what, i, v)
				}
			}
		}
		check("output_norm", m.W.OutputNorm)
		for l := range m.W.Config.Layers {
			check("attn_norm", m.W.Layers[l].attnNorm)
			check("ffn_norm", m.W.Layers[l].ffnNorm)
		}
	}
}

// TestGemmaLogitsDifferFromLlama's is the coarse check: an architecture's numerical differences
// reach the forward pass at all.
//
// It cannot isolate the gate, and that is worth stating because the first version of this test
// claimed to. Gemma differs from Llama in three ways -- norm offset, embedding scale, gate --
// and the embedding scale alone makes the logits differ, so an implementation that ignored
// FFNAct entirely still passes this. The gate is pinned directly by
// TestActivateGateFollowsTheArchitecture below; this one catches the larger failure of a table
// that is read and never used.
//
// It is kept because the two tests fail differently: the direct one fails on a single function
// and names the cause, this one fails when a difference is dropped somewhere in the forward
// path that no unit test covers.
func TestGemmaLogitsDifferFromLlama(t *testing.T) {
	logits := func(arch string) []float32 {
		t.Helper()
		path := gguftest.Model{
			Arch: arch, Layers: 2, Hidden: 32, Heads: 4, FFN: 64, NoEOS: true,
		}.Write(t, t.TempDir())
		m, err := NewWithOptions(path, Options{})
		if err != nil {
			t.Fatalf("%s: load: %v", arch, err)
		}
		cache, err := NewKVCache(m.Config(), 16)
		if err != nil {
			t.Fatalf("%s: cache: %v", arch, err)
		}
		out, err := m.Prefill(cache, []int32{int32(m.Config().BOSID), 1, 2}, 0)
		if err != nil {
			t.Fatalf("%s: prefill: %v", arch, err)
		}
		return out
	}

	asLlama := logits("llama")
	asGemma := logits("gemma")
	if len(asLlama) != len(asGemma) {
		t.Fatalf("logit counts differ: %d and %d", len(asLlama), len(asGemma))
	}
	// Three things differ between the two loads of the same tensors -- the norm offset, the
	// embedding scale and the gate -- so the logits must differ. Comparing against the
	// individual pieces is each of the tests above; this one is that they are wired at all.
	same := 0
	for i := range asLlama {
		if asLlama[i] == asGemma[i] {
			same++
		}
	}
	if same == len(asLlama) {
		t.Fatal("gemma and llama produced identical logits from identical tensors; " +
			"the architecture's numerical differences never reach the forward pass")
	}
	for i, v := range asGemma {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logit %d is %v", i, v)
		}
	}
}

// TestEmbedIntoAppliesTheScale pins the multiply that stands in for Gemma's input scaling.
//
// It is a separate test because no end-to-end comparison can see it: dropping the scale still
// leaves the gate and the norm offset differing, so a llama-versus-gemma logit comparison passes
// either way. What the scale affects is every activation in the network, and the only way to
// check the helper is applied rather than skipped is to call the helper.
//
// The two branches matter separately. The scale of 1 path is the copy every other architecture
// takes, so it has to stay a copy; the other path multiplies.
func TestEmbedIntoAppliesTheScale(t *testing.T) {
	row := []float32{1, -2, 0.5, 0}
	original := append([]float32(nil), row...)

	dst := make([]float32, len(row))
	embedInto(dst, row, 1)
	for i := range row {
		if dst[i] != row[i] {
			t.Errorf("scale 1 at %d gave %v, want %v", i, dst[i], row[i])
		}
	}

	embedInto(dst, row, 2)
	for i := range row {
		if want := row[i] * 2; dst[i] != want {
			t.Errorf("scale 2 at %d gave %v, want %v", i, dst[i], want)
		}
	}
	// The source is not modified: the same row is read for every token that maps to it, and
	// the matrix is usually the tied output projection, so writing the scale into it would
	// compound with every use.
	for i := range row {
		if row[i] != original[i] {
			t.Errorf("embedInto modified its source at %d: %v, want %v", i, row[i], original[i])
		}
	}
}

// TestActivateGateFollowsTheArchitecture pins the gate dispatch itself, which the architecture
// comparison above cannot isolate.
//
// Gemma's gate is GELU and the Llama family's is SiLU, and the two agree at 0 and near 0 -- so
// an assertion at a single point would not separate them, and an implementation that always
// called SiLU would still produce finite, plausible logits. The inputs here include both signs
// and a magnitude where the two functions are far apart.
func TestActivateGateFollowsTheArchitecture(t *testing.T) {
	in := []float32{-3, -1, -0.5, 0, 0.5, 1, 3}

	got := func(a ffnAct) []float32 {
		x := append([]float32(nil), in...)
		Config{FFNAct: a}.activateGate(x)
		return x
	}
	silu, gelu := got(ffnSiLU), got(ffnGELU)

	// The reference values, computed by the kernel functions directly rather than by
	// activateGate, so a dispatch that called neither is still caught.
	wantSilu := append([]float32(nil), in...)
	kernel.SiLU(wantSilu)
	wantGelu := append([]float32(nil), in...)
	kernel.GELU(wantGelu)

	for i := range in {
		if silu[i] != wantSilu[i] {
			t.Errorf("ffnSiLU at %v gave %v, want %v", in[i], silu[i], wantSilu[i])
		}
		if gelu[i] != wantGelu[i] {
			t.Errorf("ffnGELU at %v gave %v, want %v", in[i], gelu[i], wantGelu[i])
		}
		if silu[i] == gelu[i] && in[i] != 0 {
			t.Errorf("the two gates agree at %v (%v); the case cannot tell them apart", in[i], silu[i])
		}
	}
	// The zero value of the enum must be SiLU: LLAMA is the architecture every other one is
	// described in terms of, and a Config built without the field keeps its behaviour.
	var zero ffnAct
	if zero != ffnSiLU {
		t.Errorf("the zero ffnAct is %v, want SiLU", zero)
	}
	if zero.String() != "silu" || ffnGELU.String() != "gelu" {
		t.Errorf("gate names are %q and %q", zero.String(), ffnGELU.String())
	}
}

// TestGemmaGenerates checks the whole path end to end, since a shape or scale mistake that
// still produces finite logits would only show up as text that decodes to nothing useful.
func TestGemmaGenerates(t *testing.T) {
	path := gguftest.Model{
		Arch: "gemma", Layers: 2, Hidden: 32, Heads: 4, FFN: 64, NoEOS: true,
	}.Write(t, t.TempDir())
	m, err := NewWithOptions(path, Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	cache, err := NewKVCache(m.Config(), 16)
	if err != nil {
		t.Fatalf("cache: %v", err)
	}
	const prompt, budget = 3, 5
	out, err := m.Generate(cache, []int32{1, 2, 3}, Greedy{}, GenerateOptions{MaxTokens: budget})
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	// Generate returns the whole sequence, prompt included.
	if len(out) != prompt+budget {
		t.Errorf("returned %d tokens, want %d prompt + %d generated", len(out), prompt, budget)
	}
	for i, id := range out {
		if id < 0 || int(id) >= m.Config().VocabSize {
			t.Errorf("token %d is %d, outside the vocabulary of %d", i, id, m.Config().VocabSize)
		}
	}
}
