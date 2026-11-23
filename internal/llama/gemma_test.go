package llama

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguftest"
	"github.com/pplmx/orbix/internal/kernel"
)

// TestGemmaConfigRecordsItsDifferences pins what the architecture table says about Gemma,
// including the value a checkpoint cannot state: the embedding scale is derived from the hidden
// size.
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

	if cfg.FFNAct != ffnGELU {
		t.Errorf("FFNAct = %v, want GELU", cfg.FFNAct)
	}
	// sqrt(hidden), computed here rather than copied from the table.
	if want := float32(math.Sqrt(32)); math.Abs(float64(cfg.EmbedScale-want)) > 1e-6 {
		t.Errorf("EmbedScale = %v, want sqrt(32) = %v", cfg.EmbedScale, want)
	}
	// The adjacent pairing is what Llama's converted files want, and applying it here is the
	// mistake that looks correct: measured against llama.cpp on Gemma 2B, it put the top-8 log
	// probabilities 1.96 away where this pairing is 0.19.
	if cfg.RopeLayout != RopeNeox {
		t.Errorf("RopeLayout = %v, want the half-split pairing Gemma was trained with",
			cfg.RopeLayout)
	}
	if cfg.AttnBias {
		t.Error("AttnBias is true; Gemma has no attention bias")
	}
}

// TestGemmaSummaryNamesItsDifferences checks that the differences a reader cannot see are
// printed.
//
// The summary is what an operator has when a Gemma file generates something odd, and "gemma: 18
// layers, 2048 hidden" alone does not distinguish a correct load from one that used Llama's
// rotary pairing.
func TestGemmaSummaryNamesItsDifferences(t *testing.T) {
	path := gguftest.Model{Arch: "gemma", Layers: 2, Hidden: 32, Heads: 4, FFN: 64}.Write(t, t.TempDir())
	m, err := NewWithOptions(path, Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	got := m.Config().String()
	for _, want := range []string{"embeddings scaled by", "gelu gate"} {
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
	for _, unwanted := range []string{"embeddings scaled", "gelu"} {
		if strings.Contains(plain, unwanted) {
			t.Errorf("llama summary %q mentions %q", plain, unwanted)
		}
	}
}

// TestGemmaNormWeightsAreUsedAsStored records the convention the whole of the above depends on.
//
// Gemma's reference implementation multiplies by (1 + weight), which invites a loader to add the
// 1 at load time. This checkpoint does not need that: measured against llama.cpp, using the
// stored values directly reproduces its log probabilities to within 0.19, while adding 1 puts
// them 26 away. The conversion was written, measured, and removed -- this test is what is left
// of it, so that the next reader does not re-add it on the strength of the reference code.
func TestGemmaNormWeightsAreUsedAsStored(t *testing.T) {
	path := gguftest.Model{Arch: "gemma", Layers: 2, Hidden: 32, Heads: 4, FFN: 64}.Write(t, t.TempDir())
	m, err := NewWithOptions(path, Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	// The fixture writes 1 into every norm tensor, so a loader that left them alone reports 1
	// and one that added the 1 would report 2.
	check := func(what string, w []float32) {
		t.Helper()
		if len(w) == 0 {
			t.Fatalf("%s is empty", what)
		}
		for i, v := range w {
			if v != 1 {
				t.Fatalf("%s[%d] = %v, want 1 as stored", what, i, v)
			}
		}
	}
	check("output_norm", m.W.OutputNorm)
	for l := range m.W.Config.Layers {
		check("attn_norm", m.W.Layers[l].attnNorm)
		check("ffn_norm", m.W.Layers[l].ffnNorm)
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

// TestGemmaMatchesReferenceLogprobs is the decisive check: the architecture is right if the
// real checkpoint's distribution over the vocabulary matches llama.cpp's for the same prefix.
//
// Set GEMMA_MODEL to a Gemma 2B GGUF to run it. The expected values come from llama.cpp 0.3.37
// (llama-cpp-python, n_gpu_layers 0, default cache types) on gemma-2b.Q4_K_M, prompt "The
// capital of France is", top-8 at the first generated position. The token ids are llama.cpp's
// own, and this tokenizer encodes the same prompt to the same six ids.
//
// Measured deviations: 0.100, 0.176, 0.150, 0.047, 0.015, 0.185, 0.055, 0.185 nats, so the
// tolerance is about twice the largest. They are not systematic -- some are positive and some
// negative, and the same 8 tokens come back in nearly the same order -- which is what
// accumulated fp32-versus-fp16 rounding looks like rather than a wrong rotation.
//
// The comparison earns its place because the three mistakes it can see are all invisible in the
// shapes: the adjacent rotary pairing puts these values 1.96 away, adding one to the stored norm
// weights puts them 26 away, and using SiLU instead of GELU puts them 12 away. Every one of
// those still produces fluent English, which is exactly why a shape-only test would not do.
func TestGemmaMatchesReferenceLogprobs(t *testing.T) {
	path := os.Getenv("GEMMA_MODEL")
	if path == "" {
		t.Skip("set GEMMA_MODEL to a Gemma 2B GGUF file")
	}
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	// The expected values are Gemma 2B's. Another checkpoint gives different log probabilities
	// for reasons that have nothing to do with the architecture, so pointing this at the wrong
	// file reports a broken implementation instead of a wrong input.
	if cfg.VocabSize != 256000 || cfg.Layers != 18 {
		t.Skipf("the expected values are Gemma 2B's (vocab 256000, 18 layers); this file has vocab %d and %d layers",
			cfg.VocabSize, cfg.Layers)
	}

	cache, err := NewKVCache(cfg, 64)
	if err != nil {
		t.Fatal(err)
	}
	var logits []float32
	for pos, id := range []int{2, 651, 6037, 576, 6081, 603} {
		logits, err = m.Forward(cache, id, pos)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
	}

	// llama.cpp's top 8, keyed by token id.
	want := []struct {
		id   int32
		text string
		lp   float64
	}{
		{7127, " Paris", -1.3771851},
		{476, " a", -2.1534102},
		{573, " the", -3.0621370},
		{235248, " ", -3.0660899},
		{780, " not", -3.3337958},
		{1170, " also", -3.4196637},
		{7023, " located", -3.4358027},
		{3836, " known", -3.4994786},
	}
	const tolerance = 0.35
	for _, w := range want {
		got := logSoftmax(logits, w.id)
		if d := math.Abs(got - w.lp); d > tolerance {
			t.Errorf("logprob of %q (id %d) = %.4f, llama.cpp says %.4f, off by %.4f",
				w.text, w.id, got, w.lp, d)
		}
	}
	// The same token has to win, or the numbers above could agree while the model preferred
	// something else entirely.
	argmax := int32(0)
	for i, v := range logits {
		if v > logits[argmax] {
			argmax = int32(i)
		}
	}
	if argmax != 7127 {
		t.Errorf("greedy token is %d, want 7127 (\" Paris\")", argmax)
	}
}

// logSoftmax returns one token's log probability under the distribution logits describes.
func logSoftmax(logits []float32, id int32) float64 {
	maxv := logits[0]
	for _, v := range logits {
		if v > maxv {
			maxv = v
		}
	}
	var sum float64
	for _, v := range logits {
		sum += math.Exp(float64(v - maxv))
	}
	return float64(logits[id]-maxv) - math.Log(sum)
}
