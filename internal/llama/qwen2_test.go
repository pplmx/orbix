package llama

import (
	"os"
	"testing"
)

// These tests need a Qwen2 GGUF checkpoint. They exist because Qwen2 differs from
// Llama in three places that all fail quietly if wrong -- a rotation layout, an
// attention bias, and a BPE vocabulary -- so each has a test that would notice the
// difference rather than accepting fluent text.
//
// Set QWEN2_MODEL to a Qwen2 GGUF file to run them.

func qwenPath(t *testing.T) string {
	t.Helper()
	p := os.Getenv("QWEN2_MODEL")
	if p == "" {
		t.Skip("set QWEN2_MODEL to a Qwen2 GGUF file")
	}
	return p
}

func TestQwen2Config(t *testing.T) {
	m, err := New(qwenPath(t))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	if cfg.Arch != "qwen2" {
		t.Errorf("Arch = %q, want qwen2", cfg.Arch)
	}
	if cfg.RopeLayout != RopeNeox {
		t.Errorf("RopeLayout = %v, want neox", cfg.RopeLayout)
	}
	if !cfg.AttnBias {
		t.Error("AttnBias = false, want true for qwen2")
	}
	// The head dimension and grouped-query ratio are what the layout arithmetic
	// depends on, so they are worth pinning to real numbers for this checkpoint.
	//
	// The numbers are Qwen2.5-0.5B's, so a different Qwen2 file fails here despite
	// satisfying every architectural assertion above. The message says which file the
	// shape belongs to; without that, "unexpected shape" reads as a loader bug rather
	// than as the wrong checkpoint.
	if cfg.Hidden != 896 || cfg.Layers != 24 || cfg.Heads != 14 || cfg.KVHeads != 2 {
		t.Errorf("unexpected shape: %d hidden, %d layers, %d heads, %d kv; "+
			"this test pins Qwen2.5-0.5B (896/24/14/2)",
			cfg.Hidden, cfg.Layers, cfg.Heads, cfg.KVHeads)
	}
	if got := m.W.Layers[0].qBias; len(got) != cfg.Heads*cfg.HeadDim {
		t.Errorf("q bias is %d wide, want %d", len(got), cfg.Heads*cfg.HeadDim)
	}
	if got := m.W.Layers[0].kBias; len(got) != cfg.kvDim() {
		t.Errorf("k bias is %d wide, want %d", len(got), cfg.kvDim())
	}
}

// TestQwen2NeoxRopeIsNotNormal is the regression for the rotation layout. The two
// layouts agree on the first dimension pair and diverge after it, so a model run
// with the wrong one still produces fluent text; the only reliable check is that
// the two results differ and that the NEOX one is the one the model was trained
// with.
func TestQwen2NeoxRopeIsNotNormal(t *testing.T) {
	base := testConfig()
	cfgNeox := base
	cfgNeox.RopeLayout = RopeNeox
	cfgNormal := base
	cfgNormal.RopeLayout = RopeNormal

	neox, err := newRope(cfgNeox, 8)
	if err != nil {
		t.Fatal(err)
	}
	normal, err := newRope(cfgNormal, 8)
	if err != nil {
		t.Fatal(err)
	}
	if neox.layout == normal.layout {
		t.Fatal("both ropes were built with the same layout")
	}
	cfg := base

	x := make([]float32, cfg.HeadDim)
	for i := range x {
		x[i] = float32(i+1) / float32(cfg.HeadDim)
	}
	a := append([]float32(nil), x...)
	b := append([]float32(nil), x...)
	if err := neox.apply(a, 3); err != nil {
		t.Fatal(err)
	}
	if err := normal.apply(b, 3); err != nil {
		t.Fatal(err)
	}
	if equalFloats(a, b) {
		t.Fatal("neox and normal rotations produced identical vectors")
	}

	// Both must preserve the vector's norm: the rotation is orthonormal, and a
	// layout that mixed the wrong pairs would not be.
	const tol = 1e-5
	if got, want := norm2(a), norm2(x); absf(got-want) > tol {
		t.Errorf("neox rotation changed the norm: %v -> %v", want, got)
	}
	if got, want := norm2(b), norm2(x); absf(got-want) > tol {
		t.Errorf("normal rotation changed the norm: %v -> %v", want, got)
	}
}

func norm2(x []float32) float32 {
	var sum float32
	for _, v := range x {
		sum += v * v
	}
	return sqrt32(sum)
}

func sqrt32(v float32) float32 {
	// Newton's method is enough for a norm check and keeps this file free of a
	// math import that a reader would have to notice.
	if v <= 0 {
		return 0
	}
	x := v
	for range 20 {
		x = (x + v/x) / 2
	}
	return x
}

func absf(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

func equalFloats(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestQwen2AttentionBiasIsApplied checks the bias actually changes the projection.
// A model loaded with bias vectors but a forward pass that ignores them produces
// text that reads as slightly-off rather than as broken, so this compares the two.
func TestQwen2AttentionBiasIsApplied(t *testing.T) {
	m, err := New(qwenPath(t))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()

	run := func(m *Model) []float32 {
		cache, err := NewKVCache(cfg, cfg.ContextLen)
		if err != nil {
			t.Fatal(err)
		}
		tokens := []int32{151644, 872, 198, 11847}
		logits, err := m.Prefill(cache, tokens, 0)
		if err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		return append([]float32(nil), logits...)
	}
	withBias := run(m)

	// Zero the biases: the output must change, which proves they are read.
	for l := range m.W.Layers {
		layer := &m.W.Layers[l]
		clear(layer.qBias)
		clear(layer.kBias)
		clear(layer.vBias)
	}
	withoutBias := run(m)

	if equalFloats(withBias, withoutBias) {
		t.Fatal("zeroing the attention biases did not change the logits")
	}
}

// TestQwen2GreedyContinues checks that a real checkpoint produces a continuation
// that is on-topic, which is the end-to-end signal that the rotation, the bias, and
// the decoding are all right. "The capital of France is" is chosen because a
// wrong-but-fluent model rarely answers it with "Paris".
func TestQwen2GreedyContinues(t *testing.T) {
	path := qwenPath(t)
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	cache, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatal(err)
	}
	// Token ids for the Qwen chat template's opening, then the prompt text.
	// The tokenizer is a different package, so the ids are spelled out here from
	// the checkpoint's vocabulary. Decoded, this is
	// "<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n".
	//
	// These were wrong for a while: the ids in the middle spelled "_category skys", so the
	// test asked a different question from the one its comment described and still passed,
	// because it only checked that something was generated.
	prompt := []int32{151644, 872, 198, 785, 6722, 315, 9625, 374, 151645, 198, 151644, 77091, 198}
	out, err := m.Generate(cache, prompt, Greedy{}, GenerateOptions{MaxTokens: 6})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(out) <= len(prompt) {
		t.Fatalf("generated nothing: %v", out)
	}
	// The first generated token is "Paris" (id 59604), which is what makes this an
	// end-to-end check rather than a liveness check. A swapped RoPE layout on this
	// checkpoint produced a degenerate digit run instead, so pinning the answer is what
	// distinguishes the two.
	const paris = int32(59604)
	if got := out[len(prompt)]; got != paris {
		t.Errorf("first generated token = %d, want %d (\"Paris\")", got, paris)
	}
	// Every id must be inside the vocabulary, which a rotation or bias mistake
	// would violate only rarely -- but an out-of-range id would panic the decoder.
	for _, id := range out[len(prompt):] {
		if int(id) >= cfg.VocabSize {
			t.Errorf("generated token %d outside the vocabulary (%d)", id, cfg.VocabSize)
		}
	}
}

// TestQwen2PrefillMatchesForward pins the batched and single-token paths together on
// a model with attention bias, where a missing bias in one path but not the other
// would show up as a divergence rather than as wrong text.
func TestQwen2PrefillMatchesForward(t *testing.T) {
	m, err := New(qwenPath(t))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	tokens := []int32{151644, 872, 198, 11847, 84321}

	seq, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatal(err)
	}
	var want []float32
	for pos, tok := range tokens {
		last, err := m.Forward(seq, int(tok), pos)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
		if pos == len(tokens)-1 {
			want = append([]float32(nil), last...)
		}
	}

	batch, err := NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatal(err)
	}
	got, err := m.Prefill(batch, tokens, 0)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("logit[%d] = %v batched, %v sequential", i, got[i], want[i])
		}
	}
}

// TestQwen2ArchitectureRegistry pins the table itself: adding an architecture
// should be a visible change, and removing one should fail a test rather than a
// user's model load.
func TestQwen2ArchitectureRegistry(t *testing.T) {
	// A whole-string comparison, not a Contains per name: supportedArchs sorts because the
	// names come out of a map, and a message that reordered itself between runs would be
	// the thing the sort exists to prevent. Contains cannot see that -- removing the sort
	// leaves it green.
	if got, want := supportedArchs(), "gemma, gemma2, gemma3, llama, mistral, qwen2, qwen3"; got != want {
		t.Errorf("supportedArchs() = %q, want %q", got, want)
	}
	spec, ok := archSpecs["qwen2"]
	if !ok {
		t.Fatal("qwen2 is not registered")
	}
	if spec.rope != RopeNeox || !spec.attnBias {
		t.Errorf("qwen2 spec = %+v, want the NEOX rotation with attention bias", spec)
	}
	spec, ok = archSpecs["llama"]
	if !ok {
		t.Fatal("llama is not registered")
	}
	if spec.rope != RopeNormal || spec.attnBias {
		t.Errorf("llama spec = %+v, want the normal rotation without bias", spec)
	}
	// Mistral shares Llama's rotation and bias layout; what differs is the sliding
	// window, which comes from the config key rather than the table.
	spec, ok = archSpecs["mistral"]
	if !ok {
		t.Fatal("mistral is not registered")
	}
	if spec.rope != RopeNormal || spec.attnBias {
		t.Errorf("mistral spec = %+v, want the normal rotation without bias", spec)
	}
}
