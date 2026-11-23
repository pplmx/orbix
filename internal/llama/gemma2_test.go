package llama

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguftest"
)

// TestWindowForAlternates pins which layers are bounded, which is the part of Gemma 2's sliding
// window that is architecture rather than metadata.
//
// The window length arrives as a metadata key and is the same number for both kinds of layer;
// what the architecture decides is that only every second one uses it. A loader that applied it
// to every layer would produce a model whose long-range attention silently stops working while
// everything still runs.
func TestWindowForAlternates(t *testing.T) {
	pattern := Config{WindowLen: 4096, SlidingPattern: true}
	want := []int{4096, 0, 4096, 0, 4096, 0}
	for l, w := range want {
		if got := windowFor(pattern, l); got != w {
			t.Errorf("layer %d window = %d, want %d", l, got, w)
		}
	}

	// Without the pattern every layer is bounded, which is Mistral's shape.
	flat := Config{WindowLen: 4096}
	for l := range 4 {
		if got := windowFor(flat, l); got != 4096 {
			t.Errorf("layer %d window = %d, want 4096 on every layer", l, got)
		}
	}

	// No window at all is the Llama shape, and neither field changes that.
	none := Config{}
	for l := range 4 {
		if got := windowFor(none, l); got != 0 {
			t.Errorf("layer %d window = %d, want 0", l, got)
		}
	}
}

// TestAttentionParamsScaleAndSoftcap covers the two constants a score is computed from.
//
// The scale is 1/sqrt(head dim), where the head dim comes from the file's key_length rather than
// from Hidden/Heads: Gemma 2 has 2304 hidden and 8 heads but 256-wide queries, so deriving it
// would give 288 and scale every score by a twelfth too little.
func TestAttentionParamsScaleAndSoftcap(t *testing.T) {
	// Without an explicit scale, 1/sqrt(head dim).
	plain := Config{HeadDim: 256}
	if got, want := attentionParams(plain, 0).scale, float32(1/math.Sqrt(256)); math.Abs(float64(got-want)) > 1e-7 {
		t.Errorf("default scale = %v, want %v", got, want)
	}
	if got := attentionParams(plain, 0).softcap; got != 0 {
		t.Errorf("default softcap = %v, want none", got)
	}

	// The softcap is applied after the scale, and the order is observable: at a score large
	// enough to saturate, the result is the cap regardless of the scale, while applying the cap
	// first and scaling after would multiply it.
	capped := Config{HeadDim: 256, AttnSoftcap: 50}
	if got := attentionParams(capped, 0).apply(1e6); math.Abs(float64(got-50)) > 1e-5 {
		t.Errorf("apply(1e6) = %v, want the softcap 50", got)
	}
	// A small dot product passes through nearly unchanged: 0.5/16 through tanh(y/50)*50.
	dot := float32(0.5)
	scaled := dot / 16
	if got, want := attentionParams(capped, 0).apply(dot), float32(50*math.Tanh(float64(scaled/50))); math.Abs(float64(got-want)) > 1e-6 {
		t.Errorf("apply(%v) = %v, want %v", dot, got, want)
	}
	// The head dim decides the scale, which is the Gemma 2 case: 288 would be the derived value
	// and 256 is the file's.
	fromFile := Config{HeadDim: 256}
	derived := Config{HeadDim: 288}
	if attentionParams(fromFile, 0).scale == attentionParams(derived, 0).scale {
		t.Error("the scale ignored the head dim")
	}
}

// TestGemma2SummaryNamesItsDifferences checks that every added difference reaches the summary.
//
// They are what an operator has when a Gemma 2 file generates something odd, and the shapes alone
// do not distinguish a correct load from one that skipped the sandwich norms or applied the
// window to every layer.
func TestGemma2SummaryNamesItsDifferences(t *testing.T) {
	path := gguftest.Model{Arch: "gemma2", Layers: 2, Hidden: 32, Heads: 4, FFN: 64,
		PostNorms: true, AttnSoftcap: 50, FinalSoftcap: 30, WindowLen: 4096,
	}.Write(t, t.TempDir())
	m, err := NewWithOptions(path, Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	got := m.Config().String()
	for _, want := range []string{"softcaps 50/30", "sandwich norms", "every second layer bounded to 4096"} {
		if !strings.Contains(got, want) {
			t.Errorf("summary %q does not mention %q", got, want)
		}
	}
}

// TestPostNormsRunOnlyWhereTheArchitectureSays pins the sandwich norms.
//
// A pre-norm architecture adds a sub-layer's output to the residual untouched. Gemma 2 normalizes
// it first, and the flag is what decides rather than whether the weights happen to be there --
// otherwise a Gemma 2 file with a missing tensor would quietly become a different model.
func TestPostNormsRunOnlyWhereTheArchitectureSays(t *testing.T) {
	path := gguftest.Model{Arch: "gemma2", Layers: 2, Hidden: 32, Heads: 4, FFN: 64,
		PostNorms: true}.Write(t, t.TempDir())
	m, err := NewWithOptions(path, Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if !m.Config().PostNorms {
		t.Fatal("PostNorms is false on a gemma2 model")
	}
	for l := range m.Config().Layers {
		if len(m.W.Layers[l].attnPostNorm) == 0 || len(m.W.Layers[l].ffnPostNorm) == 0 {
			t.Fatalf("layer %d is missing its post-norm tensors", l)
		}
	}

	// The norm doubles a vector of 2s into 2s (RMSNorm of a constant vector is that constant),
	// and a mixed vector comes back with RMS 1 -- so the assertion that it ran is that the
	// result differs from the input when the input's RMS is not 1.
	v := make([]float32, m.Config().Hidden)
	for i := range v {
		v[i] = float32(i + 1)
	}
	orig := append([]float32(nil), v...)
	if err := m.postNorm(v, m.W.Layers[0].attnPostNorm); err != nil {
		t.Fatalf("postNorm: %v", err)
	}
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if rms := math.Sqrt(sum / float64(len(v))); math.Abs(rms-1) > 1e-5 {
		t.Errorf("RMS after the post-norm is %v, want 1", rms)
	}

	// A model without the flag leaves the vector alone, whatever weights it carries.
	plain := gguftest.Model{Layers: 1, Hidden: 32, Heads: 4, FFN: 64}.Write(t, t.TempDir())
	pm, err := NewWithOptions(plain, Options{})
	if err != nil {
		t.Fatalf("load plain: %v", err)
	}
	v2 := append([]float32(nil), orig...)
	if err := pm.postNorm(v2, make([]float32, len(v2))); err != nil {
		t.Fatalf("postNorm on a plain model: %v", err)
	}
	for i := range v2 {
		if v2[i] != orig[i] {
			t.Errorf("a pre-norm model changed element %d: %v, want %v", i, v2[i], orig[i])
		}
	}
}

// TestGemma2MatchesReferenceLogprobs is the decisive check for the three things Gemma 2 adds.
//
// Set GEMMA2_MODEL to a Gemma 2 2B GGUF to run it. The expected values come from llama.cpp
// 0.3.37 on gemma-2-2b-it-Q3_K_L, prompt "The capital of France is", top-8 at the first
// generated position. The prompt encodes to the same six ids on both sides.
//
// Measured deviation: 0.374 nats at the worst of the eight, and the same eight tokens come back
// in the same order, with ' **' greedy on both. That is larger than the 0.19 measured for the
// Q4_K_M Gemma 1 file, which is what a three-bit quantisation costs when two implementations
// dequantize independently.
//
// The tolerance below is set from what the test has to catch, not from the measurement:
// dropping the post-norms puts the same eight tokens 130 nats away, and using the adjacent
// rotary pairing 0.88 away, so 0.6 sits between the measurement and the nearest wrong variant.
//
// What this cannot see is the softcaps. At this position the logits are well inside their cap,
// so turning both off moves the deviation from 0.374 to 0.378 -- they are architecturally
// required and numerically inert here. Their wiring is covered by the config and kernel tests
// instead, and the honest statement is that no test in this package would notice them being
// applied in the wrong place.
func TestGemma2MatchesReferenceLogprobs(t *testing.T) {
	path := os.Getenv("GEMMA2_MODEL")
	if path == "" {
		t.Skip("set GEMMA2_MODEL to a Gemma 2 2B GGUF file")
	}
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	if cfg.VocabSize != 256000 || cfg.Layers != 26 {
		t.Skipf("the expected values are Gemma 2 2B's (vocab 256000, 26 layers); this file has vocab %d and %d layers",
			cfg.VocabSize, cfg.Layers)
	}
	// The three settings whose absence the comparison below would report as a broken model.
	if cfg.RopeLayout != RopeNeox {
		t.Errorf("RopeLayout = %v, want the half-split pairing", cfg.RopeLayout)
	}
	if !cfg.PostNorms || !cfg.SlidingPattern {
		t.Errorf("PostNorms/SlidingPattern = %v/%v, want both", cfg.PostNorms, cfg.SlidingPattern)
	}
	// Gemma 2's attention projections are not square: 2304 hidden, 8 heads, 256-wide queries.
	if cfg.HeadDim != 256 {
		t.Errorf("HeadDim = %d, want 256 from attention.key_length rather than 288 from Hidden/Heads",
			cfg.HeadDim)
	}
	if cfg.AttnSoftcap != 50 || cfg.FinalSoftcap != 30 {
		t.Errorf("softcaps = %v/%v, want 50/30", cfg.AttnSoftcap, cfg.FinalSoftcap)
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

	want := []struct {
		id   int32
		text string
		lp   float64
	}{
		{5231, " **", -1.551132},
		{235292, ":", -1.863591},
		{7127, " Paris", -2.209725},
		{955, "...", -2.256747},
		{235336, "?", -2.714611},
		{235248, " ", -3.154603},
		{3836, " known", -3.240672},
		{2804, " ...", -3.970544},
	}
	const tolerance = 0.6
	for _, w := range want {
		got := logSoftmax(logits, w.id)
		if d := math.Abs(got - w.lp); d > tolerance {
			t.Errorf("logprob of %q (id %d) = %.4f, llama.cpp says %.4f, off by %.4f",
				w.text, w.id, got, w.lp, d)
		}
	}
	argmax := int32(0)
	for i, v := range logits {
		if v > logits[argmax] {
			argmax = int32(i)
		}
	}
	if argmax != 5231 {
		t.Errorf("greedy token is %d, want 5231 (\" **\")", argmax)
	}
}

// TestAttendSoftcapFlattensTheDistribution is the precise test of the attention cap, at a
// magnitude where the cap actually binds.
//
// The real checkpoint cannot provide this: at the reference position its scores are well inside
// a cap of 50, which is why turning the cap off there moved the log probabilities by 0.004. The
// case below is chosen so both answers are computable by hand. Two keys score 200 and 100
// against a unit query, with scale 1, over values 1 and 0:
//
//	uncapped: softmax(200, 100) is 1 and 0 in float32, so the output is 1.
//	capped at 50: the scores become 50*tanh(4) and 50*tanh(2), 49.964 and 48.211, and
//	softmax over a difference of 1.753 gives weights 0.852 and 0.148, so the output is 0.852.
//
// Those two numbers are far apart, so an implementation that ignored the cap, applied it before
// the scale, or applied it to the output instead of the scores fails on a value rather than on a
// rounded comparison.
func TestAttendSoftcapFlattensTheDistribution(t *testing.T) {
	q := []float32{1, 0}
	keys := []float32{
		200, 0, // position 0: dot 200
		100, 0, // position 1: dot 100
	}
	values := []float32{
		1, 1, // position 0
		0, 0, // position 1
	}
	keysSpans := []span{{data: keys, stride: 2, rows: 2, width: 2}}
	valuesSpans := []span{{data: values, stride: 2, rows: 2, width: 2}}
	scores := make([]float32, 2)
	out := make([]float32, 2)

	if err := attend(q, keysSpans, valuesSpans, 2, scores, out, attnParams{scale: 1}); err != nil {
		t.Fatalf("attend: %v", err)
	}
	if out[0] < 0.999 {
		t.Errorf("uncapped output[0] = %v, want the first value to dominate at ~1", out[0])
	}

	if err := attend(q, keysSpans, valuesSpans, 2, scores, out, attnParams{scale: 1, softcap: 50}); err != nil {
		t.Fatalf("attend: %v", err)
	}
	// The capped scores, worked out independently of the implementation.
	s0 := 50 * math.Tanh(200.0/50)
	s1 := 50 * math.Tanh(100.0/50)
	want := float32(math.Exp(s0) / (math.Exp(s0) + math.Exp(s1)))
	if math.Abs(float64(out[0]-want)) > 1e-5 {
		t.Errorf("capped output[0] = %v, want %v", out[0], want)
	}
	// The cap's whole point: the ratio between the two keys is compressed, so the second
	// position keeps a real share of the attention rather than none.
	if out[0] > 0.95 {
		t.Errorf("capped output[0] = %v; the cap did not flatten the distribution", out[0])
	}
}

// TestFinalSoftcapBoundsTheLogits covers the other cap, which is the one an operator would notice:
// without it a model whose output projection is hot reports logits in the hundreds and a
// distribution that is all but one token.
//
// The regime is reached by scaling the final norm weight rather than by building a fixture with
// large weights: RMSNorm is linear in its weight, so multiplying it by 100 multiplies the logits
// by 100 and nothing else in the forward pass changes. The test asserts the regime, the bound,
// and that the two runs differ -- without the first, a cap that did nothing would still satisfy
// the bound.
func TestFinalSoftcapBoundsTheLogits(t *testing.T) {
	path := gguftest.Model{Arch: "gemma2", Layers: 2, Hidden: 32, Heads: 4, FFN: 64,
		PostNorms: true, AttnSoftcap: 50, FinalSoftcap: 30}.Write(t, t.TempDir())
	m, err := NewWithOptions(path, Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	// Hot output projection, reached through the last norm.
	for i := range m.W.OutputNorm {
		m.W.OutputNorm[i] = 100
	}

	run := func(cap float32) []float32 {
		t.Helper()
		m.W.Config.FinalSoftcap = cap
		cache, err := NewKVCache(m.W.Config, 32)
		if err != nil {
			t.Fatalf("cache: %v", err)
		}
		logits, err := m.Prefill(cache, []int32{1, 2}, 0)
		if err != nil {
			t.Fatalf("prefill: %v", err)
		}
		return append([]float32(nil), logits...)
	}

	uncapped := run(0)
	capped := run(30)

	var maxUncapped float32
	uncapped = uncapped[:m.Config().VocabSize]
	for _, v := range uncapped {
		if v > maxUncapped {
			maxUncapped = v
		}
	}
	if maxUncapped <= 30 {
		t.Fatalf("the largest uncapped logit is %v; the fixture is not in the regime where the"+
			" cap binds, so this test would pass without a cap", maxUncapped)
	}

	capped = capped[:m.Config().VocabSize]
	for i, v := range capped {
		if v > 30 || v < -30 {
			t.Fatalf("logit %d is %v, outside the cap of 30", i, v)
		}
	}
	// Saturated, not merely inside: the scaled projection puts the largest logit far past 30.
	if capped[0] > 30 {
		t.Errorf("logit 0 = %v, want it bounded", capped[0])
	}
	same := true
	for i := range capped {
		if capped[i] != uncapped[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("the capped and uncapped logits are identical; the cap never ran")
	}
}
