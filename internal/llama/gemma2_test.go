package llama

import (
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
	"github.com/pplmx/orbix/internal/kernel"
	"github.com/pplmx/orbix/internal/tokenize"
)

// TestWindowForAlternates pins which layers are bounded, which is the part of Gemma 2's sliding
// window that is architecture rather than metadata.
//
// The window length arrives as a metadata key and is the same number for both kinds of layer;
// what the architecture decides is that only every second one uses it. A loader that applied it
// to every layer would produce a model whose long-range attention silently stops working while
// everything still runs.
func TestWindowForAlternates(t *testing.T) {
	// A declared window of 4096 leaves 4097 positions reachable: llama.cpp masks a key when the
	// distance is >= n_swa, so the key at exactly p-4096 is still visible. These numbers are
	// counts of reachable positions, not the declared distance.
	pattern := Config{WindowLen: 4096, SlidingPattern: true}
	want := []int{4097, 0, 4097, 0, 4097, 0}
	for l, w := range want {
		if got := windowFor(pattern, l); got != w {
			t.Errorf("layer %d window = %d, want %d", l, got, w)
		}
	}

	// Without the pattern every layer is bounded, which is Mistral's shape.
	flat := Config{WindowLen: 4096}
	for l := range 4 {
		if got := windowFor(flat, l); got != 4097 {
			t.Errorf("layer %d window = %d, want 4097 on every layer", l, got)
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
	// Four layers, because the alternating rule needs more than two to say anything: over two
	// layers "every second layer" and "every layer" are the same set, and the summary reports what
	// the configuration computes rather than the rule that produced it.
	path := gguftest.Model{Arch: "gemma2", Layers: 4, Hidden: 32, Heads: 4, FFN: 64,
		PostNorms: true, AttnSoftcap: 50, FinalSoftcap: 30, WindowLen: 4096,
	}.Write(t, t.TempDir())
	m, err := NewWithOptions(path, Options{})
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	got := m.Config().String()
	for _, want := range []string{
		"softcaps 50/30",
		"sandwich norms",
		"2 of 4 layers bounded to 4096",
	} {
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

// TestGemma2WindowChangesWhatTheModelAnswers is the real-weight counterpart of the fixture test
// above.
//
// The fixture cannot show this: a constructed model's weights make every position look alike, so
// bounding attention changes the logits by 1e-6 and a window that was silently dropped looks
// identical to one that was applied. On gemma-2-2b-it the difference is 10.93 logits and a
// different argmax, which is what makes the window worth getting right and what makes the
// architecture default in TestGemma2HasAWindowWhenTheFileOmitsTheKey worth having.
//
// Set GEMMA2_MODEL to a Gemma 2 GGUF. The window is set to 32 rather than left at the file's 4096
// so the prompt can be short: 32 binds at position 32, and a 122-token prompt crosses it five
// times over.
func TestGemma2WindowChangesWhatTheModelAnswers(t *testing.T) {
	path := os.Getenv("GEMMA2_MODEL")
	if path == "" {
		t.Skip("set GEMMA2_MODEL to a Gemma 2 GGUF file")
	}
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(f)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	w, err := LoadWeights(path)
	if err != nil {
		t.Fatalf("LoadWeights: %v", err)
	}
	ids := tok.Encode(strings.Repeat("The quick brown fox jumps over the lazy dog. ", 12), true)
	if len(ids) > 128 {
		ids = ids[:128]
	}
	if len(ids) < 64 {
		t.Fatalf("the prompt is only %d tokens; a window of 32 would bind too few times", len(ids))
	}

	// bounded runs the prompt with one window, returning the logits of the last position.
	bounded := func(window int) []float32 {
		w.Config.WindowLen = window
		m := modelFromWeights(t, w)
		cache, err := NewKVCache(w.Config, len(ids)+8)
		if err != nil {
			t.Fatalf("NewKVCache: %v", err)
		}
		var logits []float32
		for pos, id := range ids {
			if logits, err = m.Forward(cache, int(id), pos); err != nil {
				t.Fatalf("Forward at %d: %v", pos, err)
			}
		}
		return append([]float32(nil), logits...)
	}
	with, without := bounded(32), bounded(0)

	worst, argW, argO := 0.0, 0, 0
	for i := range with {
		if d := math.Abs(float64(with[i] - without[i])); d > worst {
			worst = d
		}
		if with[i] > with[argW] {
			argW = i
		}
		if without[i] > without[argO] {
			argO = i
		}
	}
	if worst < 1 {
		t.Errorf("a 32-position window moved the logits by only %.4f on real weights; the fixture "+
			"cannot see this either, so a value this small would mean the window is not reaching "+
			"attention at all", worst)
	}
	if argW == argO {
		t.Logf("the window moved the logits by %.2f without changing the argmax, which is expected "+
			"on a repetitive prompt", worst)
	}
}

// TestGemma3MatchesReferenceLogprobs checks the Gemma 3 architecture against llama.cpp.
//
// Gemma 3 keeps Gemma 2's block -- sandwich norms, GeGLU, scaled embeddings -- and changes three
// things this package has to get right: per-head query and key normalization, no attention softcap,
// and a sliding window whose shape comes from the file rather than from "every second layer".
//
// The window is the one that bit: Gemma 3 270M declares a window of 512 and no pattern key at all,
// and llama.cpp reads that absence as "every layer is local" (set_swa_pattern with n_pattern 0).
// Reading it as "every second layer" left nine layers attending over the whole context, which
// loads without complaint and computes something else.
//
// Set GEMMA3_MODEL to gemma-3-270m-it-Q8_0.gguf. Expected values from llama.cpp 0.3.37, prompt
// "The capital of France is"; the prompt encodes to the same six ids on both sides and the greedy
// token agrees. The deviations over the shared tokens are 0.0001, 0.66, 0.27, 0.08, 0.16, 0.42 and
// 0.66 nats -- the low-probability tail, as with every checkpoint here, while the token the model
// actually picks agrees to four decimal places.
func TestGemma3MatchesReferenceLogprobs(t *testing.T) {
	path := os.Getenv("GEMMA3_MODEL")
	if path == "" {
		t.Skip("set GEMMA3_MODEL to a Gemma 3 GGUF file")
	}
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	if cfg.VocabSize != 262144 {
		t.Skipf("the expected values are Gemma 3 270M's; this file has a %d-token vocabulary",
			cfg.VocabSize)
	}

	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(f)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	ids := tok.Encode("The capital of France is", true)
	for i, want := range []int32{2, 818, 5279, 529, 7001, 563} {
		if ids[i] != want {
			t.Fatalf("prompt id %d = %d, want %d", i, ids[i], want)
		}
	}

	cache, err := NewKVCache(cfg, 256)
	if err != nil {
		t.Fatal(err)
	}
	var logits []float32
	for pos, id := range ids {
		logits, err = m.Forward(cache, int(id), pos)
		if err != nil {
			t.Fatalf("Forward: %v", err)
		}
	}

	// Resolved by exact text so the test does not depend on the ids staying put across files.
	toks, _ := f.GetArray[string]("tokenizer.ggml.tokens")
	find := func(p string) int32 {
		for i, s := range toks {
			if s == p {
				return int32(i)
			}
		}
		t.Fatalf("token %q is not in the vocabulary", p)
		return 0
	}
	const tolerance = 1.5
	for _, w := range []struct {
		text string
		ref  float64
	}{
		{"▁Paris", -0.000517},
		{"▁the", -9.133692},
		{"▁**", -9.396717},
		{"Paris", -10.126068},
		{"▁Marseille", -10.469614},
		{"▁La", -11.122358},
		{"▁changed", -11.231756},
	} {
		got := logSoftmax(logits, find(w.text))
		if d := math.Abs(got - w.ref); d > tolerance {
			t.Errorf("logprob of %q = %.4f, llama.cpp says %.4f, off by %.4f",
				w.text, got, w.ref, d)
		}
	}

	argmax := int32(0)
	for i, v := range logits {
		if v > logits[argmax] {
			argmax = int32(i)
		}
	}
	if argmax != find("▁Paris") {
		t.Errorf("greedy token is %d, want the id of %q", argmax, "▁Paris")
	}

	// The window shape is part of what this test is for, so it is asserted rather than left to the
	// summary string.
	if cfg.WindowLen != 512 {
		t.Errorf("window = %d, want 512", cfg.WindowLen)
	}
	// This file declares no pattern key, so the architecture's default fills the pattern in: layers
	// 0-4 of every six are local and every sixth is global. The assertion used to be that every
	// layer was bounded, on the reading that llama.cpp's absent key means "all local"; that is not
	// what it does -- gemma3.cpp calls load_swa_pattern(ml, 6) -- and the difference is worth 12.8
	// nats at 1200 tokens, see TestGemma3DefaultWindowPatternIsEverySixthLayer.
	local := 0
	for l := range cfg.Layers {
		if windowFor(cfg, l) > 0 {
			local++
		}
	}
	if want := cfg.Layers / gemma3WindowPeriod * (gemma3WindowPeriod - 1); local != want {
		t.Errorf("%d of %d layers are bounded, want %d: this file declares no pattern and a period"+
			" of %d bounds all but every %dth", local, cfg.Layers, want,
			gemma3WindowPeriod, gemma3WindowPeriod)
	}
}

// TestEveryForwardPathUsesTheLayerRotation covers the wiring, not the arithmetic.
//
// Model has four forward paths -- Forward, attentionCached, Prefill and DecodeStep -- and each has
// its own loop over heads calling rope.apply. A second table that only Forward consults would
// leave three paths rotating at the wrong rate, and no test above can see it: they check that
// ropeFor answers correctly, not that anyone asks it.
//
// The check is a value comparison rather than a coverage claim. Each path is run on a model whose
// two bases are deliberately extreme, and the result is compared against the same path run with
// the two bases swapped. If a path reads the global table unconditionally, both runs agree.
func TestEveryForwardPathUsesTheLayerRotation(t *testing.T) {
	// Every layer local (a two-layer fixture is inside the period-6 default on both layers), so
	// the local table is the one every path has to reach for.
	build := func(localBase float32) *Model {
		path := gguftest.Model{Arch: "gemma3", Layers: 2, Hidden: 32, Heads: 4, FFN: 64,
			QKNorm: true, PostNorms: true, WindowLen: 8, ContextLen: 32, NoEOS: true,
			ExtraFloatKey: "gemma3.rope.freq_base", ExtraFloatValue: 1000000,
		}.Write(t, t.TempDir())
		// The local base is patched on the config rather than written as a second key, because
		// the fixture writes one extra float and the point here is the selector, not the parsing.
		m, err := NewWithOptions(path, Options{Workers: 1})
		if err != nil {
			t.Fatalf("NewWithOptions: %v", err)
		}
		m.W.Config.RopeBaseSWA = localBase
		r, err := newLocalRope(m.W.Config, m.W.Config.ContextLen)
		if err != nil {
			t.Fatalf("newLocalRope: %v", err)
		}
		m.ropeLocal = r
		if r == nil {
			t.Fatal("the model has one table, so this test cannot tell the paths apart")
		}
		return m
	}

	tokens := []int32{1, 2, 3, 4}

	// Forward: one token at a time.
	forward := func(m *Model) []float32 {
		cache, err := NewKVCache(m.Config(), 16)
		if err != nil {
			t.Fatalf("NewKVCache: %v", err)
		}
		var logits []float32
		for pos, tok := range tokens {
			logits, err = m.Forward(cache, int(tok), pos)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
		}
		return slices.Clone(logits)
	}
	// Prefill: all four positions in one pass.
	prefill := func(m *Model) []float32 {
		cache, err := NewKVCache(m.Config(), 16)
		if err != nil {
			t.Fatalf("NewKVCache: %v", err)
		}
		logits, err := m.Prefill(cache, tokens, 0)
		if err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		return slices.Clone(logits)
	}
	// DecodeStep: four sequences advanced together, each past position 0.
	//
	// The two prefixes matter. At position 0 the rotation is the identity for every base, so a
	// DecodeStep driven on empty caches returns the same logits whatever the table holds; the
	// first version of this test did exactly that and passed for the wrong reason.
	decode := func(m *Model) []float32 {
		cs := make([]Cache, len(tokens))
		for i := range cs {
			c, err := NewKVCache(m.Config(), 16)
			if err != nil {
				t.Fatalf("NewKVCache: %v", err)
			}
			if _, err := m.Prefill(c, []int32{1, 2}, 0); err != nil {
				t.Fatalf("Prefill: %v", err)
			}
			cs[i] = c
		}
		logits, err := m.DecodeStep(cs, tokens)
		if err != nil {
			t.Fatalf("DecodeStep: %v", err)
		}
		return slices.Clone(logits)
	}
	// attentionCached: prefill a prefix, then score one more token against the stored positions.
	cached := func(m *Model) []float32 {
		pool, err := NewBlockPool(m.Config(), BlockPoolOptions{BlockSize: 4, Blocks: 16})
		if err != nil {
			t.Fatalf("NewBlockPool: %v", err)
		}
		cache := pool.New()
		if _, err := m.PrefillWithPrefix(cache, tokens[:3], 0); err != nil {
			t.Fatalf("PrefillWithPrefix: %v", err)
		}
		out, err := m.logitsForCached(cache, tokens, 3)
		if err != nil {
			t.Fatalf("logitsForCached: %v", err)
		}
		return slices.Clone(out)
	}

	for _, c := range []struct {
		name string
		run  func(*Model) []float32
	}{
		{"Forward", forward},
		{"Prefill", prefill},
		{"DecodeStep", decode},
		{"attentionCached", cached},
	} {
		t.Run(c.name, func(t *testing.T) {
			a := c.run(build(1e4))
			b := c.run(build(7e5))
			if slices.Equal(a, b) {
				t.Errorf("%s produced identical logits for local bases 1e4 and 7e5, so it is not "+
					"reading the local table at all", c.name)
			}
		})
	}
}

// modelFromWeights builds the scratch buffers a Model needs around already-loaded weights, so a
// test can vary the Config without re-reading the file. NewWithOptions is the production path and
// takes a path, not a *Weights; this exists only to run the same model under two configs.
func modelFromWeights(t *testing.T, w *Weights) *Model {
	t.Helper()
	cfg := w.Config
	r, err := newRope(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatalf("newRope: %v", err)
	}
	local, err := newLocalRope(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatalf("newLocalRope: %v", err)
	}
	return &Model{
		// Workers rather than 1: the partitionable kernels are documented as bit-identical across
		// the split (kernel/pool.go), so a test that compares two configs gets the same answer
		// either way -- and a real checkpoint on one core is minutes per prompt instead of
		// seconds.
		W: w, rope: r, ropeLocal: local, run: kernel.NewRunner(runtime.GOMAXPROCS(0)),
		x: make([]float32, cfg.Hidden), xn: make([]float32, cfg.Hidden),
		q: make([]float32, cfg.Heads*cfg.HeadDim), k: make([]float32, cfg.kvDim()),
		v: make([]float32, cfg.kvDim()), attn: make([]float32, cfg.Heads*cfg.HeadDim),
		attnOut: make([]float32, cfg.Hidden), gate: make([]float32, cfg.FFN),
		up: make([]float32, cfg.FFN), scores: make([]float32, cfg.ContextLen),
		logits: make([]float32, cfg.VocabSize),
	}
}

// TestGemma3PredictsEnglishPastTheWindowBetterThanTheWrongPattern is the evidence that needs no
// reference at all.
//
// The tests above pin the rotation and the pattern against llama.cpp and against Hugging Face's
// config, both outside this repository. This one asks the checkpoint: Gemma 3 was trained on text,
// so the interpretation it was trained with predicts held-out English better than one it was not.
// A direction, not a figure, and the direction is the thing a comparison against a wrong reference
// cannot supply -- a wrong reading is fluently wrong, which is what made both defects survive.
//
// Only positions past the window are scored, and that restriction is the measurement rather than a
// convenience. Over the whole passage the two readings are 93.7 and 105.4 perplexity, a nine-tenths
// ratio that is a real difference but a weak discriminator. Split at the window it is stark:
// positions 1..511 score 93.9 against 101.3, and positions 512 onwards score 93.3 against 124.8.
// The first half is a rotation difference seen through a context that never binds; the second is
// what the window and the local layers actually do. All of the signal is in the second half.
//
// Measured on gemma-3-270m-it-Q8_0 with testdata/gemma3-nll.txt, 639 tokens: the trained reading
// scores 93.3 past the window and the old one -- every layer local, one base of 1e6 -- scores 124.8.
func TestGemma3PredictsEnglishPastTheWindowBetterThanTheWrongPattern(t *testing.T) {
	path := os.Getenv("GEMMA3_MODEL")
	if path == "" {
		t.Skip("set GEMMA3_MODEL to a Gemma 3 GGUF file")
	}
	raw, err := os.ReadFile(filepath.Join("testdata", "gemma3-nll.txt"))
	if err != nil {
		t.Fatalf("read the passage: %v", err)
	}
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	tok, err := tokenize.New(f)
	if err != nil {
		t.Fatalf("tokenizer: %v", err)
	}
	ids := tok.Encode(string(raw), true)

	// perplexityPastWindow runs the model under one config and returns the perplexity of the
	// positions after the first window, where the two readings diverge.
	perplexityPastWindow := func(mutate func(*Config)) (float64, int) {
		w, err := LoadWeights(path)
		if err != nil {
			t.Fatalf("LoadWeights: %v", err)
		}
		cfg := w.Config
		mutate(&cfg)
		w.Config = cfg
		m := modelFromWeights(t, w)
		cache, err := NewKVCache(cfg, len(ids)+8)
		if err != nil {
			t.Fatalf("NewKVCache: %v", err)
		}
		var logits []float32
		var nll float64
		counted := 0
		for pos, id := range ids {
			// Position pos is predicted by the logits left by position pos-1, so the first
			// position past the window is the one whose predecessor is at the window's edge.
			if pos > cfg.WindowLen {
				nll -= logSoftmax(logits, id)
				counted++
			}
			if logits, err = m.Forward(cache, int(id), pos); err != nil {
				t.Fatalf("Forward at %d: %v", pos, err)
			}
		}
		if counted < 50 {
			t.Fatalf("only %d positions past the window; the passage is too short to measure a "+
				"difference there", counted)
		}
		return math.Exp(nll / float64(counted)), counted
	}

	trained, n := perplexityPastWindow(func(*Config) {})
	// The reading this package used before the fix: every layer local, one rotation for all.
	before, _ := perplexityPastWindow(func(c *Config) {
		all := make([]bool, c.Layers)
		for i := range all {
			all[i] = true
		}
		c.LocalLayers = all
		c.RopeBaseSWA = c.RopeBase
	})

	t.Logf("perplexity past the window over %d positions: trained reading %.1f, the old one %.1f",
		n, trained, before)
	// A quarter better, against the measured ratio of 0.747. The margin is set well inside the
	// measurement so that a tokenizer change or a different quantisation of the same model does not
	// fail the test, while a reading that is wrong in the way the old one was cannot pass it.
	if !(trained < before*0.9) {
		t.Errorf("the trained reading scores %.1f past the window and the old one %.1f; the "+
			"trained one has to be at least a tenth better, and that direction is what identifies "+
			"which reading this checkpoint was trained with", trained, before)
	}
}

// TestGemma3DefaultWindowPatternIsEverySixthLayer pins the pattern a Gemma 3 file gets when it
// declares a window and no pattern.
//
// The branch used to make every layer local, reading llama.cpp's load_swa_pattern fallthrough as
// "n_pattern 0 means all local". llama.cpp calls load_swa_pattern(ml, 6) for gemma3, so an absent
// key keeps the 6 it was handed: layers 0-4 of every six are local and layer 5 is global. Hugging
// Face's Gemma3TextConfig says the same thing from the other side, with sliding_window_pattern
// defaulting to 6 and layer_types marking (i+1) % 6 == 0 as full_attention.
//
// Measured on gemma-3-270m-it-Q8_0 at 1200 tokens, where the window first separates the readings:
// all-local put the top-8 log probabilities up to 12.8 nats from llama.cpp's, period-6 within 0.44.
// Prompts of 513 tokens and below cannot see it, which is why the earlier evidence missed this.
func TestGemma3DefaultWindowPatternIsEverySixthLayer(t *testing.T) {
	// Twelve layers, so the pattern repeats twice and a rule that only got the first six right
	// would still be caught.
	path := gguftest.Model{Arch: "gemma3", Layers: 12, Hidden: 32, Heads: 4, FFN: 64,
		QKNorm: true, PostNorms: true, WindowLen: 512,
	}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	cfg, err := LoadConfig(f)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}

	if len(cfg.LocalLayers) != 12 {
		t.Fatalf("LocalLayers has %d entries for 12 layers; the file declares no pattern key and "+
			"the architecture's default should have filled one in", len(cfg.LocalLayers))
	}
	// Index 5 and 11 are the global layers: every sixth, counted from zero.
	want := []bool{
		true, true, true, true, true, false,
		true, true, true, true, true, false,
	}
	for l := range want {
		if cfg.LocalLayers[l] != want[l] {
			t.Errorf("layer %d local = %v, want %v", l, cfg.LocalLayers[l], want[l])
		}
		// windowFor is the predicate everything downstream keys off -- attention bounds and, since
		// the two-frequency change, which rotation table a layer uses -- so the bit and its
		// consequence are both asserted.
		bounded := windowFor(cfg, l) > 0
		if bounded != want[l] {
			t.Errorf("layer %d windowFor = %d, want a bounded window: %v", l, windowFor(cfg, l), want[l])
		}
	}
}

// TestGemma3RotatesLocalAndGlobalLayersAtTheirOwnFrequencies covers the per-layer rotation, which
// this package used to refuse rather than implement.
//
// Gemma 3 rotates its windowed layers and its full-attention layers at different rates: Hugging
// Face's Gemma3TextConfig carries {"global": 1e6, "local": 1e4}, and llama.cpp picks between
// rope_freq_base_train_swa and the global base per layer, on the same is_swa predicate that decides
// which layers read a bounded window. A file may name the local rate as rope.freq_base_swa or leave
// it to the architecture's default.
//
// What it costs to get wrong is measured on gemma-3-270m-it-Q8_0, which declares only the global
// base: served at one rate this engine was up to 12.8 nats away from llama.cpp's top-8 log
// probabilities at 1200 tokens, and at two rates within 0.44. -- see docs/models.md.
//
// The test drives the selector directly rather than through a forward pass, because a two-layer
// fixture's window is wider than its context: nothing in a fixture can make the choice visible in
// a logit, and a test that claimed to would be measuring something else.
func TestGemma3RotatesLocalAndGlobalLayersAtTheirOwnFrequencies(t *testing.T) {
	// A gemma3 fixture with a window narrower than its context, so the period-6 default separates
	// the layers. The window is what makes a layer local; the base only says which table it uses.
	// The global base has to differ from gemma3's local default of 10000 for the second table to
	// exist at all; a fixture that leaves both at the package default has one rotation and would
	// pass every assertion below for the wrong reason.
	path := gguftest.Model{Arch: "gemma3", Layers: 6, Hidden: 32, Heads: 4, FFN: 64,
		QKNorm: true, PostNorms: true, WindowLen: 2,
		ExtraFloatKey: "gemma3.rope.freq_base", ExtraFloatValue: 1000000,
	}.Write(t, t.TempDir())
	m, err := NewWithOptions(path, Options{})
	if err != nil {
		t.Fatalf("NewWithOptions: %v", err)
	}
	cfg := m.Config()

	// The default: no key in the file, so the architecture's own second rate.
	if cfg.RopeBaseSWA != 10000 {
		t.Errorf("RopeBaseSWA = %g on a gemma3 file that names no local rate, want the 10000 its "+
			"architecture defaults to", cfg.RopeBaseSWA)
	}
	if m.ropeLocal == nil {
		t.Fatal("no second rotation table was built, so every layer would rotate at the global base")
	}

	// Layers 0..4 of each period are local and layer 5 is global, which is what makes the two
	// tables reachable at all. Asserted here because the selector's answer is meaningless if the
	// pattern is wrong: with every layer global, both tables would be correct and unused.
	wantLocal := []bool{true, true, true, true, true, false}
	for l, want := range wantLocal {
		if got := windowFor(cfg, l) > 0; got != want {
			t.Errorf("layer %d local = %v, want %v", l, got, want)
		}
		if got := m.ropeFor(l) == m.ropeLocal; got != want {
			t.Errorf("layer %d uses the local table = %v, want %v", l, got, want)
		}
	}

	// A file that names its own local rate gets that rate, over the architecture's default.
	explicit := gguftest.Model{Arch: "gemma3", Layers: 6, Hidden: 32, Heads: 4, FFN: 64,
		QKNorm: true, PostNorms: true,
		ExtraFloatKey: "gemma3.rope.freq_base_swa", ExtraFloatValue: 500000,
	}.Write(t, t.TempDir())
	got, err := NewWithOptions(explicit, Options{})
	if err != nil {
		t.Fatalf("NewWithOptions: %v", err)
	}
	if got.Config().RopeBaseSWA != 500000 {
		t.Errorf("RopeBaseSWA = %g, want the file's 500000", got.Config().RopeBaseSWA)
	}

	// A plain llama file has one rotation and must not pay for a second table.
	plain := gguftest.Model{Arch: "llama", Layers: 2, Hidden: 32, Heads: 4, FFN: 64}.Write(t, t.TempDir())
	lm, err := NewWithOptions(plain, Options{})
	if err != nil {
		t.Fatalf("NewWithOptions: %v", err)
	}
	if lm.ropeLocal != nil {
		t.Error("a llama model built a second rotation table; it declares one frequency")
	}
	if got, want := lm.Config().RopeBaseSWA, lm.Config().RopeBase; got != want {
		t.Errorf("RopeBaseSWA = %g, want the global base %g", got, want)
	}
}

// TestRopeFreqBaseSWAIsRefusedWhenNotFinite covers the guards on the second base.
//
// The two values that matter are the ones a "<= 0" test lets through: NaN becomes the base of the
// exponent and turns every table entry past position 0 into NaN, and +Inf collapses every frequency
// to 1 so the rotation stops distinguishing positions. Both are refused by name rather than served
// as a model whose output is silently wrong.
func TestRopeFreqBaseSWAIsRefusedWhenNotFinite(t *testing.T) {
	for _, v := range []float64{math.NaN(), math.Inf(1), 0, -1} {
		path := gguftest.Model{Arch: "gemma3", Layers: 2, Hidden: 32, Heads: 4, FFN: 64,
			QKNorm: true, PostNorms: true,
			ExtraFloatKey: "gemma3.rope.freq_base_swa", ExtraFloatValue: float32(v),
		}.Write(t, t.TempDir())
		_, err := NewWithOptions(path, Options{})
		if err == nil {
			t.Errorf("a file with rope.freq_base_swa = %v loaded", v)
			continue
		}
		if !strings.Contains(err.Error(), "freq_base_swa") {
			t.Errorf("error %q does not name the key that was wrong", err)
		}
	}
}

// TestSlidingWindowReachIsInclusive covers the boundary of the sliding window, which is off by one
// in the obvious reading of the metadata key.
//
// A file declares a window as a distance: llama.cpp masks a key when pos_query - pos_key >= n_swa,
// so a window of 512 leaves the key at exactly p-512 visible. Treating the number as a count of
// positions drops that one, and the loss is invisible until a prompt is long enough for the window
// to bind -- which no fixture test exercises, because a fixture's context is shorter than any real
// window.
//
// Measured on gemma-3-270m-it-Q8_0 with a 513-token prompt built from one repeated sentence, so the
// next token is known: at 513 tokens it is "brown". The count reading answered "fox" (the token
// after it); a numpy reference written directly against the mask rule answered "brown" with "> w"
// and "fox" with ">= w", which is what identified the boundary rather than the whole window as the
// fault. After the fix both llama.cpp and orbix answer "brown".
func TestSlidingWindowReachIsInclusive(t *testing.T) {
	const window = 512
	cfg := Config{WindowLen: window}
	if got := windowFor(cfg, 0); got != window+1 {
		t.Errorf("windowFor(%d) = %d, want %d: the position at exactly p-window is still reachable",
			window, got, window+1)
	}

	// The count has to be what makes a query at position `window` see positions 1..window, which is
	// window positions plus its own. Spelling the arithmetic out here rather than trusting the +1:
	// at position p the reachable keys are p-reach+1 .. p, and the oldest must be p-window.
	p, reach := window, windowFor(cfg, 0)
	oldest := p - reach + 1
	if oldest != p-window {
		t.Errorf("at position %d a reach of %d starts at %d, want %d", p, reach, oldest, p-window)
	}

	// Zero stays zero: "no window" must not become "see only yourself".
	if got := windowFor(Config{}, 0); got != 0 {
		t.Errorf("windowFor with no window = %d, want 0", got)
	}
}
