package llama

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/kernel"
)

// Config holds the hyperparameters read from a GGUF file.
//
// The field names follow the GGUF keys they come from, with two exceptions:
// KVHeads falls back to Heads when the file omits it (older models are
// multi-head, not grouped-query), and RopeDim falls back to HeadDim.
type Config struct {
	// Arch is general.architecture, the key prefix for everything else.
	Arch string

	// RopeLayout is how the rotary embedding pairs dimensions. Llama rotates
	// adjacent pairs (2i, 2i+1); Qwen2 rotates i with i+n_dims/2, which ggml calls
	// NEOX. Getting this wrong leaves a model fluent but measurably off, so it is
	// a field with a name rather than a branch hidden in the rotation.
	RopeLayout RopeLayout

	// AttnBias reports whether the attention projections have bias vectors. Qwen2
	// has q, k, and v biases; Llama has none.
	AttnBias bool

	// Name is general.name, for display.
	Name string

	// VocabSize is the number of token embeddings.
	VocabSize int

	// Hidden is the embedding dimension, llama.embedding_length.
	Hidden int

	// Layers is llama.block_count.
	Layers int

	// Heads is llama.attention.head_count.
	Heads int

	// KVHeads is llama.attention.head_count_kv, defaulting to Heads.
	KVHeads int

	// HeadDim is HeadCount-derived: Hidden/Heads.
	HeadDim int

	// FFN is llama.feed_forward_length, the width of one expert's hidden layer.
	FFN int

	// Experts is llama.expert_count, or 0 for a dense feed-forward network.
	//
	// A mixture-of-experts layer replaces the single gated FFN with Experts copies of
	// it plus a small router, and runs only ExpertUsed of them per token. The compute
	// is therefore proportional to ExpertUsed rather than Experts, which is the whole
	// point: Mixtral 8x7B runs two experts per token and costs like a 13B model while
	// carrying 47B of weights.
	//
	// Zero is the dense default and is not a special case in the forward pass: a dense
	// layer is the same code with one expert and no router, which is why this package
	// keeps one path rather than two.
	Experts int

	// ExpertUsed is llama.expert_used_count, how many experts a token routes to.
	ExpertUsed int

	// ContextLen is llama.context_length, the trained window this package
	// allocates a KV cache for.
	ContextLen int

	// RopeDim is llama.rope.dimension_count, the number of dimensions the
	// rotation is applied to. It is usually but not always equal to HeadDim.
	RopeDim int

	// RopeBase is llama.rope.freq_base, the theta of the rotation.
	RopeBase float32

	// RopeFreqs is the rope_freqs.weight tensor: one frequency divisor per dimension
	// pair, or nil when the model has none.
	//
	// It is how a model extends its context without changing freq_base. Each pair's
	// frequency is divided by its entry, so an entry of 32 makes that dimension rotate 32
	// times slower -- a 4096-position model becomes a 131072-position one. Ignoring the
	// tensor leaves the low dimensions right and the high ones rotating far too fast,
	// which reads as fluent text with positions confused past the original context.
	RopeFreqs []float32

	// RopeScale scales the rotation frequency. Some fine-tunes set it below 1
	// to extend the usable context; the default is 1.
	RopeScale float32

	// WindowLen is how many recent positions attention may read, or 0 for no limit.
	//
	// Mistral trained with a 4096-position sliding window: each token attends to the
	// previous 4096 and no further. Applying full attention instead does not fail --
	// it produces fluent text with subtly wrong probabilities, because the model never
	// learned to weigh distant tokens. Treating the window as a cache bound is also
	// wrong: the cache still has to hold the context the caller asked for, since the
	// positions come back into range for later tokens.
	WindowLen int

	// RMSNormEps is the epsilon in the RMS normalization.
	RMSNormEps float32

	// TiedEmbeddings reports whether the output projection reuses the token
	// embedding matrix. A file expresses this by omitting output.weight.
	TiedEmbeddings bool

	// EOSID is the end-of-sequence token, or -1 when the file does not define
	// one. Generation stops on it; a -1 means the caller must supply its own
	// stopping condition.
	EOSID int

	// BOSID is the beginning-of-sequence token, or -1 when undefined.
	BOSID int

	// EmbedScale is the factor token embeddings are multiplied by on the way into the network.
	// It is 1 for architectures that do not scale them.
	//
	// This is applied at the lookup rather than folded into the embedding matrix, because that
	// matrix is usually also the tied output projection and that must see the unscaled weights.
	EmbedScale float32

	// FFNAct is the activation on the gated feed-forward branch.
	FFNAct ffnAct

	// AttnSoftcap, when non-zero, bounds every attention score with
	// tanh(score/AttnSoftcap)*AttnSoftcap.
	AttnSoftcap float32
	// FinalSoftcap, when non-zero, bounds the output logits the same way.
	FinalSoftcap float32
	// PostNorms reports that each block normalises its sub-layer outputs before the residual
	// add. The weights are nil on architectures that do not.
	PostNorms bool
	// SlidingPattern reports that only every second layer is bounded by WindowLen.
	SlidingPattern bool
	// LocalLayers, when non-empty, says layer by layer whether it is bounded by WindowLen.
	//
	// It overrides SlidingPattern because a per-file array can describe shapes the alternating
	// rule cannot: Gemma 3's file states one boolean per layer, and this checkpoint's 18 layers are
	// all local. Empty means the file did not say, and the architecture's rule applies.
	LocalLayers []bool
	// QKNorm reports that each head's query and key are normalised before the rotation.
	QKNorm bool
}

// activateGate applies the architecture's gated feed-forward activation in place.
//
// The gate is the one place a Llama-shaped and a Gemma-shaped block differ numerically without
// differing in shape, so getting it wrong loads cleanly and generates text that is merely
// wrong -- which is why it is selected from the architecture table rather than hardcoded.
func (cfg Config) activateGate(x []float32) {
	if cfg.FFNAct == ffnGELU {
		kernel.GELU(x)
		return
	}
	kernel.SiLU(x)
}

// activationName names the feed-forward gate, for the model summary.
func (a ffnAct) String() string {
	if a == ffnGELU {
		return "gelu"
	}
	return "silu"
}

// RopeLayout is a rotary embedding dimension pairing.
type RopeLayout int

const (
	// RopeNormal pairs dimension i with i+1, as Llama does.
	RopeNormal RopeLayout = iota
	// RopeNeox pairs dimension i with i+dims/2, as Qwen-family models do.
	RopeNeox
)

// String names the layout, for the model summary.
func (l RopeLayout) String() string {
	if l == RopeNeox {
		return "neox"
	}
	return "normal"
}

// archSpec is what each supported architecture changes about the forward pass.
//
// Only the differences are listed. Everything not mentioned here -- the pre-norm
// block structure, RMSNorm, the gated FFN, grouped-query attention -- is shared, and
// that sharing is the point of treating these as one implementation with a spec
// rather than two copies.
// ffnAct is the activation on the gated feed-forward branch.
type ffnAct uint8

const (
	// ffnSiLU is x/(1+exp(-x)), the SwiGLU gate the Llama family uses.
	ffnSiLU ffnAct = iota
	// ffnGELU is the tanh approximation of the Gaussian error linear unit, the GeGLU gate
	// Gemma uses. Swapping it changes every feed-forward output, so it is a per-architecture
	// property rather than something a config file can set.
	ffnGELU
)

type archSpec struct {
	// rope is the dimension pairing the model was trained with.
	rope RopeLayout
	// attnBias reports whether the attention projections carry bias vectors.
	attnBias bool
	// embedScale reports that token embeddings are multiplied by sqrt(hidden) on the way in.
	//
	// It cannot be folded into the embedding matrix the way normOnes is folded into the norm
	// weights: the same matrix is usually the tied output projection, and that projection must
	// see the unscaled weights.
	embedScale bool
	// ffn is the activation on the gated feed-forward branch.
	ffn ffnAct
	// postNorms reports that each block normalises its attention and feed-forward outputs
	// before adding them to the residual, as Gemma 2 does. The usual pre-norm block adds a
	// sub-layer's output directly; this one normalises it first, which is why the tensors have
	// to be present as well as the flag set.
	postNorms bool
	// slidingPattern reports that attention alternates between a bounded window and the whole
	// context, layer by layer. The window length itself is a metadata key; this says whether it
	// applies to every layer or to every second one.
	slidingPattern bool
	// qkNorm reports that each head's query and key are RMS-normalised over their own width
	// before the rotation, which is what Qwen3 adds to Qwen2's layout. The weights are one
	// vector of head-dim length per layer, shared by every head of that layer.
	qkNorm bool
}

// archSpecs lists the architectures this package implements.
//
// A model whose architecture is not here is refused at load time. That is
// deliberate: loading a Qwen2 checkpoint with Llama's rotation produces fluent
// English with the wrong numbers, which is far harder to notice than an error
// naming the architecture.
var archSpecs = map[string]archSpec{
	"llama":   {rope: RopeNormal, attnBias: false},
	"mistral": {rope: RopeNormal, attnBias: false},
	"qwen2":   {rope: RopeNeox, attnBias: true},
	// Qwen3 keeps Qwen2's rotation and drops its attention bias, but normalises every head's
	// query and key over the head width before rotating them. That step is invisible in the
	// tensor shapes -- the two bias vectors it replaces are the same size as its two norm
	// vectors -- so loading a Qwen3 file as qwen2 would run and answer from the wrong numbers.
	"qwen3": {rope: RopeNeox, attnBias: false, qkNorm: true},
	// Gemma differs from Llama in three ways that no packet listing shows: its rotary
	// embeddings pair dimension i with i+d/2, its token embeddings are scaled on the way in,
	// and its feed-forward gate is GELU. Loading one with Llama's settings produces fluent
	// English with the wrong distribution instead of an error, which is the failure this table
	// exists to prevent.
	//
	// The pairing is the one that looks like a mistake. Llama's files are converted with their
	// query and key rows permuted so that the adjacent-pair rotation reproduces the original
	// model, and Gemma's are not, so the same rotation applied to both gets one of them wrong.
	// Measured on Gemma 2B against llama.cpp: adjacent pairing put the top-8 log probabilities
	// 1.96 away, this pairing 0.19.
	"gemma": {
		rope: RopeNeox, attnBias: false,
		embedScale: true, ffn: ffnGELU,
	},
	// Gemma 2 adds three things on top of Gemma: both attention scores and output logits are
	// bounded by a softcap, each block normalises its sub-layer outputs before the residual
	// ("sandwich" norms), and only every second block is bounded by the sliding window. The two
	// cap values and the window length are metadata keys; what belongs here is that the blocks
	// are shaped that way at all.
	"gemma2": {
		rope: RopeNeox, attnBias: false,
		embedScale: true, ffn: ffnGELU,
		postNorms: true, slidingPattern: true,
	},
	// Gemma 3 keeps Gemma 2's block shape -- sandwich norms, GeGLU, scaled embeddings -- and
	// changes three things. It normalises each head's query and key, which is Qwen3's step. It drops
	// the attention softcap (Gemma 2 bounds its scores; Gemma 3 does not) but keeps the output one.
	// And its sliding window is a per-layer pattern read from the file rather than every second
	// layer, so a checkpoint whose layers are all local is described exactly rather than rounded to
	// a rule.
	//
	// It also rotates the local and global layers at different frequencies, which is the one piece
	// this package does not implement: see docs/models.md.
	"gemma3": {
		rope: RopeNeox, attnBias: false,
		embedScale: true, ffn: ffnGELU, qkNorm: true,
		postNorms: true, slidingPattern: true,
	},
}

// LoadConfig reads the architecture parameters and validates them.
//
// Validation is strict on purpose: a shape mismatch caught here produces a clear
// error, while one caught during the forward pass produces a slice-bounds panic
// with no indication of which hyperparameter was wrong.
func LoadConfig(f *gguf.File) (Config, error) {
	arch, err := f.Arch()
	if err != nil {
		return Config{}, err
	}
	spec, ok := archSpecs[arch]
	if !ok {
		return Config{}, fmt.Errorf("llama: unsupported architecture %q (supported: %s)", arch, supportedArchs())
	}

	p := configParser{arch: arch, file: f, spec: spec}
	cfg := Config{
		Arch:       arch,
		RopeLayout: spec.rope,
		AttnBias:   spec.attnBias,
		FFNAct:     spec.ffn,
		EmbedScale: 1,
		PostNorms:  spec.postNorms, SlidingPattern: spec.slidingPattern, QKNorm: spec.qkNorm,
	}
	if name, ok := f.Get[string]("general.name"); ok {
		cfg.Name = name
	}

	// Assembly is staged by concern, one function per stage, in the order the file implies:
	// dimensions first, because everything else is checked against them; then the head
	// geometry; then the rotation, the experts and the window; then the vocabulary, which comes
	// from a tensor rather than from a key. validate runs last, because comparing every tensor
	// shape is only meaningful once the numbers it compares against exist.
	for _, stage := range []func(*Config) error{
		p.dimensions,
		p.headGeometry,
		p.rotation,
		p.experts,
		p.window,
		p.vocabulary,
	} {
		if err := stage(&cfg); err != nil {
			return Config{}, err
		}
	}
	if err := cfg.validate(f); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

// configParser reads one architecture's metadata, spelling each key the way the file does.
//
// It exists for two reasons rather than as ceremony: every stage would otherwise repeat
// arch + "." + name and the "required and positive" rule, and a stage that got either wrong
// would fail on a file that is fine.
type configParser struct {
	arch string
	file *gguf.File
	// spec is the archSpecs entry for arch, carried so a stage can read a property the table
	// records (the embedding scale is the one that matters here) without a second lookup.
	spec archSpec
}

// get reads a required positive integer, naming the key as the file spells it.
func (p configParser) get(name string) (int, error) {
	v, ok := p.file.Get[int](p.arch + "." + name)
	if !ok {
		return 0, fmt.Errorf("llama: missing %s.%s", p.arch, name)
	}
	if v <= 0 {
		return 0, fmt.Errorf("llama: %s.%s = %d, must be positive", p.arch, name, v)
	}
	return v, nil
}

// dimensions reads the four numbers every later check is relative to.
func (p configParser) dimensions(cfg *Config) error {
	// One table rather than five ifs: the five keys are read the same way and differ only in
	// which field they land in, and a table makes a new one a single line instead of a
	// paragraph that could be written with the wrong error path.
	for _, key := range []struct {
		name string
		dst  *int
	}{
		{"embedding_length", &cfg.Hidden},
		{"block_count", &cfg.Layers},
		{"attention.head_count", &cfg.Heads},
		{"feed_forward_length", &cfg.FFN},
		{"context_length", &cfg.ContextLen},
	} {
		v, err := p.get(key.name)
		if err != nil {
			return err
		}
		*key.dst = v
	}

	return nil
}

// headGeometry reads how the attention heads are laid out.
func (p configParser) headGeometry(cfg *Config) error {
	arch, f := p.arch, p.file
	_, _ = arch, f
	// GQA models list fewer KV heads than query heads. When the key is absent
	// the model is multi-head, which is the same thing with equal counts.
	if kv, ok := f.Get[int](arch + ".attention.head_count_kv"); ok {
		if kv <= 0 {
			return fmt.Errorf("llama: %s.attention.head_count_kv = %d, must be positive", arch, kv)
		}
		cfg.KVHeads = kv
	} else {
		cfg.KVHeads = cfg.Heads
	}

	if cfg.Hidden%cfg.Heads != 0 {
		return fmt.Errorf("llama: embedding_length %d is not divisible by head_count %d", cfg.Hidden, cfg.Heads)
	}
	// The per-head width is usually Hidden/Heads, but not always: Gemma 2 has 2304 hidden and 8
	// heads with 256-wide queries and keys, so the attention projections are not square. A file
	// that states key_length overrides the derived value, and the tensor shapes below then
	// validate the result.
	cfg.HeadDim = cfg.Hidden / cfg.Heads
	if v, ok := f.Get[int](arch + ".attention.key_length"); ok {
		if v <= 0 || v > cfg.Hidden {
			return fmt.Errorf("llama: %s.attention.key_length = %d, outside (0, %d]",
				arch, v, cfg.Hidden)
		}
		cfg.HeadDim = v
	}
	// One width for q, k and v is what the rest of the package assumes, so a file that asks for
	// different key and value widths is refused rather than half-supported: the attention
	// output would be assembled from the wrong slice of the wrong length.
	if v, ok := f.Get[int](arch + ".attention.value_length"); ok && v != cfg.HeadDim {
		return fmt.Errorf("llama: %s.attention.value_length = %d, but key_length is %d; "+
			"equal widths are the only shape this package implements", arch, v, cfg.HeadDim)
	}
	if cfg.Heads%cfg.KVHeads != 0 {
		return fmt.Errorf("llama: head_count %d is not divisible by head_count_kv %d", cfg.Heads, cfg.KVHeads)
	}

	return nil
}

// rotation reads every rope key.
func (p configParser) rotation(cfg *Config) error {
	arch, f := p.arch, p.file
	_, _ = arch, f
	cfg.RopeDim = cfg.HeadDim
	if v, ok := f.Get[int](arch + ".rope.dimension_count"); ok {
		if v <= 0 || v > cfg.HeadDim {
			return fmt.Errorf("llama: rope.dimension_count %d outside (0, %d]", v, cfg.HeadDim)
		}
		cfg.RopeDim = v
	}
	// rope.dimension_count must be even: the rotation pairs dimensions.
	if cfg.RopeDim%2 != 0 {
		return fmt.Errorf("llama: rope.dimension_count %d is odd", cfg.RopeDim)
	}

	cfg.RopeBase = 10000
	if v, ok := f.Get[float32](arch + ".rope.freq_base"); ok {
		// "v <= 0" is false for NaN, so a NaN base passed this guard and became the
		// exponent's base: thetaScale was NaN, so every cos and sin past position 0 was
		// NaN. +Inf also passed, and it is the quieter of the two -- every frequency
		// collapses to 1, the rotation stops distinguishing positions, and nothing reports
		// it. Both are refused by naming the actual problem rather than a range.
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || v <= 0 {
			return fmt.Errorf("llama: rope.freq_base %v must be a finite positive number", v)
		}
		cfg.RopeBase = v
	}
	cfg.RopeScale = 1
	if v, ok := f.Get[float32](arch + ".rope.scale_linear"); ok {
		// A zero or negative scale keeps the default, which is this key's documented
		// behaviour: scaling by zero would collapse every position onto one, and the schema
		// has no value that means "off", so a non-positive one is read as absent.
		//
		// A non-finite one is not that. "v > 0" let +Inf through, and it is the worst of
		// the values here: the angle is +Inf, math.Cos(+Inf) is NaN, and the rotation is NaN
		// at every position including 0 -- measured cos[1][0] NaN against 0.5403023 for
		// scale 1. NaN was silently downgraded to the default, which is at least usable but
		// still means the file said something the loader discarded without saying so. Both
		// are refused now; the range cases keep their old behaviour.
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("llama: rope.scale_linear %v is not a finite number", v)
		}
		if v > 0 {
			cfg.RopeScale = v
		}
	}
	// Any scaling scheme other than the linear one changes the rotation itself.
	// Refusing is the honest option: applying the plain rotation to a
	// YaRN-trained model produces plausible but wrong text instead of an error.
	if t, ok := f.Get[string](arch + ".rope.scaling.type"); ok && t != "" && t != "linear" && t != "none" {
		return fmt.Errorf("llama: unsupported rope scaling type %q", t)
	}

	return nil
}

// experts reads the mixture-of-experts counts.
func (p configParser) experts(cfg *Config) error {
	arch, f := p.arch, p.file
	_, _ = arch, f
	// A file states its expert counts under the architecture namespace. The
	// tensors decide whether they are used, because some converters emit the counts
	// for a model that then stores a dense FFN; validate enforces that the two agree.
	if v, ok := f.Get[int](arch + ".expert_count"); ok && v > 0 {
		cfg.Experts = v
	}
	if v, ok := f.Get[int](arch + ".expert_used_count"); ok && v > 0 {
		cfg.ExpertUsed = v
		// A file that routes to more experts than it has is corrupt, and the failure is
		// silent without this check: selectExperts clamps k to the router's width, so a
		// used_count of 8 against 4 experts runs every expert and produces plausible text
		// from a model that was never trained to combine them all. The clamp stays -- it is
		// what keeps the kernel from indexing past the end -- but the file is refused here,
		// where the numbers can still be named.
		//
		// Checked against Experts rather than against the tensor shape because the shape is
		// what established Experts, in validate; this runs before that and its message
		// carries both numbers.
		if cfg.Experts > 0 && cfg.ExpertUsed > cfg.Experts {
			return fmt.Errorf("llama: %s.expert_used_count %d exceeds expert_count %d",
				arch, cfg.ExpertUsed, cfg.Experts)
		}
	}

	return nil
}

// window reads the sliding window and its per-layer pattern.
func (p configParser) window(cfg *Config) error {
	arch, f := p.arch, p.file
	_, _ = arch, f
	// The sliding window is per-architecture, and a missing key means unbounded: a
	// Mistral fine-tune that trained with full attention sets it to 0 or omits it.
	//
	// A value larger than the context is not rejected, because it is not wrong: a window
	// wider than every position the model can reach is the same as no window, and trimSpans
	// returns the spans unchanged once keep >= total. The cost of getting there is one
	// spanRows call per head, which is why it is noted rather than guarded -- a file that
	// writes the key as uint32 0xFFFFFFFF loads with WindowLen 4294967295 and attends
	// normally.
	if v, ok := f.Get[int](arch + ".attention.sliding_window"); ok && v > 0 {
		cfg.WindowLen = v
	}

	// The two caps are per-file numbers, so they are read by key rather than taken from the
	// architecture. A negative or non-finite value would invert or destroy every score, so both
	// are refused instead of being clamped to something that runs.
	for _, capKey := range []struct {
		name string
		dst  *float32
	}{
		{"attn_logit_softcapping", &cfg.AttnSoftcap},
		{"final_logit_softcapping", &cfg.FinalSoftcap},
	} {
		v, ok := f.Get[float32](arch + "." + capKey.name)
		if !ok {
			continue
		}
		if v <= 0 || math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("llama: %s.%s = %v, must be a finite positive number",
				arch, capKey.name, v)
		}
		*capKey.dst = v
	}

	// A per-layer window pattern is a file-level fact, like the window length itself: Gemma 3
	// stores one boolean per layer, and a file whose layers are all local cannot be described by
	// the alternating rule its architecture name suggests. Read here rather than assumed there.
	// The key is written two ways: an array of one boolean per layer, or a single integer giving
	// the period of an alternating pattern. llama.cpp reads the array first and falls back to the
	// integer; a file with neither takes the integer case with a period of zero, which marks every
	// layer local.
	//
	// That last case is Gemma 3 270M: it declares a window of 512 and no pattern at all, so all 18
	// of its layers are bounded. Reading its absence as "no window" or as "every second layer"
	// both change what the model computes.
	if err := p.localLayers(cfg); err != nil {
		return err
	}

	// Gemma 3 can rotate local and global layers at different frequencies, and this package does
	// not implement that: it keeps one rotation per model. Refused rather than served at the wrong
	// rate, because a wrong rotation is invisible -- the output stays fluent and every token is
	// slightly wrong -- and the file says plainly that two rates are wanted.
	if base, ok := f.Get[float32](arch + ".rope.freq_base_swa"); ok && base != cfg.RopeBase {
		return fmt.Errorf("llama: %s.rope.freq_base_swa = %g with a global base of %g: "+
			"two rotation frequencies are not implemented", arch, base, cfg.RopeBase)
	}

	// Gemma scales the embeddings by sqrt(hidden) on the way in. Computed here, once, so the
	// three lookup sites multiply by a constant instead of taking a square root per token.
	if p.spec.embedScale {
		cfg.EmbedScale = float32(math.Sqrt(float64(cfg.Hidden)))
	}

	cfg.RMSNormEps = 1e-5
	if v, ok := f.Get[float32](arch + ".attention.layer_norm_rms_epsilon"); ok {
		// Same gap as freq_base: a NaN epsilon is not "<= 0" and reached the norm, where
		// every output became NaN. A zero epsilon is already refused by the range check.
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || v <= 0 {
			return fmt.Errorf("llama: attention.layer_norm_rms_epsilon %v must be a finite positive number", v)
		}
		cfg.RMSNormEps = v
	}

	return nil
}

// localLayers decides which layers a windowed model bounds.
//
// The key is written three ways and llama.cpp reads them in this order: an array of one boolean
// per layer, then a single integer giving the period of an alternating pattern, then -- when both
// are absent -- a per-architecture convention. The third case is the dangerous one, which is why
// it is asserted on the architecture rather than applied to any windowed file: Gemma 3 declares a
// window of 512 and no pattern at all, and its converter leaves the key out only when every layer
// is local, while Gemma 2's files predate the key entirely and are described by the alternating
// rule. Reading one convention as the other bounded the wrong layers, and the difference is
// invisible in the output text.
func (p configParser) localLayers(cfg *Config) error {
	arch, f := p.arch, p.file

	if pattern, ok := f.GetArray[bool](arch + ".attention.sliding_window_pattern"); ok {
		if len(pattern) != cfg.Layers {
			return fmt.Errorf("llama: %s.attention.sliding_window_pattern holds %d "+
				"entries for %d layers", arch, len(pattern), cfg.Layers)
		}
		cfg.LocalLayers = pattern
	} else if n, ok := f.Get[int](arch + ".attention.sliding_window_pattern"); ok {
		if n < 0 {
			return fmt.Errorf("llama: %s.attention.sliding_window_pattern = %d, must "+
				"not be negative", arch, n)
		}
		// The period counts every n-th layer as global and the rest local, starting local, which
		// is llama.cpp's dense_first=false branch.
		cfg.LocalLayers = make([]bool, cfg.Layers)
		for l := range cfg.LocalLayers {
			cfg.LocalLayers[l] = n == 0 || l%n < n-1
		}
	} else if cfg.WindowLen > 0 && arch == "gemma3" {
		// No pattern key at all: llama.cpp's n_pattern == 0 case, where every layer is local.
		//
		// This is asserted per architecture rather than for any windowed one, because "the key is
		// absent" means different things: Gemma 3's converter always writes the array and leaves
		// it out only when every layer is local, while Gemma 2's files predate the key entirely
		// and are described by the alternating rule. Applying Gemma 3's reading to Gemma 2
		// bounded every layer instead of every second one.
		cfg.LocalLayers = make([]bool, cfg.Layers)
		for l := range cfg.LocalLayers {
			cfg.LocalLayers[l] = true
		}
	}

	return nil
}

// vocabulary reads what the tokenizer namespace and the embedding tensor say.
//
// arch and f are aliased here because this stage mixes both namespaces: the vocabulary comes
// from a tensor and the special tokens from tokenizer.ggml.*, so it does not go through p.get.
func (p configParser) vocabulary(cfg *Config) error {
	arch, f := p.arch, p.file
	_, _ = arch, f
	// The vocabulary size comes from the embedding tensor, not from a
	// hyperparameter: GGUF does not store one, and the tokenizer length can
	// legitimately differ from the padded embedding rows.
	//
	// The two directions are not equally safe, and the difference is why every entry point
	// checks token < VocabSize rather than trusting the tokenizer. Embedding rows beyond the
	// token count are ordinary padding -- a checkpoint rounded 32001 pieces up to 32064 so a
	// matmul dimension is a round number -- and nothing can reach them. Embedding rows short
	// of the token count is the dangerous direction: the tokenizer can return an id the model
	// has no row for, and the failure would be a slice index panic deep in the forward pass.
	// Measured: one past the last valid id is refused at the boundary with
	// "llama: token 363 outside the vocabulary (363)", at every entry point.
	//
	// Refusing the file when the tokenizer overruns is not an option this function has: it
	// reads the config, and the tokenizer is loaded separately. That is why the check lives
	// at the entry points instead of at load time.
	emb, ok := f.Tensor("token_embd.weight")
	if !ok {
		return fmt.Errorf("llama: missing token_embd.weight")
	}
	if len(emb.Dims) != 2 {
		return fmt.Errorf("llama: token_embd.weight has %d dims, want 2", len(emb.Dims))
	}
	if got := int(emb.Dims[0]); got != cfg.Hidden {
		return fmt.Errorf("llama: token_embd.weight row length %d, want embedding_length %d", got, cfg.Hidden)
	}
	// A zero-width embedding gives VocabSize 0, and the first forward pass then produces an
	// empty logits row: samplers are written for one row per position, and Temperature.Next
	// indexes the last candidate when floating-point accumulation falls short of 1, so an
	// empty row panics there. Refused at load time, where the shape can be named, rather than
	// left for a sampler to trip over.
	if emb.Dims[1] == 0 {
		return fmt.Errorf("llama: token_embd.weight has a zero-width vocabulary")
	}
	cfg.VocabSize = int(emb.Dims[1])

	_, hasOutput := f.Tensor("output.weight")
	cfg.TiedEmbeddings = !hasOutput

	// Special tokens live under the tokenizer namespace, not the architecture
	// one, because they belong to the vocabulary rather than to the block
	// structure.
	cfg.EOSID, cfg.BOSID = -1, -1
	if v, ok := f.Get[int]("tokenizer.ggml.eos_token_id"); ok {
		cfg.EOSID = v
	}
	if v, ok := f.Get[int]("tokenizer.ggml.bos_token_id"); ok {
		cfg.BOSID = v
	}

	return nil
}

// validate checks the shapes of every tensor the forward pass will touch.
//
// Doing this once at load time is what turns "index out of range" into a message
// naming the tensor and both shapes.
func (c Config) validate(f *gguf.File) error {
	type want struct {
		name string
		dims []uint64
	}
	wants := []want{
		{"output_norm.weight", []uint64{uint64(c.Hidden)}},
		{"output.weight", []uint64{uint64(c.Hidden), uint64(c.VocabSize)}},
	}
	// Every layer contributes at least one tensor of its own, so a block count larger than the
	// file's tensor count cannot be satisfied -- and building the expectation list for it is not
	// free: the loop below allocates a few strings per layer, so a header claiming 2^30 layers
	// spends minutes and gigabytes before the first missing-tensor error can be produced. The
	// count is checked up front so the failure is immediate and the allocation never happens.
	if c.Layers > f.TensorCount() {
		return fmt.Errorf("llama: block_count %d exceeds the %d tensors in the file",
			c.Layers, f.TensorCount())
	}
	for l := range c.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		wants = append(wants,
			want{p + "attn_norm.weight", []uint64{uint64(c.Hidden)}},
			want{p + "attn_q.weight", []uint64{uint64(c.Hidden), uint64(c.Heads * c.HeadDim)}},
			want{p + "attn_k.weight", []uint64{uint64(c.Hidden), uint64(c.KVHeads * c.HeadDim)}},
			want{p + "attn_v.weight", []uint64{uint64(c.Hidden), uint64(c.KVHeads * c.HeadDim)}},
			want{p + "attn_output.weight", []uint64{uint64(c.Heads * c.HeadDim), uint64(c.Hidden)}},
			want{p + "ffn_norm.weight", []uint64{uint64(c.Hidden)}},
		)
		if c.QKNorm {
			wants = append(wants,
				want{p + "attn_q_norm.weight", []uint64{uint64(c.HeadDim)}},
				want{p + "attn_k_norm.weight", []uint64{uint64(c.HeadDim)}},
			)
		}
		if c.Experts > 0 {
			// The expert tensors stack one matrix per expert in the outermost
			// dimension, which is why their shapes carry the expert count as a third
			// dimension rather than the hidden width first.
			e := uint64(c.Experts)
			wants = append(wants,
				want{p + "ffn_gate_inp.weight", []uint64{uint64(c.Hidden), e}},
				want{p + "ffn_gate_exps.weight", []uint64{uint64(c.Hidden), uint64(c.FFN), e}},
				want{p + "ffn_up_exps.weight", []uint64{uint64(c.Hidden), uint64(c.FFN), e}},
				want{p + "ffn_down_exps.weight", []uint64{uint64(c.FFN), uint64(c.Hidden), e}},
			)
		} else {
			wants = append(wants,
				want{p + "ffn_gate.weight", []uint64{uint64(c.Hidden), uint64(c.FFN)}},
				want{p + "ffn_up.weight", []uint64{uint64(c.Hidden), uint64(c.FFN)}},
				want{p + "ffn_down.weight", []uint64{uint64(c.FFN), uint64(c.Hidden)}},
			)
		}
		if c.AttnBias {
			wants = append(wants,
				want{p + "attn_q.bias", []uint64{uint64(c.Heads * c.HeadDim)}},
				want{p + "attn_k.bias", []uint64{uint64(c.KVHeads * c.HeadDim)}},
				want{p + "attn_v.bias", []uint64{uint64(c.KVHeads * c.HeadDim)}},
			)
		}
	}
	for _, w := range wants {
		ti, ok := f.Tensor(w.name)
		if !ok {
			if w.name == "output.weight" && c.TiedEmbeddings {
				// A tied model omits this tensor by definition.
				continue
			}
			return fmt.Errorf("llama: missing tensor %s", w.name)
		}
		if len(ti.Dims) != len(w.dims) {
			return fmt.Errorf("llama: %s has %d dims %v, want %d", w.name, len(ti.Dims), ti.Dims, len(w.dims))
		}
		for i := range w.dims {
			if ti.Dims[i] != w.dims[i] {
				return fmt.Errorf("llama: %s has shape %v, want %v", w.name, ti.Dims, w.dims)
			}
		}
	}
	return nil
}

// String renders the configuration for a model summary.
func (c Config) String() string {
	tie := "separate output projection"
	if c.TiedEmbeddings {
		tie = "tied embeddings"
	}
	bias := ""
	if c.AttnBias {
		bias = ", attention bias"
	}
	// The architecture differences are named rather than left implicit in the entrypoint's
	// name, because they are exactly what a reader comparing two checkpoints cannot see: the
	// same shapes and the same tensor names describe a Llama block and a Gemma block that
	// differ in three numbers.
	extra := ""
	if c.EmbedScale != 1 && c.EmbedScale != 0 {
		extra += fmt.Sprintf(", embeddings scaled by %g", c.EmbedScale)
	}
	if c.FFNAct != ffnSiLU {
		extra += ", " + c.FFNAct.String() + " gate"
	}
	if c.AttnSoftcap != 0 || c.FinalSoftcap != 0 {
		extra += fmt.Sprintf(", softcaps %g/%g", c.AttnSoftcap, c.FinalSoftcap)
	}
	if c.QKNorm {
		extra += ", qk norm"
	}
	if c.PostNorms {
		extra += ", sandwich norms"
	}
	// The window is per layer when the file (or the architecture's rule) said so, so the summary
	// counts rather than describing a rule: "every second layer" was wrong for a file whose layers
	// are all local, and a reader cannot check a claim they cannot see the data for.
	if c.WindowLen > 0 {
		local := 0
		for l := range c.Layers {
			if windowFor(c, l) > 0 {
				local++
			}
		}
		switch local {
		case c.Layers:
			extra += fmt.Sprintf(", every layer bounded to %d", c.WindowLen)
		case 0:
			// WindowLen set but no layer bounded: worth saying, because the two together look
			// like a configuration mistake even when they are not.
			extra += fmt.Sprintf(", no layer bounded to %d", c.WindowLen)
		default:
			extra += fmt.Sprintf(", %d of %d layers bounded to %d", local, c.Layers, c.WindowLen)
		}
	}
	return fmt.Sprintf(
		"%s: %d layers, %d hidden, %d heads (%d kv), head dim %d, ffn %d, ctx %d, vocab %d, rope dim %d base %g %s, eps %g, %s%s%s",
		c.Arch, c.Layers, c.Hidden, c.Heads, c.KVHeads, c.HeadDim, c.FFN,
		c.ContextLen, c.VocabSize, c.RopeDim, c.RopeBase, c.RopeLayout, c.RMSNormEps, tie, bias, extra,
	)
}

// supportedArchs lists the architectures in sorted order, for an error message.
func supportedArchs() string {
	names := make([]string, 0, len(archSpecs))
	for name := range archSpecs {
		names = append(names, name)
	}
	slices.Sort(names)
	return strings.Join(names, ", ")
}
