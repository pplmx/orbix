package llama

import (
	"fmt"
	"slices"
	"strings"

	"github.com/pplmx/orbix/internal/gguf"
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
type archSpec struct {
	// rope is the dimension pairing the model was trained with.
	rope RopeLayout
	// attnBias reports whether the attention projections carry bias vectors.
	attnBias bool
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

	get := func(name string) (int, error) {
		v, ok := f.Get[int](arch + "." + name)
		if !ok {
			return 0, fmt.Errorf("llama: missing %s.%s", arch, name)
		}
		if v <= 0 {
			return 0, fmt.Errorf("llama: %s.%s = %d, must be positive", arch, name, v)
		}
		return v, nil
	}

	cfg := Config{Arch: arch, RopeLayout: spec.rope, AttnBias: spec.attnBias}
	if name, ok := f.Get[string]("general.name"); ok {
		cfg.Name = name
	}
	if cfg.Hidden, err = get("embedding_length"); err != nil {
		return Config{}, err
	}
	if cfg.Layers, err = get("block_count"); err != nil {
		return Config{}, err
	}
	if cfg.Heads, err = get("attention.head_count"); err != nil {
		return Config{}, err
	}
	if cfg.FFN, err = get("feed_forward_length"); err != nil {
		return Config{}, err
	}
	if cfg.ContextLen, err = get("context_length"); err != nil {
		return Config{}, err
	}

	// GQA models list fewer KV heads than query heads. When the key is absent
	// the model is multi-head, which is the same thing with equal counts.
	if kv, ok := f.Get[int](arch + ".attention.head_count_kv"); ok {
		if kv <= 0 {
			return Config{}, fmt.Errorf("llama: %s.attention.head_count_kv = %d, must be positive", arch, kv)
		}
		cfg.KVHeads = kv
	} else {
		cfg.KVHeads = cfg.Heads
	}

	if cfg.Hidden%cfg.Heads != 0 {
		return Config{}, fmt.Errorf("llama: embedding_length %d is not divisible by head_count %d", cfg.Hidden, cfg.Heads)
	}
	cfg.HeadDim = cfg.Hidden / cfg.Heads
	if cfg.Heads%cfg.KVHeads != 0 {
		return Config{}, fmt.Errorf("llama: head_count %d is not divisible by head_count_kv %d", cfg.Heads, cfg.KVHeads)
	}

	cfg.RopeDim = cfg.HeadDim
	if v, ok := f.Get[int](arch + ".rope.dimension_count"); ok {
		if v <= 0 || v > cfg.HeadDim {
			return Config{}, fmt.Errorf("llama: rope.dimension_count %d outside (0, %d]", v, cfg.HeadDim)
		}
		cfg.RopeDim = v
	}
	// rope.dimension_count must be even: the rotation pairs dimensions.
	if cfg.RopeDim%2 != 0 {
		return Config{}, fmt.Errorf("llama: rope.dimension_count %d is odd", cfg.RopeDim)
	}

	cfg.RopeBase = 10000
	if v, ok := f.Get[float32](arch + ".rope.freq_base"); ok {
		if v <= 0 {
			return Config{}, fmt.Errorf("llama: rope.freq_base %v must be positive", v)
		}
		cfg.RopeBase = v
	}
	cfg.RopeScale = 1
	if v, ok := f.Get[float32](arch + ".rope.scale_linear"); ok && v > 0 {
		cfg.RopeScale = v
	}
	// Any scaling scheme other than the linear one changes the rotation itself.
	// Refusing is the honest option: applying the plain rotation to a
	// YaRN-trained model produces plausible but wrong text instead of an error.
	if t, ok := f.Get[string](arch + ".rope.scaling.type"); ok && t != "" && t != "linear" && t != "none" {
		return Config{}, fmt.Errorf("llama: unsupported rope scaling type %q", t)
	}

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
			return Config{}, fmt.Errorf("llama: %s.expert_used_count %d exceeds expert_count %d",
				arch, cfg.ExpertUsed, cfg.Experts)
		}
	}

	// The sliding window is per-architecture, and a missing key means unbounded: a
	// Mistral fine-tune that trained with full attention sets it to 0 or omits it.
	if v, ok := f.Get[int](arch + ".attention.sliding_window"); ok && v > 0 {
		cfg.WindowLen = v
	}

	cfg.RMSNormEps = 1e-5
	if v, ok := f.Get[float32](arch + ".attention.layer_norm_rms_epsilon"); ok {
		if v <= 0 {
			return Config{}, fmt.Errorf("llama: attention.layer_norm_rms_epsilon %v must be positive", v)
		}
		cfg.RMSNormEps = v
	}

	// The vocabulary size comes from the embedding tensor, not from a
	// hyperparameter: GGUF does not store one, and the tokenizer length can
	// legitimately differ from the padded embedding rows.
	emb, ok := f.Tensor("token_embd.weight")
	if !ok {
		return Config{}, fmt.Errorf("llama: missing token_embd.weight")
	}
	if len(emb.Dims) != 2 {
		return Config{}, fmt.Errorf("llama: token_embd.weight has %d dims, want 2", len(emb.Dims))
	}
	if got := int(emb.Dims[0]); got != cfg.Hidden {
		return Config{}, fmt.Errorf("llama: token_embd.weight row length %d, want embedding_length %d", got, cfg.Hidden)
	}
	// A zero-width embedding gives VocabSize 0, and the first forward pass then produces an
	// empty logits row: samplers are written for one row per position, and Temperature.Next
	// indexes the last candidate when floating-point accumulation falls short of 1, so an
	// empty row panics there. Refused at load time, where the shape can be named, rather than
	// left for a sampler to trip over.
	if emb.Dims[1] == 0 {
		return Config{}, fmt.Errorf("llama: token_embd.weight has a zero-width vocabulary")
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

	if err := cfg.validate(f); err != nil {
		return Config{}, err
	}
	return cfg, nil
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
	return fmt.Sprintf(
		"%s: %d layers, %d hidden, %d heads (%d kv), head dim %d, ffn %d, ctx %d, vocab %d, rope dim %d base %g %s, eps %g, %s%s",
		c.Arch, c.Layers, c.Hidden, c.Heads, c.KVHeads, c.HeadDim, c.FFN,
		c.ContextLen, c.VocabSize, c.RopeDim, c.RopeBase, c.RopeLayout, c.RMSNormEps, tie, bias,
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
