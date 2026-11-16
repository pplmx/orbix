package llama

import (
	"fmt"

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

	// FFN is llama.feed_forward_length.
	FFN int

	// ContextLen is llama.context_length, the trained window this package
	// allocates a KV cache for.
	ContextLen int

	// RopeDim is llama.rope.dimension_count, the number of dimensions the
	// rotation is applied to. It is usually but not always equal to HeadDim.
	RopeDim int

	// RopeBase is llama.rope.freq_base, the theta of the rotation.
	RopeBase float32

	// RopeScale scales the rotation frequency. Some fine-tunes set it below 1
	// to extend the usable context; the default is 1.
	RopeScale float32

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
	if arch != "llama" {
		return Config{}, fmt.Errorf("llama: unsupported architecture %q", arch)
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

	cfg := Config{Arch: arch}
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
	for l := range c.Layers {
		p := fmt.Sprintf("blk.%d.", l)
		wants = append(wants,
			want{p + "attn_norm.weight", []uint64{uint64(c.Hidden)}},
			want{p + "attn_q.weight", []uint64{uint64(c.Hidden), uint64(c.Heads * c.HeadDim)}},
			want{p + "attn_k.weight", []uint64{uint64(c.Hidden), uint64(c.KVHeads * c.HeadDim)}},
			want{p + "attn_v.weight", []uint64{uint64(c.Hidden), uint64(c.KVHeads * c.HeadDim)}},
			want{p + "attn_output.weight", []uint64{uint64(c.Heads * c.HeadDim), uint64(c.Hidden)}},
			want{p + "ffn_norm.weight", []uint64{uint64(c.Hidden)}},
			want{p + "ffn_gate.weight", []uint64{uint64(c.Hidden), uint64(c.FFN)}},
			want{p + "ffn_up.weight", []uint64{uint64(c.Hidden), uint64(c.FFN)}},
			want{p + "ffn_down.weight", []uint64{uint64(c.FFN), uint64(c.Hidden)}},
		)
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
	return fmt.Sprintf(
		"%s: %d layers, %d hidden, %d heads (%d kv), head dim %d, ffn %d, ctx %d, vocab %d, rope dim %d base %g, eps %g, %s",
		c.Arch, c.Layers, c.Hidden, c.Heads, c.KVHeads, c.HeadDim, c.FFN,
		c.ContextLen, c.VocabSize, c.RopeDim, c.RopeBase, c.RMSNormEps, tie,
	)
}
