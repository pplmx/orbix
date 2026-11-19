package llama

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
)

// buildFile assembles a GGUF file in memory. The llama tests need files whose
// metadata and tensor directory disagree in specific ways, which is easier to
// express by construction than by shipping fixtures.
type kv struct {
	key string
	val gguf.Value
}

func buildFile(kvs []kv, tensors []gguf.TensorInfo, data []byte) []byte {
	var b bytes.Buffer
	u32 := func(v uint32) {
		var tmp [4]byte
		binary.LittleEndian.PutUint32(tmp[:], v)
		b.Write(tmp[:])
	}
	u64 := func(v uint64) {
		var tmp [8]byte
		binary.LittleEndian.PutUint64(tmp[:], v)
		b.Write(tmp[:])
	}
	str := func(s string) {
		u64(uint64(len(s)))
		b.WriteString(s)
	}
	b.WriteString(gguf.Magic)
	u32(gguf.Version)
	u64(uint64(len(tensors)))
	u64(uint64(len(kvs)))
	for _, e := range kvs {
		str(e.key)
		u32(uint32(e.val.Type))
		encodeValue(&b, e.val)
	}
	for _, ti := range tensors {
		str(ti.Name)
		u32(uint32(len(ti.Dims)))
		for _, d := range ti.Dims {
			u64(d)
		}
		u32(uint32(ti.Type))
		u64(ti.Offset)
	}
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	b.Write(data)
	return b.Bytes()
}

func encodeValue(b *bytes.Buffer, v gguf.Value) {
	u32 := func(x uint32) {
		var tmp [4]byte
		binary.LittleEndian.PutUint32(tmp[:], x)
		b.Write(tmp[:])
	}
	u64 := func(x uint64) {
		var tmp [8]byte
		binary.LittleEndian.PutUint64(tmp[:], x)
		b.Write(tmp[:])
	}
	switch v.Type {
	case gguf.TypeUint32:
		u32(uint32(v.Uint))
	case gguf.TypeFloat32:
		u32(math.Float32bits(float32(v.Float)))
	case gguf.TypeString:
		u64(uint64(len(v.Str)))
		b.WriteString(v.Str)
	default:
		panic("encodeValue: unsupported type")
	}
}

func strVal(s string) gguf.Value  { return gguf.Value{Type: gguf.TypeString, Str: s} }
func u32Val(v uint32) gguf.Value  { return gguf.Value{Type: gguf.TypeUint32, Uint: uint64(v)} }
func f32Val(v float32) gguf.Value { return gguf.Value{Type: gguf.TypeFloat32, Float: float64(v)} }

// tinyConfig is a 2-layer, 2-head model with a 4-wide head: small enough that a
// full tensor directory fits in a test, complete enough to exercise every shape
// check.
type tinyConfig struct {
	hidden, layers, heads, kvHeads, ffn, ctx, vocab int
	tied                                            bool
	// experts and usedAbove are non-zero for a mixture-of-experts fixture.
	experts int
	used    int
	// ropeFreqs, when non-nil, is written as a rope_freqs.weight tensor with these values.
	// Its length is whatever the caller gave, so a wrong-size tensor is expressible.
	ropeFreqs []float32
}

func (c tinyConfig) headDim() int { return c.hidden / c.heads }

func (c tinyConfig) kvs() []kv {
	out := []kv{
		{"general.architecture", strVal("llama")},
		{"general.name", strVal("tiny")},
		{"llama.embedding_length", u32Val(uint32(c.hidden))},
		{"llama.block_count", u32Val(uint32(c.layers))},
		{"llama.attention.head_count", u32Val(uint32(c.heads))},
		{"llama.attention.head_count_kv", u32Val(uint32(c.kvHeads))},
		{"llama.feed_forward_length", u32Val(uint32(c.ffn))},
		{"llama.context_length", u32Val(uint32(c.ctx))},
		{"llama.rope.dimension_count", u32Val(uint32(c.headDim()))},
		{"llama.attention.layer_norm_rms_epsilon", f32Val(1e-5)},
	}
	if c.experts > 0 {
		out = append(out,
			kv{"llama.expert_count", u32Val(uint32(c.experts))},
			kv{"llama.expert_used_count", u32Val(uint32(c.used))},
		)
	}
	return out
}

// TestExpertUsedCountMustNotExceedExpertCount covers the check that the file's two expert
// numbers agree.
//
// Nothing compared them, and the failure was silent: selectExperts clamps its k to the
// router's width, so a file declaring expert_count 4 and expert_used_count 8 ran all four
// experts for every token. That is not a crash and not garbage -- it is a mixture-of-experts
// model evaluated in a configuration it was never trained for, producing text that reads
// fine. Measured before the guard: the file loaded with Experts=4 ExpertUsed=8.
//
// used == experts is legal and must stay accepted: routing every expert is a real
// configuration, if an expensive one.
// TestZeroWidthVocabularyIsRefused covers the shape that makes every logits row empty.
//
// Nothing checked VocabSize, and a file whose token_embd.weight is [hidden, 0] loaded with
// VocabSize 0. The consequence is one layer down: a forward pass returns an empty logits
// row, and Temperature.Next panics on it at the fallback that indexes the last candidate --
// the case where float64 accumulation of the probabilities lands a hair below 1. Greedy
// returns 0 there without complaining, so the two samplers disagree about what an empty row
// means, which is a reason to refuse the shape rather than to teach both to tolerate it.
//
// Measured before the check: LoadConfig accepted the file with VocabSize=0.
func TestZeroWidthVocabularyIsRefused(t *testing.T) {
	c := tinyConfig{hidden: 8, layers: 1, heads: 2, kvHeads: 2, ffn: 16, ctx: 32, vocab: 0}
	f, err := gguf.Read(bytes.NewReader(buildFile(c.kvs(), c.tensors(), nil)))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	_, err = LoadConfig(f)
	if err == nil {
		t.Fatal("LoadConfig accepted a zero-width vocabulary; samplers index that empty row")
	}
	// The message has to name the tensor, since the shape is spread over the metadata and
	// the embedding is what actually carries the vocabulary width.
	for _, want := range []string{"token_embd.weight", "vocabulary"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}

	// A one-token vocabulary is degenerate but legal, and must still load: the rule is about
	// zero width, not about small vocabularies.
	c.vocab = 1
	f, err = gguf.Read(bytes.NewReader(buildFile(c.kvs(), c.tensors(), nil)))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	cfg, err := LoadConfig(f)
	if err != nil {
		t.Fatalf("LoadConfig refused a one-token vocabulary: %v", err)
	}
	if cfg.VocabSize != 1 {
		t.Errorf("VocabSize = %d, want 1", cfg.VocabSize)
	}
}

func TestExpertUsedCountMustNotExceedExpertCount(t *testing.T) {
	cases := []struct {
		name    string
		experts int
		used    int
		wantErr bool
		why     string
	}{
		{"sparse routing", 4, 2, false, "the ordinary case"},
		{"all experts", 4, 4, false, "routing to every expert is legal"},
		{"more used than exist", 4, 8, true, "the router cannot address experts the file does not hold"},
		{"one past the count", 4, 5, true, "the boundary is exclusive"},
		{"dense model, stray used key", 0, 3, false, "a dense FFN ignores expert_used_count entirely"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cfg := tinyConfig{
				hidden: 8, layers: 1, heads: 2, kvHeads: 2, ffn: 16, ctx: 32, vocab: 11,
				experts: c.experts, used: c.used,
			}
			f, err := gguf.Read(bytes.NewReader(buildFile(cfg.kvs(), cfg.tensors(), nil)))
			if err != nil {
				t.Fatalf("Read: %v", err)
			}
			got, err := LoadConfig(f)
			if c.wantErr {
				if err == nil {
					t.Fatalf("LoadConfig accepted expert_used_count %d against expert_count %d, "+
						"Experts=%d ExpertUsed=%d (%s)", c.used, c.experts, got.Experts, got.ExpertUsed, c.why)
				}
				// Both numbers have to appear, or the message does not say what to fix.
				for _, want := range []string{"expert_used_count", "expert_count"} {
					if !strings.Contains(err.Error(), want) {
						t.Errorf("error %q does not mention %q", err, want)
					}
				}
				return
			}
			if err != nil {
				t.Fatalf("LoadConfig refused a legal configuration: %v (%s)", err, c.why)
			}
		})
	}
}

func (c tinyConfig) tensors() []gguf.TensorInfo {
	h := uint64(c.hidden)
	out := []gguf.TensorInfo{
		{Name: "token_embd.weight", Dims: []uint64{h, uint64(c.vocab)}, Type: gguf.F32},
		{Name: "output_norm.weight", Dims: []uint64{h}, Type: gguf.F32},
	}
	if !c.tied {
		out = append(out, gguf.TensorInfo{Name: "output.weight", Dims: []uint64{h, uint64(c.vocab)}, Type: gguf.F32})
	}
	if c.ropeFreqs != nil {
		out = append(out, gguf.TensorInfo{
			Name: "rope_freqs.weight", Dims: []uint64{uint64(len(c.ropeFreqs))}, Type: gguf.F32,
		})
	}
	for i := range c.layers {
		p := "blk." + string(rune('0'+i)) + "."
		out = append(out,
			gguf.TensorInfo{Name: p + "attn_norm.weight", Dims: []uint64{h}, Type: gguf.F32},
			gguf.TensorInfo{Name: p + "attn_q.weight", Dims: []uint64{h, h}, Type: gguf.F32},
			gguf.TensorInfo{Name: p + "attn_k.weight", Dims: []uint64{h, uint64(c.kvHeads * c.headDim())}, Type: gguf.F32},
			gguf.TensorInfo{Name: p + "attn_v.weight", Dims: []uint64{h, uint64(c.kvHeads * c.headDim())}, Type: gguf.F32},
			gguf.TensorInfo{Name: p + "attn_output.weight", Dims: []uint64{h, h}, Type: gguf.F32},
			gguf.TensorInfo{Name: p + "ffn_norm.weight", Dims: []uint64{h}, Type: gguf.F32},
		)
		if c.experts > 0 {
			e := uint64(c.experts)
			out = append(out,
				gguf.TensorInfo{Name: p + "ffn_gate_inp.weight", Dims: []uint64{h, e}, Type: gguf.F32},
				gguf.TensorInfo{Name: p + "ffn_gate_exps.weight", Dims: []uint64{h, uint64(c.ffn), e}, Type: gguf.F32},
				gguf.TensorInfo{Name: p + "ffn_up_exps.weight", Dims: []uint64{h, uint64(c.ffn), e}, Type: gguf.F32},
				gguf.TensorInfo{Name: p + "ffn_down_exps.weight", Dims: []uint64{uint64(c.ffn), h, e}, Type: gguf.F32},
			)
			continue
		}
		out = append(out,
			gguf.TensorInfo{Name: p + "ffn_gate.weight", Dims: []uint64{h, uint64(c.ffn)}, Type: gguf.F32},
			gguf.TensorInfo{Name: p + "ffn_up.weight", Dims: []uint64{h, uint64(c.ffn)}, Type: gguf.F32},
			gguf.TensorInfo{Name: p + "ffn_down.weight", Dims: []uint64{uint64(c.ffn), h}, Type: gguf.F32},
		)
	}
	return out
}

func defaultTiny() tinyConfig {
	return tinyConfig{hidden: 8, layers: 2, heads: 2, kvHeads: 2, ffn: 16, ctx: 32, vocab: 11}
}

func parseTiny(t *testing.T, c tinyConfig) *gguf.File {
	t.Helper()
	f, err := gguf.Read(bytes.NewReader(buildFile(c.kvs(), c.tensors(), nil)))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	return f
}

func TestLoadConfig(t *testing.T) {
	f := parseTiny(t, defaultTiny())
	cfg, err := LoadConfig(f)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.Layers != 2 || cfg.Hidden != 8 || cfg.Heads != 2 || cfg.KVHeads != 2 {
		t.Errorf("config = %+v", cfg)
	}
	if cfg.HeadDim != 4 {
		t.Errorf("HeadDim = %d, want 4", cfg.HeadDim)
	}
	if cfg.VocabSize != 11 {
		t.Errorf("VocabSize = %d, want 11", cfg.VocabSize)
	}
	if cfg.TiedEmbeddings {
		t.Error("TiedEmbeddings = true for a model with output.weight")
	}
}

func TestLoadConfigTiedEmbeddings(t *testing.T) {
	c := defaultTiny()
	c.tied = true
	f := parseTiny(t, c)
	cfg, err := LoadConfig(f)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if !cfg.TiedEmbeddings {
		t.Error("TiedEmbeddings = false for a model without output.weight")
	}
}

func TestLoadConfigKVHeadsDefault(t *testing.T) {
	// A model that omits head_count_kv is multi-head: KV heads equal query
	// heads. Treating the absence as zero would fail validation instead.
	c := defaultTiny()
	kvs := c.kvs()
	kept := kvs[:0]
	for _, e := range kvs {
		if e.key != "llama.attention.head_count_kv" {
			kept = append(kept, e)
		}
	}
	f, err := gguf.Read(bytes.NewReader(buildFile(kept, c.tensors(), nil)))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	cfg, err := LoadConfig(f)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.KVHeads != cfg.Heads {
		t.Errorf("KVHeads = %d, want %d", cfg.KVHeads, cfg.Heads)
	}
}

func TestLoadConfigRejects(t *testing.T) {
	// Each case removes or corrupts exactly one thing, so the error message
	// has to name it; a generic failure would make a broken model file
	// indistinguishable from a bug in this package.
	tests := []struct {
		name   string
		mutate func(*tinyConfig) []kv
		want   string
	}{
		{
			name: "unsupported architecture",
			mutate: func(c *tinyConfig) []kv {
				kvs := c.kvs()
				// qwen2 is supported now; the test needs an architecture this
				// package genuinely does not implement.
				kvs[0] = kv{"general.architecture", strVal("gemma")}
				return kvs
			},
			want: "unsupported architecture",
		},
		{
			name: "missing embedding_length",
			mutate: func(c *tinyConfig) []kv {
				return removeKey(c.kvs(), "llama.embedding_length")
			},
			want: "missing llama.embedding_length",
		},
		{
			name: "zero block count",
			mutate: func(c *tinyConfig) []kv {
				return replaceKey(c.kvs(), "llama.block_count", u32Val(0))
			},
			want: "must be positive",
		},
		{
			name: "hidden not divisible by heads",
			mutate: func(c *tinyConfig) []kv {
				return replaceKey(c.kvs(), "llama.attention.head_count", u32Val(3))
			},
			want: "not divisible",
		},
		{
			// Zero is not merely invalid, it is the value the divisibility check two lines
			// below would divide by: without this guard, head_count % 0 panics rather than
			// returning an error. A file that writes the key with zero is the shape this
			// catches, and it is not hypothetical -- a converter that defaults the key
			// without reading it produces exactly that.
			name: "zero kv heads",
			mutate: func(c *tinyConfig) []kv {
				return replaceKey(c.kvs(), "llama.attention.head_count_kv", u32Val(0))
			},
			want: "must be positive",
		},
		{
			name: "kv heads not dividing heads",
			mutate: func(c *tinyConfig) []kv {
				return replaceKey(c.kvs(), "llama.attention.head_count_kv", u32Val(3))
			},
			want: "not divisible",
		},
		{
			name: "rope dim larger than head dim",
			mutate: func(c *tinyConfig) []kv {
				return replaceKey(c.kvs(), "llama.rope.dimension_count", u32Val(6))
			},
			want: "outside",
		},
		{
			name: "odd rope dim",
			mutate: func(c *tinyConfig) []kv {
				return replaceKey(c.kvs(), "llama.rope.dimension_count", u32Val(3))
			},
			want: "odd",
		},
		{
			name: "non-positive rope base",
			mutate: func(c *tinyConfig) []kv {
				return append(c.kvs(), kv{"llama.rope.freq_base", f32Val(0)})
			},
			want: "must be positive",
		},
		{
			name: "unsupported rope scaling",
			mutate: func(c *tinyConfig) []kv {
				return append(c.kvs(), kv{"llama.rope.scaling.type", strVal("yarn")})
			},
			want: "unsupported rope scaling",
		},
		{
			name: "non-positive epsilon",
			mutate: func(c *tinyConfig) []kv {
				return replaceKey(c.kvs(), "llama.attention.layer_norm_rms_epsilon", f32Val(0))
			},
			want: "must be positive",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := defaultTiny()
			kvs := tc.mutate(&c)
			f, err := gguf.Read(bytes.NewReader(buildFile(kvs, c.tensors(), nil)))
			if err != nil {
				t.Fatalf("Read: %v", err)
			}
			_, err = LoadConfig(f)
			if err == nil {
				t.Fatal("LoadConfig succeeded, want an error")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error = %v, want it to mention %q", err, tc.want)
			}
		})
	}
}

func TestLoadConfigRejectsBadTensorShapes(t *testing.T) {
	// A model whose metadata is self-consistent but whose tensors disagree must
	// be rejected at load time, naming the tensor. Catching it in the forward
	// pass would show up as a slice-bounds panic with no context.
	c := defaultTiny()
	tensors := c.tensors()
	for i := range tensors {
		if tensors[i].Name == "blk.0.attn_q.weight" {
			tensors[i].Dims = []uint64{8, 5}
		}
	}
	f, err := gguf.Read(bytes.NewReader(buildFile(c.kvs(), tensors, nil)))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	_, err = LoadConfig(f)
	if err == nil {
		t.Fatal("LoadConfig succeeded with a wrong attn_q shape")
	}
	if !strings.Contains(err.Error(), "attn_q.weight") {
		t.Errorf("error = %v, want it to name the tensor", err)
	}
}

func TestLoadConfigRejectsMissingTensor(t *testing.T) {
	c := defaultTiny()
	tensors := c.tensors()[:0]
	for _, ti := range c.tensors() {
		if ti.Name != "blk.1.ffn_down.weight" {
			tensors = append(tensors, ti)
		}
	}
	f, err := gguf.Read(bytes.NewReader(buildFile(c.kvs(), tensors, nil)))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	_, err = LoadConfig(f)
	if err == nil {
		t.Fatal("LoadConfig succeeded with a missing tensor")
	}
	if !strings.Contains(err.Error(), "blk.1.ffn_down.weight") {
		t.Errorf("error = %v, want it to name the missing tensor", err)
	}
}

func removeKey(kvs []kv, key string) []kv {
	out := kvs[:0:0]
	for _, e := range kvs {
		if e.key != key {
			out = append(out, e)
		}
	}
	return out
}

func replaceKey(kvs []kv, key string, val gguf.Value) []kv {
	out := append([]kv(nil), kvs...)
	for i := range out {
		if out[i].key == key {
			out[i].val = val
		}
	}
	return out
}

func TestConfigString(t *testing.T) {
	cfg := testConfig()
	s := cfg.String()
	// The summary is what a user sees first; it must carry the numbers that
	// explain the model's cost.
	for _, want := range []string{"llama", "8 hidden", "2 heads", "16"} {
		if !strings.Contains(s, want) {
			t.Errorf("String() = %q, missing %q", s, want)
		}
	}
}

// buildTinyWeights writes a complete tiny model to disk with deterministic
// weights, so the forward pass can be tested without a downloaded checkpoint.
func buildTinyWeights(t *testing.T, c tinyConfig) string {
	t.Helper()

	var data []byte
	next := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			// Small deterministic values: large enough to exercise the math,
			// small enough that a 2-layer model does not saturate.
			out[i] = float32((i%17)-8) / 64
		}
		data = appendFloat32s(data, out)
		return out
	}

	// Lay out tensors in the same order as the directory below.
	tensors := c.tensors()
	offset := 0
	dir := make([]gguf.TensorInfo, len(tensors))
	for i, ti := range tensors {
		dir[i] = ti
		dir[i].Offset = uint64(offset)
		dir[i].Type = gguf.F32
		n := int64(1)
		for _, d := range ti.Dims {
			n *= int64(d)
		}
		// Values a test chose override the shared sequence, so a NaN or a specific factor
		// reaches the loader unchanged. The sequence would otherwise make every tensor
		// well-formed and every value arbitrary.
		if ti.Name == "rope_freqs.weight" && c.ropeFreqs != nil {
			data = appendFloat32s(data, c.ropeFreqs)
			offset += int(n) * 4
			continue
		}
		next(int(n))
		offset += int(n) * 4
	}

	path := t.TempDir() + "/tiny.gguf"
	if err := os.WriteFile(path, buildFile(c.kvs(), dir, data), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func appendFloat32s(dst []byte, vals []float32) []byte {
	for _, v := range vals {
		var tmp [4]byte
		binary.LittleEndian.PutUint32(tmp[:], math.Float32bits(v))
		dst = append(dst, tmp[:]...)
	}
	return dst
}

func TestForwardTinyModel(t *testing.T) {
	c := defaultTiny()
	path := buildTinyWeights(t, c)
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	cache, err := NewKVCache(cfg, c.ctx)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}

	logits, err := m.Forward(cache, 0, 0)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if len(logits) != cfg.VocabSize {
		t.Fatalf("logits = %d, want %d", len(logits), cfg.VocabSize)
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logit[%d] = %v, want a finite number", i, v)
		}
	}
	if got := cache.Len(); got != 1 {
		t.Errorf("cache len = %d, want 1", got)
	}
}

func TestForwardRejectsBadArguments(t *testing.T) {
	c := defaultTiny()
	path := buildTinyWeights(t, c)
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	cache, err := NewKVCache(cfg, 4)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := m.Forward(cache, cfg.VocabSize, 0); err == nil {
		t.Error("Forward accepted a token past the vocabulary")
	}
	if _, err := m.Forward(cache, -1, 0); err == nil {
		t.Error("Forward accepted a negative token")
	}
	if _, err := m.Forward(cache, 0, 4); err == nil {
		t.Error("Forward accepted a position past the cache")
	}
	// A cache that does not hold exactly pos positions must be refused: silently
	// accepting it would attend over stale slots.
	if _, err := m.Forward(cache, 0, 2); err == nil {
		t.Error("Forward accepted a position ahead of the cache length")
	}
}

func TestForwardSequence(t *testing.T) {
	c := defaultTiny()
	path := buildTinyWeights(t, c)
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	cache, err := NewKVCache(cfg, 8)
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int{0, 1, 2, 3}
	for pos, tok := range tokens {
		if _, err := m.Forward(cache, tok, pos); err != nil {
			t.Fatalf("Forward(%d, %d): %v", tok, pos, err)
		}
	}
	if got := cache.Len(); got != len(tokens) {
		t.Errorf("cache len = %d, want %d", got, len(tokens))
	}

	// Forward returns a buffer the model owns, so the comparison needs its own
	// copy: the next call overwrites the returned slice.
	withPrefix := append([]float32(nil), m.logits...)

	// The last position's logits must differ from what a context-free pass
	// would give, which proves attention reads the cache.
	solo, err := NewKVCache(cfg, 8)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Forward(solo, tokens[len(tokens)-1], 0); err != nil {
		t.Fatal(err)
	}
	identical := true
	for i := range withPrefix {
		if m.logits[i] != withPrefix[i] {
			identical = false
			break
		}
	}
	if identical {
		t.Error("logits are unchanged by the prefix; attention ignores the cache")
	}
}

func TestForwardIsDeterministic(t *testing.T) {
	c := defaultTiny()
	path := buildTinyWeights(t, c)
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()

	run := func() []float32 {
		cache, err := NewKVCache(cfg, 4)
		if err != nil {
			t.Fatal(err)
		}
		var last []float32
		for pos, tok := range []int{0, 1, 2} {
			last, err = m.Forward(cache, tok, pos)
			if err != nil {
				t.Fatal(err)
			}
		}
		return append([]float32(nil), last...)
	}
	a, b := run(), run()
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("logit[%d] = %v then %v; the forward pass is not deterministic", i, a[i], b[i])
		}
	}
}

func TestGreedySampler(t *testing.T) {
	logits := []float32{-1, 5, 3, 5, 0}
	// Ties resolve to the first maximum, which keeps output reproducible.
	if got := (Greedy{}).Next(logits); got != 1 {
		t.Errorf("Greedy.Next = %d, want 1", got)
	}
	// A NaN must not win.
	if got := (Greedy{}).Next([]float32{1, float32(math.NaN()), 2}); got != 2 {
		t.Errorf("Greedy.Next with a NaN = %d, want 2", got)
	}
}

func TestTemperatureZeroIsGreedy(t *testing.T) {
	logits := []float32{1, 9, 2}
	s := Temperature{Temp: 0, NextFloat: func() float64 { return 0.99 }}
	if got := s.Next(logits); got != 1 {
		t.Errorf("Temperature{0}.Next = %d, want the greedy choice 1", got)
	}
}

func TestTemperatureTopKRespected(t *testing.T) {
	// With top-k 1 the sampler has only one candidate, so the result is the
	// argmax regardless of the random draw.
	logits := []float32{1, 9, 2, 3}
	for _, r := range []float64{0, 0.5, 0.999} {
		s := Temperature{Temp: 1, TopK: 1, NextFloat: func() float64 { return r }}
		if got := s.Next(logits); got != 1 {
			t.Errorf("r=%v: Next = %d, want 1", r, got)
		}
	}
}

func TestGenerateStopsAtMaxTokens(t *testing.T) {
	c := defaultTiny()
	path := buildTinyWeights(t, c)
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	cache, err := NewKVCache(cfg, 32)
	if err != nil {
		t.Fatal(err)
	}
	out, err := m.Generate(cache, []int32{0}, Greedy{}, GenerateOptions{MaxTokens: 5})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	// The prompt plus at most MaxTokens; EOS may cut it short.
	if len(out) > 6 {
		t.Errorf("Generate produced %d tokens, want at most 6", len(out))
	}
	if len(out) == 0 || out[0] != 0 {
		t.Errorf("Generate output does not start with the prompt: %v", out)
	}
}

func TestGenerateRejectsEmptyPrompt(t *testing.T) {
	c := defaultTiny()
	path := buildTinyWeights(t, c)
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cache, err := NewKVCache(m.Config(), 8)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Generate(cache, nil, Greedy{}, GenerateOptions{MaxTokens: 1}); err == nil {
		t.Error("Generate accepted an empty prompt")
	}
}

func TestGenerateStopsAtContextLimit(t *testing.T) {
	c := defaultTiny()
	path := buildTinyWeights(t, c)
	m, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	// A cache only large enough for the prompt and three more tokens.
	cache, err := NewKVCache(cfg, 4)
	if err != nil {
		t.Fatal(err)
	}
	out, err := m.Generate(cache, []int32{0}, Greedy{}, GenerateOptions{MaxTokens: 100})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(out) > 4 {
		t.Errorf("Generate produced %d tokens with a 4-position cache", len(out))
	}
}

func TestLoadConfigMistral(t *testing.T) {
	// Mistral is Llama's architecture with a sliding window and a different
	// tokenizer: same rotation, no attention biases. Registering it with Qwen2's
	// rotation would produce fluent text with the wrong numbers.
	c := defaultTiny()
	kvs := c.kvs()
	for i := range kvs {
		if kvs[i].key == "general.architecture" {
			kvs[i].val = strVal("mistral")
			continue
		}
		if after, ok := strings.CutPrefix(kvs[i].key, "llama."); ok {
			kvs[i].key = "mistral." + after
		}
	}
	kvs = append(kvs, kv{"mistral.attention.sliding_window", u32Val(4)})

	f, err := gguf.Read(bytes.NewReader(buildFile(kvs, c.tensors(), nil)))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	cfg, err := LoadConfig(f)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.Arch != "mistral" {
		t.Errorf("Arch = %q, want mistral", cfg.Arch)
	}
	if cfg.RopeLayout != RopeNormal {
		t.Errorf("RopeLayout = %v, want RopeNormal", cfg.RopeLayout)
	}
	if cfg.AttnBias {
		t.Error("AttnBias = true; Mistral has no attention biases")
	}
	if cfg.WindowLen != 4 {
		t.Errorf("WindowLen = %d, want 4", cfg.WindowLen)
	}
}

func TestLoadConfigWindowLenDefaultsToUnbounded(t *testing.T) {
	// A model that omits sliding_window attends over its whole context. Reading the
	// absence as zero and then trimming to zero positions would leave attention with
	// nothing to look at.
	f := parseTiny(t, defaultTiny())
	cfg, err := LoadConfig(f)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.WindowLen != 0 {
		t.Errorf("WindowLen = %d, want 0 (unbounded)", cfg.WindowLen)
	}
}

func TestLoadConfigWindowLenIgnoresNonPositive(t *testing.T) {
	// Published Mistral fine-tunes exist with sliding_window set to 0 to mean "no
	// window", which is the same thing as omitting it.
	c := defaultTiny()
	kvs := append(c.kvs(), kv{"llama.attention.sliding_window", u32Val(0)})
	f, err := gguf.Read(bytes.NewReader(buildFile(kvs, c.tensors(), nil)))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	cfg, err := LoadConfig(f)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.WindowLen != 0 {
		t.Errorf("WindowLen = %d, want 0", cfg.WindowLen)
	}
}

// TestBlockCountIsBoundedByTensorCount covers a header whose layer count is far larger than the
// file can possibly contain.
//
// Every layer contributes at least one tensor, so a block count above the file's tensor count
// cannot be satisfied -- and finding that out the obvious way is expensive: LoadConfig builds an
// expectation list with a handful of strings per layer, so a header claiming 2^30 layers spends
// minutes and gigabytes before the first missing-tensor error is reachable. Measured before the
// bound was added: the call had not returned after 60 seconds. After it: 10ms, naming both numbers.
//
// A corrupted download produces this, which is why the header is not assumed honest.
func TestBlockCountIsBoundedByTensorCount(t *testing.T) {
	// A file whose block count is far beyond its tensor count. The tensors are the tiny
	// fixture's, so the mismatch is the only thing wrong with it.
	c := tinyConfig{hidden: 8, layers: 1 << 20, heads: 2, kvHeads: 2, ffn: 16, ctx: 64, vocab: 8}
	f, err := gguf.Read(bytes.NewReader(buildFile(c.kvs(), tinyConfig{hidden: 8, layers: 1, heads: 2, kvHeads: 2, ffn: 16, ctx: 64, vocab: 8}.tensors(), nil)))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if n := f.TensorCount(); n >= c.layers {
		t.Fatalf("fixture has %d tensors, not fewer than %d layers", n, c.layers)
	}
	_, err = LoadConfig(f)
	if err == nil {
		t.Fatal("a block count larger than the tensor count was accepted")
	}
	if !strings.Contains(err.Error(), "block_count") {
		t.Errorf("error %q does not name block_count", err)
	}
	// Both numbers belong in the message: the declared count alone does not tell a user that the
	// file is short, which is the part that says the download is bad.
	if !strings.Contains(err.Error(), "tensors") {
		t.Errorf("error %q does not say how many tensors the file holds", err)
	}
}
