package llama

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
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
		// wantAll are values the message has to carry as well: the numbers a caller needs to
		// know which side of a comparison was wrong. Asserting only the category word leaves a
		// message that says "not divisible" and nothing else passing.
		wantAll []string
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
			// hidden=8, heads=3.
			wantAll: []string{"8", "3"},
		},
		{
			// The same hazard one line above: embedding_length % head_count with a zero
			// divisor. It is caught by the shared positivity check in get rather than by a
			// guard written for it, which is why it is worth a case of its own -- the
			// protection is a property of a helper two hundred lines away, and a later edit
			// that reads this key directly would remove it without touching anything here.
			// Measured: with head_count 0, LoadConfig returns
			// "llama.attention.head_count = 0, must be positive" and does not panic.
			name: "zero heads",
			mutate: func(c *tinyConfig) []kv {
				return replaceKey(c.kvs(), "llama.attention.head_count", u32Val(0))
			},
			want: "must be positive",
			// The key and the value, so a reader can tell which of several positivity
			// messages fired.
			wantAll: []string{"head_count", "0"},
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
			// heads=2, kv heads=3.
			wantAll: []string{"2", "3"},
		},
		{
			name: "rope dim larger than head dim",
			mutate: func(c *tinyConfig) []kv {
				return replaceKey(c.kvs(), "llama.rope.dimension_count", u32Val(6))
			},
			want: "outside",
			// The value asked for and the head dimension it must not exceed: 6 and 4.
			wantAll: []string{"6", "4"},
		},
		{
			name: "odd rope dim",
			mutate: func(c *tinyConfig) []kv {
				return replaceKey(c.kvs(), "llama.rope.dimension_count", u32Val(3))
			},
			want:    "odd",
			wantAll: []string{"3"},
		},
		{
			name: "non-positive rope base",
			mutate: func(c *tinyConfig) []kv {
				return append(c.kvs(), kv{"llama.rope.freq_base", f32Val(0)})
			},
			want: "must be a finite positive number",
		},
		{
			// "v <= 0" is false for NaN, so this used to load and make every cos and sin
			// past position 0 NaN. The exponent's base was the NaN itself.
			name: "NaN rope base",
			mutate: func(c *tinyConfig) []kv {
				return append(c.kvs(), kv{"llama.rope.freq_base", f32Val(float32(math.NaN()))})
			},
			want:    "must be a finite positive number",
			wantAll: []string{"NaN"},
		},
		{
			// The quieter half: +Inf does not produce NaN, it collapses every frequency to
			// 1, so the rotation stops telling positions apart and nothing reports it.
			name: "infinite rope base",
			mutate: func(c *tinyConfig) []kv {
				return append(c.kvs(), kv{"llama.rope.freq_base", f32Val(float32(math.Inf(1)))})
			},
			want:    "must be a finite positive number",
			wantAll: []string{"+Inf"},
		},
		{
			// The old check was "ok && v > 0", so every unusable value silently became
			// the default 1: the model ran at positions the header did not describe.
			name: "NaN rope scale",
			mutate: func(c *tinyConfig) []kv {
				return append(c.kvs(), kv{"llama.rope.scale_linear", f32Val(float32(math.NaN()))})
			},
			want:    "is not a finite number",
			wantAll: []string{"NaN"},
		},
		{
			// +Inf passed the old check and poisoned every position, including 0:
			// the angle is +Inf and math.Cos(+Inf) is NaN.
			name: "infinite rope scale",
			mutate: func(c *tinyConfig) []kv {
				return append(c.kvs(), kv{"llama.rope.scale_linear", f32Val(float32(math.Inf(1)))})
			},
			want:    "is not a finite number",
			wantAll: []string{"+Inf"},
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
			want: "must be a finite positive number",
		},
		{
			// Same gap as the rope base: a NaN epsilon is not "<=" anything, so the norm
			// divided by it and every output became NaN.
			name: "NaN epsilon",
			mutate: func(c *tinyConfig) []kv {
				return replaceKey(c.kvs(), "llama.attention.layer_norm_rms_epsilon",
					f32Val(float32(math.NaN())))
			},
			want:    "must be a finite positive number",
			wantAll: []string{"NaN"},
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
			for _, w := range tc.wantAll {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("error = %v, want it to state %q", err, w)
				}
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

// TestConfigStringVariants covers the two flags Config.String appends.
//
// The summary is the first thing a user reads -- bench and serve both print it -- and it is where
// a model's shape differences are visible. Both variants were unexecuted: TestConfigString checks
// the numbers on one config and asserts nothing about the two trailing fields, so a run whose
// summary said "separate output projection" for a tied model, or omitted the attention bias, would
// have looked correct.
//
// Tied embeddings change what is loaded and how much memory the output projection costs;
// attention bias is the Qwen2 shape and is what separates two models with otherwise identical
// numbers.
func TestConfigStringVariants(t *testing.T) {
	base := testConfig()

	t.Run("separate projection by default", func(t *testing.T) {
		c := base
		c.TiedEmbeddings, c.AttnBias = false, false
		s := c.String()
		if !strings.Contains(s, "separate output projection") {
			t.Errorf("String() = %q, want it to say the projection is separate", s)
		}
		if strings.Contains(s, "tied") {
			t.Errorf("String() = %q, mentions tying for an untied model", s)
		}
		if strings.Contains(s, "attention bias") {
			t.Errorf("String() = %q, mentions bias for a model without one", s)
		}
	})

	t.Run("tied embeddings", func(t *testing.T) {
		c := base
		c.TiedEmbeddings = true
		s := c.String()
		if !strings.Contains(s, "tied embeddings") {
			t.Errorf("String() = %q, want it to say the embeddings are tied", s)
		}
		if strings.Contains(s, "separate output projection") {
			t.Errorf("String() = %q, says both tied and separate", s)
		}
	})

	t.Run("attention bias", func(t *testing.T) {
		c := base
		c.AttnBias = true
		s := c.String()
		if !strings.Contains(s, ", attention bias") {
			t.Errorf("String() = %q, want it to name the attention bias", s)
		}
	})

	t.Run("both", func(t *testing.T) {
		// The two suffixes are appended in one format string, so the pair has to read correctly
		// too: ", attention bias" after "tied embeddings".
		c := base
		c.TiedEmbeddings, c.AttnBias = true, true
		s := c.String()
		if !strings.Contains(s, "tied embeddings, attention bias") {
			t.Errorf("String() = %q, want the two suffixes adjacent", s)
		}
	})
}

// TestRopeLayoutString covers both names RopeLayout.String can return.
//
// It lands inside Config.String, which bench and serve print, so "normal" printed for a neox
// model would misreport the layout the rope is actually using -- and the two layouts pair
// different dimensions, so the number would explain nothing about the output.
func TestRopeLayoutString(t *testing.T) {
	if got := RopeNormal.String(); got != "normal" {
		t.Errorf("RopeNormal.String() = %q, want normal", got)
	}
	if got := RopeNeox.String(); got != "neox" {
		t.Errorf("RopeNeox.String() = %q, want neox", got)
	}
	// And the value reaches the summary, which is where a reader sees it.
	c := testConfig()
	c.RopeLayout = RopeNeox
	if s := c.String(); !strings.Contains(s, "neox") {
		t.Errorf("String() = %q, want it to name the neox layout", s)
	}
}

// TestLoadConfigRejectsWrongTensorShape covers the two shape checks in Config.validate.
//
// The missing-tensor test next to this one covers the third: a name that is absent entirely.
// A tensor that is present with the wrong shape is the more likely corruption -- a truncated
// conversion, a converter that wrote 7B dimensions into a 1B file -- and neither the dimension
// count nor the dimension values had a test. Downgrading the count check to `<` and the value
// check to `<` both passed the llama and engine suites.
//
// Both failures are worth naming precisely: a loader that accepted a narrow attn_q would run a
// forward pass over a matrix whose width does not match the activation row, which reads the
// wrong memory rather than failing.
func TestLoadConfigRejectsWrongTensorShape(t *testing.T) {
	c := defaultTiny()
	cases := []struct {
		name   string
		mutate func([]gguf.TensorInfo) []gguf.TensorInfo
		want   []string
	}{
		{
			name: "a dimension value is wrong",
			mutate: func(ts []gguf.TensorInfo) []gguf.TensorInfo {
				out := append([]gguf.TensorInfo(nil), ts...)
				for i := range out {
					if out[i].Name == "blk.0.attn_q.weight" {
						// A fresh slice: the helper's dims are shared with the fixture.
						out[i].Dims = []uint64{out[i].Dims[0] + 1, out[i].Dims[1]}
					}
				}
				return out
			},
			want: []string{"blk.0.attn_q.weight", "has shape"},
		},
		{
			name: "the dimension count is wrong",
			mutate: func(ts []gguf.TensorInfo) []gguf.TensorInfo {
				out := append([]gguf.TensorInfo(nil), ts...)
				for i := range out {
					if out[i].Name == "blk.0.attn_q.weight" {
						out[i].Dims = []uint64{out[i].Dims[0], out[i].Dims[1], 1}
					}
				}
				return out
			},
			want: []string{"blk.0.attn_q.weight", "dims"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f, err := gguf.Read(bytes.NewReader(buildFile(c.kvs(), tc.mutate(c.tensors()), nil)))
			if err != nil {
				t.Fatalf("Read: %v", err)
			}
			_, err = LoadConfig(f)
			if err == nil {
				t.Fatal("LoadConfig accepted a tensor with the wrong shape")
			}
			for _, w := range tc.want {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("error %q does not mention %q", err, w)
				}
			}
		})
	}
}

// TestBlockCountBoundaryIsTheTensorCount pins the comparison, not just the refusal.
//
// The existing test uses 1<<20 layers against a handful of tensors, which any threshold in that
// neighbourhood catches -- doubling the bound to TensorCount()*2 passed the whole llama and
// engine suites. The boundary only matters where the two differ: a file whose block count is
// just past its tensor count.
//
// At exactly TensorCount() layers the guard must NOT fire, because the count is satisfiable in
// principle and the per-tensor check is what reports the real problem; at one past it the guard
// must fire, and before allocating the expectation list. The allocation is the reason the guard
// exists -- building the list for a header claiming 2^30 layers spends minutes and gigabytes
// before the first missing-tensor error can be produced.
func TestBlockCountBoundaryIsTheTensorCount(t *testing.T) {
	// Five tensors: two global weight tensors plus a couple of unrelated names, so the file
	// declares no block tensors at all and every single-layer expectation is missing.
	tensors := []gguf.TensorInfo{
		{Name: "token_embd.weight", Dims: []uint64{8, 8}, Type: gguf.F32},
		{Name: "output_norm.weight", Dims: []uint64{8}, Type: gguf.F32},
		{Name: "output.weight", Dims: []uint64{8, 8}, Type: gguf.F32},
		{Name: "unrelated_a", Dims: []uint64{8}, Type: gguf.F32},
		{Name: "unrelated_b", Dims: []uint64{8}, Type: gguf.F32},
	}
	const count = 5

	load := func(layers int) error {
		t.Helper()
		c := tinyConfig{hidden: 8, layers: layers, heads: 2, kvHeads: 2, ffn: 16, ctx: 64, vocab: 8}
		f, err := gguf.Read(bytes.NewReader(buildFile(c.kvs(), tensors, nil)))
		if err != nil {
			t.Fatalf("Read: %v", err)
		}
		if got := f.TensorCount(); got != count {
			t.Fatalf("fixture has %d tensors, want %d", got, count)
		}
		_, err = LoadConfig(f)
		return err
	}

	// Equal to the tensor count: the bound does not fire, so the failure is the per-tensor
	// check naming a block tensor.
	err := load(count)
	if err == nil {
		t.Fatal("a file with no block tensors was accepted")
	}
	if strings.Contains(err.Error(), "block_count") {
		t.Errorf("at exactly %d layers the block_count guard fired: %v", count, err)
	}
	if !strings.Contains(err.Error(), "missing tensor") {
		t.Errorf("error %q is not the missing-tensor report the guard is meant to leave room for", err)
	}

	// One past it: the guard fires, and does so before asking for any block tensor.
	err = load(count + 1)
	if err == nil {
		t.Fatal("a block count larger than the tensor count was accepted")
	}
	if !strings.Contains(err.Error(), "block_count") {
		t.Errorf("error %q does not name block_count", err)
	}
	if strings.Contains(err.Error(), "missing tensor") {
		t.Errorf("the guard did not run first: %v", err)
	}
}

// TestQwen2WithoutBiasTensorsIsRefused covers the shape check the AttnBias flag drives.
//
// The architecture table says qwen2 has attention bias, so LoadConfig requires the three bias
// tensors. A file that names qwen2 and omits them is not a file this engine can serve -- the
// projections would silently run without vectors the checkpoint was trained with, which changes
// every number and fails nothing.
//
// The missing-tensor test above removes ffn_down.weight, which the arch table requires for every
// architecture; this one is specific to the bias list, and neutering that list's shape checks left
// the suite green.
func TestQwen2WithoutBiasTensorsIsRefused(t *testing.T) {
	// AttnBias false writes a qwen2 file with no bias tensors at all.
	path := gguftest.Model{
		Arch:       "qwen2",
		AttnBias:   false,
		NoEOS:      true,
		ContextLen: 128,
	}.Write(t, t.TempDir())

	_, err := New(path)
	if err == nil {
		t.Fatal("a qwen2 file with no attention bias was accepted")
	}
	if !strings.Contains(err.Error(), "attn_q.bias") {
		t.Errorf("error %q does not name the missing bias tensor", err)
	}

	// The same file with the tensors present loads, so the refusal is about the omission rather
	// than about qwen2 being unsupported.
	ok := gguftest.Model{
		Arch:       "qwen2",
		AttnBias:   true,
		NoEOS:      true,
		ContextLen: 128,
	}.Write(t, t.TempDir())
	if _, err := New(ok); err != nil {
		t.Fatalf("a qwen2 file with attention bias did not load: %v", err)
	}
}

// TestGQABiasWidthsAreCheckedPerTensor covers the k and v bias shapes on a model whose key/value
// heads are fewer than its query heads.
//
// The check requires KVHeads*HeadDim for k and v, Heads*HeadDim for q. With Heads equal to KVHeads
// -- which every other fixture in this package has -- the two constraints are the same, so widening
// the k requirement to the query width changed nothing measurable. A 4-head model with 2 kv heads
// makes the two widths different numbers, and a file whose k bias is the query width has to be
// refused.
func TestGQABiasWidthsAreCheckedPerTensor(t *testing.T) {
	c := tinyConfig{hidden: 16, layers: 1, heads: 4, kvHeads: 2, ffn: 32, ctx: 32, vocab: 11}
	headDim := c.headDim() // 16/4 = 4, so q is 16 wide and k/v are 8 wide.

	// qwen2, because that is the architecture the bias table marks as biased.
	kvs := []kv{{"general.architecture", strVal("qwen2")}, {"general.name", strVal("gqa")}}
	for _, k := range c.kvs()[2:] {
		// Rewrite the llama.<key> prefix on the metadata the config reader looks for.
		name, ok := strings.CutPrefix(k.key, "llama.")
		if !ok {
			t.Fatalf("unexpected key %q", k.key)
		}
		kvs = append(kvs, kv{"qwen2." + name, k.val})
	}

	// Correct widths first, so a refusal below is about the shape.
	good := c.tensors()
	for i := range c.layers {
		p := "blk." + string(rune('0'+i)) + "."
		good = append(good,
			gguf.TensorInfo{Name: p + "attn_q.bias", Dims: []uint64{uint64(c.heads * headDim)}, Type: gguf.F32},
			gguf.TensorInfo{Name: p + "attn_k.bias", Dims: []uint64{uint64(c.kvHeads * headDim)}, Type: gguf.F32},
			gguf.TensorInfo{Name: p + "attn_v.bias", Dims: []uint64{uint64(c.kvHeads * headDim)}, Type: gguf.F32},
		)
	}
	if _, err := LoadConfig(mustParse(t, buildFile(kvs, good, nil))); err != nil {
		t.Fatalf("a GQA file with correct bias widths was refused: %v", err)
	}

	// The k bias widened to the query width: same value count as q, different from what k needs.
	// Only the k tensor changes, so the failure has to be attributed to it.
	wide := append([]gguf.TensorInfo(nil), good...)
	for i := range wide {
		if strings.HasSuffix(wide[i].Name, ".attn_k.bias") {
			wide[i].Dims = []uint64{uint64(c.heads * headDim)}
		}
	}
	_, err := LoadConfig(mustParse(t, buildFile(kvs, wide, nil)))
	if err == nil {
		t.Fatal("a GQA file with a query-width k bias was accepted")
	}
	if !strings.Contains(err.Error(), "attn_k.bias") {
		t.Errorf("error %q does not name the k bias tensor", err)
	}
}

// mustParse reads a built file or fails.
func mustParse(t *testing.T, raw []byte) *gguf.File {
	t.Helper()
	f, err := gguf.Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	return f
}

// TestWindowLargerThanTheContextIsAccepted pins what an oversized sliding window does.
//
// sliding_window is unsigned in GGUF, so a file can write it as 0xFFFFFFFF. Get[int] converts
// that to 4294967295 -- it is below MaxInt64, so the overflow guard does not apply -- and the
// loader keeps it rather than clamping. That is deliberate: a window wider than every position
// the model can reach is the same as no window, and trimSpans returns its input unchanged once
// keep >= total.
//
// The alternative, rejecting or clamping, would have to pick a context length to clamp against
// at load time, and the assertion here is that not doing so changes nothing observable.
func TestWindowLargerThanTheContextIsAccepted(t *testing.T) {
	c := defaultTiny()
	kvs := append(c.kvs(), kv{"llama.attention.sliding_window", u32Val(0xFFFFFFFF)})
	f, err := gguf.Read(bytes.NewReader(buildFile(kvs, c.tensors(), nil)))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	cfg, err := LoadConfig(f)
	if err != nil {
		t.Fatalf("LoadConfig refused an oversized window: %v", err)
	}
	if want := int(0xFFFFFFFF); cfg.WindowLen != want {
		t.Errorf("WindowLen = %d, want %d", cfg.WindowLen, want)
	}
	// Larger than any context the fixture declares, which is the property that makes it a
	// no-op rather than a bound.
	if cfg.WindowLen <= cfg.ContextLen {
		t.Errorf("the window (%d) is not larger than the context (%d), so this case does not "+
			"exercise what it claims", cfg.WindowLen, cfg.ContextLen)
	}
}

// TestLoadConfigRejectsBadEmbeddingShape covers the two checks on token_embd.weight.
//
// The embedding table is the one tensor whose shape the config file itself determines, so a
// disagreement means the metadata and the weights came from different models. Both messages carry
// numbers -- how many dimensions were found, how long a row is, and what was expected -- and neither
// had a test: the existing shape test corrupts attn_q instead.
//
// Catching this at load time keeps the failure here. In the forward pass it surfaces as a
// slice-bounds panic inside a matmul, which names nothing.
func TestLoadConfigRejectsBadEmbeddingShape(t *testing.T) {
	t.Run("wrong number of dimensions", func(t *testing.T) {
		c := defaultTiny()
		tensors := c.tensors()
		for i := range tensors {
			if tensors[i].Name == "token_embd.weight" {
				// A vector instead of a matrix.
				tensors[i].Dims = []uint64{uint64(c.vocab)}
			}
		}
		f, err := gguf.Read(bytes.NewReader(buildFile(c.kvs(), tensors, nil)))
		if err != nil {
			t.Fatalf("Read: %v", err)
		}
		_, err = LoadConfig(f)
		if err == nil {
			t.Fatal("LoadConfig accepted a one-dimensional token_embd.weight")
		}
		if !strings.Contains(err.Error(), "token_embd.weight") {
			t.Errorf("error = %v, want it to name the tensor", err)
		}
		// Both the count found and the count expected.
		for _, want := range []string{"1", "2"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error = %v, want it to state %q", err, want)
			}
		}
	})

	t.Run("wrong row length", func(t *testing.T) {
		c := defaultTiny()
		tensors := c.tensors()
		for i := range tensors {
			if tensors[i].Name == "token_embd.weight" {
				// Two-dimensional, but the rows are one wider than embedding_length. The shape is
				// [hidden, vocab], which is what the fixture writes at its own definition.
				tensors[i].Dims = []uint64{uint64(c.hidden + 1), uint64(c.vocab)}
			}
		}
		f, err := gguf.Read(bytes.NewReader(buildFile(c.kvs(), tensors, nil)))
		if err != nil {
			t.Fatalf("Read: %v", err)
		}
		_, err = LoadConfig(f)
		if err == nil {
			t.Fatal("LoadConfig accepted a token_embd.weight with the wrong row length")
		}
		// The row found and the embedding_length it must equal.
		for _, want := range []string{strconv.Itoa(c.hidden + 1), strconv.Itoa(c.hidden)} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error = %v, want it to state %q", err, want)
			}
		}
	})
}

// TestLoadConfigRejectsWrongDimensionCount covers the generic shape check's dims message.
//
// The check runs over every tensor whose shape the config expects, and it distinguishes two
// failures: a tensor with the wrong number of dimensions, and one with the right count but the wrong
// extent. The second is covered by the attn_q case; the first had no test, so its message -- which
// prints the dimensions actually found and the count expected -- could lose both without a failure.
//
// The shapes are hard to read at a glance, which is the reason to print them: an error naming only
// the tensor leaves its reader comparing two lists by hand.
func TestLoadConfigRejectsWrongDimensionCount(t *testing.T) {
	c := defaultTiny()
	tensors := c.tensors()
	for i := range tensors {
		if tensors[i].Name == "blk.0.attn_q.weight" {
			// Three dimensions where the weight matrix has two.
			tensors[i].Dims = []uint64{8, 5, 3}
		}
	}
	f, err := gguf.Read(bytes.NewReader(buildFile(c.kvs(), tensors, nil)))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	_, err = LoadConfig(f)
	if err == nil {
		t.Fatal("LoadConfig accepted a three-dimensional attn_q.weight")
	}
	if !strings.Contains(err.Error(), "blk.0.attn_q.weight") {
		t.Errorf("error = %v, want it to name the tensor", err)
	}
	// The count found, the dims themselves, and the count expected.
	if !strings.Contains(err.Error(), "3 dims") {
		t.Errorf("error = %v, want it to state the 3 dims found", err)
	}
	if !strings.Contains(err.Error(), "want 2") {
		t.Errorf("error = %v, want it to state the 2 expected", err)
	}
	if !strings.Contains(err.Error(), "[8 5 3]") {
		t.Errorf("error = %v, want it to print the dims it found", err)
	}
}

// TestLoadConfigReadsRopeBaseAndScale covers the two rope fields that override a default.
//
// Both were unread: the value from the metadata was assigned to a field no test ever populated from a
// file, so dropping either assignment changed nothing observable. The rope tests set the struct field
// directly, which tests the rotation and not the reading.
//
// The defaults are the other half: 10000 and 1 are what a model without the keys gets, and they are
// assigned before the lookup rather than after, so a merge bug would leave them unset instead.
func TestLoadConfigReadsRopeBaseAndScale(t *testing.T) {
	t.Run("values from the file", func(t *testing.T) {
		c := defaultTiny()
		kvs := append(c.kvs(),
			kv{"llama.rope.freq_base", f32Val(500000)},
			kv{"llama.rope.scale_linear", f32Val(4)},
		)
		f, err := gguf.Read(bytes.NewReader(buildFile(kvs, c.tensors(), nil)))
		if err != nil {
			t.Fatalf("Read: %v", err)
		}
		cfg, err := LoadConfig(f)
		if err != nil {
			t.Fatalf("LoadConfig: %v", err)
		}
		if cfg.RopeBase != 500000 {
			t.Errorf("RopeBase = %v, want the file's 500000", cfg.RopeBase)
		}
		if cfg.RopeScale != 4 {
			t.Errorf("RopeScale = %v, want the file's 4", cfg.RopeScale)
		}
	})

	t.Run("defaults without the keys", func(t *testing.T) {
		c := defaultTiny()
		f, err := gguf.Read(bytes.NewReader(buildFile(c.kvs(), c.tensors(), nil)))
		if err != nil {
			t.Fatalf("Read: %v", err)
		}
		cfg, err := LoadConfig(f)
		if err != nil {
			t.Fatalf("LoadConfig: %v", err)
		}
		if cfg.RopeBase != 10000 {
			t.Errorf("RopeBase = %v, want the default 10000", cfg.RopeBase)
		}
		if cfg.RopeScale != 1 {
			t.Errorf("RopeScale = %v, want the default 1", cfg.RopeScale)
		}
	})

	t.Run("a non-positive scale keeps the default", func(t *testing.T) {
		// The guard is `ok && v > 0`: a zero or negative scale is ignored rather than refused, since
		// scaling by zero would collapse every position onto one and the schema has no "off" value.
		c := defaultTiny()
		kvs := append(c.kvs(), kv{"llama.rope.scale_linear", f32Val(0)})
		f, err := gguf.Read(bytes.NewReader(buildFile(kvs, c.tensors(), nil)))
		if err != nil {
			t.Fatalf("Read: %v", err)
		}
		cfg, err := LoadConfig(f)
		if err != nil {
			t.Fatalf("LoadConfig: %v", err)
		}
		if cfg.RopeScale != 1 {
			t.Errorf("RopeScale = %v, want the default 1 for a zero value", cfg.RopeScale)
		}
	})
}

// TestTemperatureTopPBounds pins the two documented no-op values for the nucleus threshold.
//
// TopP's doc says a value of zero or below disables it, and so does one at or above 1,
// "because keeping every candidate is what happens without the limit". Three branches meet
// there -- the nucleus call, the sort that only the nucleus path needs, and the plain walk --
// and nothing tested either end: TopK=1 and Temp=0 had cases, the threshold did not.
//
// The property that has to hold is that no candidate is pruned. It is asserted through the
// draw rather than by inspecting the candidate list, which the sampler does not expose: with
// every candidate still present, a draw near 1 reaches a low-probability token, and with the
// tail pruned it cannot. The draws also cover the case where the limit is implemented by
// falling through to the nucleus path -- nucleus(p=1) keeps all five candidates here, so that
// implementation is correct too, which is why this test asserts the outcome rather than which
// branch produced it.
func TestTemperatureTopPBounds(t *testing.T) {
	logits := []float32{1, 5, 3, 2, 4}
	const argmax = 1

	// The control, and the threshold is chosen for its discrimination rather than its roundness:
	// measured, nucleus(p=0.5) keeps one candidate on these logits and nucleus(p=0.7) keeps two
	// ([1 4], the argmax and the next). A threshold of 0.5 cannot tell a real nucleus from one
	// that always returns a single candidate, which is exactly the regression worth catching.
	real := Temperature{Temp: 1, TopP: 0.7, NextFloat: func() float64 { return 0.5 }}
	if got := real.Next(logits); got != argmax {
		t.Fatalf("TopP=0.7 with a draw of 0.5 returned %d, want the argmax %d", got, argmax)
	}
	// With two candidates kept and a draw past the argmax's share, the second one has to win.
	// A nucleus that kept only the argmax would return it here too.
	real2 := Temperature{Temp: 1, TopP: 0.7, NextFloat: func() float64 { return 0.99 }}
	if got := real2.Next(logits); got == argmax {
		t.Errorf("TopP=0.7 kept only the argmax: a draw of 0.99 still returned %d, so the second "+
			"candidate was pruned", got)
	}

	for _, topP := range []float32{0, -1, 1, 1.5} {
		// A draw of nearly 1 must land on a token the nucleus would have pruned. With the
		// threshold disabled that is reachable; with it applied it is not.
		s := Temperature{Temp: 1, TopP: topP, NextFloat: func() float64 { return 0.999 }}
		got := s.Next(logits)
		if got < 0 || int(got) >= len(logits) {
			t.Errorf("TopP=%v returned %d, outside the candidate range", topP, got)
		}
		if got == argmax {
			t.Errorf("TopP=%v with a draw of 0.999 returned the argmax %d: the tail was pruned, "+
				"so the threshold did not disable itself", topP, got)
		}
	}

	// The other end of the draw, which is a different fact and easy to get wrong: with the
	// limit off the candidates are NOT sorted, so the walk follows index order and a draw near
	// zero returns candidate 0 -- the smallest logit, not the largest. That is the documented
	// behaviour of a disabled limit (the sort exists only for the nucleus path), so this
	// asserts it rather than the argmax. It is also what distinguishes the two no-op values
	// from a pruning threshold: under TopP=0.5 the sort puts the argmax first and a near-zero
	// draw returns it.
	for _, topP := range []float32{0, 1} {
		s := Temperature{Temp: 1, TopP: topP, NextFloat: func() float64 { return 0.0001 }}
		if got := s.Next(logits); got != 0 {
			t.Errorf("TopP=%v with a draw of 0.0001 returned %d, want candidate 0: with the limit "+
				"disabled nothing is sorted, so the walk follows index order", topP, got)
		}
	}
	// The control: the same draw under a real threshold returns the argmax, because the nucleus
	// path sorts first.
	pruned := Temperature{Temp: 1, TopP: 0.7, NextFloat: func() float64 { return 0.0001 }}
	if got := pruned.Next(logits); got != argmax {
		t.Errorf("TopP=0.7 with a draw of 0.0001 returned %d, want the argmax %d", got, argmax)
	}
}

// TestTemperatureTopKOutOfRange covers the two values outside the useful range, both of which
// must fall back to "no limit" rather than reject the request.
//
// The guard is `s.TopK > 0 && s.TopK < len(logits)`, so a TopK at or above the vocabulary size
// keeps every candidate and a non-positive one does too. A slice taken without the second half
// of that comparison would panic on the first, and the CLI passes the flag straight through:
// `--top-k 999999` on a 32k vocabulary is a value a user can type.
func TestTemperatureTopKOutOfRange(t *testing.T) {
	logits := []float32{1, 5, 3, 2, 4}
	for _, topK := range []int{-1, 0, 5, 6, 999999} {
		s := Temperature{Temp: 1, TopK: topK, NextFloat: func() float64 { return 0.5 }}
		got := s.Next(logits)
		if got < 0 || int(got) >= len(logits) {
			t.Errorf("TopK=%d returned %d, outside the vocabulary", topK, got)
		}
	}
	// TopK=1 is the one value that must constrain the result, so it is checked separately to
	// prove the loop above is not passing by accident.
	for _, r := range []float64{0, 0.9, 0.999} {
		s := Temperature{Temp: 1, TopK: 1, NextFloat: func() float64 { return r }}
		if got := s.Next(logits); got != 1 {
			t.Errorf("TopK=1 r=%v returned %d, want the argmax 1", r, got)
		}
	}
}
