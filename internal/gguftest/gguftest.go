package gguftest

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
)

// Model describes the tiny model a test wants.
//
// The defaults are chosen so a zero Config still produces a model that runs: a few
// layers, a small hidden width, and a vocabulary of single characters. A test that cares
// about a specific shape sets only the fields it needs.
type Model struct {
	// Arch is general.architecture. Empty means "llama".
	Arch string
	// Name is general.name, which the server exposes as the model id.
	Name string

	Layers int
	Hidden int
	Heads  int
	// KVHeads is attention.head_count_kv; zero means Heads.
	KVHeads int
	FFN     int
	// ContextLen is llama.context_length. Zero means 64.
	ContextLen int
	// Vocab is the token list. Empty means DefaultVocab.
	Vocab []string
	// BOSID and EOSID are the special token ids; the defaults point at the first two
	// entries of the vocabulary.
	BOSID, EOSID int
	// NoEOS writes EOSID as a sentinel outside the vocabulary, so generation never stops
	// on its own and runs to the token budget.
	//
	// A fixture's weights are constructed, not trained, so whether its argmax ever lands
	// on the end-of-sequence token is an accident of the pattern. Tests that measure how a
	// scheduler advances a long sequence need it to keep going, and "it happened to stop
	// after six tokens this build" is not a property any assertion can rest on.
	NoEOS bool
	// NoBytes omits the 256 single-byte pieces, so every token in the vocabulary is a
	// character and no token run can be an invalid byte sequence. How those pieces are
	// spelled follows TokenizerModel: "<0xC3>" for SentencePiece, the escaped rune for a
	// byte-level vocabulary.
	//
	// The stream decoder and Decode legitimately differ on a byte run that never forms a
	// character: Decode joins every pending byte before deciding how to render it, while a
	// stream has to decide when the run ends. A test comparing the two wants a model that
	// cannot produce such a run.
	//
	// The cost of omitting them is that a character the vocabulary does not contain cannot
	// be represented at all: with no byte pieces, the encoder falls back to the unknown
	// token, so "中" becomes as many unknown ids as it has bytes and decodes to nothing.
	// That is the point of the flag for the tests that use it, and a reason not to use it
	// for a test about text rather than about shapes.
	NoBytes bool
	// ChatTemplate, when set, is written to tokenizer.chat_template.
	ChatTemplate string

	// Merges is tokenizer.ggml.merges: the BPE merge list, in rank order, written as
	// "left right". Empty omits the key, which is what a SentencePiece vocabulary does
	// and what the score-ordered fallback path expects.
	Merges []string

	// AddBOS is tokenizer.ggml.add_bos_token. Nil omits the key, which is what a file
	// that accepts the default does; a non-nil value states it explicitly.
	AddBOS *bool

	// WindowLen sets <arch>.attention.sliding_window, the per-layer attention window a
	// sliding-window model like Mistral uses. Zero omits the key, which the loader reads as
	// unbounded -- the same meaning an absent key has in a real file.
	WindowLen int

	// AttnBias adds the q, k, and v bias tensors the Qwen2 family uses and Llama does
	// not. The loader infers the flag from the tensors' presence, so writing them is the
	// whole of what a fixture needs to be a Qwen2-shaped model.
	AttnBias bool

	// Pre is tokenizer.ggml.pre: the pre-tokenizer name, which selects how text is cut
	// into chunks before merging. Empty omits the key, which is what a SentencePiece
	// vocabulary does and what a byte-level model without a chunking rule does.
	Pre string

	// OmitTokenizerModel writes the vocabulary without tokenizer.ggml.model, which the
	// format allows. Used to check that a loader identifies the family from the vocabulary
	// rather than assuming SentencePiece when the key is absent.
	OmitTokenizerModel bool

	// TokenizerModel is tokenizer.ggml.model: "llama" for SentencePiece, "gpt2" for
	// byte-level BPE. Empty means "llama".
	//
	// It is separate from Merges because the two encode the same intent from different
	// directions: a merge list only makes sense with byte-level BPE, whose pieces are
	// the escaped bytes, and a fixture that pairs a merge list with a SentencePiece
	// vocabulary builds a tokenizer that reads its own pieces wrong.
	TokenizerModel string

	// vocabBase is where the byte-fallback pieces start, recorded during resolve so the
	// writer knows which pieces are bytes without searching the list again.
	vocabBase int
	// Experts and ExpertUsed make the model a mixture of experts.
	Experts    int
	ExpertUsed int
}

// DefaultVocab is a vocabulary of characters and a few words, enough for round trips and
// for generation to have somewhere to go.
//
// The first entry is <unk> because a model without an unknown token cannot encode a
// character it does not know, and the tests exercise exactly that fallback.
var DefaultVocab = []string{
	"<unk>", "<s>", "</s>",
	"\u2581", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
	"n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
	// Uppercase letters, so a capitalized word is not silently a run of unknowns.
	"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
	"N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
	".", ",", "'", "!", "?",
	// Whole words, with the space marker, so ordinary prose encodes to few tokens rather
	// than one per character. A vocabulary of bare characters makes every word a long run
	// and every generated string unreadable, which is fine for a shape test and useless
	// for one that has to find a word in the output.
	"\u2581the", "\u2581The", "\u2581once", "\u2581Once", "\u2581upon", "\u2581time", "\u2581a", "\u2581and",
	"\u2581of", "\u2581to", "\u2581in", "\u2581was", "\u2581is", "\u2581it",
	"\u2581there", "\u2581little", "\u2581girl", "\u2581she", "\u2581her",
	"\u2581so", "\u2581happy", "\u2581city", "\u2581France", "\u2581Paris",
	// Multi-character pieces for the words above, so a merge has something to build.
	"\u2581t", "\u2581th", "\u2581ti", "he", "on", "ti", "me", "up", "in", "er",
	"re", "es", "an", "ar", "is", "hi", "app", "ppy", "it", "tt", "le", "irl",
}

// resolve fills in the defaults and returns a complete description.
func (m Model) resolve() Model {
	if m.Arch == "" {
		m.Arch = "llama"
	}
	if m.Name == "" {
		m.Name = "tiny"
	}
	if m.Layers <= 0 {
		m.Layers = 2
	}
	if m.Hidden <= 0 {
		m.Hidden = 32
	}
	if m.Heads <= 0 {
		m.Heads = 2
	}
	if m.KVHeads <= 0 {
		m.KVHeads = m.Heads
	}
	if m.FFN <= 0 {
		m.FFN = 64
	}
	if m.ContextLen <= 0 {
		m.ContextLen = 64
	}
	if len(m.Vocab) == 0 {
		m.Vocab = DefaultVocab
		// A byte-level vocabulary spells a word's leading space as the escaped rune
		// (U+0120, "Ġ"), not SentencePiece's marker (U+2581, "▁"). Getting this wrong
		// does not fail anything: the pieces are simply never found, and every word
		// falls back to one token per byte. That is why it went unnoticed -- the
		// encodings were valid, just not the ones the vocabulary implies.
		if m.TokenizerModel == "gpt2" {
			m.Vocab = byteLevelSpelling(m.Vocab)
		}
	}
	// The 256 byte-fallback pieces are part of the vocabulary, not an addition to it at
	// write time: the embedding and output tensors are shaped from the declared
	// vocabulary, and a file whose metadata lists more tokens than its tensors have rows
	// fails the loader's shape check.
	m.vocabBase = len(m.Vocab)
	if !m.NoBytes {
		m.Vocab = append(append([]string(nil), m.Vocab...), m.bytePieces()...)
	}
	if m.BOSID == 0 && m.EOSID == 0 {
		m.BOSID, m.EOSID = 1, 2
	}
	return m
}

// headDim is the width of one attention head.
func (m Model) headDim() int { return m.Hidden / m.Heads }

// kvDim is the width of the combined key or value projection.
func (m Model) kvDim() int { return m.KVHeads * m.headDim() }

// kv is one metadata entry. GGUF has no exported writer, and this package is the reason
// it does not need one: a writer exists only here, for fixtures.
type kv struct {
	Key   string
	Value gguf.Value
}

// tensorSpec is one tensor's name and shape.
type tensorSpec struct {
	name string
	dims []uint64
}

// tensors lists the tensors a model of this shape needs, in the order they are laid out.
func (m Model) tensors() []tensorSpec {
	h := uint64(m.Hidden)
	out := []tensorSpec{
		{"token_embd.weight", []uint64{h, uint64(len(m.Vocab))}},
		{"output_norm.weight", []uint64{h}},
		// A separate output projection rather than tied embeddings. With tied weights
		// the final hidden state is projected onto the same rows the input embedding
		// reads, and a fixture built from one deterministic pattern lands its argmax on
		// a special token at once -- every sequence would end at its first step, which
		// makes the fixture useless for tests that need generation to continue.
		{"output.weight", []uint64{h, uint64(len(m.Vocab))}},
	}
	for i := range m.Layers {
		p := "blk." + itoa(i) + "."
		out = append(out,
			tensorSpec{p + "attn_norm.weight", []uint64{h}},
			tensorSpec{p + "attn_q.weight", []uint64{h, h}},
			tensorSpec{p + "attn_k.weight", []uint64{h, uint64(m.kvDim())}},
			tensorSpec{p + "attn_v.weight", []uint64{h, uint64(m.kvDim())}},
			tensorSpec{p + "attn_output.weight", []uint64{h, h}},
			tensorSpec{p + "ffn_norm.weight", []uint64{h}},
		)
		if m.AttnBias {
			out = append(out,
				tensorSpec{p + "attn_q.bias", []uint64{h}},
				tensorSpec{p + "attn_k.bias", []uint64{uint64(m.kvDim())}},
				tensorSpec{p + "attn_v.bias", []uint64{uint64(m.kvDim())}},
			)
		}
		if m.Experts > 0 {
			e := uint64(m.Experts)
			out = append(out,
				tensorSpec{p + "ffn_gate_inp.weight", []uint64{h, e}},
				tensorSpec{p + "ffn_gate_exps.weight", []uint64{h, uint64(m.FFN), e}},
				tensorSpec{p + "ffn_up_exps.weight", []uint64{h, uint64(m.FFN), e}},
				tensorSpec{p + "ffn_down_exps.weight", []uint64{uint64(m.FFN), h, e}},
			)
			continue
		}
		out = append(out,
			tensorSpec{p + "ffn_gate.weight", []uint64{h, uint64(m.FFN)}},
			tensorSpec{p + "ffn_up.weight", []uint64{h, uint64(m.FFN)}},
			tensorSpec{p + "ffn_down.weight", []uint64{uint64(m.FFN), h}},
		)
	}
	return out
}

// Write builds the model and returns the path to a GGUF file in dir.
//
// Every weight is filled from a small deterministic pattern. Real values are not needed:
// the assertions these fixtures serve are about wiring -- that a batched pass equals a
// serial one, that a policy reorders without changing output -- and those hold for any
// weights. Deterministic values make a failure reproducible, which random ones would not.
func (m Model) Write(t *testing.T, dir string) string {
	t.Helper()
	raw, err := m.Bytes()
	if err != nil {
		t.Fatalf("gguftest: %v", err)
	}
	path := filepath.Join(dir, "tiny.gguf")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatalf("gguftest: write %s: %v", path, err)
	}
	return path
}

// Bytes builds the model and returns the GGUF file's contents.
//
// It validates the shape rather than assuming it: a model whose hidden width does not
// divide by its head count produces a file that fails much later, inside attention, with
// a slice-bounds panic that names no hyperparameter.
func (m Model) Bytes() ([]byte, error) {
	m = m.resolve()
	if m.Hidden%m.Heads != 0 {
		return nil, fmt.Errorf("hidden %d is not divisible by heads %d", m.Hidden, m.Heads)
	}
	if m.Heads%m.KVHeads != 0 {
		return nil, fmt.Errorf("heads %d is not divisible by kv heads %d", m.Heads, m.KVHeads)
	}

	specs := m.tensors()
	var data []byte
	infos := make([]gguf.TensorInfo, len(specs))
	offset := 0
	for i, spec := range specs {
		infos[i] = gguf.TensorInfo{Name: spec.name, Dims: spec.dims, Type: gguf.F32}
		n := int64(1)
		for _, d := range spec.dims {
			n *= int64(d)
		}
		infos[i].Offset = uint64(offset)
		vals := make([]float32, n)
		fillWeights(vals, spec.name)
		data = appendFloat32s(data, vals)
		offset += int(n) * 4
	}

	return buildFile(m.kvs(), infos, data), nil
}

// fillWeights writes deterministic values into one tensor.
//
// Normalization weights are 1, so the residual stream keeps its scale; a fixture whose
// norms were also patterned would either saturate every layer or decay to zero, and
// either way a test would be comparing infinities or denormals instead of the property
// it meant to check.
//
// Everything else uses a cheap pseudo-random sequence rather than a short repeating
// pattern. A repeating pattern makes every vocabulary row differ from its neighbour by a
// fixed multiple, and the projection then has a stable preference that wins at every
// position -- generation stops after one token because the same token always comes out.
// The sequence below has a long period, so distinct rows stay distinct.
func fillWeights(vals []float32, name string) {
	if isNorm(name) {
		for i := range vals {
			vals[i] = 1
		}
		return
	}
	// xorshift32: deterministic, no allocation, and long enough that a tensor built from
	// it does not repeat within any fixture this package produces.
	state := uint32(0x9E3779B9)
	for i := range vals {
		state ^= state << 13
		state ^= state >> 17
		state ^= state << 5
		// Map to [-0.125, 0.125]: small enough to keep activations bounded over two
		// layers, varied enough that the output projection does not collapse.
		vals[i] = float32(int32(state)) / float32(1<<28) * 0.125
	}
}

// isNorm reports whether a tensor is a normalization weight.
func isNorm(name string) bool {
	for _, suffix := range []string{"attn_norm.weight", "ffn_norm.weight", "output_norm.weight"} {
		if len(name) >= len(suffix) && name[len(name)-len(suffix):] == suffix {
			return true
		}
	}
	return false
}

// kvs builds the metadata a loader reads.
func (m Model) kvs() []kv {
	a := m.Arch
	out := []kv{
		strKV("general.architecture", a),
		strKV("general.name", m.Name),
		u32KV(a+".embedding_length", m.Hidden),
		u32KV(a+".block_count", m.Layers),
		u32KV(a+".attention.head_count", m.Heads),
		u32KV(a+".attention.head_count_kv", m.KVHeads),
		u32KV(a+".feed_forward_length", m.FFN),
		u32KV(a+".context_length", m.ContextLen),
		u32KV(a+".rope.dimension_count", m.headDim()),
		f32KV(a+".attention.layer_norm_rms_epsilon", 1e-5),
	}
	if m.Experts > 0 {
		out = append(out,
			u32KV(a+".expert_count", m.Experts),
			u32KV(a+".expert_used_count", m.ExpertUsed),
		)
	}
	if m.WindowLen > 0 {
		out = append(out, u32KV(a+".attention.sliding_window", m.WindowLen))
	}
	model := m.TokenizerModel
	if model == "" {
		model = "llama"
	}
	// The tokenizer is SentencePiece-of-characters: model "llama", an explicit
	// add_space_prefix, and the 256 byte-fallback pieces a SentencePiece model always
	// carries.
	//
	// The byte pieces are not decoration. Without them a character outside the vocabulary
	// encodes to <unk>, and the stream decoder has nothing to reassemble -- so a test of
	// the byte-token path would find no split tokens to check and quietly skip. They are
	// written as SentencePiece names them, <0xNN>, with type 6, which is what the
	// tokenizer recognizes as a raw byte.
	vocab := m.Vocab
	tokens := make([]gguf.Value, len(vocab))
	types := make([]gguf.Value, len(vocab))
	scores := make([]gguf.Value, len(vocab))
	for i, tok := range vocab {
		tokens[i] = gguf.Value{Type: gguf.TypeString, Str: tok}
		typ := int64(1) // normal
		switch {
		case i < 3:
			typ = 3 // control
		case i >= m.vocabBase:
			typ = 6 // byte
		}
		types[i] = gguf.Value{Type: gguf.TypeInt32, Int: typ}
		// The length decides the score, which is what lets a whole-word piece win over the
		// characters it is built from. Scoring by position instead -- the first version --
		// gave every piece after the first few a score too low to ever be chosen, so a
		// word the vocabulary contained still came out as one token per character.
		scores[i] = gguf.Value{Type: gguf.TypeFloat32, Float: float64(len([]rune(tok))) / 32}
	}

	if !m.OmitTokenizerModel {
		out = append(out, strKV("tokenizer.ggml.model", model))
	}
	out = append(out,
		arrayKV("tokenizer.ggml.tokens", gguf.TypeString, tokens),
		arrayKV("tokenizer.ggml.token_type", gguf.TypeInt32, types),
		arrayKV("tokenizer.ggml.scores", gguf.TypeFloat32, scores),
		u32KV("tokenizer.ggml.bos_token_id", m.BOSID),
		u32KV("tokenizer.ggml.eos_token_id", m.EOSID),
		u32KV("tokenizer.ggml.unknown_token_id", 0),
	)
	// A SentencePiece vocabulary prepends a space during training, so the key is written
	// for it. A byte-level BPE vocabulary does not -- the space is part of the text --
	// and writing it there makes the encoder add a leading space the model never saw.
	if model == "llama" {
		out = append(out, boolKV("tokenizer.ggml.add_space_prefix", true))
	}
	if m.AddBOS != nil {
		out = append(out, boolKV("tokenizer.ggml.add_bos_token", *m.AddBOS))
	}
	if m.Pre != "" {
		out = append(out, strKV("tokenizer.ggml.pre", m.Pre))
	}
	if len(m.Merges) > 0 {
		vals := make([]gguf.Value, len(m.Merges))
		for i, merge := range m.Merges {
			vals[i] = gguf.Value{Type: gguf.TypeString, Str: merge}
		}
		out = append(out, arrayKV("tokenizer.ggml.merges", gguf.TypeString, vals))
	}
	if m.ChatTemplate != "" {
		out = append(out, strKV("tokenizer.chat_template", m.ChatTemplate))
	}
	if m.NoEOS {
		// A config-side sentinel rather than an absent key: the loader requires an EOS
		// id, and reusing a real token would make the model stop on something a test
		// cannot predict.
		for i, e := range out {
			if e.Key == m.Arch+".eos_token_id" {
				out[i] = u32KV(e.Key, NoEOSSentinel)
				continue
			}
			if e.Key == "tokenizer.ggml.eos_token_id" {
				out[i] = u32KV(e.Key, NoEOSSentinel)
			}
		}
	}
	return out
}

// NoEOSSentinel is an EOS id that no vocabulary contains, so a generated sequence never
// reaches it.
const NoEOSSentinel = 1 << 30

// spaceMarker and byteLevelSpace are the two spellings of a piece's leading space.
const (
	spaceMarker    = "\u2581"
	byteLevelSpace = "\u0120"
)

// byteLevelSpelling rewrites SentencePiece's space marker to the byte-level escape.
//
// The two vocabularies describe the same vocabulary of words and differ only in how the
// leading space is spelled, so the transformation is exact: replace the marker rune with
// the escaped space rune in every piece that starts with it.
func byteLevelSpelling(vocab []string) []string {
	out := make([]string, len(vocab))
	for i, piece := range vocab {
		out[i] = strings.Replace(piece, spaceMarker, byteLevelSpace, 1)
	}
	return out
}

// bytePieces returns the 256 single-byte pieces, spelled the way this vocabulary family
// spells them.
//
// The two families name a byte differently and a file that uses the wrong spelling has no
// piece for the bytes it needs. A SentencePiece vocabulary writes "<0xC3>" and the tokenizer
// recognises it as a byte token; a byte-level vocabulary writes the escaped rune GPT-2's
// table assigns to the byte, so 0xC3 is "Ã". Getting this wrong is invisible in ASCII --
// every ASCII byte is its own escaped rune, so both spellings agree -- and produced a
// fixture whose non-ASCII round trips came back as mojibake: "¡" encoded to the escaped
// runes for 0xC2 and 0xA1, found no such pieces, and fell back to byte tokens it could not
// merge.
func (m Model) bytePieces() []string {
	out := make([]string, 256)
	if m.TokenizerModel == "gpt2" {
		for i := range out {
			out[i] = string(gpt2ByteRunes[i])
		}
		return out
	}
	for i := range out {
		out[i] = fmt.Sprintf("<0x%02X>", i)
	}
	return out
}

// gpt2ByteRunes is GPT-2's bytes_to_unicode, written out here because a fixture must not
// borrow the implementation's table: a test that shares the code under test cannot catch it
// being wrong, and this table being wrong is exactly the bug the round-trip tests were added
// to find.
var gpt2ByteRunes = func() [256]rune {
	var keep [256]bool
	for _, r := range [][2]int{{0x21, 0x7E}, {0xA1, 0xAC}, {0xAE, 0xFF}} {
		for b := r[0]; b <= r[1]; b++ {
			keep[b] = true
		}
	}
	var table [256]rune
	next := 0
	for b := range 256 {
		if keep[b] {
			table[b] = rune(b)
		} else {
			table[b] = rune(256 + next)
			next++
		}
	}
	return table
}()

// Default is the standard fixture: two layers, 32 hidden, a character vocabulary.
func Default() Model { return Model{} }

// Path builds a default model and returns its path.
func Path(t *testing.T) string {
	t.Helper()
	return Default().Write(t, t.TempDir())
}

// buildFile renders a complete GGUF: magic, version, counts, metadata, directory, then
// aligned tensor data.
func buildFile(kvs []kv, tensors []gguf.TensorInfo, data []byte) []byte {
	var b bytes.Buffer
	b.WriteString(gguf.Magic)
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

	u32(gguf.Version)
	u64(uint64(len(tensors)))
	u64(uint64(len(kvs)))

	for _, e := range kvs {
		str(e.Key)
		u32(uint32(e.Value.Type))
		encodeValue(&b, e.Value)
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
	// Tensor data starts at the next 32-byte boundary from the end of the directory.
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	b.Write(data)
	return b.Bytes()
}

// encodeValue writes one metadata value.
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
	f32 := func(x float32) {
		u32(math.Float32bits(x))
	}
	switch v.Type {
	case gguf.TypeString:
		u64(uint64(len(v.Str)))
		b.WriteString(v.Str)
	case gguf.TypeBool:
		if v.Bool {
			b.WriteByte(1)
			return
		}
		b.WriteByte(0)
	case gguf.TypeUint8:
		b.WriteByte(byte(v.Uint))
	case gguf.TypeUint32:
		u32(uint32(v.Uint))
	case gguf.TypeUint64:
		u64(v.Uint)
	case gguf.TypeInt8:
		b.WriteByte(byte(v.Int))
	case gguf.TypeInt32:
		u32(uint32(v.Int))
	case gguf.TypeInt64:
		u64(uint64(v.Int))
	case gguf.TypeFloat32:
		f32(float32(v.Float))
	case gguf.TypeFloat64:
		u64(math.Float64bits(v.Float))
	case gguf.TypeArray:
		u32(uint32(v.ElemType))
		u64(uint64(len(v.Array)))
		for _, e := range v.Array {
			encodeValue(b, e)
		}
	}
}

// appendFloat32s appends floats in little-endian order.
func appendFloat32s(dst []byte, vals []float32) []byte {
	for _, v := range vals {
		var tmp [4]byte
		binary.LittleEndian.PutUint32(tmp[:], math.Float32bits(v))
		dst = append(dst, tmp[:]...)
	}
	return dst
}

// itoa formats a small non-negative int.
func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var buf [8]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}

// strKV, u32KV, f32KV, boolKV, and arrayKV build metadata entries.
func strKV(k, v string) kv {
	return kv{Key: k, Value: gguf.Value{Type: gguf.TypeString, Str: v}}
}

func u32KV(k string, v int) kv {
	return kv{Key: k, Value: gguf.Value{Type: gguf.TypeUint32, Uint: uint64(v), Num: float64(v)}}
}

func f32KV(k string, v float32) kv {
	return kv{Key: k, Value: gguf.Value{Type: gguf.TypeFloat32, Float: float64(v)}}
}

func boolKV(k string, v bool) kv {
	return kv{Key: k, Value: gguf.Value{Type: gguf.TypeBool, Bool: v}}
}

func arrayKV(k string, elem gguf.ValueType, vals []gguf.Value) kv {
	return kv{Key: k, Value: gguf.Value{Type: gguf.TypeArray, ElemType: elem, Array: vals}}
}
