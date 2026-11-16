package tokenize

import (
	"bytes"
	"encoding/binary"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
)

// Byte-level BPE and SentencePiece differ in how a space is represented before the
// merges run, and getting it wrong does not fail: the encoder produces valid ids for
// every word and silently drops every space. These tests pin both families.
//
// The fixture vocabularies below are hand-built rather than taken from a checkpoint so
// the expected ids are readable; the real-model check lives in the round-trip test that
// runs against a Qwen2 file when one is available.

// gpt2Vocab is a byte-level vocabulary with pieces for a leading space.
var gpt2Vocab = []string{
	"<|endoftext|>", // 0, control
	"Ġ",             // 1, the escaped space alone
	"Ġthe",          // 2
	"Ġcat",          // 3
	"the",           // 4
	"cat",           // 5
	"Ġ",             // 6, duplicate to keep ids explicit
	"t",             // 7
	"h",             // 8
	"e",             // 9
	"c",             // 10
	"a",             // 11
}

// spmVocab is the SentencePiece counterpart: one marker folded into the piece, and a
// single-character fallback for every letter so the unigram merge has something to
// build from. A vocabulary that only holds whole words cannot encode anything the
// trainer did not see as one piece.
var spmVocab = []string{
	"<unk>", // 0
	"<s>",   // 1
	"▁",     // 2
	"▁the",  // 3
	"▁cat",  // 4
	"the",   // 5
	"cat",   // 6
	"t",     // 7
	"h",     // 8
	"e",     // 9
	"c",     // 10
	"a",     // 11
}

// buildTokenizerGGUF assembles a minimal GGUF whose only content is a vocabulary, so
// New can be exercised on the metadata path it actually uses.
func buildTokenizerGGUF(t *testing.T, model string, tokens []string, types []int32) *gguf.File {
	t.Helper()

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
	u64(0) // tensors
	u64(3) // metadata entries

	// tokenizer.ggml.model
	str("tokenizer.ggml.model")
	u32(uint32(gguf.TypeString))
	str(model)

	// tokenizer.ggml.tokens
	str("tokenizer.ggml.tokens")
	u32(uint32(gguf.TypeArray))
	u32(uint32(gguf.TypeString))
	u64(uint64(len(tokens)))
	for _, tok := range tokens {
		str(tok)
	}

	// tokenizer.ggml.token_type
	str("tokenizer.ggml.token_type")
	u32(uint32(gguf.TypeArray))
	u32(uint32(gguf.TypeInt32))
	u64(uint64(len(types)))
	for _, typ := range types {
		u32(uint32(typ))
	}

	f, err := gguf.Read(bytes.NewReader(b.Bytes()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	return f
}

// normalTypes returns a token-type array with the first entry control and the rest
// normal, which is the shape a real vocabulary has.
func normalTypes(n int) []int32 {
	out := make([]int32, n)
	out[0] = int32(Control)
	for i := 1; i < n; i++ {
		out[i] = int32(Normal)
	}
	return out
}

func TestNewDetectsByteLevel(t *testing.T) {
	f := buildTokenizerGGUF(t, "gpt2", gpt2Vocab, normalTypes(len(gpt2Vocab)))
	tok, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !tok.byteLevel {
		t.Error("byteLevel = false for a gpt2 vocabulary")
	}
	// A byte-level vocabulary does not add a leading space: text arrives as written.
	if tok.addSpacePrefix {
		t.Error("addSpacePrefix = true for a gpt2 vocabulary; byte-level BPE does not add one")
	}
}

func TestNewDetectsSentencePiece(t *testing.T) {
	f := buildTokenizerGGUF(t, "llama", spmVocab, normalTypes(len(spmVocab)))
	tok, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if tok.byteLevel {
		t.Error("byteLevel = true for a llama vocabulary")
	}
	if !tok.addSpacePrefix {
		t.Error("addSpacePrefix = false for a SentencePiece vocabulary")
	}
}

func TestNewRejectsUnknownVocabFamily(t *testing.T) {
	// An unknown family must be reported rather than guessed: picking the wrong space
	// convention drops every space in silence, which is the bug this field prevents.
	f := buildTokenizerGGUF(t, "wordpiece", spmVocab, normalTypes(len(spmVocab)))
	if _, err := New(f); err == nil {
		t.Fatal("New accepted an unknown tokenizer.ggml.model")
	}
}

// TestByteLevelEncodeKeepsSpaces is the regression: a byte-level vocabulary must carry
// spaces through encoding as escaped spaces, not fold them into U+2581.
func TestByteLevelEncodeKeepsSpaces(t *testing.T) {
	f := buildTokenizerGGUF(t, "gpt2", gpt2Vocab, normalTypes(len(gpt2Vocab)))
	tok, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ids := tok.Encode("the cat", false)
	if len(ids) == 0 {
		t.Fatal("Encode returned nothing")
	}
	if got := tok.Decode(ids); got != "the cat" {
		t.Errorf("round trip = %q, want %q (ids %v)", got, "the cat", ids)
	}
}

// TestSentencePieceEncodeKeepsSpaces is the counterpart, so a change to one family
// cannot quietly break the other.
func TestSentencePieceEncodeKeepsSpaces(t *testing.T) {
	f := buildTokenizerGGUF(t, "llama", spmVocab, normalTypes(len(spmVocab)))
	tok, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ids := tok.Encode("the cat", false)
	if got := tok.Decode(ids); got != "the cat" {
		t.Errorf("round trip = %q, want %q (ids %v)", got, "the cat", ids)
	}
}

// TestByteLevelUsesEscapedSpacePiece checks the marker itself rather than only the
// round trip: an encoder that dropped the space and a decoder that re-inserted one
// could agree with each other and still be wrong about what the model sees.
func TestByteLevelUsesEscapedSpacePiece(t *testing.T) {
	f := buildTokenizerGGUF(t, "gpt2", gpt2Vocab, normalTypes(len(gpt2Vocab)))
	tok, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ids := tok.Encode("the cat", false)
	for _, id := range ids {
		if int(id) >= len(gpt2Vocab) {
			t.Fatalf("id %d outside the vocabulary", id)
		}
	}
	// The encoded run must contain a piece whose surface starts with the escaped space,
	// which is how the model was trained to see a word boundary.
	hasEscaped := false
	for _, id := range ids {
		if len(gpt2Vocab[id]) > 0 && gpt2Vocab[id][0] == 0xC4 && len(gpt2Vocab[id]) > 1 && gpt2Vocab[id][1] == 0xA0 {
			hasEscaped = true
		}
	}
	if !hasEscaped {
		t.Errorf("no piece carries the escaped space U+0120; ids %v with surfaces %v",
			ids, surfaces(gpt2Vocab, ids))
	}
}

// surfaces renders the pieces behind a token run, for failure messages.
func surfaces(vocab []string, ids []int32) []string {
	out := make([]string, len(ids))
	for i, id := range ids {
		out[i] = vocab[id]
	}
	return out
}
