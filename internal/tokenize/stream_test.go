package tokenize

import (
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
)

// fixtureTokenizer is a tokenizer over a constructed vocab: these tests check how a
// stream of tokens decodes, which depends on the byte-level pieces and the special-token
// types, not on the vocabulary being a trained one.
func fixtureTokenizer(t *testing.T) *Tokenizer {
	t.Helper()
	f, err := gguf.Open(gguftest.Path(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return tok
}

// TestStreamDecoderMatchesDecode is the contract for streaming: feeding a
// sequence one token at a time must produce exactly the text Decode produces for
// the whole sequence. This is the property that makes streaming safe to build on,
// and it is where the naive per-token approach fails -- it strips the space before
// every word.
func TestStreamDecoderMatchesDecode(t *testing.T) {
	tok := fixtureTokenizer(t)
	texts := []string{
		"Once upon a time, there was a little girl named Lily.",
		"Hello world",
		"  leading and   repeated   spaces",
		"Mixed 中文 and ASCII",
		"punct.,;:!?()[]{}",
	}
	for _, text := range texts {
		ids := tok.Encode(text, false)
		if len(ids) == 0 {
			continue
		}
		want := tok.Decode(ids)

		dec := tok.Seq()
		var got strings.Builder
		for _, id := range ids {
			got.WriteString(dec.Token(id))
		}
		if got.String() != want {
			t.Errorf("streamed %q, Decode gives %q (input %q)", got.String(), want, text)
		}
	}
}

// TestStreamDecoderReassemblesSplitCharacters checks the byte-token path: a
// character whose UTF-8 bytes span several tokens must not be rendered as a
// replacement character in the middle of a stream.
func TestStreamDecoderReassemblesSplitCharacters(t *testing.T) {
	tok := fixtureTokenizer(t)
	// A CJK character is three UTF-8 bytes; a vocabulary without it as a single
	// piece encodes it as three byte tokens.
	const text = "中文"
	ids := tok.Encode(text, false)
	if len(ids) < 2 {
		t.Skipf("%q encoded to %d tokens; no split to test", text, len(ids))
	}

	dec := tok.Seq()
	var got strings.Builder
	for i, id := range ids {
		piece := dec.Token(id)
		if strings.ContainsRune(piece, '\uFFFD') && i < len(ids)-1 {
			t.Fatalf("token %d (%d) emitted a replacement character mid-stream: %q", i, id, piece)
		}
		got.WriteString(piece)
	}
	if got.String() != text {
		t.Errorf("streamed %q, want %q", got.String(), text)
	}
}

// TestStreamDecoderSkipsControlTokens pins that control tokens contribute nothing,
// since a chat template's markers must not appear as text.
func TestStreamDecoderSkipsControlTokens(t *testing.T) {
	tok := fixtureTokenizer(t)
	ids := tok.Encode("hello", true) // with BOS, if the model defines one
	dec := tok.Seq()
	var got strings.Builder
	for _, id := range ids {
		piece := dec.Token(id)
		if tok.TokenType(id) == Control && piece != "" {
			t.Errorf("control token %d emitted %q", id, piece)
		}
		got.WriteString(piece)
	}
	if got.String() != tok.Decode(ids) {
		t.Errorf("streamed %q, Decode gives %q", got.String(), tok.Decode(ids))
	}
}

// TestStreamDecoderSingleToken covers a one-token sequence, where the leading
// space still has to be dropped exactly once.
//
// The token is taken by its piece rather than by encoding a word: the fixture vocabulary
// has no whole-word piece for "Once", so encoding it produced five tokens and this test
// skipped on every run. What it is about is a single piece that carries a space marker, and
// the vocabulary has those directly.
func TestStreamDecoderSingleToken(t *testing.T) {
	tok := fixtureTokenizer(t)
	id, ok := tok.index["\u2581the"]
	if !ok {
		t.Skip("the fixture has no piece for \"\u2581the\"")
	}
	ids := []int32{id}

	dec := tok.Seq()
	if got, want := dec.Token(ids[0]), tok.Decode(ids); got != want {
		t.Errorf("streamed %q, Decode gives %q", got, want)
	}
	// The marker is dropped, and only once: the piece spells "\u2581the", not "the".
	if got := tok.Decode(ids); got != "the" {
		t.Errorf("Decode(%d) = %q, want %q", ids, got, "the")
	}
}

// TestDecodeUndoesBPEByteEscaping covers GPT-2 byte-level BPE pieces, which the
// Qwen family uses instead of SentencePiece. A decoder that only knows the
// SentencePiece marker renders "Ġthe" as "the" with no space and "Ċ" as a literal
// two-character string, which reads as a sampling quirk rather than as a bug.
func TestDecodeUndoesBPEByteEscaping(t *testing.T) {
	// The escaping is a property of the vocabulary, not of a checkpoint, so a
	// synthetic tokenizer is enough and this test needs no model file.
	tok := &Tokenizer{
		tokens:    []string{"ĠOnce", "Ġupon", "Ġa", "Ċ", "Ġtime", "ĉtab", "hello"},
		types:     []TokenType{Normal, Normal, Normal, Normal, Normal, Normal, Normal},
		index:     map[string]int32{},
		byteLevel: true,
	}
	// The leading space stays: this vocabulary is byte-level, so every piece carries the
	// caller's own spaces and both forms of a word exist. llama.cpp's detokenize trims the
	// first piece only when the model declares a metaspace prepend scheme that asks for it
	// (default NEVER), so a byte-level decode keeps what it was given.
	got := tok.Decode([]int32{0, 1, 2, 4})
	if want := " Once upon a time"; got != want {
		t.Errorf("Decode = %q, want %q", got, want)
	}
	// "Ġupon" carries its own space, so the newline is followed by one space.
	if got := tok.Decode([]int32{0, 3, 1}); got != " Once\n upon" {
		t.Errorf("Decode with a newline token = %q, want %q", got, " Once\n upon")
	}
	// A SentencePiece vocabulary is the opposite case: its encoder prepends a space, so the
	// marker on the first piece is not the caller's and decoding takes it off.
	spm := &Tokenizer{
		tokens:         []string{"\u2581Once", "\u2581upon", "\u2581a", "\u2581time"},
		types:          []TokenType{Normal, Normal, Normal, Normal},
		index:          map[string]int32{},
		addSpacePrefix: true,
	}
	if got, want := spm.Decode([]int32{0, 1, 2, 3}), "Once upon a time"; got != want {
		t.Errorf("SPM Decode = %q, want %q", got, want)
	}
	if got := tok.Decode([]int32{5}); got != "\ttab" {
		t.Errorf("Decode with a tab token = %q, want %q", got, "\ttab")
	}
	// A plain ASCII piece passes through unchanged.
	if got := tok.Decode([]int32{6}); got != "hello" {
		t.Errorf("Decode = %q, want hello", got)
	}
}

// TestStreamDecoderMatchesBPEVocabulary checks the incremental path handles the
// same vocabulary, since streaming is where a wrong escape becomes a visible
// artifact in the middle of a response.
func TestStreamDecoderMatchesBPEVocabulary(t *testing.T) {
	tok := &Tokenizer{
		tokens:    []string{"ĠOnce", "Ġupon", "Ġa", "Ċ", "Ġtime"},
		types:     []TokenType{Normal, Normal, Normal, Normal, Normal},
		index:     map[string]int32{},
		byteLevel: true,
	}
	ids := []int32{0, 1, 2, 4, 3}
	// The streams a caller renders are completions, so the whole-sequence reference is
	// DecodeCompletion: it drops the leading space that a byte-level vocabulary puts on the
	// first word of a continuation, and a stream must drop the same one.
	want := tok.DecodeCompletion(ids)
	dec := tok.Seq()
	var got strings.Builder
	for _, id := range ids {
		got.WriteString(dec.Token(id))
	}
	if got.String() != want {
		t.Errorf("streamed %q, Decode gives %q", got.String(), want)
	}
}

// TestDecodeAndDecodeCompletionDifferOnTheLeadingSpace pins the split between the two
// decoders, which is the distinction llama.cpp makes with an lstrip argument.
//
// Decode is the inverse of Encode, so a round-trip test can use it; it therefore keeps a
// leading space that came from the caller's own text. DecodeCompletion is what a caller
// renders, and for a byte-level vocabulary it takes that space off, because a generated
// continuation that starts a word spells it "Ġword" and the space is not something the user
// asked to see. Collapsing the two was wrong in a way neither unit test noticed: the
// round-trip lost a space, and generation grew one.
func TestDecodeAndDecodeCompletionDifferOnTheLeadingSpace(t *testing.T) {
	// A byte-level vocabulary: pieces carry the caller's spaces, and both forms of a word
	// exist, so the leading space in the pieces is real.
	tok := &Tokenizer{
		tokens:    []string{"Ġa", "Ġb", "a"},
		types:     []TokenType{Normal, Normal, Normal},
		index:     map[string]int32{},
		byteLevel: true,
	}
	ids := []int32{0, 1}
	if got, want := tok.Decode(ids), " a b"; got != want {
		t.Errorf("Decode = %q, want %q: the round-trip must keep the caller's space", got, want)
	}
	if got, want := tok.DecodeCompletion(ids), "a b"; got != want {
		t.Errorf("DecodeCompletion = %q, want %q", got, want)
	}

	// A SentencePiece vocabulary already loses its marker space inside Decode, so the two
	// agree -- and trimming again would eat a space the caller wrote.
	spm := &Tokenizer{
		tokens:         []string{"\u2581a", "\u2581b"},
		types:          []TokenType{Normal, Normal},
		index:          map[string]int32{},
		addSpacePrefix: true,
	}
	if got, want := spm.Decode([]int32{0, 1}), "a b"; got != want {
		t.Errorf("SPM Decode = %q, want %q", got, want)
	}
	if got, want := spm.DecodeCompletion([]int32{0, 1}), "a b"; got != want {
		t.Errorf("SPM DecodeCompletion = %q, want %q", got, want)
	}
}
