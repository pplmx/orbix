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
func TestStreamDecoderSingleToken(t *testing.T) {
	tok := fixtureTokenizer(t)
	ids := tok.Encode("Once", false)
	if len(ids) != 1 {
		t.Skipf("\"Once\" encoded to %d tokens", len(ids))
	}
	dec := tok.Seq()
	if got, want := dec.Token(ids[0]), tok.Decode(ids); got != want {
		t.Errorf("streamed %q, Decode gives %q", got, want)
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
		tokens: []string{"ĠOnce", "Ġupon", "Ġa", "Ċ", "Ġtime", "ĉtab", "hello"},
		types:  []TokenType{Normal, Normal, Normal, Normal, Normal, Normal, Normal},
		index:  map[string]int32{},
	}
	got := tok.Decode([]int32{0, 1, 2, 4})
	if want := "Once upon a time"; got != want {
		t.Errorf("Decode = %q, want %q", got, want)
	}
	// "Ġupon" carries its own space, so the newline is followed by one space.
	if got := tok.Decode([]int32{0, 3, 1}); got != "Once\n upon" {
		t.Errorf("Decode with a newline token = %q, want %q", got, "Once\n upon")
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
		tokens: []string{"ĠOnce", "Ġupon", "Ġa", "Ċ", "Ġtime"},
		types:  []TokenType{Normal, Normal, Normal, Normal, Normal},
		index:  map[string]int32{},
	}
	ids := []int32{0, 1, 2, 4, 3}
	want := tok.Decode(ids)
	dec := tok.Seq()
	var got strings.Builder
	for _, id := range ids {
		got.WriteString(dec.Token(id))
	}
	if got.String() != want {
		t.Errorf("streamed %q, Decode gives %q", got.String(), want)
	}
}
