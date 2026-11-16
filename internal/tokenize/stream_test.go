package tokenize

import (
	"os"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
)

// loadReal tokenizer from MODEL, skipping when unset.
func loadReal(t *testing.T) *Tokenizer {
	t.Helper()
	p := os.Getenv("MODEL")
	if p == "" {
		t.Skip("set MODEL to a GGUF file")
	}
	f, err := gguf.Open(p)
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
	tok := loadReal(t)
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
	tok := loadReal(t)
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
	tok := loadReal(t)
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
	tok := loadReal(t)
	ids := tok.Encode("Once", false)
	if len(ids) != 1 {
		t.Skipf("\"Once\" encoded to %d tokens", len(ids))
	}
	dec := tok.Seq()
	if got, want := dec.Token(ids[0]), tok.Decode(ids); got != want {
		t.Errorf("streamed %q, Decode gives %q", got, want)
	}
}
