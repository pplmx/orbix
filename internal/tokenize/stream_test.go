package tokenize

import (
	"os"
	"strings"
	"testing"
	"unicode/utf8"

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

// TestStreamDecoderMatchesDecodeOnSplitBytes covers the case the fixture above cannot reach.
//
// TestStreamDecoderMatchesDecode runs on a constructed vocabulary, and its comment says a
// synthetic model that lacks byte pieces cannot produce a byte run that never forms a
// character. A real byte-level vocabulary can: Qwen2's has 1457 pieces that are a bare
// continuation byte, so a single character routinely spans tokens that no fixture vocabulary
// would pair that way. Streaming a Chinese reply from such a model emitted U+FFFD as its first
// character while the non-streaming body had the real one -- the exact divergence the contract
// test exists to prevent, invisible to it because the fixture has no such pieces.
//
// Gated on QWEN2_MODEL because the property is a fact about that vocabulary.
func TestStreamDecoderMatchesDecodeOnSplitBytes(t *testing.T) {
	path := os.Getenv("QWEN2_MODEL")
	if path == "" {
		t.Skip("set QWEN2_MODEL to a byte-level GGUF with split sequences")
	}
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !tok.byteLevel {
		t.Skipf("%s is not a byte-level vocabulary", path)
	}

	// Concrete token sequences first: these are the shapes that used to diverge, and they
	// do not depend on what Encode happens to produce for a string. id 2858 ends in a lead
	// byte and 94/95 are bare continuation bytes, which together form one character; the
	// stream used to drop the space the first piece carried and render the rest as U+FFFD.
	for _, seq := range [][]int32{
		{2858, 94, 95},
		{2858, 94},
		{94, 95},
	} {
		want := tok.DecodeCompletion(seq)
		dec := tok.Seq()
		var got strings.Builder
		for _, id := range seq {
			got.WriteString(dec.Token(id))
		}
		got.WriteString(dec.Flush())
		if got.String() != want {
			t.Errorf("for ids %v streamed %q, Decode gives %q", seq, got.String(), want)
		}
	}

	for _, text := range []string{
		"北京是中国的首都。",
		"请用中文写一句话介绍北京。",
		"中文和 ASCII 混排 mixed",
	} {
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
		// The bug showed as a replacement character at the very start, so assert the
		// stronger property too rather than only the equality: an equality that both sides
		// got wrong would pass otherwise.
		if strings.HasPrefix(want, "\uFFFD") && !strings.HasPrefix(text, "\uFFFD") {
			t.Errorf("Decode of %q starts with a replacement character", text)
		}
	}
}

// TestStreamDecoderSplitBytesBuiltVocab covers the split-sequence defect on a vocabulary that
// is constructed rather than downloaded, so it runs in CI.
//
// TestStreamDecoderMatchesDecodeOnSplitBytes above needs QWEN2_MODEL because it asserts against
// a real model's pieces, and a skipped test protects nothing: the defect it guards shipped
// despite that test existing. A byte-level vocabulary only has to spell a character's bytes as
// separate pieces to reach the same code path, and gguftest can do that.
//
// The failure this catches is worse than the one on the real model. With the byte-level pieces
// rendered one at a time, a truncated character reached the client as its raw bytes -- "ä" for
// 中's first byte -- instead of as U+FFFD. That is invalid UTF-8 in a JSON body.
// TestRenderPieceIsUnreachableForByteLevel pins why bpeUnescape has no production caller.
//
// renderPiece has a byte-level branch, but both of its callers -- Decode and
// StreamDecoder.Token -- route a byte-level piece to bpeUnescapeBytes first, because a
// character split across tokens has to be held back rather than emitted as bytes. So the
// branch is correct and unreachable, and coverage reports the function below at 0%.
//
// The test states both halves: production code never reaches bpeUnescape, and renderPiece
// still answers correctly if it ever did. Without the second claim the branch looks like
// dead code to delete; without the first, bpeUnescape looks like a path that needs testing.
func TestRenderPieceIsUnreachableForByteLevel(t *testing.T) {
	path := gguftest.Model{
		Vocab:          []string{"<unk>", "<s>", "</s>", "ä", "¸", "Ń"},
		TokenizerModel: "gpt2",
	}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !tok.byteLevel {
		t.Fatal("the fixture is not byte-level, so it cannot exercise the branch")
	}

	// The direct call does render the byte, so the branch is right.
	if got := tok.renderPiece(tok.tokens[3]); got != "\xe4" {
		t.Errorf("renderPiece(byte piece) = %q, want the single byte \xe4", got)
	}

	// And the production path bypasses it: the same token through Decode reads the whole
	// character once the other two bytes arrive, which a per-byte rendering could not do.
	if got := tok.Decode([]int32{3, 4, 5}); got != "中" {
		t.Errorf("Decode([3 4 5]) = %q, want 中", got)
	}
}

func TestStreamDecoderSplitBytesBuiltVocab(t *testing.T) {
	// 中 is e4 b8 ad. These pieces are those bytes: ä is rune 0xE4, ¸ is rune 0xB8, and
	// Ń is rune 0x143, which GPT-2's table maps to byte 0xAD.
	path := gguftest.Model{
		Vocab:          []string{"<unk>", "<s>", "</s>", "ä", "¸", "Ń"},
		TokenizerModel: "gpt2",
		NoBytes:        false,
	}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !tok.byteLevel {
		t.Fatal("the fixture is not a byte-level vocabulary, so it cannot split a character")
	}

	cases := []struct {
		ids []int32
		why string
	}{
		{[]int32{3}, "one byte of a three-byte character"},
		{[]int32{3, 4}, "two bytes, still short"},
		{[]int32{3, 4, 5}, "all three, which must read as the character"},
	}
	for _, c := range cases {
		want := tok.DecodeCompletion(c.ids)
		dec := tok.Seq()
		var got strings.Builder
		for _, id := range c.ids {
			got.WriteString(dec.Token(id))
		}
		got.WriteString(dec.Flush())
		if got.String() != want {
			t.Errorf("%s: ids %v streamed %q, DecodeCompletion gives %q",
				c.why, c.ids, got.String(), want)
		}
		// The streamed text is assembled into a JSON body, so it has to be valid UTF-8 on
		// its own. The defect's signature was raw bytes reaching this point.
		if !utf8.ValidString(got.String()) {
			t.Errorf("%s: streamed %q is not valid UTF-8", c.why, got.String())
		}
	}
}

// TestTokenTypeOutOfRange covers both bounds of TokenType's guard.
//
// It is an exported accessor over the type table, so a caller walking ids -- a UI listing the
// control tokens, say -- can reach either bound with a value it computed rather than one it
// meant. Returning Undefined is what keeps that a lookup miss instead of a panic, and it is
// the same value a token the file marks 0 gets, which is the honest answer for both.
func TestTokenTypeOutOfRange(t *testing.T) {
	path := gguftest.Model{NoEOS: true}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if got := tok.TokenType(-1); got != Undefined {
		t.Errorf("TokenType(-1) = %v, want Undefined", got)
	}
	if got := tok.TokenType(int32(len(tok.types))); got != Undefined {
		t.Errorf("TokenType(%d) = %v, want Undefined at the past-the-end index",
			len(tok.types), got)
	}
	// Large as well as one-past, in case the guard is written as an equality.
	if got := tok.TokenType(1 << 20); got != Undefined {
		t.Errorf("TokenType(1<<20) = %v, want Undefined", got)
	}
	// And the in-range case still answers from the table, so this is a bound test and not a
	// test that everything returns Undefined.
	if len(tok.types) > 0 {
		if got := tok.TokenType(0); got != tok.types[0] {
			t.Errorf("TokenType(0) = %v, want the table's %v", got, tok.types[0])
		}
	}
}
