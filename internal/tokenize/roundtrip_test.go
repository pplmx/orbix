package tokenize

import (
	"os"
	"testing"
	"unicode/utf8"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
)

// Round-trip tests over a real checkpoint.
//
// These exist because two bugs in a row survived the fixture-based tests and were found the
// only way they could be found: by encoding text to a real vocabulary and decoding it back.
// The fixture vocabularies are built from ASCII pieces, so every byte above 0x7E was
// untested, and a table that mapped 94 of them wrongly passed a green suite.
//
// The tests are gated on MODEL because a round trip is only meaningful against a vocabulary
// that was actually trained: a fixture can round-trip by construction and still miss a
// convention the real file uses.

// realTokenizer loads the checkpoint named by MODEL.
func realTokenizer(t *testing.T) *Tokenizer {
	t.Helper()
	return realTokenizerAt(t, os.Getenv("MODEL"))
}

// realTokenizerAt loads path, or skips when it is empty so a machine without that
// checkpoint still runs the rest of the suite.
func realTokenizerAt(t *testing.T, path string) *Tokenizer {
	t.Helper()
	if path == "" {
		t.Skip("set MODEL to a GGUF file to run the round-trip checks")
	}
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open(%s): %v", path, err)
	}
	tok, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return tok
}

// TestRoundTripEveryRuneByteLevel is TestRoundTripEveryRune against a byte-level vocabulary.
//
// The whole-code-point sweep is the check that found the escaping-table bugs, and it only
// has teeth on a byte-level vocabulary: bpeBytes is read by the GPT-2 path alone, so a wrong
// entry there is invisible to a SentencePiece checkpoint. Verified by swapping the table's
// 0xC3 and 0xC4 entries: MODEL=stories15M (SentencePiece) passed all 1,112,064 code points,
// while MODEL=qwen2 reported "U+00C0 \"À\" round-trips to \"Ā\"" and the rest of the block.
//
// The documented gated run sets MODEL to the stories model, which means the table was only
// ever checked there -- on the one vocabulary that cannot fail. This runs the same sweep
// against QWEN2_MODEL, so both vocabulary families are covered by the standard command.
//
// Gated on QWEN2_MODEL and skipped when the file at MODEL is the byte-level one, so a
// machine holding a single byte-level checkpoint does not run the sweep twice.
func TestRoundTripEveryRuneByteLevel(t *testing.T) {
	path := os.Getenv("QWEN2_MODEL")
	tok := realTokenizerAt(t, path)
	if !tok.byteLevel {
		t.Skipf("%s is not a byte-level vocabulary", path)
	}
	if path == os.Getenv("MODEL") {
		t.Skip("MODEL and QWEN2_MODEL are the same file; TestRoundTripEveryRune covers it")
	}
	roundTripEveryRune(t, tok)
}

// roundTripEveryRune is the sweep both vocabulary families run.
func roundTripEveryRune(t *testing.T, tok *Tokenizer) {
	t.Helper()
	// A failure per code point would be thousands of lines; report a bounded sample and a
	// count, which is enough to see whether it is one mapping or a whole block.
	const maxReport = 12
	failures, reported := 0, 0
	for r := rune(1); r <= 0x10FFFF; r++ {
		if r >= 0xD800 && r <= 0xDFFF {
			continue // surrogates are not valid UTF-8 and cannot appear in text
		}
		if !utf8.ValidRune(r) {
			continue
		}
		// U+2581 is SentencePiece's space marker: a literal one in the input is
		// indistinguishable from a space inside a piece, so this vocabulary renders it as a
		// space. That is the marker working as designed, not a lost character, and it is why
		// the check skips the marker itself.
		if r == 0x2581 {
			continue
		}
		s := string(r)
		got := tok.Decode(tok.Encode(s, false))
		if got == s {
			continue
		}
		failures++
		if reported < maxReport {
			t.Errorf("U+%04X %q round-trips to %q", r, s, got)
			reported++
		}
	}
	if failures > reported {
		t.Errorf("%d code points failed in total (%d shown)", failures, reported)
	}
}

// TestRoundTripEveryRune encodes and decodes each code point on its own.
//
// One rune at a time is the right granularity for this: it isolates a mapping error to the
// character that has it, and it exercises every byte of the vocabulary's escaping table
// without needing a corpus. A mismatch names the code point, which is what turns "some text
// comes back wrong" into a fixable report.
func TestRoundTripEveryRune(t *testing.T) {
	roundTripEveryRune(t, realTokenizer(t))
}

// TestRoundTripText checks whole strings, which is where a mapping that is right per
// character can still be wrong in sequence: chunk boundaries, merges, and the space
// conventions all only exist between characters.
func TestRoundTripText(t *testing.T) {
	tok := realTokenizer(t)
	inputs := []string{
		// The one that found the leading-space split.
		"a", " a", "  a", " ", "  ", "a ", " a ", "a b", "a  b",
		// The one that found the escaping table: a letter plus a combining mark.
		"a\u0301", "cafe\u0301", "e\u0301", "cafe",
		// Composed Latin-1, which is where the missed table ranges lived.
		"naïve", "Über", "àéîõü", "ÆØÅ", "ß",
		// Scripts whose pieces are multi-byte or multi-char.
		"中文", "日本語", "한국어", "русский", "ελληνικά", "עברית", "العربية",
		// Mixed and punctuation-heavy.
		"Hello, 世界! 123", "def f(x):\n\treturn x", "a/b/c", "…—…",
		// Emoji, including one outside the BMP.
		"😀", "👨‍👩‍👧", "🇯🇵",
		// Control and whitespace characters.
		"\n", "\t", "\r\n", "\u00a0",
	}
	for _, s := range inputs {
		if got := tok.Decode(tok.Encode(s, false)); got != s {
			t.Errorf("%q round-trips to %q", s, got)
		}
	}
}

// TestDecodeIsInverseOfEncodeForLeadingSpace pins the split between Decode and
// DecodeCompletion on a real vocabulary, in both directions.
func TestDecodeIsInverseOfEncodeForLeadingSpace(t *testing.T) {
	tok := realTokenizer(t)
	for _, s := range []string{" a", "  a", " ", " a b"} {
		if got := tok.Decode(tok.Encode(s, false)); got != s {
			t.Errorf("Decode round trip of %q = %q, want the input back", s, got)
		}
	}
	// DecodeCompletion is for rendering a continuation, so it takes one leading space off.
	ids := tok.Encode(" a", false)
	if got := tok.DecodeCompletion(ids); got != "a" {
		t.Errorf("DecodeCompletion(%q) = %q, want the leading space dropped", " a", got)
	}
	// Text without a leading space is unaffected either way.
	if got := tok.DecodeCompletion(tok.Encode("a b", false)); got != "a b" {
		t.Errorf("DecodeCompletion(\"a b\") = %q, want it unchanged", got)
	}
}

// TestNonASCIILatinSurvivesDecode is the regression for the shared decode path.
//
// The fix for the byte table made it cover the whole byte range, and decode applied it to
// every vocabulary. A SentencePiece piece is plain text: "\u2581¡" is a marker and an
// exclamation, but with the wider table the exclamation looked like byte 0xA1 and decoded to
// a lone byte. A whole Latin-1 block broke that way, and only a round trip against a real
// SentencePiece vocabulary showed it -- the byte-level models were fine, which is what made
// the bug look like a table problem rather than a routing one.
func TestNonASCIILatinSurvivesDecode(t *testing.T) {
	tok := realTokenizer(t)
	// U+00A1..U+00AC is where the two byte ranges meet, so a piece's own character is most
	// likely to be mistaken for an escape there.
	for r := rune(0xA1); r <= 0xFF; r++ {
		if r >= 0xD800 && r <= 0xDFFF {
			continue
		}
		s := string(r)
		if got := tok.Decode(tok.Encode(s, false)); got != s {
			t.Errorf("U+%04X %q round-trips to %q", r, s, got)
		}
	}
}

// TestFixtureRoundTripsNonASCII is the same check as TestRoundTripText, run against a
// constructed byte-level vocabulary instead of a checkpoint.
//
// It exists because the checkpoint versions are gated on MODEL and therefore do not run in
// CI, while the bugs they caught were in code CI does run. The fixture could not catch them
// before for a specific reason: its byte pieces were spelled the SentencePiece way ("<0xC3>")
// even for a byte-level vocabulary, which writes the escaped rune instead. Every ASCII byte
// is its own escaped rune, so the two spellings agree there and disagreed only above 0x7E --
// exactly the range that had gone untested.
func TestFixtureRoundTripsNonASCII(t *testing.T) {
	tok := bpeFixture(t, gguftest.Model{TokenizerModel: "gpt2"})
	if !tok.byteLevel {
		t.Fatal("the fixture is not byte-level, so this is testing the wrong path")
	}
	inputs := []string{
		"a", "¡", "¢", "Über", "naïve", "àéîõü", "ÆØÅ", "ß",
		"中文", "日本語", "русский", "ελληνικά",
		"a\u0301", "cafe\u0301",
		"Hello, 世界! 123", "a/b/c",
		"😀", "…—…",
		" ", "  ", " a", "\n\t",
	}
	for _, s := range inputs {
		if got := tok.Decode(tok.Encode(s, false)); got != s {
			t.Errorf("%q round-trips to %q", s, got)
		}
	}
}

// TestFixtureBytePiecesAreEscapedRunes checks the fixture spells its byte pieces the way a
// byte-level vocabulary does, which is what makes the test above able to fail.
//
// A "<0xNN>" piece in a byte-level vocabulary would still be found by the byte fallback, so
// nothing reports an error; the bytes simply never merge with their neighbours, and
// multi-byte characters decode as mojibake. Pinning the spelling is cheaper than noticing
// the mojibake.
func TestFixtureBytePiecesAreEscapedRunes(t *testing.T) {
	tok := bpeFixture(t, gguftest.Model{TokenizerModel: "gpt2"})
	// A byte-level vocabulary's piece for 0xC3 must be the escaped rune U+00C3, and the
	// piece for 0x81 must be the rune the table assigns it.
	if id, ok := tok.index["Ã"]; !ok || tok.tokens[id] != "Ã" {
		t.Errorf("no escaped-rune piece for byte 0xC3 in a byte-level fixture vocabulary")
	}
	if _, ok := tok.index["<0xC3>"]; ok {
		t.Error("the byte-level fixture vocabulary contains a SentencePiece byte piece")
	}
	// The SentencePiece fixture is the other way round.
	spm := bpeFixture(t, gguftest.Model{TokenizerModel: "llama"})
	if _, ok := spm.index["<0xC3>"]; !ok {
		t.Error("the SentencePiece fixture vocabulary has no <0xC3> piece")
	}
}

// TestInvalidByteRunsBecomeReplacementRunes pins what happens to a byte sequence that never
// forms a character, which is the one case where Decode is deliberately not the inverse of
// Encode.
//
// A byte-fallback vocabulary can express any byte, including the start of a multi-byte
// character without its continuation. Decoding must still produce valid UTF-8: text
// downstream -- a JSON body, a terminal -- cannot carry the raw bytes. So the run collapses
// into U+FFFD, and the original is not recoverable. That is the correct trade and it is
// silent, which is why it is worth an assertion rather than a comment.
func TestInvalidByteRunsBecomeReplacementRunes(t *testing.T) {
	tok := bpeFixture(t, gguftest.Model{TokenizerModel: "gpt2"})
	// 0xEF starts a three-byte character; 0x4D cannot continue it.
	lead, ok := tok.index[string(bpeRunes[0xEF])]
	if !ok {
		t.Skip("the fixture has no piece for byte 0xEF")
	}
	cont, ok := tok.index[string(bpeRunes[0x4D])]
	if !ok {
		t.Skip("the fixture has no piece for byte 0x4D")
	}
	got := tok.Decode([]int32{lead, cont})
	want := "\uFFFD" + "M"
	if got != want {
		t.Errorf("Decode(0xEF, 0x4D) = %q, want %q", got, want)
	}
	if !utf8.ValidString(got) {
		t.Errorf("Decode produced invalid UTF-8: % x", got)
	}

	// A valid character reassembled from its bytes is not replaced.
	// "中" is three bytes: E4 B8 AD.
	var ids []int32
	for _, b := range []byte("中") {
		id, ok := tok.index[string(bpeRunes[b])]
		if !ok {
			t.Skipf("the fixture has no piece for byte 0x%02X", b)
		}
		ids = append(ids, id)
	}
	if got, want := tok.Decode(ids), "中"; got != want {
		t.Errorf("Decode(% x) = %q, want %q", ids, got, want)
	}
}
