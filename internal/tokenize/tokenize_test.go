package tokenize

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
)

// newTestTokenizer builds a Tokenizer directly from the fixture vocabulary,
// bypassing GGUF since the fixture is already in the shape New expects.
func newTestTokenizer(tb testing.TB) *Tokenizer {
	tb.Helper()
	tok := &Tokenizer{
		tokens:         spmTokens,
		scores:         spmScores,
		types:          make([]TokenType, len(spmTokens)),
		index:          make(map[string]int32, len(spmTokens)),
		byteID:         fillNeg1(),
		unknownID:      0,
		bosID:          1,
		eosID:          2,
		addSpacePrefix: true,
	}
	for i, p := range spmTokens {
		tok.index[p] = int32(i)
	}
	for i := range tok.types {
		tok.types[i] = Normal
	}
	tok.types[0] = Unknown
	tok.types[1] = Control
	tok.types[2] = Control
	return tok
}

func TestEncodeMatchesReference(t *testing.T) {
	tok := newTestTokenizer(t)
	for _, tc := range spmCases {
		got := tok.Encode(tc.text, false)
		if len(got) != len(tc.ids) {
			t.Errorf("Encode(%q) = %v (%d tokens), want %v (%d tokens)",
				tc.text, got, len(got), tc.ids, len(tc.ids))
			continue
		}
		for i := range got {
			if got[i] != tc.ids[i] {
				t.Errorf("Encode(%q)[%d] = %d, want %d (got %v, want %v)",
					tc.text, i, got[i], tc.ids[i], got, tc.ids)
				break
			}
		}
	}
}

func TestDecodeRoundTrip(t *testing.T) {
	// Decoding what Encode produced gives back the original text, for text
	// the vocabulary can represent.
	tok := newTestTokenizer(t)
	for _, text := range []string{
		"Once upon a time",
		"The cat sat on the mat.",
		"Tom went to the park with his dog.",
		"a",
	} {
		if got := tok.Decode(tok.Encode(text, false)); got != text {
			t.Errorf("round trip %q -> %q", text, got)
		}
	}
}

func TestDecodeUnknownTokens(t *testing.T) {
	// Characters outside the vocabulary become <unk> and decode to the
	// reference's "⁇" surface form. This is lossy by construction: the
	// model genuinely cannot represent '!' or ',' with this vocabulary, and
	// pretending otherwise would hide the data loss.
	tok := newTestTokenizer(t)
	got := tok.Decode(tok.Encode("Hello, world!", false))
	const want = "Hello \u2047  world \u2047 "
	if got != want {
		t.Errorf("Decode(Hello, world!) = %q, want %q", got, want)
	}
}

func TestDecodeSkipsControlAndByteTokens(t *testing.T) {
	tok := newTestTokenizer(t)
	ids := append([]int32{1}, tok.Encode("Once upon a time", false)...)
	ids = append(ids, 2)
	if got, want := tok.Decode(ids), "Once upon a time"; got != want {
		t.Errorf("Decode with BOS/EOS = %q, want %q", got, want)
	}
}

func TestDecodeAfterEncodeRoundTrips(t *testing.T) {
	// Encoding then decoding must reproduce the input for text the vocabulary
	// can represent, which is the property a chat REPL depends on.
	tok := newTestTokenizer(t)
	for _, text := range []string{
		"Once upon a time there was a little girl named Lily.",
		"The cat sat on the mat.",
		"Tom went to the park with his dog.",
	} {
		if got := tok.Decode(tok.Encode(text, false)); got != text {
			t.Errorf("round trip %q -> %q", text, got)
		}
	}
}

func TestEncodeBOS(t *testing.T) {
	tok := newTestTokenizer(t)
	got := tok.Encode("a", true)
	if len(got) == 0 || got[0] != 1 {
		t.Fatalf("Encode(with special) = %v, want a leading BOS (1)", got)
	}
	// Without the flag there must be no BOS, or a chat template that adds its
	// own would emit it twice.
	for _, id := range tok.Encode("a", false) {
		if id == 1 {
			t.Errorf("Encode(without special) = %v, want no BOS", got)
		}
	}
}

func TestEncodeEmpty(t *testing.T) {
	tok := newTestTokenizer(t)
	if got := tok.Encode("", false); len(got) != 0 {
		t.Errorf("Encode(\"\") = %v, want empty", got)
	}
	// With the special flag an empty prompt is still just BOS.
	tok.addSpacePrefix = false
	if got := tok.Encode("", true); len(got) != 1 || got[0] != 1 {
		t.Errorf("Encode(\"\", true) = %v, want [1]", got)
	}
}

func TestEncodeControlTokensAreNotProduced(t *testing.T) {
	// A user typing "<s>" must not be able to inject the BOS control token.
	tok := newTestTokenizer(t)
	for _, id := range tok.Encode("<s>", false) {
		if id == 1 || id == 2 {
			t.Fatalf("Encode(\"<s>\") produced control token %d", id)
		}
	}
}

func TestByteFallback(t *testing.T) {
	// A byte token vocabulary lets any input be encoded, including characters
	// the model never saw during training. The byte tokens are decoded back
	// as one UTF-8 sequence, not byte by byte.
	tokens := []string{"<unk>", "<s>", "</s>", "a", "▁a"}
	for i := 0; i < 256; i++ {
		tokens = append(tokens, byteTokenName(i))
	}
	tok := &Tokenizer{
		tokens:         tokens,
		scores:         make([]float32, len(tokens)),
		types:          make([]TokenType, len(tokens)),
		index:          make(map[string]int32, len(tokens)),
		byteID:         fillNeg1(),
		unknownID:      0,
		bosID:          1,
		eosID:          2,
		addSpacePrefix: false,
	}
	for i, p := range tokens {
		tok.index[p] = int32(i)
		tok.types[i] = Normal
		if b, ok := parseByteToken(p); ok {
			tok.byteID[b] = int32(i)
			tok.types[i] = Byte
		}
	}

	// "é" is two UTF-8 bytes and is not in the vocabulary.
	const text = "é"
	ids := tok.Encode(text, false)
	if len(ids) != 2 {
		t.Fatalf("Encode(%q) = %v, want 2 byte tokens", text, ids)
	}
	if got := tok.Decode(ids); got != text {
		t.Errorf("Decode = %q, want %q", got, text)
	}
}

func TestDecodeInvalidUTF8(t *testing.T) {
	// A truncated multi-byte sequence must decode to the replacement
	// character rather than panicking or producing invalid UTF-8.
	tokens := []string{"<unk>", "<s>", "</s>", byteTokenName(0xE4), byteTokenName(0xB8)}
	tok := &Tokenizer{
		tokens:         tokens,
		scores:         make([]float32, len(tokens)),
		types:          []TokenType{Unknown, Control, Control, Byte, Byte},
		index:          map[string]int32{},
		byteID:         fillNeg1(),
		addSpacePrefix: false,
	}
	for i, p := range tokens {
		tok.index[p] = int32(i)
		if b, ok := parseByteToken(p); ok {
			tok.byteID[b] = int32(i)
		}
	}
	got := tok.Decode([]int32{3, 4})
	if got != "\uFFFD" {
		t.Errorf("Decode(truncated UTF-8) = %q, want the replacement character", got)
	}
}

func TestParseByteToken(t *testing.T) {
	tests := map[string]struct {
		b  byte
		ok bool
	}{
		"<0x00>": {0, true},
		"<0xFF>": {0xff, true},
		"<0xff>": {0xff, true},
		"<0x0A>": {0x0a, true},
		"<0x>":   {0, false},
		"<0xGG>": {0, false},
		"<s>":    {0, false},
		"hello":  {0, false},
	}
	for in, want := range tests {
		b, ok := parseByteToken(in)
		if ok != want.ok || (ok && b != want.b) {
			t.Errorf("parseByteToken(%q) = %#x, %v; want %#x, %v", in, b, ok, want.b, want.ok)
		}
	}
}

func byteTokenName(b int) string {
	const hex = "0123456789ABCDEF"
	return "<0x" + string(hex[b>>4]) + string(hex[b&0xf]) + ">"
}

func TestNewFromGGUF(t *testing.T) {
	// New must reject a file without a vocabulary rather than building a
	// tokenizer that silently maps everything to nothing.
	if _, err := New(&gguf.File{Metadata: map[string]gguf.Value{}}); err == nil {
		t.Fatal("New(no tokens) succeeded, want error")
	}
}

func BenchmarkEncode(b *testing.B) {
	tok := newTestTokenizer(b)
	const text = "Once upon a time there was a little girl named Lily."
	b.SetBytes(int64(len(text)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.Encode(text, true)
	}
}

// A byte-level BPE vocabulary ships a merge list, and the order of that list is the
// definition of the tokenization: lower rank applies first. A scoring heuristic over the
// pieces cannot reproduce it, which is why a model with 151387 rules needs the list read
// rather than approximated. These tests use a small vocabulary where the two orders
// disagree, so a regression in either direction shows up as a different token sequence.

// mergeFixture builds a byte-level BPE tokenizer over a vocabulary chosen so that rank
// order and score order disagree.
func mergeFixture(t *testing.T, merges []string) *Tokenizer {
	t.Helper()
	// Pieces: single characters, the space marker, and the merged runs the tests expect.
	// Scores descend with position, so a score-ordered merge prefers the *later* piece --
	// the opposite of what rank order does in the case below.
	path := gguftest.Model{
		Vocab: []string{
			"<unk>", "<s>", "</s>",
			"\u0120", "a", "b", "c", "d",
			"ab", "bc", "abc", "bcd", "abcd",
			"\u0120a", "\u0120ab",
		},
		Merges: merges,
		// A merge list belongs to a byte-level BPE vocabulary: the pieces are escaped
		// bytes, and the two together are what a merge operates on. Pairing the list with
		// a SentencePiece vocabulary builds a tokenizer that reads its own pieces wrong.
		TokenizerModel: "gpt2",
		NoBytes:        true,
	}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return tok
}

// TestMergeRankOrderDecides checks the property the merge list exists for: two rules that
// could both apply are applied in rank order, not in an order derived from anything else.
func TestMergeRankOrderDecides(t *testing.T) {
	// "ab" is ranked before "bc", so "abc" forms as ("ab")+"c", which is not a piece, and
	// the encoder has to fall back to "ab"+"c". If "bc" were applied first the result
	// would be "a"+"bc" instead.
	tok := mergeFixture(t, []string{"a b", "b c"})
	ids := tok.Encode("abc", false)
	if tok.byteLevel != true {
		t.Fatalf("fixture is not byte-level; the merge list would not apply")
	}
	want := []int32{tok.index["ab"], tok.index["c"]}
	if len(ids) != len(want) {
		t.Fatalf("Encode(abc) = %v, want %v", ids, want)
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("Encode(abc) = %v, want %v (rank order, not insertion order)", ids, want)
		}
	}
}

// TestMergeReversingRankOrderChangesOutput is the other half: the same vocabulary with the
// ranks swapped must tokenize differently. Without this, an encoder that ignored the list
// and used its own preference would pass the test above.
func TestMergeReversingRankOrderChangesOutput(t *testing.T) {
	ab := mergeFixture(t, []string{"a b", "b c"})
	ba := mergeFixture(t, []string{"b c", "a b"})
	first := ab.Encode("abc", false)
	second := ba.Encode("abc", false)

	// The exact tokens, not merely "the two differ". A difference-only assertion passes
	// for any bug that changes one side: an encoder that dropped a piece, or merged a
	// whole run into "abc" on one order, would look like evidence rank order worked.
	//
	// When "a b" outranks "b c", "abc" can only form as "ab"+"c" -- the pieces the
	// vocabulary holds for that split. Reversing the ranks gives "a"+"bc" instead.
	wantFirst := []int32{ab.index["ab"], ab.index["c"]}
	wantSecond := []int32{ba.index["a"], ba.index["bc"]}
	for _, want := range []struct {
		name string
		got  []int32
		exp  []int32
	}{
		{"a b before b c", first, wantFirst},
		{"b c before a b", second, wantSecond},
	} {
		if len(want.got) != len(want.exp) {
			t.Errorf("%s: Encode(abc) = %v, want %v", want.name, want.got, want.exp)
			continue
		}
		for j := range want.exp {
			if want.got[j] != want.exp[j] {
				t.Errorf("%s: Encode(abc) = %v, want %v", want.name, want.got, want.exp)
				break
			}
		}
	}
}

// TestMergeListOmitsUnlistedPairs checks that a pair absent from the list never merges,
// however the pieces are scored.
func TestMergeListOmitsUnlistedPairs(t *testing.T) {
	// "c d" is listed, "b c" is not, so "bcd" cannot form by merging bc then d.
	tok := mergeFixture(t, []string{"c d"})
	ids := tok.Encode("bcd", false)
	for _, id := range ids {
		if id == tok.index["bcd"] {
			t.Fatalf("Encode(bcd) = %v includes bcd, which needs an unlisted merge", ids)
		}
	}
}

// TestNoMergeListFallsBackToScores checks that a vocabulary without a merge list still
// tokenizes: that is the SentencePiece path, and it must not become a regression of the
// byte-level support.
func TestNoMergeListFallsBackToScores(t *testing.T) {
	tok := mergeFixture(t, nil)
	if tok.merges != nil {
		t.Fatal("a vocabulary with no merges key set a merge table")
	}
	ids := tok.Encode("abc", false)
	if len(ids) == 0 {
		t.Fatal("Encode(abc) produced nothing without a merge list")
	}
	if got := tok.Decode(ids); got != "abc" {
		t.Errorf("round trip = %q, want abc", got)
	}
}

// TestMalformedMergeIsReported keeps a corrupt list from tokenizing silently: a merge
// entry with no separator means the vocabulary is damaged, and the loader says so.
func TestMalformedMergeIsReported(t *testing.T) {
	path := gguftest.Model{
		Vocab:   gguftest.DefaultVocab,
		Merges:  []string{"a b", "broken"},
		NoBytes: true,
	}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if _, err := New(f); err == nil {
		t.Fatal("New accepted a merge entry with no separator")
	}
}

// TestMergeRankBeatsScoreOrder checks the merge path against the score path on inputs
// where they disagree, using expected token ids from applying the merge list by hand.
//
// This is the test that would have caught the original bug. A scoring heuristic over the
// pieces produces valid ids and fluent text on ordinary prose, so nothing looked wrong;
// on the words below it produces a different token sequence, which is a prompt the model
// was never trained on.
//
// The vocabulary is constructed so the two orderings must differ: rank order folds
// "anti"+"dis" first, while the descending piece score prefers the longer "dis"+"est".
func TestMergeRankBeatsScoreOrder(t *testing.T) {
	// Pieces have descending scores with position, so a score-ordered merge prefers the
	// pair whose right piece is later in the vocabulary.
	path := gguftest.Model{
		Vocab: []string{
			"<unk>", "<s>", "</s>", "\u0120",
			"a", "b", "c", "d", "e", "f",
			"ab", "bc", "cd", "de", "ef",
			"abcd", "cdef",
		},
		// "a b" outranks "b c": with the list, "abc" must form as ab+c. By score the
		// longer "abcd" piece would dominate and split differently.
		Merges:         []string{"a b", "c d"},
		TokenizerModel: "gpt2",
		NoBytes:        true,
	}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	withMerges, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got := withMerges.Encode("abcd", false)

	// The list says "a b" then "c d", so the result is ab+cd, not the abcd piece.
	want := []int32{withMerges.index["ab"], withMerges.index["cd"]}
	if len(got) != len(want) {
		t.Fatalf("Encode(abcd) = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Encode(abcd) = %v, want %v from the merge list, not the abcd piece", got, want)
		}
	}

	// The same tokenizer with the list dropped must reach a different answer, which is
	// what makes the assertion above about the list rather than about the pieces.
	//
	// Dropping the table by hand is a faithful stand-in for a vocabulary that has no merges
	// key: measured on this fixture, the patched tokenizer and one built from a file with no
	// merges key both encode "abcd" as [15], against [10 12] with the list. The alternative --
	// two fixtures differing only in that key -- would compare two files rather than one, and
	// the point of the contrast is that everything except the table is identical.
	scored, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	scored.merges = nil
	if byScore := scored.Encode("abcd", false); len(byScore) == 2 && byScore[0] == want[0] && byScore[1] == want[1] {
		t.Error("the score path produced the same tokens; this fixture does not distinguish the two")
	}
}

// TestEncodeWithSpecialsKeepsOneLeadingSpace covers the dummy space prefix around markers.
//
// SentencePiece models are trained on text that begins with a space, so the encoder adds one
// before segmenting. That prefix belongs to the sequence, not to each piece of it: Encode
// cuts the text at its special markers and encodes the gaps separately, and giving every gap
// its own prefix turns the space in "the<|m|> garden" into a second leading space. The output
// still tokenizes and still decodes to legible text, and says "the  garden" -- a doubled
// space in every chat prompt with a space after a marker.
//
// The assertion counts the free-standing space marker rather than decoding, because Decode
// drops a single leading space by design (llama.cpp's token_to_piece does the same). One
// marker means one prefix, which is what the sequence should carry.
//
// The byte-level family is unaffected, which is why the existing marker tests did not catch
// it: byte-level BPE adds no prefix, so the two paths never differed there.
func TestEncodeWithSpecialsKeepsOneLeadingSpace(t *testing.T) {
	tok := newTestTokenizer(t) // SentencePiece: addSpacePrefix is true
	tok.specials = append(tok.specials, specialToken{text: "<|m|>", id: 1})

	spaceMarkers := func(ids []int32) int {
		n := 0
		for _, id := range ids {
			if int(id) < len(tok.tokens) && tok.tokens[id] == spacePrefix {
				n++
			}
		}
		return n
	}

	// Cases where a prefix belongs: the marker is at the start, so the gap after it opens
	// the sequence. Exactly one space marker, however many markers precede it.
	for _, in := range []string{"<|m|> the garden", "<|m|><|m|> the"} {
		if got := spaceMarkers(tok.EncodeWithSpecials(in, false)); got != 1 {
			t.Errorf("EncodeWithSpecials(%q) has %d leading space markers, want 1", in, got)
		}
	}
	// Cases where the sequence has already started: the text after the marker must not get
	// a prefix of its own. Zero free-standing markers, the same as the text without a marker.
	for _, in := range []string{"the<|m|> garden", "the<|m|>garden"} {
		if got := spaceMarkers(tok.EncodeWithSpecials(in, false)); got != 0 {
			t.Errorf("EncodeWithSpecials(%q) has %d leading space markers, want 0", in, got)
		}
	}
	// The doubling was visible in the text, which is how a user would notice it.
	if got, want := tok.Decode(tok.EncodeWithSpecials("the<|m|> garden", false)), "the garden"; got != want {
		t.Errorf("EncodeWithSpecials(%q) decodes to %q, want %q", "the<|m|> garden", got, want)
	}
}

// TestByteLevelVocabularyWithoutModelKeyIsRecognised covers a file that omits
// tokenizer.ggml.model.
//
// The key is optional in the format, and treating its absence as SentencePiece -- which is what
// llama.cpp does -- is wrong for a byte-level vocabulary. The pieces are spelled with the
// escaped runes, so only the byte-level decoder turns "Ġthe" back into " the"; read as
// SentencePiece the same id decodes to the literal characters, and a model answers in a stream
// of "Ċ" and "Ġ" instead of prose.
//
// Found on a real Qwen2 conversion whose metadata lost the key: it produced "ĊA.ĠTĊB.ĠFĊ"
// where the same weights through a correct file produced "Paris. It is the largest city".
//
// The vocabulary identifies itself: byte-level spelling uses U+0120 for a space and U+010A for
// a newline, and SentencePiece uses U+2581 for both.
func TestByteLevelVocabularyWithoutModelKeyIsRecognised(t *testing.T) {
	base := gguftest.Model{
		TokenizerModel:     "gpt2",
		OmitTokenizerModel: true,
		ContextLen:         64,
	}
	path := base.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !tok.byteLevel {
		t.Fatal("a byte-level vocabulary without the model key was read as SentencePiece")
	}
	// The decisive check is the round trip: a byte-level vocabulary decodes " the" for the
	// piece spelling it, not the literal "Ġthe".
	text := " the"
	ids := tok.Encode(text, false)
	if got := tok.Decode(ids); got != text {
		t.Errorf("round trip %q gave %q", text, got)
	}

	// A SentencePiece vocabulary still takes the default, so the inference does not flip it.
	spPath := gguftest.Model{OmitTokenizerModel: true, ContextLen: 64}.Write(t, t.TempDir())
	spF, err := gguf.Open(spPath)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	spTok, err := New(spF)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if spTok.byteLevel {
		t.Error("a SentencePiece vocabulary without the model key was read as byte-level")
	}
}

// TestBpeUnescapePassthrough covers the two arms of bpeUnescape that a real byte-level
// vocabulary never produces.
//
// The function is documented to handle both families, and its two escape tables cover all 256
// GPT-2 bytes plus the SentencePiece marker. A piece that is neither -- a literal space, or text
// outside both tables such as CJK -- is passed through as itself, which is the right answer for a
// vocabulary that spells its pieces as plain text. Those arms are what keeps this function from
// mangling such a file, and neither was executed: every fixture piece is in the tables.
func TestBpeUnescapePassthrough(t *testing.T) {
	cases := []struct{ in, want string }{
		// In the bpeBytes table, so this is the ordinary path, kept as a control.
		{"\u0120", " "},
		// The SentencePiece marker.
		{"\u2581hello", " hello"},
		// A literal space: r == ' ' arm.
		{"a b", "a b"},
		// Outside both tables: the utf8 append arm.
		{"\u4e2d", "\u4e2d"},
		{"\u4e2d\u6587", "\u4e2d\u6587"},
		// A piece holding both markers takes the slow path (because U+0120 is a table entry
		// above 0x7e) and must still treat U+2581 as a space there, not pass it through.
		{"\u0120\u2581x", "  x"},
		// A literal space inside a piece that also has a table entry above 0x7e: the slow
		// path's own space arm, which the fast path covers by ReplaceAll instead.
		{"\u0120 y", "  y"},
		// And a CJK rune in a piece that is on the slow path: the WriteRune arm, which is what
		// keeps a vocabulary that mixes escaped bytes with plain non-ASCII from losing the
		// plain part.
		{"\u0120\u4e2d", " \u4e2d"},
	}
	for _, c := range cases {
		if got := bpeUnescape(c.in); got != c.want {
			t.Errorf("bpeUnescape(%q) = %q, want %q", c.in, got, c.want)
		}
		// The byte form must agree, since the decoder routes by vocabulary family rather than
		// by piece shape and the two have to produce the same text.
		if got := string(bpeUnescapeBytes(c.in)); got != c.want {
			t.Errorf("bpeUnescapeBytes(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestBigramQueueTieBreaks covers the two left-index tie-breaks in bigramQueue.less.
//
// Both orderings end by comparing left, which is what makes the merge order deterministic: two
// pairs with the same rank or the same score would otherwise come out in whatever order the heap
// happened to arrange them, and the tokenization of text with repeated pairs would stop being
// reproducible. The score-mode tie-break was reached by the existing encode tests; the rank-mode
// one was not, because no fixture has two distinct pairs sharing a rank.
//
// The first branch -- two entries disagreeing about byRank -- is left uncovered on purpose. The
// comment says why: byRank comes from the tokenizer, so one queue never mixes the two orderings,
// and the comparison exists only to stay total if it ever did.
func TestBigramQueueTieBreaks(t *testing.T) {
	t.Run("rank mode ties break on left", func(t *testing.T) {
		q := bigramQueue{
			{byRank: true, rank: 5, left: 10},
			{byRank: true, rank: 5, left: 3},
		}
		if !q.less(1, 0) {
			t.Error("left=3 did not sort before left=10 at equal rank")
		}
		if q.less(0, 1) {
			t.Error("left=10 sorted before left=3")
		}
		// Rank still outranks left, so a lower rank wins even with a larger left.
		q2 := bigramQueue{
			{byRank: true, rank: 9, left: 0},
			{byRank: true, rank: 1, left: 99},
		}
		if !q2.less(1, 0) {
			t.Error("rank 1 did not sort before rank 9")
		}
	})

	t.Run("score mode ties break on left", func(t *testing.T) {
		q := bigramQueue{
			{score: 2.5, left: 10},
			{score: 2.5, left: 3},
		}
		if !q.less(1, 0) {
			t.Error("left=3 did not sort before left=10 at equal score")
		}
		// A larger score wins, which is the unigram direction.
		q2 := bigramQueue{
			{score: 1, left: 0},
			{score: 2, left: 99},
		}
		if !q2.less(1, 0) {
			t.Error("score 2 did not sort before score 1")
		}
	})
}

// TestNewRejectsParallelArraysOfTheWrongLength covers the two length checks between the vocabulary
// and the arrays stored beside it.
//
// A file whose scores or token types do not line up with the tokens describes a vocabulary it cannot
// mean: index i would carry one piece's text and another piece's score. Both checks report the two
// counts, which is what makes the file diagnosable -- "mismatch" alone leaves the reader counting
// arrays by hand.
//
// The fixture builder always writes arrays of the right length, so these are constructed as a File
// directly.
func TestNewRejectsParallelArraysOfTheWrongLength(t *testing.T) {
	vocab := func() []gguf.Value {
		return []gguf.Value{
			{Type: gguf.TypeString, Str: "a"},
			{Type: gguf.TypeString, Str: "b"},
			{Type: gguf.TypeString, Str: "c"},
		}
	}
	base := func() map[string]gguf.Value {
		return map[string]gguf.Value{
			"tokenizer.ggml.tokens": {Type: gguf.TypeArray, Array: vocab()},
		}
	}

	t.Run("short scores", func(t *testing.T) {
		md := base()
		md["tokenizer.ggml.scores"] = gguf.Value{Type: gguf.TypeArray, Array: []gguf.Value{
			{Type: gguf.TypeFloat32, Float: 1},
		}}
		_, err := New(&gguf.File{Metadata: md})
		if err == nil {
			t.Fatal("New accepted one score for three tokens")
		}
		// Both counts belong in the message: the reader has to know which array is short.
		for _, want := range []string{"1", "3"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error %q does not state %s", err, want)
			}
		}
	})

	t.Run("short token types", func(t *testing.T) {
		md := base()
		md["tokenizer.ggml.token_type"] = gguf.Value{Type: gguf.TypeArray, Array: []gguf.Value{
			{Type: gguf.TypeInt32, Int: 1},
			{Type: gguf.TypeInt32, Int: 1},
		}}
		_, err := New(&gguf.File{Metadata: md})
		if err == nil {
			t.Fatal("New accepted two token types for three tokens")
		}
		for _, want := range []string{"2", "3"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error %q does not state %s", err, want)
			}
		}
	})

	// The same file with matching arrays builds, so the refusals are about the lengths.
	t.Run("matching arrays are accepted", func(t *testing.T) {
		md := base()
		scores := make([]gguf.Value, 3)
		types := make([]gguf.Value, 3)
		for i := range 3 {
			scores[i] = gguf.Value{Type: gguf.TypeFloat32, Float: float64(i)}
			types[i] = gguf.Value{Type: gguf.TypeInt32, Int: 1}
		}
		md["tokenizer.ggml.scores"] = gguf.Value{Type: gguf.TypeArray, Array: scores}
		md["tokenizer.ggml.token_type"] = gguf.Value{Type: gguf.TypeArray, Array: types}
		if _, err := New(&gguf.File{Metadata: md}); err != nil {
			t.Fatalf("New refused a matching file: %v", err)
		}
	})
}

// TestEncodeAddsTheDummyPrefixUnconditionally pins a rule that looks wrong until it is
// checked against SentencePiece itself.
//
// " hello" encodes to one more token than "hello", which reads like a doubled space: the
// prefix function is conditional on `first` and nothing else, so a caller who wrote a leading
// space appears to get a second one. It was checked against the reference implementation
// rather than reasoned about -- sentencepiece 0.2.3, a BPE model trained with
// add_dummy_prefix=True and remove_extra_whitespaces=False, gives
//
//	'hello'   -> ['▁hello']
//	' hello'  -> ['▁', '▁hello']
//	'  hello' -> ['▁', '▁', '▁hello']
//
// so the prefix is unconditional there too, and the extra piece is the caller's own space
// standing alone. The counts are asserted rather than the ids because the ids belong to the
// fixture vocabulary; the property is that each additional leading space costs one token.
//
// A run of spaces on its own follows the same rule and is the case that makes it obvious:
// n spaces encode to n+1 pieces, all of them the marker.
func TestEncodeAddsTheDummyPrefixUnconditionally(t *testing.T) {
	tok := newTestTokenizer(t)
	if !tok.addSpacePrefix {
		t.Skip("this fixture does not use a SentencePiece vocabulary")
	}

	// One token per character of "hello" plus the prefix.
	base := len(tok.Encode("hello", false))
	for _, c := range []struct {
		text  string
		extra int
	}{
		{"hello", 0},
		{" hello", 1},
		{"  hello", 2},
		{"   hello", 3},
	} {
		got := len(tok.Encode(c.text, false))
		if got != base+c.extra {
			t.Errorf("Encode(%q) is %d tokens, want %d (%d + %d leading-space pieces)",
				c.text, got, base+c.extra, base, c.extra)
		}
	}

	// A string of nothing but spaces: every one is its own piece after the prefix.
	for _, n := range []int{1, 2, 5} {
		text := strings.Repeat(" ", n)
		if got := len(tok.Encode(text, false)); got != n+1 {
			t.Errorf("Encode(%d spaces) is %d tokens, want %d", n, got, n+1)
		}
		// And the round trip still returns exactly what was written, prefix included.
		if back := tok.Decode(tok.Encode(text, false)); back != text {
			t.Errorf("Decode(Encode(%q)) = %q", text, back)
		}
	}
}

// TestControlMarkersEncodeButDoNotDecode pins the asymmetry between the two directions for a
// chat template's markers.
//
// EncodeWithSpecials turns <|m|> into its own id, because a template means the token and not
// the eight characters it is spelled from. Decode then drops that id, because a control piece
// is markup rather than prose: printing it back would put "<|im_start|>" into a user's reply.
// So the pair is deliberately not a round trip, and stating it here keeps a later reader from
// "fixing" either half.
//
// The surface form is asserted too, not just the id: without the special the same text encodes
// to the characters separately, which is what a user typing the marker by hand gets, and the
// two must differ. That difference is the reason the function exists.
func TestControlMarkersEncodeButDoNotDecode(t *testing.T) {
	tok := newTestTokenizer(t)
	const marker = "<|m|>"
	tok.specials = append(tok.specials, specialToken{text: marker, id: 1})

	// Plain words on purpose: the fixture vocabulary has no newline piece, so a text with
	// one would decode to the unknown surface and the assertion would be about that instead.
	const body = "user hi assistant"
	// As markup: one id, and the text around it survives it.
	ids := tok.EncodeWithSpecials(marker+body+marker, false)
	occurrences := 0
	for _, id := range ids {
		if id == 1 {
			occurrences++
		}
	}
	if occurrences != 2 {
		t.Errorf("the marker became %d ids in %v, want 2", occurrences, ids)
	}
	// As prose: the marker is spelled out, so it cannot be one id.
	plain := tok.Encode(marker+body+marker, false)
	for _, id := range plain {
		if id == 1 {
			t.Errorf("Encode produced the control id for a marker typed as text: %v", plain)
		}
	}
	if len(plain) <= len(ids) {
		t.Errorf("the same text encoded to %d ids as markup and %d as prose; the markup form "+
			"should be shorter because the marker collapses", len(ids), len(plain))
	}

	// Decoding drops the control ids and keeps what they separated.
	if got, want := tok.Decode(ids), body; got != want {
		t.Errorf("Decode = %q, want %q: a control piece is markup, not output", got, want)
	}
	// Two adjacent markers produce no tokens at all between them, which is what keeps a
	// template's spacing intact: an empty gap is not text.
	adjacent := tok.EncodeWithSpecials(marker+marker, false)
	if len(adjacent) != 2 {
		t.Errorf("two adjacent markers gave %d ids %v, want exactly 2 and nothing between",
			len(adjacent), adjacent)
	}
	if got := tok.Decode(adjacent); got != "" {
		t.Errorf("Decode of two adjacent markers = %q, want empty", got)
	}
}

// TestChatEncodingResolvesControlPiecesInMessageText records a behaviour that is currently
// reachable from the HTTP API and that nobody has decided about.
//
// /v1/chat/completions encodes its prompt with EncodeWithSpecials, because a rendered chat
// template is markup: the "<|im_start|>" the template writes has to become one control token
// rather than six characters. The side effect is that the same resolution applies to text the
// client wrote, and the server cannot tell the two apart -- renderChat concatenates the
// template and the message content into one string before this call.
//
// Measured on the fixture vocabulary: a message containing "<s>", "</s>" or "<unk>" encodes
// to a run that contains the control id (1, 2, 0 respectively), three ids in total, where the
// same text through Encode -- the path /v1/completions takes -- is five to seven ordinary
// characters. llama.cpp's llama_tokenize defaults to parse_special=false for exactly this
// reason, so a client that wants a marker taken literally has no way to ask for it here.
//
// This test pins the current behaviour rather than endorsing it. Fixing it means encoding the
// template's own markers separately from the interpolated content, which changes the renderer's
// contract and both call sites (the server and the CLI's chat command); that is a decision for
// the maintainer, not a silent change. If that is done, this test should fail and be rewritten
// to assert the safe behaviour.
func TestChatEncodingResolvesControlPiecesInMessageText(t *testing.T) {
	tok := newTestTokenizer(t)

	// newTestTokenizer marks pieces Control in types but does not build the specials list --
	// that happens in New -- so EncodeWithSpecials would fall through to plain encoding and
	// this test would measure nothing. Build it the way New does.
	type control struct {
		text string
		id   int32
	}
	var controls []control
	for id := range tok.tokens {
		if tok.types[id] == Control {
			controls = append(controls, control{text: tok.tokens[id], id: int32(id)})
			tok.specials = append(tok.specials, specialToken{text: tok.tokens[id], id: int32(id)})
		}
	}
	if len(controls) == 0 {
		t.Skip("the fixture vocabulary has no control pieces")
	}

	for _, c := range controls {
		t.Run(c.text, func(t *testing.T) {
			// A marker embedded in text the way a client would send it.
			const text = "please quote THIS literally"
			withMarker := "please quote " + c.text + " literally"

			chat := tok.EncodeWithSpecials(withMarker, false)
			plain := tok.Encode(withMarker, false)

			// The marker resolves to the control id on the chat path...
			found := false
			for _, id := range chat {
				if id == c.id {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("EncodeWithSpecials(%q) = %v, which does not contain the control id %d: "+
					"this test records the behaviour, so a change here is the fix landing",
					withMarker, chat, c.id)
			}
			// ...and to ordinary characters on the completions path, which is the difference
			// that makes the chat endpoint an injection surface rather than a spelling rule.
			for _, id := range plain {
				if id == c.id {
					t.Errorf("Encode(%q) contains the control id %d; the completions path is "+
						"documented as taking text literally", withMarker, c.id)
				}
			}
			if len(plain) <= len(chat) {
				t.Errorf("the same text is %d ids as prose and %d as markup; the marker should "+
					"collapse to one id on the markup path", len(plain), len(chat))
			}
			// The control-free control: text without the marker must not gain one.
			if ids := tok.EncodeWithSpecials(text, false); containsID(ids, c.id) {
				t.Errorf("text without the marker encoded to %v, which contains the control id %d",
					ids, c.id)
			}
		})
	}
}

// containsID reports whether ids holds want.
func containsID(ids []int32, want int32) bool {
	for _, id := range ids {
		if id == want {
			return true
		}
	}
	return false
}

// TestEncodeSurvivesADocumentSizedInput is the property that makes this encoder safe on a
// large prompt: it handles a megabyte in one chunk without recursing.
//
// Nothing pinned this. The merge loop is iterative -- a symbol list with prev/next links and a
// priority queue, no recursion anywhere -- and measured on the fixture, a one-megabyte run of a
// single character encodes to 1,000,001 tokens (the extra one is the dummy prefix) in 110ms and
// 168 MiB. The count is the assertion: a recursive merge, or one that splits the chunk first,
// either changes how the text is cut or overflows the stack, and both show up as a different
// token count or a dead test.
//
// A timing assertion was written first and removed: replacing the heap pop with a linear scan --
// genuinely quadratic, same answers -- still passed, because the fixture's vocabulary produces
// short queues and the quadratic term never dominates. A test that cannot tell the two apart
// would assert a property it does not check, which is worse than not asserting it.
func TestEncodeSurvivesADocumentSizedInput(t *testing.T) {
	tok := newTestTokenizer(t)
	text := strings.Repeat("x", 1<<20)
	ids := tok.Encode(text, false)
	// One token per character plus the dummy prefix. An implementation that recursed would
	// either overflow the stack or, in a language without one, take long enough to be obvious;
	// one that chunked the text differently would produce a different count.
	if want := len(text) + 1; len(ids) != want {
		t.Errorf("%d tokens for %d characters, want %d", len(ids), len(text), want)
	}

	// The same input through EncodeWithSpecials takes the other pre-tokenizer path and must
	// handle it just as well.
	if got := tok.EncodeWithSpecials(text, false); len(got) != len(ids) {
		t.Errorf("EncodeWithSpecials produced %d tokens against Encode's %d for text with no "+
			"markers in it", len(got), len(ids))
	}
}

// TestNewRejectsATokenTypeOutsideTheFormat pins the range the token_type array may hold.
//
// GGUF defines seven roles, 0 through 6. The loader copied whatever integers it found straight
// into the type slice, and every comparison in this package is an exact match against Control
// or Unknown, so an unknown value behaved as Normal without anything saying so. Measured on a
// hand-built vocabulary whose types were all 99: it encoded and decoded exactly like an
// all-Normal one, while the same file with all 3 (Control) decoded to the empty string. A
// damaged Control entry is therefore the silent case -- a marker that should be suppressed
// starts reaching the text.
//
// The package's own rule is that a parser reports an unknown enum rather than guessing, which
// is why this refuses instead of defaulting to Normal.
func TestNewRejectsATokenTypeOutsideTheFormat(t *testing.T) {
	// build writes a vocabulary of four pieces with every type set to v.
	build := func(t *testing.T, v int32) *gguf.File {
		t.Helper()
		var b []byte
		u32 := func(x uint32) { var w [4]byte; binary.LittleEndian.PutUint32(w[:], x); b = append(b, w[:]...) }
		u64 := func(x uint64) { var w [8]byte; binary.LittleEndian.PutUint64(w[:], x); b = append(b, w[:]...) }
		str := func(s string) { u64(uint64(len(s))); b = append(b, s...) }
		tokens := []string{"<unk>", "<s>", "a", "b"}
		b = append(b, []byte("GGUF")...)
		u32(3)
		u64(0) // no tensors
		u64(3) // three metadata entries
		str("tokenizer.ggml.tokens")
		u32(uint32(gguf.TypeArray))
		u32(uint32(gguf.TypeString))
		u64(uint64(len(tokens)))
		for _, tk := range tokens {
			str(tk)
		}
		str("tokenizer.ggml.token_type")
		u32(uint32(gguf.TypeArray))
		u32(uint32(gguf.TypeInt32))
		u64(uint64(len(tokens)))
		for range tokens {
			u32(uint32(v))
		}
		str("tokenizer.ggml.model")
		u32(uint32(gguf.TypeString))
		str("llama")
		f, err := gguf.Read(bytes.NewReader(b))
		if err != nil {
			t.Fatalf("Read: %v", err)
		}
		return f
	}

	// Every defined role must still load, so the check cannot be satisfied by refusing
	// everything.
	for _, v := range []int32{0, 1, 2, 3, 4, 5, 6} {
		if _, err := New(build(t, v)); err != nil {
			t.Errorf("New rejected the defined type %d: %v", v, err)
		}
	}

	// And the values just outside the range must be named, with the offending value and the
	// range in the message -- "invalid token type" would leave a reader guessing which entry.
	for _, c := range []struct {
		v    int32
		want string
	}{
		{7, "7"}, {99, "99"}, {-1, "-1"},
	} {
		_, err := New(build(t, c.v))
		if err == nil {
			t.Errorf("New accepted the undefined type %d; it would behave as Normal", c.v)
			continue
		}
		msg := err.Error()
		if !strings.Contains(msg, c.want) {
			t.Errorf("error %q does not name the value %s", msg, c.want)
		}
		if !strings.Contains(msg, "0-6") {
			t.Errorf("error %q does not name the valid range", msg)
		}
	}
}

// TestLooksByteLevel pins the inference used when tokenizer.ggml.model is absent.
//
// The one indirect test covers the case that matters most -- a real byte-level vocabulary
// with the key missing -- but nothing asserted the conditions themselves, and the function
// decides between two decodings that produce different text from the same ids. The comment
// on the caller records what that costs: a Qwen2 conversion without the key produced
// "ĊA.ĠTĊB.ĠFĊ" where the correct file produced a sentence.
//
// Both markers are required and the SentencePiece marker vetoes them, so the rule cannot be
// satisfied by a vocabulary that merely contains one of the three by accident.
func TestLooksByteLevel(t *testing.T) {
	cases := []struct {
		name   string
		tokens []string
		want   bool
		why    string
	}{
		{"both markers", []string{"a", "\u0120", "\u010a"}, true,
			"a space and a newline marker together identify the byte-level spelling"},
		{"space only", []string{"a", "\u0120"}, false,
			"one marker is not enough; a piece spelling a space can appear in other schemes"},
		{"newline only", []string{"a", "\u010a"}, false, "the same, for the other marker"},
		{"both plus SentencePiece", []string{"\u0120", "\u010a", "\u2581"}, false,
			"U+2581 vetoes: a vocabulary carrying it is SentencePiece whatever else it holds"},
		{"SentencePiece only", []string{"\u2581"}, false, "the ordinary SentencePiece case"},
		{"neither", []string{"a", "b"}, false,
			"no evidence is not evidence for byte-level, so the SentencePiece default stands"},
		{"empty", nil, false, "an empty vocabulary identifies nothing"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := looksByteLevel(c.tokens); got != c.want {
				t.Errorf("looksByteLevel(%q) = %v, want %v: %s", c.tokens, got, c.want, c.why)
			}
		})
	}

	// The scan is bounded, and the bound is a real limit rather than a detail: markers past
	// it are not seen, so such a vocabulary falls back to the SentencePiece default. Every
	// real vocabulary puts them in the first few hundred entries -- measured, the fixture's
	// are at 117 and 139 -- which is the claim the bound rests on. Asserted so the
	// consequence is recorded rather than to suggest the bound is wrong.
	late := make([]string, 5000)
	late[4500], late[4501] = "\u0120", "\u010a"
	if looksByteLevel(late) {
		t.Error("markers past the scan limit were seen, so the bound is not 4096")
	}
}
