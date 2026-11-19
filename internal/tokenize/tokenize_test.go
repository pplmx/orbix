package tokenize

import (
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
	if len(first) == len(second) && len(first) == 2 &&
		first[0] == second[0] && first[1] == second[1] {
		t.Errorf("reversing the merge ranks produced the same tokens %v; "+
			"rank order is not being used", first)
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
