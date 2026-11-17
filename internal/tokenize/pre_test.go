package tokenize

import (
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
)

// The pre-tokenizer decides where a merge may not cross. These tests use the chunk
// boundaries rather than token ids, because the boundaries are the rule and the ids follow
// from them -- a table of ids would pass for the wrong reason if the vocabulary changed.

func TestQwen2Chunks(t *testing.T) {
	// Boundaries taken from the Qwen2 expression's behaviour: a contraction is its own
	// chunk, a single digit is its own chunk, and the space before a word belongs to the
	// word.
	cases := []struct {
		in   string
		want []string
	}{
		{"hello  world", []string{"hello", " ", " world"}},
		{"   spaces   ", []string{"  ", " spaces", "   "}},
		{"hello\nworld", []string{"hello", "\n", "world"}},
		{"don't stop", []string{"don", "'t", " stop"}},
		{"123abc", []string{"1", "2", "3", "abc"}},
		{"abc123", []string{"abc", "1", "2", "3"}},
		{"x=1+2", []string{"x", "=", "1", "+", "2"}},
	}
	for _, c := range cases {
		got := (&Tokenizer{pre: preQwen2}).preSplit(c.in)
		if len(got) != len(c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
			continue
		}
		for i := range c.want {
			if got[i] != c.want[i] {
				t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
				break
			}
		}
	}
}

// TestQwen2ChunksCoverInput checks that splitting loses nothing and duplicates nothing.
// A splitter that drops a character is silent: the text still tokenizes, just not the
// text that was passed in.
func TestQwen2ChunksCoverInput(t *testing.T) {
	for _, in := range []string{
		" ", "  ", "a", "a b", "hello  world", "mixed \n\t spaces",
		"'.", "don't", "  \n\n  ", "日本語", "abc123def", "a\u00a0b",
	} {
		// The empty string is excluded above: its one "chunk" is the empty chunk, which
		// joins back to the input like any other.
		var joined string
		for _, chunk := range (&Tokenizer{pre: preQwen2}).preSplit(in) {
			if chunk == "" {
				t.Fatalf("preSplit(%q) produced an empty chunk", in)
			}
			joined += chunk
		}
		if joined != in {
			t.Errorf("preSplit(%q) loses text: joined = %q", in, joined)
		}
	}
}

// TestNoPreTokenizerKeepsOneChunk pins the default: a vocabulary without the key must
// behave as it did before the key was read.
func TestNoPreTokenizerKeepsOneChunk(t *testing.T) {
	for _, pre := range []preType{preNone} {
		got := (&Tokenizer{pre: pre}).preSplit("hello  world")
		if len(got) != 1 || got[0] != "hello  world" {
			t.Errorf("preSplit with no pre-tokenizer = %q, want the whole text", got)
		}
	}
}

// TestQwen2Contractions checks the clitic rule directly: it is case-insensitive and only
// fires at a chunk start.
func TestQwen2Contractions(t *testing.T) {
	for _, in := range []string{"'s", "'T", "'re", "'VE", "'m", "'LL", "'d"} {
		if n := qwen2ContractionLen(in); n != len(in) {
			t.Errorf("qwen2ContractionLen(%q) = %d, want %d", in, n, len(in))
		}
	}
	if n := qwen2ContractionLen("'x"); n != 0 {
		t.Errorf("qwen2ContractionLen(\"'x\") = %d, want 0", n)
	}
}

// The Llama 3 family rule is the Qwen2 rule with digit groups of up to three, plus two
// behaviours that are not part of the expression at all. Both are tested here because
// both are invisible when wrong: the tokenizer still produces valid ids.

// TestLlama3DigitGroups checks the one rule the two families disagree about.
func TestLlama3DigitGroups(t *testing.T) {
	llama := &Tokenizer{pre: preLlama3}
	qwen := &Tokenizer{pre: preQwen2}
	cases := []struct {
		in          string
		llama, qwen []string
	}{
		{"999", []string{"999"}, []string{"9", "9", "9"}},
		{"123456", []string{"123", "456"}, []string{"1", "2", "3", "4", "5", "6"}},
		{"a1234b", []string{"a", "123", "4", "b"}, []string{"a", "1", "2", "3", "4", "b"}},
		{"x=42", []string{"x", "=", "42"}, []string{"x", "=", "4", "2"}},
	}
	for _, c := range cases {
		if got := llama.preSplit(c.in); !sameStrings(got, c.llama) {
			t.Errorf("llama3 preSplit(%q) = %q, want %q", c.in, got, c.llama)
		}
		if got := qwen.preSplit(c.in); !sameStrings(got, c.qwen) {
			t.Errorf("qwen2 preSplit(%q) = %q, want %q", c.in, got, c.qwen)
		}
	}
}

// TestLlama3IgnoresMergesForWholePieces checks the behaviour that is not in the
// expression: a chunk the vocabulary spells as one piece is emitted as that piece, even
// when the merge ranks would have split it.
func TestLlama3IgnoresMergesForWholePieces(t *testing.T) {
	tok := bpeFixture(t, gguftest.Model{
		Vocab: []string{"<unk>", "<s>", "</s>", "\u0120", "a", "b", "c", "ab", "abc"},
		// "a b" outranks "b c", so the merge path forms ab+c rather than taking the
		// "abc" piece that exists.
		Merges:         []string{"a b", "b c"},
		TokenizerModel: "gpt2",
		NoBytes:        true,
		Pre:            "llama-bpe",
	})
	if !tok.ignoreMerges {
		t.Fatal("a llama-bpe vocabulary did not set ignoreMerges")
	}
	got := tok.Encode("abc", false)
	want := []int32{tok.index["abc"]}
	if len(got) != 1 || got[0] != want[0] {
		t.Errorf("Encode(abc) = %v, want [%d] (the whole piece, not the merges)", got, want[0])
	}

	// Dropping the flag must change the answer, which is what makes this a test of the
	// flag rather than of the vocabulary.
	tok.ignoreMerges = false
	if split := tok.Encode("abc", false); len(split) == 1 && split[0] == want[0] {
		t.Error("without ignoreMerges the result is unchanged; this fixture does not exercise it")
	}
}

// TestAddBOSIsOverridableByTheFile checks tokenizer.ggml.add_bos_token in both
// directions.
//
// The default is to add it, which is what every checkpoint that names a BOS token expects.
// A file that states add_bos_token = false must suppress it -- and that override has to be
// read rather than assumed from the family, because a pre-tokenizer and a file that
// disagreed silently would produce a prompt with one token too many or too few.
func TestAddBOSIsOverridableByTheFile(t *testing.T) {
	base := gguftest.Model{
		Vocab:          []string{"<unk>", "<s>", "</s>", "\u0120", "a"},
		TokenizerModel: "gpt2",
		NoBytes:        true,
		Pre:            "llama-bpe",
		BOSID:          1,
	}

	defaults := bpeFixture(t, base)
	if ids := defaults.Encode("a", true); len(ids) == 0 || ids[0] != 1 {
		t.Errorf("Encode(a, addSpecial) = %v, want BOS 1 first by default", ids)
	}

	off := base
	off.AddBOS = ptrTo(false)
	if ids := bpeFixture(t, off).Encode("a", true); len(ids) > 0 && ids[0] == 1 {
		t.Errorf("add_bos_token = false was ignored: %v starts with BOS", ids)
	}
}

// ptrTo returns a pointer to v, for the tri-state fixture fields.
func ptrTo[T any](v T) *T { return &v }

// bpeFixture builds a tokenizer over a fixture model.
func bpeFixture(t *testing.T, m gguftest.Model) *Tokenizer {
	t.Helper()
	f, err := gguf.Open(m.Write(t, t.TempDir()))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return tok
}

// sameStrings reports whether two string slices are equal.
func sameStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestUnknownPreTokenizerIsInert checks that a name this package does not know falls back
// to no chunking rather than to a guess. A wrong chunking rule produces text the model was
// never trained on with nothing to report it, while no rule is what this package did
// before it read the key at all.
func TestUnknownPreTokenizerIsInert(t *testing.T) {
	tok := bpeFixture(t, gguftest.Model{
		Vocab:          []string{"<unk>", "<s>", "</s>", "\u0120", "a", "b", "ab"},
		Merges:         []string{"a b"},
		TokenizerModel: "gpt2",
		NoBytes:        true,
		Pre:            "no-such-pre-tokenizer",
	})
	if tok.pre != preNone {
		t.Errorf("unknown pre-tokenizer mapped to %v, want preNone", tok.pre)
	}
	if got := tok.preSplit("a  b"); !sameStrings(got, []string{"a  b"}) {
		t.Errorf("preSplit with an unknown pre-tokenizer = %q, want the whole text", got)
	}
	if tok.ignoreMerges {
		t.Error("an unknown pre-tokenizer set ignoreMerges")
	}
}

// The GPT-2 family is the same seven-alternative shape as qwen2 with three differences:
// an optional *space* prefix rather than an optional any-non-letter, digits as one run,
// and no newline alternative. Those three are what the tests below pin.

// TestGPT2Chunks checks the chunk boundaries against the expression llama.cpp stores:
// "'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)".
func TestGPT2Chunks(t *testing.T) {
	tok := &Tokenizer{pre: preGPT2}
	cases := []struct {
		in   string
		want []string
	}{
		// The space belongs to the word after it, which is what " ?" does.
		{"hello  world", []string{"hello", " ", " world"}},
		{"don't stop", []string{"don", "'t", " stop"}},
		// A number is one chunk, unlike qwen2's single digit.
		{"123456", []string{"123456"}},
		{"x=42", []string{"x", "=", "42"}},
		// A punctuation mark is its own chunk, unlike qwen2 where it prefixes the word.
		{"a.b,c", []string{"a", ".", "b", ",", "c"}},
		{"   spaces   ", []string{"  ", " spaces", "   "}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("gpt2 preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestGPT2DiffersFromQwen2 is the other half: the two families must not produce the same
// chunks, or the family parameter is not being read.
func TestGPT2DiffersFromQwen2(t *testing.T) {
	gpt := &Tokenizer{pre: preGPT2}
	qwen := &Tokenizer{pre: preQwen2}
	for _, in := range []string{"123456", "a.b", "x=42"} {
		if sameStrings(gpt.preSplit(in), qwen.preSplit(in)) {
			t.Errorf("%q splits the same under gpt-2 and qwen2; the family is ignored", in)
		}
	}
}

// TestPreSplitNeverLosesText pins the safety property that makes the hand-written splitter
// acceptable: every byte of the input appears in exactly one chunk.
//
// This is not a formality. The expression llama.cpp stores for GPT-2 does not cover every
// position -- "a.\nb" has no alternative that matches the newline -- so a splitter that
// simply took the matches would drop it. llama.cpp reads the same expression through a
// hand-written splitter whose last branch is "_add_token(++pos)", consuming one character
// when nothing matched. Taking one byte on a gap is this package's equivalent, and it is
// why a chunk list is always a partition rather than a subset.
func TestPreSplitNeverLosesText(t *testing.T) {
	// The empty string is excluded: its single chunk is the empty chunk, and joining that
	// back gives the input like any other, which is the contract the other coverage test
	// states too.
	inputs := []string{
		"a.\nb", "x!\ny", "hi. \nthere", "a\n\nb", "\n", "\n\n", "\r\n",
		"é", "café", "日本語", "a\u0301", // a plus a combining acute
		" ", "  \t  ", "\x00", "a\x00b", "abc123", "a.b,c", "   x   ",
	}
	for _, fam := range []preType{preGPT2, preQwen2, preLlama3} {
		tok := &Tokenizer{pre: fam}
		for _, in := range inputs {
			var joined string
			for _, chunk := range tok.preSplit(in) {
				if chunk == "" {
					t.Errorf("pre=%v preSplit(%q) produced an empty chunk", fam, in)
				}
				joined += chunk
			}
			if joined != in {
				t.Errorf("pre=%v preSplit(%q) = %q, which does not cover the input", fam, in, joined)
			}
		}
	}
}

// The starcoder family is the GPT-2 expression with a bare `\p{N}` applied first.
//
// llama.cpp applies a family's expressions in sequence, each cutting the previous round's
// chunks further rather than picking one of them. That is why the result is one chunk per
// digit and not a different digit count: by the time the main rule sees a digit it is
// already alone, so its " ?\p{N}+" alternative has nothing to join.
func TestStarCoderChunks(t *testing.T) {
	tok := &Tokenizer{pre: preStarCoder}
	cases := []struct {
		in   string
		want []string
	}{
		{"123456", []string{"1", "2", "3", "4", "5", "6"}},
		{"x=42", []string{"x", "=", "4", "2"}},
		{"abc123def", []string{"abc", "1", "2", "3", "def"}},
		// Everything that is not a digit keeps the GPT-2 grouping.
		{"hello  world", []string{"hello", " ", " world"}},
		{"a.b,c", []string{"a", ".", "b", ",", "c"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("starcoder preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestStarCoderDiffersFromGPT2OnlyInDigits pins the relationship between the two families:
// the same expression, plus a digit cut. Asserting the difference is what would catch the
// first expression being dropped.
func TestStarCoderDiffersFromGPT2OnlyInDigits(t *testing.T) {
	sc := &Tokenizer{pre: preStarCoder}
	gpt := &Tokenizer{pre: preGPT2}

	// A number: the two must disagree.
	if sameStrings(sc.preSplit("123456"), gpt.preSplit("123456")) {
		t.Error("starcoder and gpt-2 agree on a number; the digit cut is not happening")
	}
	// Text without digits: the two must agree, since only the digit rule differs.
	for _, in := range []string{"hello  world", "don't stop", "a.b,c", "   spaces   "} {
		if !sameStrings(sc.preSplit(in), gpt.preSplit(in)) {
			t.Errorf("%q splits differently under starcoder and gpt-2, but neither has a "+
				"digit in it", in)
		}
	}
}

// TestPreSplitDigitsIsTwoPasses checks the property that distinguishes the starcoder
// implementation from "the digit count is one": the digit cut happens before the main rule,
// not as a parameter of it.
//
// A single pass with digits=1 would also produce one digit per chunk for "123", so that
// case does not tell the two apart. These do: the digit rule is a separate expression, so
// digits that the main rule would have grouped with a neighbour are already separated when
// it runs.
func TestPreSplitDigitsIsTwoPasses(t *testing.T) {
	sc := &Tokenizer{pre: preStarCoder}

	// "1 " -- the main rule's " ?\p{N}+" would take the space with the digit if the digit
	// cut had not already isolated it.
	if got := sc.preSplit(" 1"); !sameStrings(got, []string{" ", "1"}) {
		t.Errorf("preSplit(%q) = %q, want [\" \" \"1\"]: the space prefix must not be "+
			"available once the digit is alone", " 1", got)
	}
	// A digit adjacent to a letter is separated by the first pass, and the letter keeps its
	// own grouping from the second.
	if got := sc.preSplit("ab1cd"); !sameStrings(got, []string{"ab", "1", "cd"}) {
		t.Errorf("preSplit(%q) = %q, want [\"ab\" \"1\" \"cd\"]", "ab1cd", got)
	}
}

// TestPreTokenizerStates pins the three-way distinction the accessors exist for.
//
// A file naming a family this package has not implemented splits as one chunk, which is
// silent: the text still tokenizes, just not the text the model was trained on. A file with
// a typo in the key does the same. Both look identical from the output, so the package has
// to be asked, and the answer is only useful if the two cases are separate -- one is work
// remaining, the other is a broken file.
func TestPreTokenizerStates(t *testing.T) {
	cases := []struct {
		pre       string
		wantImpl  bool
		wantKnown bool
		why       string
	}{
		{"", true, true, "no key: llama.cpp applies no rule to such a file either"},
		{"default", true, true, `"default" means no rule, which not splitting matches`},
		{"qwen2", true, true, "implemented"},
		{"llama-bpe", true, true, "implemented"},
		{"gpt-2", true, true, "implemented"},
		{"starcoder", true, true, "implemented"},
		{"deepseek-v3", true, true, "implemented"},
		{"hunyuan-dense", true, true, "same rule as deepseek-v3"},
		{"tekken", true, true, "implemented"},
		{"gpt-4o", true, true, "the GPT4O family: tekken's shape with 1-3 digit groups and a clitic suffix"},
		{"llama4", true, true, "same rule as gpt-4o"},
		{"qwen-2", false, false, "a typo: not a family name at all"},
		{"QWEN2", false, false, "the names are case-sensitive"},
	}
	for _, c := range cases {
		path := gguftest.Model{
			Vocab:          []string{"<unk>", "<s>", "</s>", "\u2581", "a"},
			Merges:         []string{"a a"},
			TokenizerModel: "gpt2",
			NoBytes:        true,
			Pre:            c.pre,
			ContextLen:     64,
		}.Write(t, t.TempDir())
		f, err := gguf.Open(path)
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		tok, err := New(f)
		if err != nil {
			t.Fatalf("New(%q): %v", c.pre, err)
		}
		if got := tok.PreImplemented(); got != c.wantImpl {
			t.Errorf("%q PreImplemented() = %v, want %v (%s)", c.pre, got, c.wantImpl, c.why)
		}
		if got := tok.PreKnown(); got != c.wantKnown {
			t.Errorf("%q PreKnown() = %v, want %v (%s)", c.pre, got, c.wantKnown, c.why)
		}
	}
}

// TestKnownPreNamesHasNoDuplicates keeps the table a set: a duplicate would mean a name was
// added twice while implementing two families, which hides that one of them is not actually
// reachable.
func TestKnownPreNamesHasNoDuplicates(t *testing.T) {
	seen := make(map[string]bool, len(knownPreNames))
	for _, name := range knownPreNames {
		if seen[name] {
			t.Errorf("%q appears twice in knownPreNames", name)
		}
		seen[name] = true
	}
	if len(knownPreNames) < 50 {
		t.Errorf("knownPreNames has %d entries; llama.cpp maps about 100 names", len(knownPreNames))
	}
}

// The deepseek-v3 family is the first with three expressions, and the first whose main rule
// names the punctuation categories rather than their complement.
func TestDeepseek3Chunks(t *testing.T) {
	tok := &Tokenizer{pre: preDeepseek3}
	cases := []struct {
		in   string
		want []string
	}{
		// \p{N}{1,3} first, so digits group in threes rather than singly or as a run.
		{"123456", []string{"123", "456"}},
		{"x=42", []string{"x", "=", "42"}},
		// The punctuation-then-letters alternative comes before the letter alternative, so
		// a mark attaches to the following ASCII word.
		{"a.b,c", []string{"a", ".b", ",c"}},
		{"def f(x):", []string{"def", " f", "(x", "):"}},
		// The CJK expression cuts its own runs.
		{"日本", []string{"日本"}},
		{"日a本", []string{"日", "a", "本"}},
		// \p{L}\p{M} -- a combining mark joins the letter run.
		{"cafe\u0301", []string{"cafe\u0301"}},
		{"hello  world", []string{"hello", " ", " world"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("deepseek-v3 preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestDeepseek3DiffersFromQwen2InPunctuation is the sharpest difference between the two:
// qwen2's leading character is "[^\r\n\p{L}\p{N}]?", so a mark prefixes the next word;
// deepseek's excludes punctuation, and pairs it with letters through a separate alternative
// that only runs while the letters are ASCII.
func TestDeepseek3DiffersFromQwen2InPunctuation(t *testing.T) {
	ds := &Tokenizer{pre: preDeepseek3}
	qw := &Tokenizer{pre: preQwen2}
	if got, want := ds.preSplit("a.b"), []string{"a", ".b"}; !sameStrings(got, want) {
		t.Errorf("deepseek-v3 preSplit(\"a.b\") = %q, want %q", got, want)
	}
	if got, want := qw.preSplit("a.b"), []string{"a", ".b"}; !sameStrings(got, want) {
		t.Errorf("qwen2 preSplit(\"a.b\") = %q, want %q", got, want)
	}
	// A digit after a mark separates them: qwen2's digit rule has no prefix, and deepseek's
	// number grouping starts its own chunk.
	if got, want := ds.preSplit("x=42"), []string{"x", "=", "42"}; !sameStrings(got, want) {
		t.Errorf("deepseek-v3 preSplit(\"x=42\") = %q, want %q", got, want)
	}
	if got, want := qw.preSplit("x=42"), []string{"x", "=", "4", "2"}; !sameStrings(got, want) {
		t.Errorf("qwen2 preSplit(\"x=42\") = %q, want %q", got, want)
	}
}

// The tekken family is GPT-4o's, and it is the first whose letter rule has internal
// structure: a run is cut at a lower-to-upper case boundary.
//
// The subtlety is which side takes the trailing lower-case stretch. A run of capitals keeps
// one ("HTTPServer" is one chunk), so the rule is not "split camel case" -- it is a boundary
// that only fires when a lower-case letter is followed by an upper-case one.
func TestTekkenChunks(t *testing.T) {
	tok := &Tokenizer{pre: preTekken}
	cases := []struct {
		in   string
		want []string
	}{
		{"HelloWorld", []string{"Hello", "World"}},
		{"camelCase", []string{"camel", "Case"}},
		{"abcDEFghi", []string{"abc", "DEFghi"}},
		// A run of capitals keeps its trailing word, so these do not split.
		{"HTTPServer", []string{"HTTPServer"}},
		{"ABCdef", []string{"ABCdef"}},
		{"HELLO", []string{"HELLO"}},
		// But a capital after a lower-case letter does.
		{"HTTPServerX", []string{"HTTPServer", "X"}},
		{"aB", []string{"a", "B"}},
		// Single digits, and the punctuation tail that takes slashes.
		{"123456", []string{"1", "2", "3", "4", "5", "6"}},
		{"a/b", []string{"a", "/b"}},
		{"a//b", []string{"a", "//", "b"}},
		{"x!\n/", []string{"x", "!\n/"}},
		// The ordinary prefix behaviour still applies to punctuation.
		{"a.b", []string{"a", ".b"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("tekken preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestTekkenCaseSplitNeedsLowerBeforeUpper is the boundary rule stated as a property: a
// capital preceded by a capital must not start a new chunk, and a capital preceded by a
// lower-case letter must.
func TestTekkenCaseSplitNeedsLowerBeforeUpper(t *testing.T) {
	tok := &Tokenizer{pre: preTekken}
	for _, c := range []struct {
		in   string
		want int
		why  string
	}{
		{"ABCD", 1, "all capitals"},
		{"ABCDef", 1, "capitals then a word"},
		{"aBCD", 2, "a lower-case letter then capitals"},
		// "Bc" is the capital run with its trailing word; the following "D" starts a new
		// chunk because "c" is lower case. Verified against a reference implementation of
		// the expression rather than reasoned out -- the first expectation here was 2.
		{"aBcD", 3, "a, the capital run Bc, then D"},
	} {
		if got := len(tok.preSplit(c.in)); got != c.want {
			t.Errorf("tekken preSplit(%q) gave %d chunks, want %d (%s)",
				c.in, got, c.want, c.why)
		}
	}
}

// The two families below are single expressions rather than the ordered alternatives the
// others use, so they get their own splitter and their own tests. The expected values were
// produced by running the expressions from llama.cpp's llama-vocab.cpp through Python's re
// module, not by reading the Go code: a test written from the implementation would agree
// with it whatever it did.

// TestWhitespacePreTokenizer covers jina-embeddings-v2-base-zh's rule, which is just \S+.
func TestWhitespacePreTokenizer(t *testing.T) {
	tok := bpeFixture(t, gguftest.Model{Pre: "whitespace"})
	if !tok.PreImplemented() {
		t.Fatal("the whitespace pre-tokenizer is not implemented")
	}
	cases := []struct {
		in   string
		want []string
	}{
		{"Hello world  123", []string{"Hello", "world", "123"}},
		{" a b ", []string{"a", "b"}},
		{"a\tb\nc", []string{"a", "b", "c"}},
		{"中文 测试", []string{"中文", "测试"}},
	}
	for _, c := range cases {
		got := tok.preSplit(c.in)
		if len(got) != len(c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
				break
			}
		}
	}
}

// TestBloomPreTokenizer covers " ?[^(\s|.,!?…。，、।۔،)]+", which BLOOM, PORO and
// gpt3-finnish share.
//
// The separators are dropped rather than emitted alone, which is what a findall over the
// expression produces: the expression matches runs of non-separator characters, and the
// separators between them are not matches at all. Writing the splitter the other way -- one
// chunk per character when nothing matches -- yields extra punctuation pieces and still
// covers every byte of the input, so it looks correct until compared.
func TestBloomPreTokenizer(t *testing.T) {
	tok := bpeFixture(t, gguftest.Model{Pre: "bloom"})
	if !tok.PreImplemented() {
		t.Fatal("the bloom pre-tokenizer is not implemented")
	}
	if !bpeFixture(t, gguftest.Model{Pre: "gpt3-finnish"}).PreImplemented() {
		t.Error("gpt3-finnish shares BLOOM's expression but is not implemented")
	}
	cases := []struct {
		in   string
		want []string
	}{
		{"Hello world.", []string{"Hello", " world"}},
		{" hi there!", []string{" hi", " there"}},
		{"a,b.c", []string{"a", "b", "c"}},
		{"x (y) z", []string{"x", "y", " z"}},
		{"中文，测试", []string{"中文", "测试"}},
		{"  two spaces", []string{" two", " spaces"}},
	}
	for _, c := range cases {
		got := tok.preSplit(c.in)
		if len(got) != len(c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
				break
			}
		}
	}
}

// TestStableLM2SharesQwen2Rule checks the alias rather than the rule: llama.cpp puts
// stablelm2 in the same branch as qwen2, so the two must chunk identically.
func TestStableLM2SharesQwen2Rule(t *testing.T) {
	a := bpeFixture(t, gguftest.Model{Pre: "qwen2"})
	b := bpeFixture(t, gguftest.Model{Pre: "stablelm2"})
	if !b.PreImplemented() {
		t.Fatal("stablelm2 is not implemented")
	}
	for _, s := range []string{"HelloWorld 123", "don't stop", "a,b.c!"} {
		got, want := b.preSplit(s), a.preSplit(s)
		if len(got) != len(want) {
			t.Errorf("preSplit(%q): stablelm2 %q, qwen2 %q", s, got, want)
			continue
		}
		for i := range got {
			if got[i] != want[i] {
				t.Errorf("preSplit(%q): stablelm2 %q, qwen2 %q", s, got, want)
				break
			}
		}
	}
}

// TestDeepseekCoderPreTokenizer covers the one family whose expressions are applied in
// sequence rather than as alternatives within a single match: newline, then letters, then
// punctuation, then CJK, then one digit at a time.
//
// Two details are easy to get wrong and both were, in the first attempt. Only the letter and
// punctuation expressions carry the optional \s?, so a space before a digit or a CJK run
// stays its own chunk; and \p{N} has no +, so digits come out one per chunk rather than as a
// run. The expected values were produced by applying the five expressions to these inputs
// with Python's re, not by reading the Go code.
func TestDeepseekCoderPreTokenizer(t *testing.T) {
	tok := bpeFixture(t, gguftest.Model{Pre: "deepseek-coder"})
	if !tok.PreImplemented() {
		t.Fatal("the deepseek-coder pre-tokenizer is not implemented")
	}
	cases := []struct {
		in   string
		want []string
	}{
		{"Hello world", []string{"Hello", " world"}},
		{"a,b.c", []string{"a", ",", "b", ".", "c"}},
		{"x1y22", []string{"x", "1", "y", "2", "2"}},
		{"code_here", []string{"code", "_", "here"}},
		{"a\nb", []string{"a", "\n", "b"}},
		{"中文 test", []string{"中文", " test"}},
		{" x", []string{" x"}},
		{"x 1", []string{"x", " ", "1"}},
		{"x 中", []string{"x", " ", "中"}},
		{"a  b", []string{"a", " ", " b"}},
	}
	for _, c := range cases {
		got := tok.preSplit(c.in)
		if len(got) != len(c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
				break
			}
		}
	}
}

// TestFalconPreTokenizer covers falcon's three expressions: a punctuation-and-symbol run,
// the GPT-2 expression, then groups of exactly three digits.
//
// The order is what makes it falcon rather than GPT-2 with a digit rule: the punctuation
// expression runs first, so an apostrophe never forms a contraction and "don't" splits as
// ["don", "'", "t"] where GPT-2 gives ["don", "'t"]. The third expression matches digit
// triples only, so a run of four digits becomes "123" then "4", and text it does not match
// stays whole.
//
// Expected values from applying the three expressions to each input with Python's re.
func TestFalconPreTokenizer(t *testing.T) {
	tok := bpeFixture(t, gguftest.Model{Pre: "falcon"})
	if !tok.PreImplemented() {
		t.Fatal("the falcon pre-tokenizer is not implemented")
	}
	cases := []struct {
		in   string
		want []string
	}{
		{"Hello world", []string{"Hello", " world"}},
		{"a,b.c", []string{"a", ",", "b", ".", "c"}},
		{"x1234y", []string{"x", "123", "4", "y"}},
		{"'s test", []string{"'", "s", " test"}},
		{"if (x) {y}", []string{"if", " ", "(", "x", ")", " ", "{", "y", "}"}},
		{"a  b", []string{"a", " ", " b"}},
		{"123456", []string{"123", "456"}},
		{"don't", []string{"don", "'", "t"}},
	}
	for _, c := range cases {
		got := tok.preSplit(c.in)
		if len(got) != len(c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
				break
			}
		}
	}
}

// TestFamilyAliasesAreEquivalent pins the names llama.cpp groups with a family this package
// implements but that were missing from the switch.
//
// They cost no new logic: jina-v5-nano is in llama.cpp's llama3 branch, minerva-7b is in the
// branch that applies \p{N} before the GPT-2 expression (starcoder's shape), and poro-chat
// shares BLOOM's expression exactly. What they need is to be recognised, because a name that
// falls through to preNone tokenizes as one chunk and says nothing.
func TestFamilyAliasesAreEquivalent(t *testing.T) {
	pairs := []struct{ alias, family string }{
		{"jina-v5-nano", "llama-bpe"},
		{"minerva-7b", "starcoder"},
		{"poro-chat", "bloom"},
		// llama.cpp's GPT2 branch also holds these four. roberta-bpe, jina-v1-en and
		// jina-v2-code set add_sep there, which is a caller-facing flag for joining
		// sequences rather than part of the split; exaone4 sets clean_spaces, which only
		// tidies decoded text. Neither changes where a chunk ends.
		{"roberta-bpe", "gpt-2"},
		{"jina-v1-en", "gpt-2"},
		{"jina-v2-code", "gpt-2"},
		{"exaone4", "gpt-2"},
		{"megrez", "qwen2"},
		{"pixtral", "tekken"},
	}
	inputs := []string{"Hello world", "a,b.c", "x1234y", "don't stop", "a  b"}
	for _, p := range pairs {
		alias := bpeFixture(t, gguftest.Model{Pre: p.alias})
		family := bpeFixture(t, gguftest.Model{Pre: p.family})
		if !alias.PreImplemented() {
			t.Errorf("%q is not implemented, though llama.cpp maps it to %q", p.alias, p.family)
			continue
		}
		for _, in := range inputs {
			got, want := alias.preSplit(in), family.preSplit(in)
			if len(got) != len(want) {
				t.Errorf("%q chunks differently from %q on %q: %q vs %q",
					p.alias, p.family, in, got, want)
				continue
			}
			for i := range got {
				if got[i] != want[i] {
					t.Errorf("%q chunks differently from %q on %q: %q vs %q",
						p.alias, p.family, in, got, want)
					break
				}
			}
		}
	}
}

// TestGPT4OPreTokenizer covers the GPT4O family, which llama.cpp shares between gpt-4o,
// llama4, inkling2, kanana2, talkie and minimax-m2.
//
// Its expression is tekken's shape with two differences: \p{N}{1,3} instead of \p{N}, and
// an optional clitic suffix on the letter alternatives. Both change visible boundaries, so
// gpt-4o was previously mapped to tekken and chunked wrongly -- a three-digit run came out
// as three single digits, and "don't" split into a word and a separate "'t".
//
// Expected values from applying the expression to each input with Python's re.
func TestGPT4OPreTokenizer(t *testing.T) {
	tok := bpeFixture(t, gguftest.Model{Pre: "gpt-4o"})
	if !tok.PreImplemented() {
		t.Fatal("the GPT4O family is not implemented")
	}
	cases := []struct {
		in   string
		want []string
	}{
		{"HelloWorld", []string{"Hello", "World"}},
		{"x1234y", []string{"x", "123", "4", "y"}},
		{"12345", []string{"123", "45"}},
		{"don't", []string{"don't"}},
		{"DON'T", []string{"DON'T"}},
		{"HelloWorld's", []string{"Hello", "World's"}},
		{"a  b", []string{"a", " ", " b"}},
		{"x/y", []string{"x", "/y"}},
		{"Hello\nWorld", []string{"Hello", "\n", "World"}},
		{"'s test", []string{"'s", " test"}},
	}
	for _, c := range cases {
		got := tok.preSplit(c.in)
		if len(got) != len(c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
				break
			}
		}
	}
}

// TestGPT4OAliasesAgree checks the five other names llama.cpp puts on the same branch.
func TestGPT4OAliasesAgree(t *testing.T) {
	ref := bpeFixture(t, gguftest.Model{Pre: "gpt-4o"})
	for _, name := range []string{"llama4", "inkling2", "kanana2", "talkie", "minimax-m2"} {
		alias := bpeFixture(t, gguftest.Model{Pre: name})
		if !alias.PreImplemented() {
			t.Errorf("%q is not implemented, though llama.cpp maps it to GPT4O", name)
			continue
		}
		for _, in := range []string{"x1234y", "don't", "HelloWorld", "x/y", "a  b"} {
			got, want := alias.preSplit(in), ref.preSplit(in)
			bad := len(got) != len(want)
			for i := 0; !bad && i < len(got); i++ {
				if got[i] != want[i] {
					bad = true
				}
			}
			if bad {
				t.Errorf("%q chunks differently from gpt-4o on %q: %q vs %q", name, in, got, want)
			}
		}
	}
}

// TestEveryHandledNameIsAKnownName checks the switch against the list it is supposed to be
// a subset of.
//
// The two are maintained separately -- knownPreNames is transcribed from llama.cpp's name
// switch, and the case groups are written by hand -- so a name can appear in one and not the
// other. That happened: the starcoder group listed "minerva", which llama.cpp does not have,
// next to the real "minerva-7b". It cost nothing at runtime, because the bogus entry matched
// nothing and the real name was in the same group, which is exactly why it went unnoticed for
// several rounds: a dead entry looks like a working one.
func TestEveryHandledNameIsAKnownName(t *testing.T) {
	// Every name the tokenizer recognises, discovered the way a user would: by asking what
	// each known name resolves to.
	known := make(map[string]bool, len(knownPreNames))
	for _, n := range knownPreNames {
		known[n] = true
	}
	// The switch's own contents are not exported, so this drives them through New and
	// checks the names it accepts are ones the list contains.
	for _, name := range knownPreNames {
		path := gguftest.Model{Pre: name, NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
		f, err := gguf.Open(path)
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		if _, err := New(f); err != nil {
			t.Fatalf("New(pre=%q): %v", name, err)
		}
	}
	// A name that is not in the list must not be accepted as a family by the switch. The
	// bogus entry above was an alias, so this catches the shape of the mistake rather than
	// the exact string: a pre value that resolves to an implemented family has to be one
	// llama.cpp could actually emit.
	for _, bogus := range []string{"minerva", "qwen-2", "QWEN2"} {
		path := gguftest.Model{Pre: bogus, NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
		f, err := gguf.Open(path)
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		tok, err := New(f)
		if err != nil {
			t.Fatalf("New(pre=%q): %v", bogus, err)
		}
		if tok.PreImplemented() && !known[bogus] {
			t.Errorf("%q is treated as an implemented family, but it is not a name llama.cpp "+
				"maps to one -- it is dead code in the switch", bogus)
		}
	}
}

// TestImplementedNameCountPinsTheReadmeNumber keeps the README's "covering 63 of the 100
// names" checkable.
//
// It is a number a reader can compare against llama.cpp, and it changes by one every time a
// family is added -- which is exactly the kind of figure that goes stale silently, since
// nothing fails when it drifts. The two figures are separate assertions because they move
// for different reasons: the table grows when llama.cpp adds a family, the implemented count
// when one is written here.
func TestImplementedNameCountPinsTheReadmeNumber(t *testing.T) {
	// A real tokenizer per name, so the lookup under test is the one a file goes through:
	// PreImplemented reads the parsed rule, not the name.
	implemented := 0
	for _, name := range knownPreNames {
		path := gguftest.Model{Pre: name, NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
		f, err := gguf.Open(path)
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		tok, err := New(f)
		if err != nil {
			t.Fatalf("New(pre=%q): %v", name, err)
		}
		if !tok.PreKnown() {
			t.Errorf("%q is in knownPreNames but PreKnown reports false", name)
		}
		if tok.PreImplemented() {
			implemented++
		}
	}

	const (
		wantNames       = 100
		wantImplemented = 63
	)
	if got := len(knownPreNames); got != wantNames {
		t.Errorf("knownPreNames has %d entries, README says %d", got, wantNames)
	}
	if implemented != wantImplemented {
		t.Errorf("%d names are implemented, README says %d: update the README count in the "+
			"same commit as the family", implemented, wantImplemented)
	}
}
