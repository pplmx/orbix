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
		{"tekken", false, true, "a real family, not implemented here"},
		{"llama4", false, true, "a real family, not implemented here"},
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
