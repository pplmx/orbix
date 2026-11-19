package tokenize

import (
	"testing"
	"unicode/utf8"

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
// to be asked, and the answer is only useful if the two cases are separate -- one would be
// work remaining, the other is a broken file.
//
// Every name llama.cpp maps to a family is implemented, so no case below can produce
// "known but not implemented" any more. The distinction is kept because it is what makes the
// claim checkable: TestEveryKnownNameResolves fails the moment a name is in knownPreNames
// without a rule, and that is the only way the middle state can come back.
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

// TestKnownPreNamesAllReachAFamily checks the two name lists against each other: every
// name knownPreNames claims is a name the mapping switch actually recognises.
//
// The two can disagree in the direction that hides a bug. PreKnown falls back to
// knownPreNames when preFamilyOf fails, so a name added to that list without a case in the
// switch reports itself as known while the file is split as one chunk -- the exact silent
// degradation the family rules exist to avoid, arriving through the field meant to detect
// it. The reverse direction matters too: a case added without a knownPreNames entry makes
// inspect report a name it does not recognise as known.
//
// This drives the real constructor for all 100 names rather than comparing the lists
// textually, because the switch is the authority and static extraction would have to
// reimplement it -- my first attempt at that picked up "clean_up_tokenization_spaces" out
// of a comment and reported a mismatch that did not exist.
func TestKnownPreNamesAllReachAFamily(t *testing.T) {
	for _, name := range knownPreNames {
		t.Run(name, func(t *testing.T) {
			path := gguftest.Model{
				Vocab:          []string{"<unk>", "<s>", "</s>", "\u2581", "a"},
				Merges:         []string{"a a"},
				TokenizerModel: "gpt2",
				NoBytes:        true,
				Pre:            name,
				ContextLen:     64,
			}.Write(t, t.TempDir())
			f, err := gguf.Open(path)
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			tok, err := New(f)
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			if !tok.PreImplemented() {
				t.Errorf("%q is in knownPreNames but has no rule: it would split the file as one chunk", name)
			}
			// "default" is the one name whose correct answer is preNone: it means llama.cpp
			// applies no regex to this file, which is what not splitting does, and
			// PreImplemented says so explicitly. Every other known name must reach a family
			// with a rule, or the name is in the list without being wired up.
			if name == "default" {
				if tok.pre != preNone {
					t.Errorf(`"default" resolved to %v, want preNone`, tok.pre)
				}
				return
			}
			if _, ok := preFamilyOf(tok.pre); !ok {
				t.Errorf("%q resolved to %v, which has no preRule", name, tok.pre)
			}
		})
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
	// The expected splits were checked against llama.cpp's expression with the text between
	// matches kept: \S+ matches the runs, and a splitter that deletes the gaps it skipped
	// loses the spaces outright. The earlier expectations here listed only the runs.
	tok := bpeFixture(t, gguftest.Model{Pre: "whitespace"})
	if !tok.PreImplemented() {
		t.Fatal("the whitespace pre-tokenizer is not implemented")
	}
	cases := []struct {
		in   string
		want []string
	}{
		// The spaces are chunks too: \S+ matches the runs, and the text between matches is
		// carried through rather than deleted. Consecutive spaces are one chunk.
		{"Hello world  123", []string{"Hello", " ", "world", "  ", "123"}},
		{" a b ", []string{" ", "a", " ", "b", " "}},
		{"a\tb\nc", []string{"a", "\t", "b", "\n", "c"}},
		{"中文 测试", []string{"中文", " ", "测试"}},
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
	// The expected splits were rechecked against llama.cpp's expression: the characters the
	// rule stops at are not separators to be dropped, they are text the splitter carries
	// through as their own chunks, and the gap between two matches is one chunk rather than
	// one per character. The earlier expectations here dropped the stop characters outright,
	// which is what the implementation did at the time.
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
		{"Hello world.", []string{"Hello", " world", "."}},
		{" hi there!", []string{" hi", " there", "!"}},
		{"a,b.c", []string{"a", ",", "b", ".", "c"}},
		{"x (y) z", []string{"x", " (", "y", ")", " z"}},
		{"中文，测试", []string{"中文", "，", "测试"}},
		{"  two spaces", []string{" ", " two", " spaces"}},
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

// TestImplementedNameCountPinsTheReadmeNumber keeps the README's "covering all 100 names"
// checkable.
//
// It is a number a reader can compare against llama.cpp, and it changes by one every time a
// family is added -- which is exactly the kind of figure that goes stale silently, since
// nothing fails when it drifts. The two figures are separate assertions because they move
// for different reasons: the table grows when llama.cpp adds a family, the implemented count
// when one is written here. Both read 100 today; the comment said "covering 63 of the 100"
// until every family was written, and that is the drift this test exists to catch.
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
		wantImplemented = 100
	)
	if got := len(knownPreNames); got != wantNames {
		t.Errorf("knownPreNames has %d entries, README says %d", got, wantNames)
	}
	if implemented != wantImplemented {
		t.Errorf("%d names are implemented, README says %d: update the README count in the "+
			"same commit as the family", implemented, wantImplemented)
	}
}

// TestDbrxAndSmaugShareTheLlama3Rule checks the two names llama.cpp maps to llama3's
// expression verbatim.
//
// Its DBRX and SMAUG cases fall through to one regex and the source says so; the only thing
// that differs is clean_up_tokenization_spaces, a decoding-side setting. Mapping the names is
// therefore the whole change, and the risk is not that the rule is wrong but that a name is
// accepted and resolves to something else.
//
// The expected chunks come from transcribing that regex into Python's regex module and
// running it over the same strings -- an implementation that shares no code with this one,
// which is what the repository asks for when a result is being cross-checked.
func TestDbrxAndSmaugShareTheLlama3Rule(t *testing.T) {
	llama := &Tokenizer{pre: preLlama3}
	cases := []struct {
		in   string
		want []string
	}{
		{"a1234b", []string{"a", "123", "4", "b"}},
		{"Hello world", []string{"Hello", " world"}},
		{"don't", []string{"don", "'t"}},
		{"x=42", []string{"x", "=", "42"}},
		{"  indented", []string{" ", " indented"}},
		{"line\n\ntwo", []string{"line", "\n\n", "two"}},
		{"999", []string{"999"}},
		{"a.b", []string{"a", ".b"}},
	}
	for _, c := range cases {
		if got := llama.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	// The names have to reach that rule through the file, which is how a model gets here.
	for _, name := range []string{"dbrx", "smaug-bpe"} {
		path := gguftest.Model{Pre: name, NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
		f, err := gguf.Open(path)
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		tok, err := New(f)
		if err != nil {
			t.Fatalf("New(pre=%q): %v", name, err)
		}
		if !tok.PreImplemented() {
			t.Errorf("%q is not reported as implemented", name)
		}
		if got := tok.preSplit("a1234b"); !sameStrings(got, []string{"a", "123", "4", "b"}) {
			t.Errorf("%q preSplit(\"a1234b\") = %q, want the llama3 split", name, got)
		}
	}
}

// TestDeepseekLLMChunks pins the deepseek-llm rule, whose letter class is written out as
// ranges rather than as \p{L}.
//
// The expected splits come from applying llama.cpp's six expressions in sequence in Python's
// regex module, then comparing code point by code point against this implementation over 500
// random strings drawn from the class boundaries. They agree; these are the cases that make
// the rule's shape visible.
func TestDeepseekLLMChunks(t *testing.T) {
	tok := &Tokenizer{pre: preDeepseekLLM}
	cases := []struct {
		in   string
		want []string
	}{
		// Letters keep their run, punctuation is its own chunk, and one preceding space is
		// taken with the run it precedes.
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"don't", []string{"don", "'", "t"}},
		{"x=y", []string{"x", "=", "y"}},
		// Numbers group as a run, not in threes: this is what separates the family from
		// llama3 and from deepseek-coder.
		{"123456", []string{"123456"}},
		{"a1b2", []string{"a", "1", "b", "2"}},
		// Each newline is its own chunk, so CRLF is two.
		{"a\r\nb", []string{"a", "\r", "\n", "b"}},
		// CJK is its own chunk. The range starts at U+0800 rather than at a Han character,
		// so the scripts between are taken by this alternative and stay whole.
		{"你好世界", []string{"你好世界"}},
		{"日本語テスト", []string{"日本語テスト"}},
		// The letter class covers Latin, Greek and Cyrillic but not Arabic, which none of
		// the expressions match -- so it stays as one gap chunk instead of being cut.
		{"Привет мир", []string{"Привет", " мир"}},
		{"مرحبا", []string{"مرحبا"}},
		// One space before a run belongs to it; the others stand alone.
		{"  hello", []string{" ", " hello"}},
		{"a  b", []string{"a", " ", " b"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestDeepseekLLMRangesAreSorted guards the property inRanges depends on.
//
// The class in llama.cpp is not written in ascending order -- the fullwidth punctuation
// ranges come before the curly quotes -- and a table used as-is makes the binary search miss
// entries. That is how the first version of this family merged a quote into the following CJK
// ideograph: U+201E was in the class but unreachable.
func TestDeepseekLLMRangesAreSorted(t *testing.T) {
	for _, c := range []struct {
		name   string
		ranges [][2]rune
	}{
		{"deepseekLLMLetterRanges", deepseekLLMLetterRanges},
		{"deepseekLLMPunctRanges", deepseekLLMPunctRanges},
	} {
		for i := 1; i < len(c.ranges); i++ {
			prev, cur := c.ranges[i-1], c.ranges[i]
			if prev[0] > prev[1] {
				t.Errorf("%s[%d] = %U..%U is reversed", c.name, i-1, prev[0], prev[1])
			}
			if cur[0] <= prev[1] {
				t.Errorf("%s[%d] = %U..%U overlaps or touches %U..%U",
					c.name, i, cur[0], cur[1], prev[0], prev[1])
			}
		}
	}
	// Every rune of the class has to be reachable, which a lookup at the edges shows.
	for _, r := range []rune{0x201E, 0x0021, 0xFF5E, 0x10400, 0x03A9, 0x00B5} {
		if !isDeepseekLLMLetter(r) && !isDeepseekLLMPunct(r) {
			t.Errorf("%U is in neither class but was expected to be", r)
		}
	}
}

// TestMMBertChunks pins the mmbert rule, which splits less than any other family: the only
// boundary is between newlines and everything else, so a whole sentence with punctuation and
// spaces is one chunk.
//
// The expected splits come from running the expression through Python's regex module and
// keeping the gaps, which is the same independent check the other families were held to.
func TestMMBertChunks(t *testing.T) {
	tok := &Tokenizer{pre: preMMBert}
	cases := []struct {
		in   string
		want []string
	}{
		// Nothing but a newline separates a chunk, so punctuation and spaces stay inside.
		{"Hello, world!", []string{"Hello, world!"}},
		{" hi 5. ", []string{" hi 5. "}},
		{"中文测试", []string{"中文测试"}},
		// A run of newlines is one chunk however long it is.
		{"a\nb", []string{"a", "\n", "b"}},
		{"line\n\ntwo", []string{"line", "\n\n", "two"}},
		{"\n\n\n", []string{"\n\n\n"}},
		{"x", []string{"x"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	// And the name has to reach that rule through a file, which is how a model gets here.
	path := gguftest.Model{Pre: "mmbert", NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	real, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !real.PreImplemented() {
		t.Error("mmbert is not reported as implemented")
	}
	if got := real.preSplit("Hello, world!"); !sameStrings(got, []string{"Hello, world!"}) {
		t.Errorf("mmbert preSplit(\"Hello, world!\") = %q, want one chunk", got)
	}
}

// TestVikingChunks pins the viking family, which is BLOOM's expression with a single-digit
// cut after it.
//
// The two steps have to run in that order: the stop-set pass leaves "a1b2" whole, and the
// digit pass then takes each digit out. Doing it the other way would cut the digits first and
// then let the stop-set pass see runs it would have grouped differently.
func TestVikingChunks(t *testing.T) {
	tok := &Tokenizer{pre: preViking}
	cases := []struct {
		in   string
		want []string
	}{
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		// Each digit is its own chunk, not a run: this is what separates viking from bloom.
		{"a1b2", []string{"a", "1", "b", "2"}},
		{"123456", []string{"1", "2", "3", "4", "5", "6"}},
		{"x y", []string{"x", " y"}},
		{" hi 5. ", []string{" hi", " ", "5", ". "}},
		{"中文测试", []string{"中文测试"}},
		{"he said…", []string{"he", " said", "…"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	// bloom must not have gained the digit pass: the two families share an expression but
	// not a rule.
	bloom := &Tokenizer{pre: preBloom}
	if got := bloom.preSplit("a1b2"); !sameStrings(got, []string{"a1b2"}) {
		t.Errorf("bloom preSplit(\"a1b2\") = %q, want it whole", got)
	}

	path := gguftest.Model{Pre: "viking", NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	real, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !real.PreImplemented() {
		t.Error("viking is not reported as implemented")
	}
	if got := real.preSplit("a1b2"); !sameStrings(got, []string{"a", "1", "b", "2"}) {
		t.Errorf("viking preSplit(\"a1b2\") = %q, want the digits cut", got)
	}
}

// TestSuperBPEChunks pins the superbpe rule, whose digit groups are counted from the right.
//
// Every other digit-taking family groups from the left, so a remainder lands at the end; this
// one's lookahead asserts a multiple of three digits remain, which puts the remainder at the
// start instead. "1234" is ["1", "234"] here and ["123", "4"] under llama3.
func TestSuperBPEChunks(t *testing.T) {
	tok := &Tokenizer{pre: preSuperBPE}
	cases := []struct {
		in   string
		want []string
	}{
		{"123456789", []string{"123", "456", "789"}},
		{"123", []string{"123"}},
		// The remainder leads.
		{"1234", []string{"1", "234"}},
		{"12345", []string{"12", "345"}},
		// Fewer than three digits is one chunk.
		{"12", []string{"12"}},
		// Non-digits are kept where they are.
		{"abc123456", []string{"abc", "123", "456"}},
		{"a1b2", []string{"a", "1", "b", "2"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	// The other digit families must not have gained the right-to-left cut.
	llama := &Tokenizer{pre: preLlama3}
	if got := llama.preSplit("1234"); !sameStrings(got, []string{"123", "4"}) {
		t.Errorf("llama3 preSplit(\"1234\") = %q, want the remainder last", got)
	}

	path := gguftest.Model{Pre: "superbpe", NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	real, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !real.PreImplemented() {
		t.Error("superbpe is not reported as implemented")
	}
	if got := real.preSplit("1234"); !sameStrings(got, []string{"1", "234"}) {
		t.Errorf("superbpe preSplit(\"1234\") = %q, want the remainder first", got)
	}
}

// TestSopranoChunks pins the soprano rule, whose main expression is a partition.
//
// Every other family's main expression is a run with an optional leading character, so a
// space joins the word after it. Here the classes are disjoint and no class takes a character
// from another, which is why "Hello, world!" has a separate " " chunk and why "x_y" stays
// whole -- the underscore is a word character.
//
// The digit expression runs first, so digits are cut out before the partition sees them.
func TestSopranoChunks(t *testing.T) {
	tok := &Tokenizer{pre: preSoprano}
	cases := []struct {
		in   string
		want []string
	}{
		// Whitespace is its own class, so the space is a chunk rather than a prefix.
		{"Hello, world!", []string{"Hello", ",", " ", "world", "!"}},
		{"hi there", []string{"hi", " ", "there"}},
		{"  two  ", []string{"  ", "two", "  "}},
		// Underscore is a word character and joins the run.
		{"x_y z", []string{"x_y", " ", "z"}},
		// Digits are cut one at a time before the partition runs.
		{"a1b2", []string{"a", "1", "b", "2"}},
		{"123456", []string{"1", "2", "3", "4", "5", "6"}},
		{"a-b.c", []string{"a", "-", "b", ".", "c"}},
		{"中文测试", []string{"中文测试"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	path := gguftest.Model{Pre: "soprano", NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	real, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !real.PreImplemented() {
		t.Error("soprano is not reported as implemented")
	}
	if got := real.preSplit("hi there"); !sameStrings(got, []string{"hi", " ", "there"}) {
		t.Errorf("soprano preSplit(\"hi there\") = %q, want the space on its own", got)
	}
}

// TestChatGLMNamesShareTheLlama3Rule checks the two names llama.cpp maps to CHATGLM4, whose
// expression is llama3's verbatim.
//
// The branch differs only in settings this package does not model -- a null BOS id,
// ignore_merges, and clean_up_tokenization_spaces -- so mapping the names is the whole change.
// As with dbrx and smaug-bpe, the risk is not that the rule is wrong but that a name resolves
// to something else.
func TestChatGLMNamesShareTheLlama3Rule(t *testing.T) {
	llama := &Tokenizer{pre: preLlama3}
	for _, name := range []string{"chatglm-bpe", "glm5"} {
		path := gguftest.Model{Pre: name, NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
		f, err := gguf.Open(path)
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		tok, err := New(f)
		if err != nil {
			t.Fatalf("New(pre=%q): %v", name, err)
		}
		if !tok.PreImplemented() {
			t.Errorf("%q is not reported as implemented", name)
		}
		// The same inputs, split the same way as llama3: digits in threes, a punctuation
		// prefix on the following word, contractions whole.
		for _, in := range []string{"a1234b", "Hello, world!", "don't", "中文测试"} {
			want := llama.preSplit(in)
			if got := tok.preSplit(in); !sameStrings(got, want) {
				t.Errorf("%q preSplit(%q) = %q, want llama3's %q", name, in, got, want)
			}
		}
	}
}

// TestSharedRuleNamesResolveToTheirBaseFamily covers the names that llama.cpp puts in another
// family's branch verbatim.
//
// These are mapping-only changes, so the failure they can have is a name resolving to the
// wrong rule rather than the rule being wrong. The check is therefore a comparison against the
// base family's own splitter over inputs that separate the families from each other, instead
// of a table of expected chunks that would have to be maintained per name.
func TestSharedRuleNamesResolveToTheirBaseFamily(t *testing.T) {
	// Inputs that distinguish the families: digit grouping, punctuation prefix, contractions,
	// newlines, and CJK.
	inputs := []string{"a1234b", "Hello, world!", "don't", "line\n\ntwo", "中文测试", "x_y z"}

	check := func(t *testing.T, names []string, base preType) {
		t.Helper()
		want := (&Tokenizer{pre: base})
		for _, name := range names {
			path := gguftest.Model{Pre: name, NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
			f, err := gguf.Open(path)
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			tok, err := New(f)
			if err != nil {
				t.Fatalf("New(pre=%q): %v", name, err)
			}
			if !tok.PreImplemented() {
				t.Errorf("%q is not reported as implemented", name)
				continue
			}
			for _, in := range inputs {
				if got, wantSplit := tok.preSplit(in), want.preSplit(in); !sameStrings(got, wantSplit) {
					t.Errorf("%q preSplit(%q) = %q, want the base family's %q", name, in, got, wantSplit)
				}
			}
		}
	}

	// llama.cpp's CHATGLM4 and DBRX/SMAUG cases hold llama3's expression verbatim.
	check(t, []string{"chatglm-bpe", "glm4", "glm5", "dbrx", "smaug-bpe"}, preLlama3)
	// GEMMA4, DOMYN_SMALL and SARVAM_MOE hold mmbert's: the text is cut only at newlines.
	check(t, []string{"gemma4", "granite-embed-multi-311m", "domyn-small", "sarvam-moe"}, preMMBert)
	// grok-2 writes \s*[\r\n]+ where bailingmoe has \s*[\r\n]; both consume the whole run,
	// so the rules agree on every newline input tried.
	check(t, []string{"grok-2", "bailingmoe", "bailingmoe2", "llada-moe"}, preBailingMoE)
	// GRANITE_EMB_MULTI holds GPT4O's expression verbatim. The name is easy to confuse with
	// granite-embed-multi-311m, which is in the GEMMA4 branch and maps to mmbert instead: the
	// two share a prefix but not a rule.
	check(t, []string{"granite-embed-multi-97m", "gpt-4o"}, preGPT4O)
}

// TestBailingMoEChunks pins the bailingmoe rule, which is llama3's shape with a bare \p{N}.
//
// The one difference that matters is digits: llama3 takes up to three at a time, this takes
// one, so "1234" is ["1", "2", "3", "4"] here and ["123", "4"] there. The newline
// alternatives differ on paper -- \s*[\r\n] against \s*[\r\n]+ -- but both consume the run,
// so "line\n\ntwo" splits the same way under each.
func TestBailingMoEChunks(t *testing.T) {
	tok := &Tokenizer{pre: preBailingMoE}
	cases := []struct {
		in   string
		want []string
	}{
		{"a1234b", []string{"a", "1", "2", "3", "4", "b"}},
		{"12", []string{"1", "2"}},
		{"x=42", []string{"x", "=", "4", "2"}},
		// The rest of the expression is llama3's, so these match it exactly.
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"don't", []string{"don", "'t"}},
		{"line\n\ntwo", []string{"line", "\n\n", "two"}},
		{"中文测试", []string{"中文测试"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	// llama3 must not have gained the single-digit cut: the two rules differ by one field and
	// a careless edit could move both.
	llama := &Tokenizer{pre: preLlama3}
	if got := llama.preSplit("1234"); !sameStrings(got, []string{"123", "4"}) {
		t.Errorf("llama3 preSplit(\"1234\") = %q, want up to three digits", got)
	}

	// All three names llama.cpp puts in the BAILINGMOE branch reach this rule.
	for _, name := range []string{"bailingmoe", "bailingmoe2", "llada-moe"} {
		path := gguftest.Model{Pre: name, NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
		f, err := gguf.Open(path)
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		real, err := New(f)
		if err != nil {
			t.Fatalf("New(pre=%q): %v", name, err)
		}
		if !real.PreImplemented() {
			t.Errorf("%q is not reported as implemented", name)
		}
		if got := real.preSplit("1234"); !sameStrings(got, []string{"1", "2", "3", "4"}) {
			t.Errorf("%q preSplit(\"1234\") = %q, want single digits", name, got)
		}
	}
}

// TestTinyAyaChunks pins the tiny_aya rule: a digit lookahead that groups from the right and
// stops at a word boundary, then a case-splitting letter rule.
//
// Two things separate it from the other right-grouping family, superbpe. First, the lookahead
// ends in \b, so a digit run whose right neighbour is a word character is not matched at all
// and stays gap text -- "a1b2" has no group boundary because "1" touches letters on both
// sides, and the letter rule then cuts it. Second, a main rule runs after the digit pass, so
// the pieces it leaves are split further by case.
func TestTinyAyaChunks(t *testing.T) {
	tok := &Tokenizer{pre: preTinyAya}
	cases := []struct {
		in   string
		want []string
	}{
		// Grouped from the right, remainder first: the same shape as superbpe.
		{"1234", []string{"1", "234"}},
		{"123456", []string{"123", "456"}},
		// The boundary rule: "1234a" ends on a letter, so the digit pass matches nothing and
		// the letter rule then cuts from the left. Without the rule this would be
		// ["1", "234", "a"], which is the only input found where the two differ.
		{"1234a", []string{"123", "4", "a"}},
		// A run touching a letter is whole for the digit pass, then cut by the letter rule.
		{"a1b2", []string{"a", "1", "b", "2"}},
		// A run between punctuation reaches a word boundary and groups.
		{"x=42", []string{"x", "=", "42"}},
		// Case boundaries split a letter run.
		{"helloWorld", []string{"hello", "World"}},
		{"HelloWorld", []string{"Hello", "World"}},
		{"HELLO", []string{"HELLO"}},
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"don't", []string{"don't"}},
		{"中文测试", []string{"中文测试"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	// superbpe has no boundary rule: the same run groups there.
	super := &Tokenizer{pre: preSuperBPE}
	if got := super.preSplit("a1b2"); !sameStrings(got, []string{"a", "1", "b", "2"}) {
		t.Errorf("superbpe preSplit(\"a1b2\") = %q, want the digits cut", got)
	}

	// Both names in llama.cpp's TINY_AYA branch reach this rule.
	for _, name := range []string{"cohere2moe", "tiny_aya"} {
		path := gguftest.Model{Pre: name, NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
		f, err := gguf.Open(path)
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		real, err := New(f)
		if err != nil {
			t.Fatalf("New(pre=%q): %v", name, err)
		}
		if !real.PreImplemented() {
			t.Errorf("%q is not reported as implemented", name)
		}
		if got := real.preSplit("1234"); !sameStrings(got, []string{"1", "234"}) {
			t.Errorf("%q preSplit(\"1234\") = %q, want the remainder first", name, got)
		}
	}
}

// TestLagunaChunks pins the laguna rule: a newline pass, then bailingmoe's expression.
//
// The newline pass is not redundant with the main rule. The main rule's whitespace
// alternative takes " \n" as one chunk, while the pass cuts at the newline first, so
// " x \n y " gives [" x", " ", "\n", " y", " "]. Every other input splits the same way
// either path, which is why a mapping to the bailingmoe rule was rejected after trying it.
func TestLagunaChunks(t *testing.T) {
	tok := &Tokenizer{pre: preLaguna}
	cases := []struct {
		in   string
		want []string
	}{
		// The discriminating input: whitespace before a newline.
		{" x \n y ", []string{" x", " ", "\n", " y", " "}},
		{"ab\ncd ef", []string{"ab", "\n", "cd", " ef"}},
		{"line\n\ntwo", []string{"line", "\n\n", "two"}},
		// Everything after the pass is bailingmoe's: single digits, contractions, CJK.
		{"a1234b", []string{"a", "1", "2", "3", "4", "b"}},
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"don't", []string{"don", "'t"}},
		{"中文测试", []string{"中文测试"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	// bailingmoe has no newline pass, so it keeps the space attached to the newline.
	bm := &Tokenizer{pre: preBailingMoE}
	if got := bm.preSplit(" x \n y "); !sameStrings(got, []string{" x", " \n", " y", " "}) {
		t.Errorf("bailingmoe preSplit(\" x \\n y \") = %q, want the space kept with the newline", got)
	}

	path := gguftest.Model{Pre: "laguna", NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	real, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !real.PreImplemented() {
		t.Error("laguna is not reported as implemented")
	}
}

// TestFlintChunks pins the flint rule: digit groups and the CJK block first, then a
// case-splitting letter rule, then a punctuation run named by category.
//
// The digit pass comes first, so the letter rule has no digit alternative of its own -- a
// difference from tiny_aya, whose main rule includes one. The punctuation alternative names
// \p{P}\p{S} rather than a range, and the leading character is any non-letter, so a period
// attaches to the word after it: "a.b" is ["a", ".b"].
func TestFlintChunks(t *testing.T) {
	tok := &Tokenizer{pre: preFlint}
	cases := []struct {
		in   string
		want []string
	}{
		// Digits group from the left in threes, remainder last.
		{"1234", []string{"123", "4"}},
		{"a1234b", []string{"a", "123", "4", "b"}},
		// The leading character absorbs a period; the letter rule then splits on case.
		{"a.b", []string{"a", ".b"}},
		{"helloWorld", []string{"hello", "World"}},
		{"HelloWorld", []string{"Hello", "World"}},
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"don't", []string{"don't"}},
		{"x=42", []string{"x", "=", "42"}},
		{"中文测试", []string{"中文测试"}},
		{" x \n y ", []string{" x", " \n", " y", " "}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	path := gguftest.Model{Pre: "flint", NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	real, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !real.PreImplemented() {
		t.Error("flint is not reported as implemented")
	}
}

// TestUfakzekaChunks pins the ufakzeka rule: qwen2's expression with the contraction
// alternative removed.
//
// That removal is the whole reason the family exists. llama.cpp's comment says it keeps
// Turkish apostrophe suffixes attached, and it does so through the leading character: with no
// contraction alternative to match first, "'da" is taken as an apostrophe plus a letter run,
// giving ["İstanbul", "'da"]. Every family whose expression carries the alternative splits it
// at "'d" instead.
func TestUfakzekaChunks(t *testing.T) {
	tok := &Tokenizer{pre: preUfakzeka}
	cases := []struct {
		in   string
		want []string
	}{
		// The discriminating input: the suffix stays whole.
		{"İstanbul'da", []string{"İstanbul", "'da"}},
		// English contractions still split, because the leading character takes the
		// apostrophe and the letters after it form the run.
		{"don't", []string{"don", "'t"}},
		{"can't", []string{"can", "'t"}},
		{"we're", []string{"we", "'re"}},
		{"I'll", []string{"I", "'ll"}},
		// The rest is qwen2's: single digits, a punctuation prefix, newline runs, CJK.
		{"a1234b", []string{"a", "1", "2", "3", "4", "b"}},
		{"a.b", []string{"a", ".b"}},
		{"x=42", []string{"x", "=", "4", "2"}},
		{"line\n\ntwo", []string{"line", "\n\n", "two"}},
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"中文测试", []string{"中文测试"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	// bailingmoe carries the contraction alternative, so it splits the same suffix at "'d".
	bm := &Tokenizer{pre: preBailingMoE}
	if got := bm.preSplit("İstanbul'da"); !sameStrings(got, []string{"İstanbul", "'d", "a"}) {
		t.Errorf("bailingmoe preSplit(\"İstanbul'da\") = %q, want the suffix split at \"'d\"", got)
	}

	path := gguftest.Model{Pre: "ufakzeka", NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	real, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !real.PreImplemented() {
		t.Error("ufakzeka is not reported as implemented")
	}
}

// TestQwen35Chunks pins the qwen35 rule: qwen2 with combining marks in the letter class.
//
// The mark matters for input whose base letter is followed by a combining character: the two
// stay in one run here, where a class of \p{L} alone would cut before the mark and leave it
// to the punctuation alternative.
func TestQwen35Chunks(t *testing.T) {
	tok := &Tokenizer{pre: preQwen35}
	cases := []struct {
		in   string
		want []string
	}{
		// The discriminating input: a base letter and a combining acute are one chunk.
		{"e\u0301clair", []string{"e\u0301clair"}},
		// The punctuation alternative still absorbs newlines, as qwen2's does.
		{",,,\n\nx", []string{",,,\n\n", "x"}},
		// The rest is qwen2's.
		{"a1234b", []string{"a", "1", "2", "3", "4", "b"}},
		{"don't", []string{"don", "'t"}},
		{"a.b", []string{"a", ".b"}},
		{"x=42", []string{"x", "=", "4", "2"}},
		{"line\n\ntwo", []string{"line", "\n\n", "two"}},
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"中文测试", []string{"中文测试"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	// qwen2 has no mark in its letter class, so the same input cuts differently there.
	q2 := &Tokenizer{pre: preQwen2}
	if got := q2.preSplit("e\u0301clair"); !sameStrings(got, []string{"e", "\u0301clair"}) {
		t.Errorf("qwen2 preSplit(\"e\\u0301clair\") = %q, want the mark cut off", got)
	}

	path := gguftest.Model{Pre: "qwen35", NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	real, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !real.PreImplemented() {
		t.Error("qwen35 is not reported as implemented")
	}
}

// TestSeedCoderChunks pins the seed-coder rule, whose punctuation alternative excludes
// newlines and carries no newline tail.
//
// Newlines still group through the "\s*[\r\n]+" alternative, so a run of them is one chunk.
// What changes is that a punctuation run stops before a newline instead of absorbing it:
// ",,,\n\nx" is [",,,", "\n\n", "x"] here and [",,,\n\n", "x"] under qwen2.
func TestSeedCoderChunks(t *testing.T) {
	tok := &Tokenizer{pre: preSeedCoder}
	cases := []struct {
		in   string
		want []string
	}{
		// The discriminating inputs: punctuation does not take the newlines after it.
		{",,,\n\nx", []string{",,,", "\n\n", "x"}},
		{"!?\n x", []string{"!?", "\n", " x"}},
		// A run of newlines is still one chunk.
		{"line\n\ntwo", []string{"line", "\n\n", "two"}},
		{"a\n\nb", []string{"a", "\n\n", "b"}},
		// The rest is qwen2's.
		{"a1234b", []string{"a", "1", "2", "3", "4", "b"}},
		{"don't", []string{"don", "'t"}},
		{"a.b", []string{"a", ".b"}},
		{"x=42", []string{"x", "=", "4", "2"}},
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"中文测试", []string{"中文测试"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	// qwen2 absorbs the newline into the punctuation run, which is what separates them.
	q2 := &Tokenizer{pre: preQwen2}
	if got := q2.preSplit(",,,\n\nx"); !sameStrings(got, []string{",,,\n\n", "x"}) {
		t.Errorf("qwen2 preSplit(\",,,\\n\\nx\") = %q, want the newlines absorbed", got)
	}

	path := gguftest.Model{Pre: "seed-coder", NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	real, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !real.PreImplemented() {
		t.Error("seed-coder is not reported as implemented")
	}
}

// TestMiniCPM5Chunks pins the minicpm5 rule: digit groups first, then a llama3-shaped rule
// whose digit alternative is a plain run.
//
// The two passes matter for where a remainder lands. The first cuts groups of up to three, so
// "1234567" becomes ["123", "456", "7"]; the second never sees a digit run longer than three,
// which is why its alternative is "\p{N}+" rather than llama3's "\p{N}{1,3}" even though the
// result on short runs is the same.
func TestMiniCPM5Chunks(t *testing.T) {
	tok := &Tokenizer{pre: preMiniCPM5}
	cases := []struct {
		in   string
		want []string
	}{
		{"1234", []string{"123", "4"}},
		{"1234567", []string{"123", "456", "7"}},
		{"a1234b", []string{"a", "123", "4", "b"}},
		// A short run reaches the second pass whole and stays one chunk there.
		{"x=42", []string{"x", "=", "42"}},
		// The rest is llama3's shape.
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"don't", []string{"don", "'t"}},
		{"line\n\ntwo", []string{"line", "\n\n", "two"}},
		{"a.b", []string{"a", ".b"}},
		{"中文测试", []string{"中文测试"}},
		{",,,\n\nx", []string{",,,\n\n", "x"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	// llama3 groups up to three with no leading pass, so a long run splits differently there.
	l3 := &Tokenizer{pre: preLlama3}
	if got := l3.preSplit("1234567"); !sameStrings(got, []string{"123", "456", "7"}) {
		t.Errorf("llama3 preSplit(\"1234567\") = %q", got)
	}

	path := gguftest.Model{Pre: "minicpm5", NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	real, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !real.PreImplemented() {
		t.Error("minicpm5 is not reported as implemented")
	}
}

// TestZonos3Chunks pins the zonos3 rule, which is the only family here whose alternatives do
// not cover letters at all.
//
// Three alternatives: digit groups of up to three, runs of two or more non-space non-letter
// non-digit characters, and a space run that reaches the end of the text. Letters are gap
// text, so a letter run stays whole -- "Hello, world!" is one chunk, and the comma is not long
// enough for the {2,} run. Everything about the rule follows from that gap: "x=42" keeps "x="
// because a single "=" is too short to match, and a newline matches nothing because the
// whitespace alternative is a literal space, not \s.
func TestZonos3Chunks(t *testing.T) {
	tok := &Tokenizer{pre: preZonos3}
	cases := []struct {
		in   string
		want []string
	}{
		// Letters are gap text, so a sentence with one punctuation mark stays whole.
		{"Hello, world!", []string{"Hello, world!"}},
		{"don't", []string{"don't"}},
		{"a.b", []string{"a.b"}},
		{"a, b", []string{"a, b"}},
		{"hi there", []string{"hi there"}},
		{"中文测试", []string{"中文测试"}},
		// A newline is not a space, so these are gap text too.
		{"line\n\ntwo", []string{"line\n\ntwo"}},
		{"a\n\nb", []string{"a\n\nb"}},
		// Digits group in threes; the remainder stays with the following gap.
		{"a1234b", []string{"a", "123", "4", "b"}},
		{"x=42", []string{"x=", "42"}},
		// Two or more punctuation marks match, with any newlines or slashes after them.
		{"!?!", []string{"!?!"}},
		{"a!!b", []string{"a", "!!", "b"}},
		{"!?/\n", []string{"!?/\n"}},
		{"..x", []string{"..", "x"}},
		// A space run that ends the text matches; in front of a non-space it gives back the
		// last space, which then leads the following chunk.
		{"a   ", []string{"a", "   "}},
		{"  ", []string{"  "}},
		{"a  b", []string{"a", " ", " b"}},
		{"a   b", []string{"a", "  ", " b"}},
		{"a b", []string{"a b"}},
		{"ab  cd", []string{"ab", " ", " cd"}},
		{"  \n  ", []string{"  ", "\n", "  "}},
		{"a \n b", []string{"a", " ", "\n b"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	path := gguftest.Model{Pre: "zonos3", NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	real, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !real.PreImplemented() {
		t.Error("zonos3 is not reported as implemented")
	}
}

// TestJais2Chunks pins the jais-2 rule: llama3's expression with the whitespace tail built
// from power-of-two tiers.
//
// Each tier carries the same lookahead as qwen2's tail, so a run followed by a non-space gives
// up its last space to the following chunk and the tiers only cover the rest. Tier sizes are
// tried largest first, which is why the pieces are the run's binary decomposition: 100 spaces
// become 64, 32 and 4, then " b".
func TestJais2Chunks(t *testing.T) {
	spaces := func(n int) string {
		out := make([]byte, n)
		for i := range out {
			out[i] = ' '
		}
		return string(out)
	}
	tok := &Tokenizer{pre: preJais2}
	cases := []struct {
		in   string
		want []string
	}{
		// The tiering: one space always leads the next chunk, so n spaces give the tiers for
		// n-1 followed by " b".
		{"a" + spaces(1) + "b", []string{"a", " b"}},
		{"a" + spaces(2) + "b", []string{"a", " ", " b"}},
		{"a" + spaces(3) + "b", []string{"a", "  ", " b"}},
		{"a" + spaces(4) + "b", []string{"a", "  ", " ", " b"}},
		{"a" + spaces(5) + "b", []string{"a", "    ", " b"}},
		{"a" + spaces(8) + "b", []string{"a", "    ", "  ", " ", " b"}},
		{"a" + spaces(17) + "b", []string{"a", spaces(16), " b"}},
		{"a" + spaces(100) + "b", []string{"a", spaces(64), spaces(32), spaces(2), spaces(1), " b"}},
		// A run that ends the text consumes all of it, in the same largest-first tiers.
		{"a" + spaces(3), []string{"a", spaces(2), spaces(1)}},
		{"a" + spaces(4), []string{"a", spaces(4)}},
		// The rest is llama3's shape.
		{"a1234b", []string{"a", "123", "4", "b"}},
		{"don't", []string{"don", "'t"}},
		{"line\n\ntwo", []string{"line", "\n\n", "two"}},
		{"a.b", []string{"a", ".b"}},
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	// llama3 has the single tail, so a run of four spaces is one chunk there.
	l3 := &Tokenizer{pre: preLlama3}
	if got := l3.preSplit("a" + spaces(4) + "b"); !sameStrings(got, []string{"a", spaces(3), " b"}) {
		t.Errorf("llama3 preSplit with four spaces = %q, want one run", got)
	}

	path := gguftest.Model{Pre: "jais-2", NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	real, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !real.PreImplemented() {
		t.Error("jais-2 is not reported as implemented")
	}
}

// TestExaoneMoEChunks pins the exaone-moe rule, the only family here whose letter run can
// cross a space.
//
// The alternative is "(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+": a single space followed by a
// letter continues the run, so "hi there" is one chunk. Two spaces end it, because the inner
// group consumes exactly one, which is why "a  b" splits as ["a", " ", " b"] -- the same way
// every other family treats a space -- and why the difference only shows on single spaces.
func TestExaoneMoEChunks(t *testing.T) {
	tok := &Tokenizer{pre: preExaoneMoE}
	cases := []struct {
		in   string
		want []string
	}{
		// The discriminating inputs: one space continues the run, two do not.
		{"hi there", []string{"hi there"}},
		{"a b c", []string{"a b c"}},
		{"one two three", []string{"one two three"}},
		{"a  b", []string{"a", " ", " b"}},
		{"a  b  c", []string{"a", " ", " b", " ", " c"}},
		// Punctuation still ends the run, so the space after it leads the next chunk.
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"a.b", []string{"a", ".b"}},
		// Marks join the run.
		{"e\u0301clair", []string{"e\u0301clair"}},
		// The rest is llama3's shape with a single-digit alternative.
		{"a1234b", []string{"a", "1", "2", "3", "4", "b"}},
		{"x=42", []string{"x", "=", "4", "2"}},
		{"don't", []string{"don", "'t"}},
		{"line\n\ntwo", []string{"line", "\n\n", "two"}},
		{"中文测试", []string{"中文测试"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	// bailingmoe has no such alternative, so the same phrase splits at every space there.
	bm := &Tokenizer{pre: preBailingMoE}
	if got := bm.preSplit("hi there"); !sameStrings(got, []string{"hi", " there"}) {
		t.Errorf("bailingmoe preSplit(\"hi there\") = %q, want the space kept as a prefix", got)
	}

	path := gguftest.Model{Pre: "exaone-moe", NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	real, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !real.PreImplemented() {
		t.Error("exaone-moe is not reported as implemented")
	}
}

// TestTekkenNum3Chunks pins the tekken-num3 rule: tekken's expression with its digit
// alternative replaced by a lookahead that groups three at a time from the right.
//
// That is the same right-grouping tiny_aya and superbpe use, but with no word boundary in the
// lookahead, so "a1234b" groups where tiny_aya's "1234a" would not: the difference between the
// two is the \b, not the grouping.
func TestTekkenNum3Chunks(t *testing.T) {
	tok := &Tokenizer{pre: preTekkenNum3}
	cases := []struct {
		in   string
		want []string
	}{
		// Grouped from the right, remainder first.
		{"1234", []string{"1", "234"}},
		{"123456", []string{"123", "456"}},
		{"1234567", []string{"1", "234", "567"}},
		{"12", []string{"12"}},
		// No word boundary in the lookahead, so a run touching a letter still groups.
		{"a1234b", []string{"a", "1", "234", "b"}},
		// The rest is tekken's: case boundaries, contractions, the slash tail, newline runs.
		{"helloWorld", []string{"hello", "World"}},
		{"HelloWorld", []string{"Hello", "World"}},
		{"don't", []string{"don", "'t"}},
		{"line\n\ntwo", []string{"line", "\n\n", "two"}},
		{"a/b", []string{"a", "/b"}},
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"中文测试", []string{"中文测试"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	// tekken gives one digit per chunk, which is what the digit alternative here replaces.
	tk := &Tokenizer{pre: preTekken}
	if got := tk.preSplit("1234"); !sameStrings(got, []string{"1", "2", "3", "4"}) {
		t.Errorf("tekken preSplit(\"1234\") = %q, want single digits", got)
	}

	path := gguftest.Model{Pre: "tekken-num3", NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	real, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !real.PreImplemented() {
		t.Error("tekken-num3 is not reported as implemented")
	}
}

// TestYoutuChunks pins the youtu rule: four scripts cut out first, then a case-splitting letter
// rule on what is left.
//
// The leading pass matters even where the main rule would reach a similar answer, because the
// two stages differ in what they group. Hangul, CJK punctuation, bopomofo and CJK are four
// separate alternatives, so a change of class ends a run -- and a punctuation run reaches the
// main rule whole, which then splits it.
func TestYoutuChunks(t *testing.T) {
	tok := &Tokenizer{pre: preYoutu}
	cases := []struct {
		in   string
		want []string
	}{
		// Each script is its own run, and a Latin word after Hangul starts a new chunk.
		{"\uc548\ub155\ud558\uc138\uc694", []string{"\uc548\ub155\ud558\uc138\uc694"}},
		{"\ud55c\uad6d\uc5b4test", []string{"\ud55c\uad6d\uc5b4", "test"}},
		{"\u3105\u3106\u3107", []string{"\u3105\u3106\u3107"}},
		{"\u4e2d\u6587\u6d4b\u8bd5", []string{"\u4e2d\u6587\u6d4b\u8bd5"}},
		{"\u4e2d\u6587 test", []string{"\u4e2d\u6587", " test"}},
		// A punctuation run arrives whole and the main rule then cuts it.
		{"\uff01\uff1f", []string{"\uff01", "\uff1f"}},
		// The main rule is tiny_aya's: single digits, case boundaries, a clitic suffix.
		{"a1234b", []string{"a", "1", "2", "3", "4", "b"}},
		{"helloWorld", []string{"hello", "World"}},
		{"don't", []string{"don't"}},
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"line\n\ntwo", []string{"line", "\n\n", "two"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	path := gguftest.Model{Pre: "youtu", NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	real, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !real.PreImplemented() {
		t.Error("youtu is not reported as implemented")
	}
}

// TestSpark25Chunks pins the spark2_5 rule, whose four expressions put a bare \p{N} *after*
// the main rule rather than before it.
//
// Three details separate it from the families it resembles. The leading digit pass groups in
// threes, but the trailing \p{N} then cuts every group apart, so a number ends one digit per
// chunk -- the outcome matches starcoder's, by a longer route. The letter rule does not split
// on case and takes any non-letter as its leading character, so a period attaches: "a.b" is
// ["a", ".b"]. And the newline alternative is a single [\r\n], so a pair is two chunks where
// most families would keep them together.
func TestSpark25Chunks(t *testing.T) {
	tok := &Tokenizer{pre: preSpark25}
	cases := []struct {
		in   string
		want []string
	}{
		// Grouped first, cut again: the trailing \p{N} wins.
		{"1234", []string{"1", "2", "3", "4"}},
		{"1234567", []string{"1", "2", "3", "4", "5", "6", "7"}},
		{"a1234b", []string{"a", "1", "2", "3", "4", "b"}},
		// No case split.
		{"helloWorld", []string{"helloWorld"}},
		// The leading character takes a step.
		{"a.b", []string{"a", ".b"}},
		// A newline pair is two chunks.
		{"line\n\ntwo", []string{"line", "\n", "\n", "two"}},
		// The punct run names the categories and has no newline tail.
		{"!?abc", []string{"!?", "abc"}},
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"x=42", []string{"x", "=", "4", "2"}},
		{"a  b", []string{"a", " ", " b"}},
		{"don't", []string{"don", "'t"}},
		{"e\u0301clair", []string{"e\u0301clair"}},
		// The punctuation-letter alternative claims an apostrophe that starts a chunk before
		// the contraction rule reads a clitic from it, so "'sa" is one chunk while the clitic
		// inside "don't" above still separates.
		{"'sa", []string{"'sa"}},
		{"'t", []string{"'t"}},
		// The newline alternative is a bare [\r\n], so a CRLF pair is two chunks.
		{"\r\n", []string{"\r", "\n"}},
		{"a\r\nb", []string{"a", "\r", "\n", "b"}},
		{"\n\n", []string{"\n", "\n"}},
		// The leading character excludes punctuation and symbols, so a mark after a
		// punctuation character is not taken as that character's prefix.
		{",\u0301", []string{",", "\u0301"}},
		{".\u0301", []string{".", "\u0301"}},
		{"\u4e2d\u6587\u6d4b\u8bd5", []string{"\u4e2d\u6587\u6d4b\u8bd5"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	path := gguftest.Model{Pre: "spark2_5", NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	real, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !real.PreImplemented() {
		t.Error("spark2_5 is not reported as implemented")
	}
}

// TestChameleonChunks pins the chameleon rule, which is the only family in llama.cpp whose
// expressions compose as passes rather than as alternatives.
//
// Four things separate it from every other family here, and each is a chunk boundary that
// would be wrong if the expression were read as one main rule:
//
//   - Punctuation is cut one character at a time, and before the main rule, so "a--b" is
//     ["a", "-", "-", "b"] and an apostrophe never forms a contraction: "don't" is
//     ["don", "'", "t"] where llama3 gives ["don", "'t"].
//   - Two and four spaces are alternatives matched at each position, so eight spaces are two
//     chunks of four and three spaces are two-plus-one.
//   - Digits are cut by the full \p{N} category rather than \p{Nd}, which is why the
//     superscript "²" is its own chunk and not a run with its neighbour.
//   - The image alternative matches inside ordinary text, so "IMGIMGABZX" is
//     ["IMGIMGABZ", "X"].
//
// Every expectation below was produced by transcribing llama.cpp's six expressions into
// Python's regex module and applying them in order, which shares no code with this package.
func TestChameleonChunks(t *testing.T) {
	tok := &Tokenizer{pre: preChameleon}
	cases := []struct {
		in   string
		want []string
	}{
		// The image alternative, which is the one pass with a multi-character match.
		{"IMGIMGABZ", []string{"IMGIMGABZ"}},
		{"IMGIMGABZX", []string{"IMGIMGABZ", "X"}},
		{"xIMGIMGABZ", []string{"x", "IMGIMGABZ"}},
		{"IMGIMGACDBZ", []string{"IMGIMGACDBZ"}},
		{"IMGIMGAAAAAZ", []string{"IMGIMGAAAAAZ"}},
		{"IMGIMGAB", []string{"IMGIMGAB"}},
		// Tab and newline are single chunks; a CR is left to the main rule, so a CRLF pair
		// is not the same shape as a pair of newlines.
		{"a\tb", []string{"a", "\t", "b"}},
		{"a\nb", []string{"a", "\n", "b"}},
		{"a\rb", []string{"a", "\r", "b"}},
		{"a\r\nb", []string{"a", "\r", "\n", "b"}},
		// Two and four spaces are their own alternatives, tried at each position.
		{"a b", []string{"a", " b"}},
		{"a  b", []string{"a", "  ", "b"}},
		{"a   b", []string{"a", "  ", " b"}},
		{"a    b", []string{"a", "    ", "b"}},
		{"a     b", []string{"a", "    ", " b"}},
		{"a      b", []string{"a", "    ", "  ", "b"}},
		{"a       b", []string{"a", "    ", "  ", " b"}},
		{"  indented", []string{"  ", "indented"}},
		{"a  ", []string{"a", "  "}},
		{"a ", []string{"a", " "}},
		{"x  y  z", []string{"x", "  ", "y", "  ", "z"}},
		// The digit pass runs over \p{N}, not \p{Nd}.
		{"1234", []string{"1", "2", "3", "4"}},
		{"a1234b", []string{"a", "1", "2", "3", "4", "b"}},
		{"3.14", []string{"3", ".", "1", "4"}},
		{"\u00b2\u00b3", []string{"\u00b2", "\u00b3"}},
		{"\u0661\u0662\u0663", []string{"\u0661", "\u0662", "\u0663"}},
		// Punctuation is one character per chunk, and no contraction survives it.
		{"a--b", []string{"a", "-", "-", "b"}},
		{"a--->b", []string{"a", "-", "-", "-", ">", "b"}},
		{"a!?b", []string{"a", "!", "?", "b"}},
		{"++", []string{"+", "+"}},
		{"x=42", []string{"x", "=", "4", "2"}},
		{"x_y z", []string{"x", "_", "y", " z"}},
		{"don't", []string{"don", "'", "t"}},
		{"'s", []string{"'", "s"}},
		{"they're", []string{"they", "'", "re"}},
		{"x's", []string{"x", "'", "s"}},
		// Symbols are not punctuation, so the main rule groups them.
		{"\u20ac5", []string{"\u20ac", "5"}},
		{"\u20ac\u20ac", []string{"\u20ac\u20ac"}},
		{"a\u20ac\u20acb", []string{"a", "\u20ac\u20ac", "b"}},
		{"a\U0001F600b", []string{"a", "\U0001F600", "b"}},
		// The main rule is GPT-2's: no case split, no mark joining, and a letter run keeps
		// the space before it.
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"HelloWorld", []string{"HelloWorld"}},
		{"e\u0301clair", []string{"e", "\u0301", "clair"}},
		{"\u4e2d\u6587\u6d4b\u8bd5", []string{"\u4e2d\u6587\u6d4b\u8bd5"}},
		{"a.b", []string{"a", ".", "b"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	// The family reaches its rule through the file, which is how a model gets here.
	path := gguftest.Model{Pre: "chameleon", NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	real, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !real.PreImplemented() {
		t.Error("chameleon is not reported as implemented")
	}
}

// TestChameleonDiffersFromGPT2 pins why chameleon needs its own splitter rather than an
// alias: the main rule is GPT-2's expression verbatim, so the passes are the whole family.
//
// Without them the punctuation, digit and space cases below would all come out as one
// chunk each -- the failure mode where a model still answers, on text it never saw.
func TestChameleonDiffersFromGPT2(t *testing.T) {
	cham := &Tokenizer{pre: preChameleon}
	gpt := &Tokenizer{pre: preGPT2}
	for _, in := range []string{"a--b", "don't", "a1234b", "a    b", "IMGIMGABZX", "3.14"} {
		if sameStrings(cham.preSplit(in), gpt.preSplit(in)) {
			t.Errorf("%q splits the same under chameleon and gpt-2; the passes are not running", in)
		}
	}
}

// TestWhitespaceTailMeasuresTheLastSpace checks the shared whitespace tail on a run that
// mixes character widths.
//
// The tail is "\s+(?!\S)|\s+", and its lookahead makes it give up the *last* character of the
// run so the next chunk can take it as a leading space. Measuring that last character as the
// byte length of the first one cuts the run mid-character when the widths differ: a run
// holding an em space and a plain space would emit a lone 0xE2 and leave the rest of the
// encoding to the next chunk, which is text no model was trained on and no error reports.
// Every family with the shared tail is affected, which is why the cases run over three of
// them rather than the one the bug was found on.
//
// The expectations come from the reference expressions in Python's regex module.
func TestWhitespaceTailMeasuresTheLastSpace(t *testing.T) {
	cases := []string{
		"a \u2003b",      // plain then em
		"a\u2003 b",      // em then plain
		"a \u00a0b",      // plain then no-break
		"a\u00a0\u2003b", // two wide spaces
		"a \u2003 b",     // run surrounded by text
		"a\r\u00a0b",     // the tail after a newline alternative
		"x\u2003\u2003y",
	}
	// qwen2 and llama3 share the tail and agree on every case here.
	for _, fam := range []preType{preQwen2, preLlama3} {
		tok := &Tokenizer{pre: fam}
		for _, in := range cases {
			for _, chunk := range tok.preSplit(in) {
				if !utf8.ValidString(chunk) {
					t.Errorf("pre=%v preSplit(%q) = %q: chunk %q is not valid UTF-8",
						fam, in, tok.preSplit(in), chunk)
				}
			}
		}
	}
	// The GPT-2 tail has no run grouping, so its chunks are narrower; spelling two of them
	// out keeps the test from passing on a splitter that simply cuts everything.
	gpt := &Tokenizer{pre: preGPT2}
	want := map[string][]string{
		"a \u2003b":      {"a", " ", "\u2003", "b"},
		"a\u2003\u2003y": {"a", "\u2003", "\u2003", "y"},
		"a \u2003 b":     {"a", " \u2003", " b"},
	}
	for in, expect := range want {
		if got := gpt.preSplit(in); !sameStrings(got, expect) {
			t.Errorf("gpt-2 preSplit(%q) = %q, want %q", in, got, expect)
		}
	}
	qwen := &Tokenizer{pre: preQwen2}
	if got, expect := qwen.preSplit("a \u2003b"), []string{"a", " ", "\u2003b"}; !sameStrings(got, expect) {
		t.Errorf("qwen2 preSplit(%q) = %q, want %q", "a \u2003b", got, expect)
	}
}

// TestK2HorizonChunks pins the k2-horizon rule, which is qwen35's shape with two changes to
// the letter alternative.
//
// The first is that the run takes the joining controls U+200C and U+200D. Neither is a letter
// or a mark, so without the flag they fall to the punctuation alternative and split the word --
// and a word split inside a zero-width joiner is text the vocabulary never saw. The second is
// digit grouping of up to three, where qwen35 cuts every digit on its own.
//
// The expectations come from a transcription of the expression into Python's regex module,
// which shares no code with this package.
func TestK2HorizonChunks(t *testing.T) {
	tok := &Tokenizer{pre: preK2Horizon}
	cases := []struct {
		in   string
		want []string
	}{
		// The joining controls are class members, so they neither end a run nor start one.
		{"a\u200cb", []string{"a\u200cb"}},
		{"a\u200db", []string{"a\u200db"}},
		{"ab\u200c\u200dcd", []string{"ab\u200c\u200dcd"}},
		{"\u200ca", []string{"\u200ca"}},
		{"a\u200c", []string{"a\u200c"}},
		// A space before a joiner-led run stays with the run, as the leading-character
		// alternative allows for any non-letter.
		{"x \u200cy", []string{"x", " \u200cy"}},
		// Combining marks join the run, and a mark may open one.
		{"e\u0301clair", []string{"e\u0301clair"}},
		{"cafe\u0301", []string{"cafe\u0301"}},
		{"a\u0301\u0302b", []string{"a\u0301\u0302b"}},
		{"\u0301a", []string{"\u0301a"}},
		{"\u0301\u0302a", []string{"\u0301\u0302a"}},
		{"a\u0301 b", []string{"a\u0301", " b"}},
		// Digits group up to three, where qwen35 cuts each one.
		{"1234", []string{"123", "4"}},
		{"1234567", []string{"123", "456", "7"}},
		{"a1234b", []string{"a", "123", "4", "b"}},
		{"12", []string{"12"}},
		// \p{N} is the whole number category, so a superscript groups with what follows.
		{"\u00b2\u00b3", []string{"\u00b2\u00b3"}},
		// The rest is qwen35's expression: clitics, punctuation runs, newlines, whitespace.
		{"don't", []string{"don", "'t"}},
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"a--b", []string{"a", "--", "b"}},
		{"a,,\n\nb", []string{"a", ",,\n\n", "b"}},
		{"line\n\ntwo", []string{"line", "\n\n", "two"}},
		{"a\r\nb", []string{"a", "\r\n", "b"}},
		{"a  b", []string{"a", " ", " b"}},
		{"a \u2003b", []string{"a", " ", "\u2003b"}},
		{"HelloWorld", []string{"HelloWorld"}},
		{"\u4e2d\u6587\u6d4b\u8bd5", []string{"\u4e2d\u6587\u6d4b\u8bd5"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	// qwen35 must not have gained either change: the two rules are one field apart, and a
	// careless edit could move both.
	qwen := &Tokenizer{pre: preQwen35}
	if got, want := qwen.preSplit("a\u200cb"), []string{"a", "\u200cb"}; !sameStrings(got, want) {
		t.Errorf("qwen35 preSplit(\"a\\u200cb\") = %q, want the joiner cut off", got)
	}
	if got, want := qwen.preSplit("1234"), []string{"1", "2", "3", "4"}; !sameStrings(got, want) {
		t.Errorf("qwen35 preSplit(\"1234\") = %q, want one digit per chunk", got)
	}

	path := gguftest.Model{Pre: "k2-horizon", NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	real, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !real.PreImplemented() {
		t.Error("k2-horizon is not reported as implemented")
	}
}

// TestLetterRunOpensOnAMark covers the families whose letter class admits more than letters.
//
// The expression is "[\p{L}\p{M}]+", not "\p{L}\p{M}*", so a combining mark can *open* a run:
// "\u0301a" is one chunk. Reading the mark as the leading-character alternative instead yields
// ["\u0301", "a"], which is a boundary the vocabulary never saw. The same applies to
// k2-horizon's joining controls, which are class members rather than a suffix on the run.
func TestLetterRunOpensOnAMark(t *testing.T) {
	cases := []struct {
		fam  preType
		in   string
		want []string
	}{
		{preQwen35, "\u0301a", []string{"\u0301a"}},
		{preQwen35, "\u0301\u0302a", []string{"\u0301\u0302a"}},
		{preQwen35, "a\u0301\u0302b", []string{"a\u0301\u0302b"}},
		{preDeepseek3, "\u0301a", []string{"\u0301a"}},
		{preK2Horizon, "\u200ca", []string{"\u200ca"}},
		{preK2Horizon, "\u200c\u200da", []string{"\u200c\u200da"}},
		// The distinguishing shape: a leading character *then* a mark. Taking the mark as
		// a prefix as well leaves an empty run, so the run cannot start and the mark is
		// emitted as its own chunk -- ["\u0301", "a"]-style output where the reference has
		// [" \u0301a"].
		{preQwen35, " \u0301a", []string{" \u0301a"}},
		{preQwen35, ".\u0301a", []string{".\u0301a"}},
		{preQwen35, "x \u0301y", []string{"x", " \u0301y"}},
		{preK2Horizon, " \u0301a", []string{" \u0301a"}},
		{preK2Horizon, "x \u0301y", []string{"x", " \u0301y"}},
	}
	for _, c := range cases {
		tok := &Tokenizer{pre: c.fam}
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("pre=%v preSplit(%q) = %q, want %q", c.fam, c.in, got, c.want)
		}
	}
	// A family without marks in its class reaches the same chunks by a different route, and
	// the difference still shows on text where the mark is between two words: qwen2's leading
	// character takes the mark as a prefix, so "a\u0301b" is ["a", "\u0301b"] there and
	// ["a\u0301b"] under qwen35. Both are what the reference does.
	qwen := &Tokenizer{pre: preQwen2}
	if got, want := qwen.preSplit("a\u0301b"), []string{"a", "\u0301b"}; !sameStrings(got, want) {
		t.Errorf("qwen2 preSplit(%q) = %q, want the mark taken as a prefix", "a\u0301b", got)
	}
	q35 := &Tokenizer{pre: preQwen35}
	if got, want := q35.preSplit("a\u0301b"), []string{"a\u0301b"}; !sameStrings(got, want) {
		t.Errorf("qwen35 preSplit(%q) = %q, want one run", "a\u0301b", got)
	}
}

// TestNumberClassIsWholeCategory pins "\p{N}" as the whole number category rather than
// \p{Nd}.
//
// The expressions write \p{N}, which also holds letter numbers (Roman numerals), superscripts,
// and vulgar fractions. A decimal-digit reading changes boundaries silently on text that does
// contain them: under qwen2 "²³" must be two chunks and under llama-bpe one, and the two
// families disambiguate the class from opposite directions.
//
// Expected chunks come from the reference expressions in Python's regex module.
func TestNumberClassIsWholeCategory(t *testing.T) {
	qwen2 := &Tokenizer{pre: preQwen2}
	llama3 := &Tokenizer{pre: preLlama3}

	// qwen2's bare \p{N} cuts each number character on its own.
	for _, c := range []struct {
		in   string
		want []string
	}{
		{"\u00b2\u00b3", []string{"\u00b2", "\u00b3"}},
		{"1\u00b22", []string{"1", "\u00b2", "2"}},
		{"\u00bd\u00bcn", []string{"\u00bd", "\u00bc", "n"}},
		{"a\u2163b", []string{"a", "\u2163", "b"}},
	} {
		if got := qwen2.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("qwen2 preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}
	// llama-bpe's \p{N}{1,3} groups them up to three.
	for _, c := range []struct {
		in   string
		want []string
	}{
		{"\u00b2\u00b3", []string{"\u00b2\u00b3"}},
		{"1\u00b22", []string{"1\u00b22"}},
		{"\u00bd\u00bcn", []string{"\u00bd\u00bc", "n"}},
	} {
		if got := llama3.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("llama-bpe preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestSplitByRuleKeepsGapTextTogether covers what happens when no alternative matches at a
// position: the character is gap text, and llama.cpp's splitter emits the gap before a match
// as *one* fragment.
//
// deepseek-v3's punctuation alternative takes "\p{P}\p{S}", which excludes the zero-width
// joiner, so "\u200c\u200c" matches nothing and is one chunk. Emitting gap characters one at a
// time would put a boundary between two identical controls, which is a split the vocabulary
// never saw -- and an earlier version went further and emitted one *byte* at a time, cutting
// a multi-byte rune in half.
func TestSplitByRuleKeepsGapTextTogether(t *testing.T) {
	d3 := &Tokenizer{pre: preDeepseek3}
	cases := []struct {
		in   string
		want []string
	}{
		{"A\u200d\u200d", []string{"A", "\u200d\u200d"}},
		{"0\t#\u00b2\t`+:\u200c\u200c", []string{"0", "\t", "#", "\u00b2", "\t", "`+:", "\u200c\u200c"}},
		{"x\u200c", []string{"x", "\u200c"}},
	}
	for _, c := range cases {
		if got := d3.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("deepseek-v3 preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
		for _, chunk := range d3.preSplit(c.in) {
			if !utf8.ValidString(chunk) {
				t.Errorf("deepseek-v3 preSplit(%q) chunk %q is not valid UTF-8", c.in, chunk)
			}
		}
	}
}

// TestPunctuationTailTakesOneCharacter covers exaone-moe's "[\\r\\n/]?" tail, which takes at
// most one character.
//
// The neighbouring families write "[\\r\\n]*" (tekken) or "\\s*[\\r\\n]+", and both consume the
// whole run. Copying that shape here would merge ",\r\n" into one chunk, where the reference
// gives [",\r", "\n"].
func TestPunctuationTailTakesOneCharacter(t *testing.T) {
	exa := &Tokenizer{pre: preExaoneMoE}
	cases := []struct {
		in   string
		want []string
	}{
		{"#,\n\u3042", []string{"#,\n", "\u3042"}},
		{"\r{a\u2163'\n'0^$", []string{"\r", "{a", "\u2163", "'\n", "'", "0", "^$"}},
		{"a,\r\nb", []string{"a", ",\r", "\n", "b"}},
	}
	for _, c := range cases {
		if got := exa.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("exaone-moe preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}
	// tekken's "*" must still consume the whole run: the two tails are not interchangeable.
	tekken := &Tokenizer{pre: preTekken}
	if got, want := tekken.preSplit("a,\r\nb"), []string{"a", ",\r\n", "b"}; !sameStrings(got, want) {
		t.Errorf("tekken preSplit(%q) = %q, want %q", "a,\r\nb", got, want)
	}
}

// TestPunctuationClassExcludesMarksOnlyForQwen35 covers the one family whose punctuation
// complement names \p{M}.
//
// qwen35 writes "[^\\s\\p{L}\\p{M}\\p{N}]", so a combining mark ends a punctuation run and the
// letter alternative takes it: "#\u0301" is ["#", "\u0301"]. The other families write
// "[^\\s\\p{L}\\p{N}]", where a mark is neither a letter nor a number and so joins the run.
func TestPunctuationClassExcludesMarksOnlyForQwen35(t *testing.T) {
	q35 := &Tokenizer{pre: preQwen35}
	// The mark ends the punctuation *run*, but the leading-character alternative then takes
	// it as a prefix on the letter run that follows, so a mark with a letter after it still
	// joins: "#\u0301a" is one chunk. A mark with no letter after it is its own chunk.
	for _, c := range []struct {
		in   string
		want []string
	}{
		{"#,#\u0301", []string{"#,#", "\u0301"}},
		{"*)\u0301", []string{"*)", "\u0301"}},
		{"#\u0301a", []string{"#\u0301a"}},
	} {
		if got := q35.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("qwen35 preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}
	// llama-bpe's complement does not name \p{M}, so the mark stays with the punctuation.
	llama := &Tokenizer{pre: preLlama3}
	if got, want := llama.preSplit("#\u0301"), []string{"#\u0301"}; !sameStrings(got, want) {
		t.Errorf("llama-bpe preSplit(%q) = %q, want %q", "#\u0301", got, want)
	}
}

// TestWordSpaceRunNeedsALetterAtTheStart covers exaone-moe's letter alternative,
// "(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+".
//
// The marks are a suffix on each word rather than members of one class, so a mark cannot open
// the run: "\u0301|" is ["\u0301", "|"] here, where qwen35's "[\p{L}\p{M}]+" makes
// ["\u0301|"] of it. The distinction shows whenever a mark follows text the run did not
// take -- after a digit, say -- and the earlier code accepted the mark as a start.
func TestWordSpaceRunNeedsALetterAtTheStart(t *testing.T) {
	exa := &Tokenizer{pre: preExaoneMoE}
	// A mark after a digit ends the digit's chunk and starts nothing, so it joins the
	// punctuation-like gap that follows rather than leading a run of its own.
	if got, want := exa.preSplit("1\u0301%"), []string{"1", "\u0301%"}; !sameStrings(got, want) {
		t.Errorf("exaone-moe preSplit(%q) = %q, want %q", "1\u0301%", got, want)
	}
	if got, want := exa.preSplit("\u0301<"), []string{"\u0301<"}; !sameStrings(got, want) {
		t.Errorf("exaone-moe preSplit(%q) = %q, want %q", "\u0301<", got, want)
	}
	// The same mark inside a word stays with it, which is what the suffix means.
	if got, want := exa.preSplit("cafe\u0301"), []string{"cafe\u0301"}; !sameStrings(got, want) {
		t.Errorf("exaone-moe preSplit(%q) = %q, want %q", "cafe\u0301", got, want)
	}
	// qwen35's class admits a mark, so a mark that starts the text is a run there: it is
	// the leading character that is excluded when it is a digit or a newline, not a mark.
	// That makes the two rules differ on exactly the inputs above.
	if got, want := (&Tokenizer{pre: preQwen35}).preSplit("\u0301a"), []string{"\u0301a"}; !sameStrings(got, want) {
		t.Errorf("qwen35 preSplit(%q) = %q, want %q", "\u0301a", got, want)
	}
}

// TestAfMoEChunks pins the afmoe rule, which is three passes: right-aligned digit groups, a
// wide class of Asian scripts, then deepseek-v3's main expression.
//
// The digit pass is llama.cpp's custom matcher rather than a regex -- a run is cut into groups
// of three counted from the right, with a one- or two-digit remainder leading -- so "12345" is
// ["12", "345"] where deepseek-v3's leading "\p{N}{1,3}" gives ["123", "45"]. The Asian class
// is far wider than the three ranges the other families name: it reaches extension A, the
// Kangxi radicals, halfwidth katakana, and the Thai, Lao, Khmer, Myanmar, Cham, Javanese,
// Roman-numeral-free Hangul and jamo runs.
//
// Expectations come from a transcription of the three expressions into Python's regex module,
// with the digit pass written out by hand from llama.cpp's custom matcher.
func TestAfMoEChunks(t *testing.T) {
	tok := &Tokenizer{pre: preAfMoE}
	cases := []struct {
		in   string
		want []string
	}{
		// Right-aligned groups of three.
		{"1234", []string{"1", "234"}},
		{"12345", []string{"12", "345"}},
		{"1234567", []string{"1", "234", "567"}},
		{"1234567890", []string{"1", "234", "567", "890"}},
		{"12", []string{"12"}},
		{"12\u0663\u0664", []string{"1", "2\u0663\u0664"}},
		{"a1234b", []string{"a", "1", "234", "b"}},
		{"\u0661\u0662\u0663\u0664", []string{"\u0661", "\u0662\u0663\u0664"}},
		// The wide Asian class: one run per script, and each stays whole.
		{"\u4e00\u4e01", []string{"\u4e00\u4e01"}},
		{"\u3400\u3401", []string{"\u3400\u3401"}},
		{"\u3041\u3042", []string{"\u3041\u3042"}},
		{"\u30a1\u30a2", []string{"\u30a1\u30a2"}},
		{"\u2f00\u2f01", []string{"\u2f00\u2f01"}},
		{"\u0e40\u0e41", []string{"\u0e40\u0e41"}},
		{"\u0e80\u0e81", []string{"\u0e80\u0e81"}},
		{"\u1780\u1781", []string{"\u1780\u1781"}},
		{"\u1000\u1001", []string{"\u1000\u1001"}},
		{"\uaa60\uaa61", []string{"\uaa60\uaa61"}},
		{"\ua9e0\ua9e1", []string{"\ua9e0\ua9e1"}},
		{"\uac00\uac01", []string{"\uac00\uac01"}},
		{"\u1100\u1101", []string{"\u1100\u1101"}},
		{"a\u3400b", []string{"a", "\u3400", "b"}},
		// Halfwidth katakana is in the class but its members are not all letters, so the
		// main rule then cuts it: the reference gives the same two chunks.
		{"\uff65\uff66", []string{"\uff65", "\uff66"}},
		// The main expression is deepseek-v3's: the punctuation-letter run comes first, and
		// there is no contraction alternative, so an apostrophe leads a word.
		{"'ta", []string{"'ta"}},
		{"don't", []string{"don", "'t"}},
		{"a.b", []string{"a", ".b"}},
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"a--b", []string{"a", "--", "b"}},
		{"a,,\n\nb", []string{"a", ",,\n\n", "b"}},
		// "\s*[\r\n]+" consumes a whitespace run that ends on a newline.
		{"\n\t\r", []string{"\n\t\r"}},
		{"\n  \n", []string{"\n  \n"}},
		{"\n\t\u00b2", []string{"\n", "\t", "\u00b2"}},
		{"a b", []string{"a", " b"}},
		{"a \u2003b", []string{"a", " ", "\u2003b"}},
		{"e\u0301clair", []string{"e\u0301clair"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	// The digit grouping is the opposite of deepseek-v3's, so an alias would be wrong.
	d3 := &Tokenizer{pre: preDeepseek3}
	if got, want := d3.preSplit("12345"), []string{"123", "45"}; !sameStrings(got, want) {
		t.Errorf("deepseek-v3 preSplit(%q) = %q, want %q", "12345", got, want)
	}

	path := gguftest.Model{Pre: "afmoe", NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	real, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !real.PreImplemented() {
		t.Error("afmoe is not reported as implemented")
	}
}

// TestNewlineAlternativeEndsOnANewline covers "\s*[\r\n]+", which is greedy and anchored: the
// match runs to the farthest newline its leading \s* can reach, and whitespace after that is
// left to the "\s+" alternatives.
//
// Reading it as "leading spaces then newlines" instead splits "\n  \n" into pieces and glues
// "\n\t" together, both of which the reference contradicts: the first is one chunk because
// the run ends on a newline, the second is two because it does not.
func TestNewlineAlternativeEndsOnANewline(t *testing.T) {
	tok := &Tokenizer{pre: preLlama3}
	cases := []struct {
		in   string
		want []string
	}{
		{"\n  \n", []string{"\n  \n"}},
		{"\n\n  \n", []string{"\n\n  \n"}},
		{"  \n", []string{"  \n"}},
		{"\n\t", []string{"\n", "\t"}},
		{"\n\t\u00b2", []string{"\n", "\t", "\u00b2"}},
		{"a\n\t\rb", []string{"a", "\n\t\r", "b"}},
		// The two spaces after the newline do not continue the chunk: the run cannot end on
		// them, so the "\s+(?!\S)" alternative takes the first and the space before the digit
		// run is the next chunk's leading character.
		{"a\n  12345b", []string{"a", "\n", " ", " ", "123", "45", "b"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("llama-bpe preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestPunctuationLetterRunHasNoContraction covers the families whose first alternative is
// "[!..~][A-Za-z]+" and whose expression has no contraction alternative at all.
//
// The engine applies the contraction rule unconditionally unless a family opts out, so
// deepseek-v3 read "'ta" as ["'t", "a"]. Its expression has no "'s|'t|..." alternative: the
// apostrophe is the punctuation character of the first alternative and the letters after it
// follow, which is one chunk.
func TestPunctuationLetterRunHasNoContraction(t *testing.T) {
	for _, fam := range []preType{preDeepseek3, preAfMoE} {
		tok := &Tokenizer{pre: fam}
		if got, want := tok.preSplit("'ta"), []string{"'ta"}; !sameStrings(got, want) {
			t.Errorf("pre=%v preSplit(%q) = %q, want %q", fam, "'ta", got, want)
		}
		// A contraction the expression *does* cover is still cut, because the apostrophe
		// and the letters after it are the punctuation-letter run.
		if got, want := tok.preSplit("don't"), []string{"don", "'t"}; !sameStrings(got, want) {
			t.Errorf("pre=%v preSplit(%q) = %q, want %q", fam, "don't", got, want)
		}
	}
	// A family with the alternative must still produce the clitic chunk.
	llama := &Tokenizer{pre: preLlama3}
	if got, want := llama.preSplit("'ta"), []string{"'t", "a"}; !sameStrings(got, want) {
		t.Errorf("llama-bpe preSplit(%q) = %q, want %q", "'ta", got, want)
	}
}

// TestEveryKnownNameResolves pins the claim the README now makes: every name llama.cpp maps to
// a family resolves to a rule here.
//
// The middle state -- a name in knownPreNames with no rule -- is the one that produces a
// silent split as one chunk, so it is the one worth failing on directly. Without this test,
// adding a name from a newer llama.cpp and forgetting the rule would only surface as
// unexpectedly poor output.
func TestEveryKnownNameResolves(t *testing.T) {
	var missing []string
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
		if !tok.PreImplemented() {
			missing = append(missing, name)
		}
	}
	if len(missing) > 0 {
		t.Errorf("%d names llama.cpp maps to a family have no rule here: %q -- a file naming "+
			"one splits as a single chunk, which is silent", len(missing), missing)
	}
}

// TestKimiK2Chunks pins llama.cpp's hand-written matcher for this family, which departs from
// the regex comment above it in three ways.
//
// The matcher is what runs, so it is what these expectations follow:
//
//   - There is no case boundary. "camelCase" and "HTTPServer" are one chunk each, though the
//     comment's two letter alternatives would cut them.
//   - A combining mark ends a word rather than joining it: "a\u0301" is ["a", "\u0301"]. A
//     mark may still be the word's leading character, so "\u0301a" is one chunk.
//   - The leading character is taken before a clitic is looked for, so an apostrophe at the
//     start of a chunk is that character plus the word -- "'scamelCase" is one chunk where a
//     prefix-contraction rule would cut "'s" off the front. A clitic after letters is
//     absorbed, so "don't" stays whole.
//
// Han is cut out first and excluded from the word alternative, which is the family's purpose:
// a Han character never joins a neighbouring word.
//
// Expectations come from a line-by-line transcription of unicode_regex_split_custom_kimi_k2.
func TestKimiK2Chunks(t *testing.T) {
	tok := &Tokenizer{pre: preKimiK2}
	cases := []struct {
		in   string
		want []string
	}{
		// Han runs are cut first and stay whole.
		{"\u4e2d\u6587\u6d4b\u8bd5", []string{"\u4e2d\u6587\u6d4b\u8bd5"}},
		{"\u4e00\u4e01", []string{"\u4e00\u4e01"}},
		{"a\u4e2db", []string{"a", "\u4e2d", "b"}},
		{"\u4e2da", []string{"\u4e2d", "a"}},
		// No case boundary: the matcher consumes every non-Han letter in one run.
		{"HTTPServer", []string{"HTTPServer"}},
		{"camelCase", []string{"camelCase"}},
		{"helloWorld", []string{"helloWorld"}},
		{"aB", []string{"aB"}},
		// The leading character is any non-newline, non-letter, non-number character.
		{"a.b", []string{"a", ".b"}},
		{".b", []string{".b"}},
		{" x", []string{" x"}},
		{"-y", []string{"-y"}},
		{"(z", []string{"(z"}},
		// A clitic after letters is absorbed; one at the start is a leading character.
		{"don't", []string{"don't"}},
		{"it's", []string{"it's"}},
		{"they're", []string{"they're"}},
		{"we'll", []string{"we'll"}},
		{"I've", []string{"I've"}},
		{"x's", []string{"x's"}},
		{"'scamelCase", []string{"'scamelCase"}},
		{"'recamelCase", []string{"'recamelCase"}},
		// A mark ends the word, and may lead one.
		{"a\u0301", []string{"a", "\u0301"}},
		{"\u0301a", []string{"\u0301a"}},
		// Digits group up to three.
		{"1234", []string{"123", "4"}},
		{"1234567", []string{"123", "456", "7"}},
		{"a1234b", []string{"a", "123", "4", "b"}},
		// Punctuation, newline and whitespace alternatives.
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"a--b", []string{"a", "--", "b"}},
		{"a,,\n\nb", []string{"a", ",,\n\n", "b"}},
		{"!?abc", []string{"!?", "abc"}},
		{"line\n\ntwo", []string{"line", "\n\n", "two"}},
		{"a\r\nb", []string{"a", "\r\n", "b"}},
		{"a b", []string{"a", " b"}},
		{"a  b", []string{"a", " ", " b"}},
		{"\n  \n", []string{"\n  \n"}},
		{"\n\t", []string{"\n", "\t"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	// The Han exclusion is what separates this family from the ones that cut a CJK *range*:
	// a Han character must not join a neighbouring Latin word.
	if got, want := tok.preSplit("a\u4e2db"), []string{"a", "\u4e2d", "b"}; !sameStrings(got, want) {
		t.Errorf("preSplit(%q) = %q, want the Han character cut out", "a\u4e2db", got)
	}
	// The manifest reaches its rule through the file.
	path := gguftest.Model{Pre: "kimi-k2", NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	real, err := New(f)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !real.PreImplemented() {
		t.Error("kimi-k2 is not reported as implemented")
	}
}

// TestSpark25NewlineIsOneCharacter covers spark2_5's "[\\r\\n]", which is a single newline
// character rather than the "\\s*[\\r\\n]+" run most families use.
//
// The two differ on a CRLF pair and on a doubled newline: the run form keeps either together
// as one chunk, while the single-character form cuts them apart. The field that selects this
// existed and was set but was never read, so the rule silently behaved like the run families
// -- a boundary the vocabulary was not trained to see.
func TestSpark25NewlineIsOneCharacter(t *testing.T) {
	tok := &Tokenizer{pre: preSpark25}
	cases := []struct {
		in   string
		want []string
	}{
		{"\r\n", []string{"\r", "\n"}},
		{"\n\n", []string{"\n", "\n"}},
		{"a\r\nb", []string{"a", "\r", "\n", "b"}},
		// The whitespace before a newline still travels with it, which is what the
		// "\\s+(?!\\S)" alternative does for the run that ends the text.
		{"  \r\n", []string{"  \r\n"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}
	// A family with the run form keeps a pair together, which is what makes this a
	// per-family rule rather than a global one.
	llama := &Tokenizer{pre: preLlama3}
	if got, want := llama.preSplit("a\n\nb"), []string{"a", "\n\n", "b"}; !sameStrings(got, want) {
		t.Errorf("llama-bpe preSplit(%q) = %q, want %q", "a\n\nb", got, want)
	}
}

// TestSpark25HasNoContractionAlternative covers the apostrophe at the start of a word.
//
// spark2_5's letter rule has no contraction branch: it is the punctuation-letter run, then
// the letter run, then the punctuation run. An apostrophe that opens a word is therefore the
// punctuation character of the first branch and the letters after it follow, so "'sa" is one
// chunk. The shared engine applies a contraction rule whenever a family does not opt out, and
// with it "'s" is cut off the front of a word that has no clitic -- a boundary the vocabulary
// never saw.
//
// The clitic *inside* a word is unaffected and must stay split, which is why the fix is
// per-family rather than a change to the contraction rule.
func TestSpark25HasNoContractionAlternative(t *testing.T) {
	tok := &Tokenizer{pre: preSpark25}
	cases := []struct {
		in   string
		want []string
	}{
		{"'sa", []string{"'sa"}},
		{"'s", []string{"'s"}},
		{"'t", []string{"'t"}},
		// A clitic after a letter is the punctuation-letter branch, not a contraction rule.
		{"don't", []string{"don", "'t"}},
		{"it's", []string{"it", "'s"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}
	// A family that does have the alternative keeps the clitic reading, which is what makes
	// this a per-family decision.
	llama := &Tokenizer{pre: preLlama3}
	if got, want := llama.preSplit("'sa"), []string{"'s", "a"}; !sameStrings(got, want) {
		t.Errorf("llama-bpe preSplit(%q) = %q, want %q", "'sa", got, want)
	}
}

// TestSpark25PunctuationIsNotAPrefix covers the leading character of the letter alternative.
//
// spark2_5 writes "[^\\r\\n\\p{L}\\p{P}\\p{S}]?", which excludes punctuation and symbols.
// Reading it as qwen2's "[^\\r\\n\\p{L}\\p{N}]?" lets a punctuation mark take a combining
// mark as its own tail: ",<combining acute>" became one chunk instead of the comma and the
// mark. The two forms agree on a plain word, which is why the difference needs a mark to
// show up.
func TestSpark25PunctuationIsNotAPrefix(t *testing.T) {
	tok := &Tokenizer{pre: preSpark25}
	cases := []struct {
		in   string
		want []string
	}{
		{",\u0301", []string{",", "\u0301"}},
		{".\u0301", []string{".", "\u0301"}},
		{"-\u0301", []string{"-", "\u0301"}},
		// A mark is not punctuation, so it may still lead a letter run.
		{"\u0301a", []string{"\u0301a"}},
		{"e\u0301clair", []string{"e\u0301clair"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestFalconGroupsOnlyASCIIDigits covers falcon's third expression, "[0-9][0-9][0-9]".
//
// It is narrower than both \d and \p{N}: no other script's digits match it and neither does a
// Roman numeral or a superscript. Reading it as \p{N} made a superscript part of a group and,
// worse, grouped runs the expression leaves alone -- "99\u00b2" is one chunk because only "99"
// is in the class and two digits do not fill a group.
//
// The grouping also reads byte offsets rather than a count, so a run containing a multi-byte
// character cannot be sliced mid-character.
func TestFalconGroupsOnlyASCIIDigits(t *testing.T) {
	tok := &Tokenizer{pre: preFalcon}
	cases := []struct {
		in   string
		want []string
	}{
		// Only the two ASCII digits are in the class, so no group forms.
		{"99\u00b2", []string{"99\u00b2"}},
		{"\u2163\u2163\u2163", []string{"\u2163\u2163\u2163"}},
		// Three ASCII digits do form a group.
		{"1234", []string{"123", "4"}},
		{"12345678", []string{"123", "456", "78"}},
		{"1234567\u00b2", []string{"123", "456", "7\u00b2"}},
	}
	for _, c := range cases {
		got := tok.preSplit(c.in)
		if !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
		for _, piece := range got {
			if !utf8.ValidString(piece) {
				t.Errorf("preSplit(%q) produced invalid UTF-8 %q", c.in, piece)
			}
		}
	}
}

// TestCaseSplitUsesTheExpressionsOwnClasses covers the two-alternative letter rule.
//
// The case-splitting families write the run as "upper* lower+|upper+ lower*" with
// upper = [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}] and lower = [\p{Ll}\p{Lm}\p{Lo}\p{M}]. Those classes
// overlap, and the alternatives are ordered, so the split is not a lower-to-upper transition:
//
//   - "\u6587A" is ["\u6587", "A"]: the ideograph is both classes, so the first alternative's
//     upper* backtracks and lets lower+ take it, and the A starts a new run.
//   - "b\u0e01" is one chunk: the Thai letter is in both classes, so lower+ takes both.
//   - "aB" is ["a", "B"]: the first alternative's lower+ stops at the capital.
//
// A single greedy scan testing unicode.IsUpper gets all three wrong.
func TestCaseSplitUsesTheExpressionsOwnClasses(t *testing.T) {
	tok := &Tokenizer{pre: preTinyAya}
	cases := []struct {
		in   string
		want []string
	}{
		{"\u6587A", []string{"\u6587", "A"}},
		{"&\u6587A", []string{"&\u6587", "A"}},
		{"b\u0e01", []string{"b\u0e01"}},
		{"\u4e2d\u4e2da1", []string{"\u4e2d\u4e2da", "1"}},
		{"aB", []string{"a", "B"}},
		{"ABc", []string{"ABc"}},
		{"A\u6587", []string{"A\u6587"}},
		{"helloWorld", []string{"hello", "World"}},
		{"HTTPServer", []string{"HTTPServer"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestStopSetTreatsAllWhitespaceAsStopping covers the " ?[^(\\s|...)]+" expression.
//
// The class excludes "\\s" -- every Unicode whitespace -- but the stop set listed only the
// ASCII four. A no-break space or an em space is neither in that list nor a letter or digit,
// so it rode along inside a run instead of ending it: ";=*{\u4e2d\u00a0" came back as one
// chunk where the reference gives [";=*{\u4e2d", "\u00a0"].
func TestStopSetTreatsAllWhitespaceAsStopping(t *testing.T) {
	tok := &Tokenizer{pre: preViking}
	cases := []struct {
		in   string
		want []string
	}{
		{";=*{\u4e2d\u00a0", []string{";=*{\u4e2d", "\u00a0"}},
		{"a\u00a0b", []string{"a", "\u00a0", "b"}},
		{"a\u2003b", []string{"a", "\u2003", "b"}},
		// An ordinary space still attaches to the run that follows it.
		{"abc def", []string{"abc", " def"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestCurlyQuantifierCountsCharacters covers zonos3's " ?[^\\s\\p{L}\\p{N}]{2,}".
//
// "{2,}" counts characters; the check compared byte offsets, so a single three-byte U+200C
// satisfied it and became a run of its own where the reference leaves it as gap text attached
// to its neighbour.
func TestCurlyQuantifierCountsCharacters(t *testing.T) {
	tok := &Tokenizer{pre: preZonos3}
	cases := []struct {
		in   string
		want []string
	}{
		// One non-ASCII character: not a run, so it stays with the gap around it.
		{";\r\n\u200c9*", []string{";\r\n\u200c", "9", "*"}},
		{"\u200c", []string{"\u200c"}},
		// Two ASCII characters do satisfy the quantifier.
		{";;", []string{";;"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestDeepseekFamiliesSplitTrailingAndRepeatedNewlines covers two rules that the run-based
// splitter got wrong.
//
// deepseek-llm's fourth expression is "\s+$", anchored to the end of the *text* rather than of
// each piece, so a trailing tab is its own chunk. Applied per piece it would cut every piece
// that ends in whitespace, which is why the earlier code left it out entirely.
//
// deepseek-coder's first expression is "[\r\n]", one character, with no "+" -- so a pair of
// carriage returns is two chunks. The step was declared without the flag that says so, and
// both families are silent about it: the text tokenizes either way, just not to the pieces
// the vocabulary was trained on.
func TestDeepseekFamiliesSplitTrailingAndRepeatedNewlines(t *testing.T) {
	llm := &Tokenizer{pre: preDeepseekLLM}
	llmCases := []struct {
		in   string
		want []string
	}{
		{"\U0001F600\t", []string{"\U0001F600", "\t"}},
		{"\u200c\u4e2d\u2003", []string{"\u200c\u4e2d", "\u2003"}},
		// A space in the middle is not trailing, so the run keeps it.
		{"a b", []string{"a", " b"}},
	}
	for _, c := range llmCases {
		if got := llm.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("deepseek-llm preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}

	coder := &Tokenizer{pre: preDeepseekCoder}
	coderCases := []struct {
		in   string
		want []string
	}{
		{"\r\n", []string{"\r", "\n"}},
		{"\r\r", []string{"\r", "\r"}},
		{"a\r\nb", []string{"a", "\r", "\n", "b"}},
	}
	for _, c := range coderCases {
		if got := coder.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("deepseek-coder preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestTinyAyaDigitClassIsNarrowerThanNumber covers where the two digit classes differ.
//
// tiny_aya's digit expression is "\d{1,3}(?=(?:\d{3})*\b)", which names \d -- the Unicode
// decimal digits -- not \p{N}. The difference is the superscripts, the Roman numerals and the
// vulgar fractions: those are numbers, and are not decimal digits. Reading the lookahead as
// \p{N} grouped them with their neighbours, and, because the match is anchored on a word
// boundary, a run whose right neighbour is a word character is not matched at all and has to
// stay whole.
//
// These expectations are the expression's own output on the same strings.
func TestTinyAyaDigitClassIsNarrowerThanNumber(t *testing.T) {
	tok := &Tokenizer{pre: preTinyAya}
	cases := []struct {
		in   string
		want []string
	}{
		// No word boundary after the Roman numeral, so no match: the pair is one chunk.
		{"0\u2163", []string{"0\u2163"}},
		{"\u0661\u2163", []string{"\u0661\u2163"}},
		{"\u2163\u0661\u2163", []string{"\u2163\u0661\u2163"}},
		// A boundary exists after the decimal run, so it groups.
		{"1234", []string{"1", "234"}},
		{"\u0661\u0662\u0663\u0664", []string{"\u0661", "\u0662\u0663\u0664"}},
		{"a\u2163b", []string{"a", "\u2163", "b"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestTinyAyaRunsTwoDigitPasses covers the family's two ordered expressions.
//
// The first is "\d{1,3}(?=(?:\d{3})*\b)": a *decimal* digit run cut into groups of three from
// the right, but only where a word boundary follows. The second is the case-splitting letter
// rule, whose own digit alternative is "\p{N}{1,3}".
//
// The order matters because the two name different classes. A superscript is "\p{N}" but not
// "\d", and not a word character either, so the first pass sees a boundary after the digit
// before it and cuts there -- "1\u00b2" is ["1", "\u00b2"] even though the second pass would have
// grouped them. A Roman numeral is both "\p{N}" and a word character, so the first pass sees no
// boundary and leaves the pair alone: "0\u2163" is one chunk.
//
// A single combined digit pass cannot produce both, which is why the family is a function.
func TestTinyAyaRunsTwoDigitPasses(t *testing.T) {
	tok := &Tokenizer{pre: preTinyAya}
	cases := []struct {
		in   string
		want []string
	}{
		// Boundary after the digit: cut, and the number the first pass ignored stays with
		// the second.
		{"1\u00b2", []string{"1", "\u00b2"}},
		{"0\u00b2", []string{"0", "\u00b2"}},
		// No boundary: the words stay together however the classes differ.
		{"0\u2163", []string{"0\u2163"}},
		{"\u2163\u0661\u2163", []string{"\u2163\u0661\u2163"}},
		// A join is a word character, so the first pass finds no boundary and does not cut.
		// The second pass then leaves the join on its own: "\\p{M}" does not admit it, and
		// it is neither a number nor punctuation, so it is gap text.
		{"0\u200d", []string{"0", "\u200d"}},
		// Grouping from the right, three at a time.
		{"1234", []string{"1", "234"}},
		{"1234567", []string{"1", "234", "567"}},
		{"123", []string{"123"}},
		{"a1234b", []string{"a", "123", "4", "b"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestFalconGroupsOnlyASCIIRuns covers the third expression, "[0-9][0-9][0-9]", and the gap
// rule around it.
//
// The class is ASCII: no other script's digits match, and neither does a Roman numeral or a
// superscript. Reading it as \p{N} grouped runs the expression leaves alone -- "99\u00b2" is
// one chunk because only "99" is in the class and two digits do not fill a group.
//
// Text the expression does not match is gap, and llama.cpp emits the text after the last match
// as one fragment attached to what precedes it. So a trailing superscript rides with the digit
// before it: "1234567\u00b2" is ["123", "456", "7\u00b2"], not ["123", "456", "7", "\u00b2"].
func TestFalconGroupsOnlyASCIIRuns(t *testing.T) {
	tok := &Tokenizer{pre: preFalcon}
	cases := []struct {
		in   string
		want []string
	}{
		{"99\u00b2", []string{"99\u00b2"}},
		{"\u2163\u2163\u2163", []string{"\u2163\u2163\u2163"}},
		{"1234", []string{"123", "4"}},
		{"12345678", []string{"123", "456", "78"}},
		{"1234567\u00b2", []string{"123", "456", "7\u00b2"}},
	}
	for _, c := range cases {
		got := tok.preSplit(c.in)
		if !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
		for _, piece := range got {
			if !utf8.ValidString(piece) {
				t.Errorf("preSplit(%q) produced invalid UTF-8 %q", c.in, piece)
			}
		}
	}
}

// TestTrailingSpaceAppliesToEveryFragment covers deepseek-llm's fourth expression, "\s+$".
//
// Each expression in a pre-tokenizer is applied to the fragments the previous one produced, so
// "$" means the end of that *fragment*, not of the original string. Running the rule once on
// the last piece instead made the split depend on where in the string the piece sat: a piece
// ending in a no-break space was split when it came last and left whole when it did not.
//
// The expected values are the expression's own output on these strings.
func TestTrailingSpaceAppliesToEveryFragment(t *testing.T) {
	tok := &Tokenizer{pre: preDeepseekLLM}
	cases := []struct {
		in   string
		want []string
	}{
		// The combining mark is gap, so it ends its own fragment and the no-break space that
		// follows becomes a fragment of its own ending in whitespace -- which the expression
		// then splits, even though more text follows in the original string.
		{";]&.%\u0301\u00a0\u2003!", []string{";]&.%", "\u0301", "\u00a0", "\u2003", "!"}},
		{"\u0301\u00a0\u2003!", []string{"\u0301", "\u00a0", "\u2003", "!"}},
		{"\u0301\u00a0", []string{"\u0301", "\u00a0"}},
		// A space in the middle of a letter run still belongs to the run after it.
		{"a\u00a0b", []string{"a", "\u00a0b"}},
	}
	for _, c := range cases {
		if got := tok.preSplit(c.in); !sameStrings(got, c.want) {
			t.Errorf("preSplit(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestPreReferenceCases checks the rules against llama.cpp's expressions on the inputs that
// have been wrong here before.
//
// The table in pre_reference_cases.go holds one representative per defect class rather than a
// random sample, because that is what makes it cheap to keep: 126 strings that between them
// cover every mistake the port has made -- a digit class that was too wide, a boundary test
// that looked at the wrong side, a run split by count instead of offset, a rule applied to the
// sequence instead of to each fragment.
//
// Every expectation comes from applying llama.cpp's own expressions as sequential passes. A
// failure means a rule has drifted from the reference, which is the only thing this package
// promises about a family.
func TestPreReferenceCases(t *testing.T) {
	if len(preReferenceCases) == 0 {
		t.Fatal("the reference table is empty")
	}
	for _, fc := range preReferenceCases {
		tok := &Tokenizer{pre: fc.pre}
		if len(fc.cases) == 0 {
			t.Errorf("%s: no cases", fc.name)
			continue
		}
		for text, want := range fc.cases {
			if got := tok.preSplit(text); !sameStrings(got, want) {
				t.Errorf("%s: preSplit(%q) = %q, want %q", fc.name, text, got, want)
			}
		}
	}
}

// TestStartsLetterRunMarkBranch covers the arm of startsLetterRun that lets a combining mark
// begin a letter run.
//
// A family whose class is "[\p{L}\p{M}]+" accepts a mark at the start, so "\u0301abc" is one
// chunk. The families that spell the run as "(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+" need a letter
// first and set wordSpaces, which returns before the mark test. Both readings are in preRule,
// and the mark arm had no coverage: every test string began with a letter.
//
// Called on the predicate directly rather than through preSplit. preSplit reaches it only via
// its leadAnyNonLetter branch, and a fixture change there would make this test pass for a
// reason that has nothing to do with the arm under test.
func TestStartsLetterRunMarkBranch(t *testing.T) {
	const mark = "\u0301" // COMBINING ACUTE ACCENT
	// The ordinary arm: a letter begins the run under any family.
	if !startsLetterRun("abc", preRule{}) {
		t.Error(`startsLetterRun("abc") = false, want true`)
	}
	withMarks := preRule{letterMarks: true}
	if !startsLetterRun(mark+"abc", withMarks) {
		t.Errorf("startsLetterRun(%q, letterMarks) = false, want true: a mark may begin the run",
			mark+"abc")
	}
	// The same text under a family that requires a letter first: wordSpaces returns false
	// before the mark test runs, which is the arm this one is distinguished from.
	letterFirst := preRule{letterMarks: true, wordSpaces: true}
	if startsLetterRun(mark+"abc", letterFirst) {
		t.Errorf("startsLetterRun(%q, wordSpaces) = true, want false", mark+"abc")
	}
	// And without letterMarks the mark is not part of any run.
	if startsLetterRun(mark+"abc", preRule{}) {
		t.Errorf("startsLetterRun(%q, plain) = true, want false", mark+"abc")
	}

	// The joiner arm: k2-horizon writes "(?:\p{L}|\p{M}|\u200C|\u200D)+", so a run may
	// begin with a zero-width joiner. Neither character is a letter or a mark, so without the
	// flag it falls to the punctuation alternative and the word is split at it.
	withJoiners := preRule{joiners: true}
	for _, joiner := range []string{"\u200c", "\u200d"} {
		if !startsLetterRun(joiner+"abc", withJoiners) {
			t.Errorf("startsLetterRun(%q, joiners) = false, want true", joiner+"abc")
		}
	}
	if startsLetterRun("\u200cabc", preRule{}) {
		t.Error("a joiner began a run without the joiners flag")
	}
}
