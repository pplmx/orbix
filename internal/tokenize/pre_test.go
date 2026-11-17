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
