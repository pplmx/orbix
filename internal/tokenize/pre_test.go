package tokenize

import (
	"testing"
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
