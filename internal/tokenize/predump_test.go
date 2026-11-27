package tokenize

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"
)

// TestDumpPreSplit writes the pre-split results for scripts/pre_xcheck.py to compare.
//
// The reference table in pre_reference_cases.go holds expectations transcribed from llama.cpp's
// expressions, one representative per defect class. This is the other instrument: the script
// applies GPT-2's published pattern to the same inputs and compares, so the two sides are written
// from different sources rather than one restating the other.
//
// Skipped unless ORBIX_PREDUMP names an output file, so it costs a normal run nothing.
func TestDumpPreSplit(t *testing.T) {
	out := os.Getenv("ORBIX_PREDUMP")
	if out == "" {
		t.Skip("set ORBIX_PREDUMP to write a comparison corpus")
	}
	// One entry per family the script has a published pattern for. Adding a family here means
	// adding its pattern there; the script fails on a name it does not know rather than skipping
	// it, so the two lists cannot drift apart silently.
	families := map[string]preType{
		"gpt2": preGPT2, "llama3": preLlama3, "qwen2": preQwen2, "tekken": preTekken,
	}
	inputs := []string{
		"hello world", "a.b", "it's 42", "  x  ", "", " ", "\n\n", "a\tb",
		"Hello, World!", "12345", "3.14", "don't", "I'll", "we've",
		"中文测试", "emoji😀here", "mixed中英文text", "a1b2c3",
		"  leading", "trailing  ", "multi   spaces", "\t\ttabs",
		"a\nb", "line1\nline2\n", "!!!", "...", "@#$%",
		"'s", "'re", "x'y", "O'Brien", "café", "naïve",
		// Boundary shapes the rules are written to disagree about: a letter run taking a
		// non-letter prefix, digits meeting letters, combining marks, and CR next to LF.
		"abc123def", "123abc", "(a)", "-x", "'94", "a'b'c",
		"x\u0301", "e\u0308", "   \n   ", "a\r\nb", "\r\n", "\r",
		"x\r\ny", "!?", "?!", "...", "end.", ".start", "a.b.c", "12 34 56",
		// Where the four families disagree with each other. Qwen2 takes one digit per chunk where
		// GPT-2 takes the run and Llama 3 takes up to three, so a plain number separates them; the
		// contraction list is spelled out rather than case-folded there, so mixed case does too.
		"1234567", "abc123456def", "1", "0",
		"It'S", "it'S", "DON'T",
		// Tekken cuts a letter run at a lower-to-upper boundary, so a CamelCase word is several
		// chunks where the other three families treat it as one; its punctuation tail also takes
		// slashes, and an uppercase run followed by lowercase splits at the join.
		"camelCaseWord", "HTTPServer", "x/y", "path/to/file", "a/b\n",
		"A", "aA", "Aa", "ABCdef",
	}
	got := make(map[string]map[string][]string, len(families))
	for name, pre := range families {
		tok := &Tokenizer{pre: pre}
		per := make(map[string][]string, len(inputs))
		for _, in := range inputs {
			per[in] = tok.preSplit(in)
		}
		got[name] = per
	}
	b, err := json.MarshalIndent(got, "", " ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(out, b, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	fmt.Printf("wrote %d cases to %s\n", len(inputs), out)
}
