package tokenize

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
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
		"jais2": preJais2, "bailingmoe": preBailingMoE,
		"superbpe": preSuperBPE, "laguna": preLaguna,
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
		// jais2's whitespace tiers are tried largest first, so a run is cut into power-of-two
		// pieces and the last space leads what follows. These lengths straddle every tier
		// boundary the expression names, including the two that are adjacent in value.
		"   x", "    x", "        x", "    y", "     y",
		strings.Repeat(" ", 16) + "z", strings.Repeat(" ", 17) + "z",
		strings.Repeat(" ", 100) + "z", strings.Repeat(" ", 513) + "z",
		// BailingMoE is llama3's shape with a bare \p{N}, so a plain number separates them and a
		// mixed-case contraction checks that its spelled-out alternatives fold.
		"DON'T", "I'LL", "we'VE",
		// superBPE groups a digit run from the right, so every length around a multiple of three
		// is worth having, and a digit run inside punctuation shows that only \p{N}+ is grouped.
		"1", "12", "123", "1234", "12345", "123456", "1234567", "a123456", "3.14159", "000123",
		// laguna runs a newline pass first, so whitespace before a newline is its own chunk rather
		// than joining the newline: " x \n y " splits as [" x", " ", "\n", " y", " "].
		" x \n y ", "x \n", "  \n  x", "a\n\nb",
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
