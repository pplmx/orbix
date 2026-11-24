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
	tok := &Tokenizer{pre: preGPT2}
	inputs := []string{
		"hello world", "a.b", "it's 42", "  x  ", "", " ", "\n\n", "a\tb",
		"Hello, World!", "12345", "3.14", "don't", "I'll", "we've",
		"中文测试", "emoji😀here", "mixed中英文text", "a1b2c3",
		"  leading", "trailing  ", "multi   spaces", "\t\ttabs",
		"a\nb", "line1\nline2\n", "!!!", "...", "@#$%",
		"'s", "'re", "x'y", "O'Brien", "café", "naïve",
	}
	got := make(map[string][]string, len(inputs))
	for _, in := range inputs {
		got[in] = tok.preSplit(in)
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
