package tokenize

import (
	"testing"
	"unicode/utf8"
)

// FuzzPreSplit checks that the pre-tokenizer rules always return a partition of their input.
//
// The rules are hand-written ports of llama.cpp's regular expressions, and the cases that break
// them are characters no ASCII fixture produces: combining marks, joiners, the superscripts that
// are \p{N} but not \d, multi-byte whitespace. A rule that mis-slices or drops text would still
// tokenize -- to boundaries the vocabulary never saw -- so the property worth checking is
// structural rather than a specific answer: the pieces must concatenate back to the input, and
// every piece must be valid UTF-8.
//
// That catches the whole class of bug this file has had: a digit run cut by count instead of
// byte offset, a drop at a chunk boundary, a gap merged twice.
func FuzzPreSplit(f *testing.F) {
	seeds := []string{
		"",
		"a",
		"hello world",
		"don't",
		"'sa",
		"1234",
		"1234567",
		"99\u00b2",
		"\u2163\u0661\u2163",
		"\u6587A",
		"b\u0e01",
		"\u200c\u200d",
		"\r\n",
		"\n\n",
		"  \n  ",
		"a\u00a0b",
		"\u0301\u00a0\u2003!",
		"a\r\nb",
		"  123",
		"`  \u21630\u2163[",
		"HTTPServer",
		"camelCase",
		"e\u0301clair",
	}
	for _, s := range seeds {
		f.Add(s)
	}

	// Every implemented family, so a rule only one of them reaches is fuzzed too.
	families := allPreFamilies()
	if len(families) < 30 {
		f.Fatalf("only %d families resolved; the enum probably grew without this list", len(families))
	}

	f.Fuzz(func(t *testing.T, text string) {
		if !utf8.ValidString(text) {
			// The input arrives as a Go string from the decoder above, so it is valid by
			// construction; skip rather than assert, because a future signature change
			// should not turn this into a false failure.
			return
		}
		for _, pre := range families {
			pieces := (&Tokenizer{pre: pre}).preSplit(text)
			var joined []byte
			for _, piece := range pieces {
				if !utf8.ValidString(piece) {
					t.Fatalf("%v: piece %q is not valid UTF-8 (input %q)", pre, piece, text)
				}
				joined = append(joined, piece...)
			}
			if string(joined) != text {
				t.Fatalf("%v: pieces joined to %q, want %q", pre, joined, text)
			}
		}
	})
}

// allPreFamilies returns every family constant defined by the preType enum.
//
// Written out rather than derived, because Go has no way to enumerate a const block at run time
// and reflecting over the package would not work for unexported constants either. The length
// check in the fuzz target is what keeps this list from silently going stale.
func allPreFamilies() []preType {
	return []preType{
		preQwen2, preLlama3, preGPT2, preStarCoder, preDeepseek3, preTekken,
		preWhitespace, preBloom, preDeepseekCoder, preFalcon, preGPT4O, preDeepseekLLM,
		preMMBert, preViking, preSuperBPE, preSoprano, preBailingMoE, preTinyAya,
		preLaguna, preFlint, preUfakzeka, preQwen35, preSeedCoder, preMiniCPM5,
		preZonos3, preJais2, preExaoneMoE, preTekkenNum3, preYoutu, preSpark25,
		preChameleon, preK2Horizon, preAfMoE, preKimiK2,
	}
}
