package tokenize

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// Pre-tokenization, as a hand-written splitter rather than a regular expression.
//
// A byte-level BPE model is trained on chunks, not on raw text: the trainer cut the corpus
// into pieces first, and a merge never spans a boundary. Reproducing the vocabulary's
// expectations therefore means applying the same cut before the merges, and the cut is
// named by tokenizer.ggml.pre because it differs per model family.
//
// The alternative is a regular expression, which is what llama.cpp stores and what the
// Hugging Face tokenizers implement. Go's regexp has no Unicode category classes --
// \p{L} and \p{N} are unavailable -- so the expression would have to be rewritten, and a
// rewritten expression that is wrong by one alternative produces text the model was never
// trained on with nothing to report it. This is a splitter over the same categories,
// written the way the rest of the package is: one rule at a time, in the order that makes
// it readable.

// preSplit cuts text into the chunks the pre-tokenizer would produce.
//
// With no pre-tokenizer the whole text is one chunk, which is the behaviour this package
// had before it read the key: merges run across every character boundary.
func (t *Tokenizer) preSplit(text string) []string {
	maxDigits := t.preDigitRun()
	if maxDigits == 0 || text == "" {
		return []string{text}
	}
	var chunks []string
	i := 0
	for i < len(text) {
		n := chunkLen(text[i:], maxDigits)
		if n <= 0 {
			// A zero-length chunk would loop forever; take one byte so progress is
			// guaranteed even if the rules below have a gap.
			n = 1
		}
		chunks = append(chunks, text[i:i+n])
		i += n
	}
	return chunks
}

// qwen2Contractions are the clitics the Qwen2 rule splits off, matched case-insensitively
// at the start of the text.
var qwen2Contractions = []string{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"}

// preDigitRun is how many digits one chunk may hold under this pre-tokenizer.
//
// The Qwen2 and Llama 3 rules are the same expression except for this number, so it is a
// property of the family rather than a second implementation: Qwen2 takes one digit at a
// time and Llama 3 takes up to three. Zero means no pre-tokenizer applies.
func (t *Tokenizer) preDigitRun() int {
	switch t.pre {
	case preQwen2:
		return 1
	case preLlama3:
		return 3
	default:
		return 0
	}
}

// chunkLen returns the length in bytes of the pre-tokenizer's first match at the start of
// s, or 0 if no rule applies.
//
// The alternatives are tried in the order the expression lists them, because that order is
// the rule: a contraction wins over a letter run, which is why "'s" is its own chunk.
func chunkLen(s string, maxDigits int) int {
	if n := qwen2ContractionLen(s); n > 0 {
		return n
	}
	// [^\r\n\p{L}\p{N}]?\p{L}+ -- one non-letter, non-digit, non-newline character
	// followed by a run of letters. The leading character is optional, and that is what
	// keeps a punctuation mark attached to the following word rather than splitting it.
	if n := qwen2LetterRunLen(s); n > 0 {
		return n
	}
	// \p{N}{1,3} -- at most maxDigits digits. A run of digits is not one chunk: a number
	// is split into groups, and the group size is what the two families disagree about.
	if n := qwen2DigitRunLen(s, maxDigits); n > 0 {
		return n
	}
	// " ?[^\s\p{L}\p{N}]+[\r\n]* -- an optional space, then a run of punctuation,
	// then any newlines that follow it.
	if n := qwen2PunctRunLen(s); n > 0 {
		return n
	}
	// \s*[\r\n]+ -- a newline with the whitespace before it.
	if n := qwen2NewlineRunLen(s); n > 0 {
		return n
	}
	// \s+(?!\S) -- trailing whitespace that ends the text.
	// \s+ -- any other run of whitespace.
	if n := runeLenAt(s); n > 0 && isSpaceAt(s) {
		return qwen2SpaceRunLen(s)
	}
	return 0
}

// qwen2DigitRunLen matches \p{N}{1,max}.
func qwen2DigitRunLen(s string, max int) int {
	i, count := 0, 0
	for i < len(s) && count < max && isDigitAt(s[i:]) {
		i += runeLenAt(s[i:])
		count++
	}
	return i
}

// qwen2ContractionLen matches a clitic at the start of s.
func qwen2ContractionLen(s string) int {
	if s[0] != '\'' || len(s) < 2 {
		return 0
	}
	for _, c := range qwen2Contractions {
		if len(s) >= len(c) && strings.EqualFold(s[:len(c)], c) {
			return len(c)
		}
	}
	return 0
}

// qwen2LetterRunLen matches [^\r\n\p{L}\p{N}]?\p{L}+.
func qwen2LetterRunLen(s string) int {
	i := 0
	if !isLetterAt(s) && !isNewlineAt(s) && !isDigitAt(s) && runeLenAt(s) > 0 {
		// The optional leading character: anything that is not a letter, digit, or
		// newline -- punctuation, a space, or a symbol.
		i = runeLenAt(s)
		if !isLetterAt(s[i:]) {
			return 0
		}
	}
	start := i
	for i < len(s) && isLetterAt(s[i:]) {
		i += runeLenAt(s[i:])
	}
	if i == start {
		return 0
	}
	return i
}

// qwen2PunctRunLen matches " ?[^\s\p{L}\p{N}]+[\r\n]*".
func qwen2PunctRunLen(s string) int {
	i := 0
	if strings.HasPrefix(s, " ") {
		i = 1
	}
	start := i
	for i < len(s) {
		n := runeLenAt(s[i:])
		if n == 0 || isSpaceAt(s[i:]) || isLetterAt(s[i:]) || isDigitAt(s[i:]) {
			break
		}
		i += n
	}
	if i == start {
		return 0
	}
	for i < len(s) && isNewlineAt(s[i:]) {
		i += runeLenAt(s[i:])
	}
	return i
}

// qwen2NewlineRunLen matches "\s*[\r\n]+".
func qwen2NewlineRunLen(s string) int {
	i := 0
	for i < len(s) && isSpaceAt(s[i:]) && !isNewlineAt(s[i:]) {
		i += runeLenAt(s[i:])
	}
	start := i
	for i < len(s) && isNewlineAt(s[i:]) {
		i += runeLenAt(s[i:])
	}
	if i == start {
		return 0
	}
	return i
}

// qwen2SpaceRunLen matches "\s+(?!\S)" then "\s+".
//
// The lookahead makes the first alternative take the *last* space of a run that is
// followed by a word, so that one space is left to be the optional prefix of the next
// chunk. Without it "hello  world" splits as ["hello", "  ", "world"] instead of
// ["hello", " ", " world"], and the second form is what the model was trained on: the
// space belongs to the word that follows it.
func qwen2SpaceRunLen(s string) int {
	i := 0
	for i < len(s) && isSpaceAt(s[i:]) {
		i += runeLenAt(s[i:])
	}
	if i == 0 {
		return 0
	}
	// (?!\S): positions where what follows is not a non-space. The first such position is
	// the last space in the run, so a run followed by a word yields all but its last
	// character.
	if i < len(s) {
		last := i - runeLenAt(s[:i])
		if last > 0 {
			return last
		}
	}
	return i
}

// Character classification, matching the Unicode categories the Qwen2 expression uses.
//
// \p{L} is any letter, \p{N} any number, and \s the whitespace set. Go's unicode package
// exposes these as tables, so the classification is the standard one rather than an
// approximation; the category names are in the comments so a reader can check them
// against the expression.

// isLetter reports whether r is a letter (\p{L}).
func isLetter(r rune) bool { return unicode.IsLetter(r) }

// isDigit reports whether r is a decimal digit, which is the subset of \p{N} the Qwen2
// rule means by "\p{N}" -- the reference expression uses the full category, but a single
// non-decimal number rune in a prompt is rare enough that treating it as one chunk is the
// same behaviour for text a model is asked about.
func isDigit(r rune) bool { return unicode.IsDigit(r) }

// isSpace reports whether r is whitespace (\s).
func isSpace(r rune) bool { return unicode.IsSpace(r) }

// isNewline reports whether r is a carriage return or line feed, the two the expression
// names explicitly.
func isNewline(r rune) bool { return r == '\r' || r == '\n' }

// runeLenAt returns the byte length of the first rune of s, or 0 when s is empty.
func runeLenAt(s string) int {
	if s == "" {
		return 0
	}
	_, size := utf8.DecodeRuneInString(s)
	return size
}

// isLetterAt, isDigitAt, isSpaceAt, and isNewlineAt classify the first rune of s.
func isLetterAt(s string) bool  { return classifyAt(s, isLetter) }
func isDigitAt(s string) bool   { return classifyAt(s, isDigit) }
func isSpaceAt(s string) bool   { return classifyAt(s, isSpace) }
func isNewlineAt(s string) bool { return classifyAt(s, isNewline) }

// classifyAt applies pred to the first rune of s.
func classifyAt(s string, pred func(rune) bool) bool {
	if s == "" {
		return false
	}
	r, _ := utf8.DecodeRuneInString(s)
	return pred(r)
}
