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
	fam, ok := preFamilyOf(t.pre)
	if !ok || text == "" {
		return []string{text}
	}
	// One family is not an expression over alternatives but a single class: \S+ keeps each
	// run of non-space characters and drops the spaces between them. Expressing that
	// through preRule would need a rule type that matches the complement of a class, which
	// the other seven do not want.
	if fam.whitespaceOnly {
		return splitWhitespaceRuns(text)
	}
	// The BLOOM expression is the same shape: an optional leading space, then a run of
	// characters outside a fixed stop set. The set contains no letter or digit, so it is
	// the punctuation-and-space complement rather than a category.
	if fam.stopSet != "" {
		return splitAtStopSet(text, fam.stopSet)
	}
	// The family's earlier expressions cut first, in the order llama.cpp lists them, and the
	// main rule then runs on every piece that leaves.
	pieces := []string{text}
	if fam.preDigits > 0 {
		pieces = splitDigitGroups(pieces, fam.preDigits)
	}
	if fam.cjk {
		pieces = splitCJK(pieces)
	}
	if len(pieces) == 1 && pieces[0] == text {
		return splitByRule(text, fam)
	}
	var out []string
	for _, piece := range pieces {
		out = append(out, splitByRule(piece, fam)...)
	}
	return out
}

// splitWhitespaceRuns cuts text into runs of non-space characters, the \S+ rule.
//
// The spaces between runs are not emitted: llama.cpp's whitespace pre-tokenizer
// (jina-embeddings-v2-base-zh) produces only the runs, and the vocabulary has no separate
// space piece to give them.
func splitWhitespaceRuns(text string) []string {
	var out []string
	start := -1
	for i, r := range text {
		if unicode.IsSpace(r) {
			if start >= 0 {
				out = append(out, text[start:i])
				start = -1
			}
			continue
		}
		if start < 0 {
			start = i
		}
	}
	if start >= 0 {
		out = append(out, text[start:])
	}
	return out
}

// splitAtStopSet cuts text into chunks of an optional leading space plus a run of
// characters outside stop.
//
// The optional space is part of the following chunk, not a chunk of its own, which is what
// makes " hello" one piece where "hello" is also one piece.
func splitAtStopSet(text, stop string) []string {
	var out []string
	i := 0
	for i < len(text) {
		start := i
		// One optional leading space belongs to the chunk, but only when a character that
		// can follow it does: a lone space before a stop character is not a match, and
		// emitting it alone would produce a chunk the expression never yields.
		if text[i] == ' ' {
			if i+1 >= len(text) || isStopRune(text[i+1:], stop) {
				_, size := utf8.DecodeRuneInString(text[i:])
				i += size
				continue
			}
			i++
		}
		for i < len(text) {
			r, n := utf8.DecodeRuneInString(text[i:])
			if r == utf8.RuneError || strings.ContainsRune(stop, r) {
				break
			}
			i += n
		}
		if i == start {
			// The character at start is in the stop set, so no match begins here. A
			// findall-style splitter drops it rather than emitting it alone: the stop
			// characters are separators, and the vocabulary has its own pieces for them
			// that the caller assembles. Advancing keeps progress guaranteed.
			_, n := utf8.DecodeRuneInString(text[i:])
			i += n
			continue
		}
		out = append(out, text[start:i])
	}
	return out
}

// isStopRune reports whether text begins with a rune in stop.
func isStopRune(text, stop string) bool {
	r, _ := utf8.DecodeRuneInString(text)
	return r == utf8.RuneError || strings.ContainsRune(stop, r)
}

// splitDigitGroups cuts each piece at digit groups of at most max digits.
func splitDigitGroups(pieces []string, max int) []string {
	var out []string
	for _, piece := range pieces {
		out = append(out, splitDigits(piece, max)...)
	}
	return out
}

// splitCJK cuts each piece so runs of CJK characters stand alone.
func splitCJK(pieces []string) []string {
	var out []string
	for _, piece := range pieces {
		out = append(out, splitCJKRuns(piece)...)
	}
	return out
}

// splitDigits cuts text at digit groups of at most max digits, keeping the runs between
// them intact. A max of 0 is not allowed; callers check first.
func splitDigits(text string, max int) []string {
	var pieces []string
	start := 0
	i := 0
	for i < len(text) {
		n := runeLenAt(text[i:])
		if n == 0 {
			n = 1
		}
		if !isDigitAt(text[i:]) {
			i += n
			continue
		}
		// Found a digit: emit what came before, then the group.
		if i > start {
			pieces = append(pieces, text[start:i])
		}
		groupStart := i
		count := 0
		for i < len(text) && isDigitAt(text[i:]) && (max <= 0 || count < max) {
			i += runeLenAt(text[i:])
			count++
		}
		pieces = append(pieces, text[groupStart:i])
		start = i
	}
	if start < len(text) {
		pieces = append(pieces, text[start:])
	}
	return pieces
}

// splitCJKRuns cuts text so each run of CJK characters is its own piece.
func splitCJKRuns(text string) []string {
	var pieces []string
	start := 0
	i := 0
	for i < len(text) {
		n := runeLenAt(text[i:])
		if n == 0 {
			n = 1
		}
		if !isCJKAt(text[i:]) {
			i += n
			continue
		}
		if i > start {
			pieces = append(pieces, text[start:i])
		}
		runStart := i
		for i < len(text) && isCJKAt(text[i:]) {
			i += runeLenAt(text[i:])
		}
		pieces = append(pieces, text[runStart:i])
		start = i
	}
	if start < len(text) {
		pieces = append(pieces, text[start:])
	}
	return pieces
}

// splitByRule cuts text once with the family's main expression.
func splitByRule(text string, fam preRule) []string {
	var chunks []string
	i := 0
	for i < len(text) {
		n := chunkLen(text[i:], fam)
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

// preRule is the shape of one pre-tokenizer expression, as the handful of choices its
// alternatives actually make.
//
// The families differ in four places and share everything else, so the differences are
// data rather than a function each. Reading them side by side is what makes it possible to
// check one against llama.cpp's expression without re-deriving the other seven rules.
type preRule struct {
	// digits is the most digits one chunk may hold. 0 means a run of any length.
	digits int
	// lead is what a letter or digit run may take as a prefix.
	//
	// The two expressions spell it differently and the difference matters: GPT-2 writes
	// " ?\p{L}+", an optional *space*, so a word after a space is one chunk; qwen2 writes
	// "[^\r\n\p{L}\p{N}]?\p{L}+", an optional *any-non-letter*, so a punctuation mark
	// joins the following word instead.
	lead leadKind
	// letterMarks is whether combining marks join a letter run (\p{L}\p{M} rather than
	// \p{L}). The deepseek and stablelm families set it.
	letterMarks bool
	// newlinesAfterPunct is whether a punctuation run may absorb the newlines that follow
	// it. qwen2 does; gpt-2 ends the chunk at the punctuation.
	newlinesAfterPunct bool
	// punctSlashTail is whether the punctuation run also absorbs following slashes, which is
	// tekken's "[\r\n/]*" tail. It only applies together with newlinesAfterPunct.
	punctSlashTail bool
	// newlineChunk is whether the expression has a "\s*[\r\n]+" alternative: a newline
	// with the spaces before it as one chunk. The GPT-2 expression has no such rule, so a
	// newline there falls through to the whitespace alternative and groups differently.
	newlineChunk bool
	// preDigits is the length of each digit group the family cuts out *before* the main
	// expression, or 0 for none.
	//
	// llama.cpp applies a family's expressions in sequence, each cutting the previous round's
	// chunks further. A leading "\p{N}" therefore isolates digits before the main rule runs,
	// so the main rule's digit alternative has nothing left to join: starcoder's bare
	// "\p{N}" gives one chunk per digit while deepseek's "\p{N}{1,3}" gives groups of up to
	// three, and neither is the same as setting the main rule's digit count.
	preDigits int
	// punctIsCategory is whether the punctuation alternative names the categories
	// (\p{P}\p{S}) rather than taking everything that is not a letter, digit, or space.
	//
	// The two differ on characters that are neither: an emoji or a currency sign is a symbol,
	// so it belongs to the run under either spelling; a combining mark is neither, so the
	// complement form swallows it and the category form does not.
	punctIsCategory bool
	// caseSplit is whether a letter run is cut where a lower-case letter is followed by an
	// upper-case one, which is the tekken (GPT-4o) rule.
	//
	// The expression spells it as two lookahead-guarded alternatives, but the effect is a
	// boundary rule: a run of capitals may take one trailing lower-case stretch
	// ("HTTPServer" stays whole), and a lower-case run ends where a capital begins
	// ("camelCase" splits). Writing it as a boundary is what makes it one check inside the
	// letter loop instead of a second copy of the letter rule.
	caseSplit bool
	// whitespaceOnly marks a family whose whole rule is \S+: runs of non-space
	// characters, with the separators dropped. It is handled before the rule engine
	// because the other fields describe alternatives within a run, and this one is about
	// where runs end.
	whitespaceOnly bool
	// stopSet, when non-empty, is the set of characters a chunk may not contain: the whole
	// rule is an optional leading space followed by a run from the complement. It is a set
	// literal rather than a category because BLOOM's expression lists the punctuation it
	// stops at, and a category would stop at more.
	stopSet string
	// punctLetterRun is whether the expression's first alternative is
	// "[!-/:-@\\[-`{-~][A-Za-z]+": one ASCII punctuation character followed by a run of
	// ASCII letters, as a single chunk.
	//
	// It takes precedence over the letter alternative, so ".b" and "(x" are one chunk each.
	// The leading-character alternative does not cover this case -- it excludes punctuation --
	// which is why the rule has to exist separately rather than falling out of it.
	punctLetterRun bool
	// cjk is whether a run of CJK characters is its own chunk, from the family's
	// "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+" expression. Without it those
	// characters go through the letter rule, which groups them the same way -- but the family
	// applies the CJK expression first, so the grouping is independent of what the letter
	// rule would have done with neighbouring text.
	cjk bool
}

// leadKind is what a run may take as its prefix.
type leadKind int

// leadKind's zero value takes no prefix, which is a real choice a family can make; the
// families this package implements both take one.
const (
	// leadSpace takes one space, which is GPT-2's " ?".
	leadSpace leadKind = iota + 1
	// leadAnyNonLetter takes one character that is not a letter, digit, or newline, which
	// is qwen2's "[^\\r\\n\\p{L}\\p{N}]?".
	leadAnyNonLetter
	// leadNotLetterPunctSymbol takes one character that is not a letter, punctuation, or
	// symbol and not a newline, which is deepseek's "[^\\r\\n\\p{L}\\p{P}\\p{S}]?".
	//
	// It is narrower than leadAnyNonLetter: punctuation is excluded, so a mark like "." is
	// its own chunk instead of a prefix on the word after it. That difference is why
	// "a.b" splits as ["a", ".b"] under qwen2 and ["a", ".", "b"] under deepseek.
	leadNotLetterPunctSymbol
)

// preFamilyOf returns the rule for a pre-tokenizer, or false when this package does not
// implement it.
func preFamilyOf(p preType) (preRule, bool) {
	switch p {
	case preQwen2:
		return preRule{
			digits: 1, lead: leadAnyNonLetter,
			newlinesAfterPunct: true, newlineChunk: true,
		}, true
	case preLlama3:
		return preRule{
			digits: 3, lead: leadAnyNonLetter,
			newlinesAfterPunct: true, newlineChunk: true,
		}, true
	case preGPT2:
		// "'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"
		// -- an optional space rather than an optional any-non-letter, digits as a run,
		// and no newline alternative.
		return preRule{digits: 0, lead: leadSpace}, true
	case preStarCoder:
		// The GPT-2 expression with a bare `\p{N}` applied first, so every digit is its own
		// chunk before the main rule runs. starcoder, refact, command-r, smollm, and the
		// others llama.cpp groups with them all mean exactly this.
		return preRule{digits: 0, lead: leadSpace, preDigits: 1}, true
	case preTekken:
		// "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+"
		// "|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-Z]))*"
		// "|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
		//
		// The letter alternatives cut at a lower-to-upper boundary; the punctuation tail takes
		// slashes as well as newlines.
		return preRule{
			digits: 1, lead: leadAnyNonLetter, letterMarks: true, caseSplit: true,
			newlinesAfterPunct: true, punctSlashTail: true, newlineChunk: true,
		}, true
	case preWhitespace:
		return preRule{whitespaceOnly: true}, true
	case preBloom:
		// " ?[^(\s|.,!?…。，、।۔،)]+" -- the parentheses and inner pipes are literal in
		// llama.cpp's expression, so the stop set is those characters, the pipe itself,
		// and whitespace.
		return preRule{stopSet: " \t\r\n|()" + ".,!?…。，、।۔،"}, true
	case preDeepseek3:
		// Three expressions in sequence:
		//   \p{N}{1,3}
		//   [\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+
		//   [!..~][A-Za-z]+ | [^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+
		//     | ?[\\p{P}\\p{S}]+[\\r\\n]* | \\s*[\\r\\n]+ | \\s+(?!\\S) | \\s+
		//
		// llama.cpp groups deepseek-v3, hunyuan-dense, joyai-llm, and hy_v4 on it.
		return preRule{
			digits: 0, lead: leadNotLetterPunctSymbol, letterMarks: true,
			punctIsCategory: true, newlinesAfterPunct: true, newlineChunk: true,
			preDigits: 3, cjk: true, punctLetterRun: true,
		}, true
	default:
		return preRule{}, false
	}
}

// chunkLen returns the length in bytes of the pre-tokenizer's first match at the start of
// s, or 0 if no rule applies.
//
// The alternatives are tried in the order the expression lists them, because that order is
// the rule: a contraction wins over a letter run, which is why "'s" is its own chunk.
func chunkLen(s string, fam preRule) int {
	if n := qwen2ContractionLen(s); n > 0 {
		return n
	}
	// [!-/:-@\\[-`{-~][A-Za-z]+ -- one ASCII punctuation character followed by ASCII
	// letters. It comes first, so it wins over the letter alternative's own prefix handling.
	if fam.punctLetterRun {
		if n := punctLetterRunLen(s); n > 0 {
			return n
		}
	}
	// [^\r\n\p{L}\p{N}]?\p{L}+ -- one non-letter, non-digit, non-newline character
	// followed by a run of letters. The leading character is optional, and that is what
	// keeps a punctuation mark attached to the following word rather than splitting it.
	// The GPT-2 rule has no such alternative, which is why a punctuation mark and the word
	// after it are two chunks there.
	if n := letterRunLen(s, fam); n > 0 {
		return n
	}
	// \p{N} or \p{N}{1,3} or \p{N}+ -- a digit chunk. Whether a number is one chunk or
	// several groups is per family, and it takes the same prefix a letter run does.
	if n := digitRunLen(s, fam); n > 0 {
		return n
	}
	// " ?[^\s\p{L}\p{N}]+[\r\n]* -- an optional space, then a run of punctuation,
	// then any newlines that follow it. GPT-2's variant stops at the punctuation.
	if n := punctRunLen(s, fam); n > 0 {
		return n
	}
	// \s*[\r\n]+ -- a newline with the whitespace before it, for the families whose
	// expression has that alternative.
	if fam.newlineChunk {
		if n := qwen2NewlineRunLen(s); n > 0 {
			return n
		}
	}
	// \s+(?!\S) -- trailing whitespace that ends the text.
	// \s+ -- any other run of whitespace.
	if n := runeLenAt(s); n > 0 && isSpaceAt(s) {
		return qwen2SpaceRunLen(s)
	}
	return 0
}

// digitRunLen matches the digit alternative: at most max digits when max is positive, or
// any run when max is zero.
//
// Only the space-prefix family takes a prefix here. GPT-2 writes " ?\p{N}+", but qwen2
// writes a bare "\p{N}" -- its optional non-letter prefix belongs to the letter
// alternative alone, so a punctuation mark before a digit is its own chunk.
func digitRunLen(s string, fam preRule) int {
	i := 0
	if fam.lead == leadSpace && strings.HasPrefix(s, " ") {
		i = 1
	}
	if i > 0 && !isDigitAt(s[i:]) {
		return 0
	}
	start := i
	count := 0
	for i < len(s) && isDigitAt(s[i:]) {
		if fam.digits > 0 && count >= fam.digits {
			break
		}
		i += runeLenAt(s[i:])
		count++
	}
	if i == start {
		return 0
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

// punctLetterRunLen matches "[!-/:-@\\[-`{-~][A-Za-z]+": one ASCII punctuation character
// then one or more ASCII letters.
func punctLetterRunLen(s string) int {
	n := runeLenAt(s)
	if n != 1 || s[0] >= 0x80 {
		return 0
	}
	c := s[0]
	// The class is ASCII punctuation: the printable range minus letters, digits, and space.
	isPunct := c >= '!' && c <= '/' || c >= ':' && c <= '@' ||
		c >= '[' && c <= '`' || c >= '{' && c <= '~'
	if !isPunct {
		return 0
	}
	i := 1
	for i < len(s) && s[i] < 0x80 && (s[i] >= 'A' && s[i] <= 'Z' || s[i] >= 'a' && s[i] <= 'z') {
		i++
	}
	if i == 1 {
		return 0
	}
	return i
}

// letterRunLen matches the letter-run alternative for one family.
func letterRunLen(s string, fam preRule) int {
	i := 0
	switch fam.lead {
	case leadSpace:
		if strings.HasPrefix(s, " ") {
			i = 1
		}
	case leadAnyNonLetter:
		if !isLetterAt(s) && !isNewlineAt(s) && !isDigitAt(s) && runeLenAt(s) > 0 {
			i = runeLenAt(s)
		}
	case leadNotLetterPunctSymbol:
		if !isLetterAt(s) && !isNewlineAt(s) && !isPunctOrSymbolAt(s) && runeLenAt(s) > 0 {
			i = runeLenAt(s)
		}
	}
	if i > 0 && !isLetterAt(s[i:]) {
		return 0
	}
	start := i
	// prevLower records whether the previous letter was lower case, which is what the
	// boundary test needs.
	prevLower := false
	for i < len(s) {
		if isLetterAt(s[i:]) {
			r, _ := utf8.DecodeRuneInString(s[i:])
			// The boundary is a lower-to-upper transition. A capital after a capital does
			// not split, which is why a run of capitals keeps its trailing word.
			if fam.caseSplit && prevLower && unicode.IsUpper(r) {
				break
			}
			prevLower = unicode.IsLower(r)
			i += runeLenAt(s[i:])
			continue
		}
		if fam.letterMarks && isMarkAt(s[i:]) {
			// A mark does not reset the case state: it belongs to the letter before it.
			i += runeLenAt(s[i:])
			continue
		}
		break
	}
	if i == start {
		return 0
	}
	return i
}

// punctRunLen matches " ?[^\s\p{L}\p{N}]+[\r\n]*", with the newline tail only when the
// family asks for it.
func punctRunLen(s string, fam preRule) int {
	i := 0
	if strings.HasPrefix(s, " ") {
		i = 1
	}
	start := i
	for i < len(s) {
		n := runeLenAt(s[i:])
		if n == 0 || isSpaceAt(s[i:]) {
			break
		}
		if fam.punctIsCategory {
			// ?[\p{P}\p{S}]+ -- only the two categories. A combining mark is neither, so it
			// ends the run rather than joining it, which is what lets the letter alternative
			// take it instead.
			if !isPunctOrSymbolAt(s[i:]) {
				break
			}
		} else {
			// ?[^\s\p{L}\p{N}]+ -- everything that is not a space, letter, or digit.
			if isLetterAt(s[i:]) || isDigitAt(s[i:]) {
				break
			}
		}
		i += n
	}
	if i == start {
		return 0
	}
	if fam.newlinesAfterPunct {
		for i < len(s) && isNewlineAt(s[i:]) {
			i += runeLenAt(s[i:])
		}
	}
	if fam.punctSlashTail {
		for i < len(s) && s[i] == '/' {
			i++
		}
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

// isMark reports whether r is a combining mark (\p{M}). A family with letterMarks lets one
// join a letter run, which is what keeps "café" one chunk when the accent arrives as a
// separate code point rather than as a precomposed character.
func isMark(r rune) bool { return unicode.IsMark(r) }

func isMarkAt(s string) bool { return classifyAt(s, isMark) }

// isCJK reports whether r falls in the ranges the CJK alternative names: CJK Unified
// Ideographs (U+4E00-U+9FA5), Hiragana (U+3040-U+309F), and Katakana (U+30A0-U+30FF).
//
// These are listed explicitly rather than taken from unicode.Is(unicode.Han) because the
// expression names ranges, not scripts: Han covers far more of the ideograph blocks, and
// using it would group characters this rule leaves to the letter alternative.
func isCJK(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FA5) || (r >= 0x3040 && r <= 0x309F) || (r >= 0x30A0 && r <= 0x30FF)
}

func isCJKAt(s string) bool { return classifyAt(s, isCJK) }

// isPunctOrSymbol reports whether r is punctuation (\p{P}) or a symbol (\p{S}).
func isPunctOrSymbol(r rune) bool { return unicode.IsPunct(r) || unicode.IsSymbol(r) }

func isPunctOrSymbolAt(s string) bool { return classifyAt(s, isPunctOrSymbol) }

// classifyAt applies pred to the first rune of s.
func classifyAt(s string, pred func(rune) bool) bool {
	if s == "" {
		return false
	}
	r, _ := utf8.DecodeRuneInString(s)
	return pred(r)
}

// knownPreNames is every tokenizer.ggml.pre value llama.cpp maps to a family, taken from
// its llama-vocab.cpp name switch.
//
// It exists so that "a family this package has not implemented" and "a name that is not a
// family" are distinguishable. Both tokenize as one chunk, and both are silent, but only
// the first means the model is being fed text it was never trained on. Keeping the whole
// list rather than only the implemented names is the point: adding a family removes a name
// from the unimplemented set, while a typo stays outside it entirely.
var knownPreNames = []string{
	"afmoe",
	"a.x-4.0",
	"bailingmoe",
	"bailingmoe2",
	"bloom",
	"chameleon",
	"chatglm-bpe",
	"codeshell",
	"cohere2moe",
	"command-r",
	"dbrx",
	"deepseek-coder",
	"deepseek-llm",
	"deepseek-r1-qwen",
	"deepseek-v3",
	"default",
	"domyn-small",
	"exaone",
	"exaone4",
	"exaone-moe",
	"f2llmv2",
	"falcon",
	"falcon3",
	"falcon-h1",
	"flint",
	"gemma4",
	"gigachat",
	"glm4",
	"glm5",
	"gpt-2",
	"gpt3-finnish",
	"gpt-4o",
	"granite-docling",
	"granite-embed-multi-311m",
	"granite-embed-multi-97m",
	"grok-2",
	"hunyuan",
	"hunyuan-dense",
	"hy_v4",
	"inkling2",
	"jais",
	"jais-2",
	"jina-de",
	"jina-es",
	"jina-v1-en",
	"jina-v2-code",
	"jina-v2-de",
	"jina-v2-es",
	"jina-v5-nano",
	"joyai-llm",
	"k2-horizon",
	"kanana2",
	"kimi-k2",
	"kormo",
	"laguna",
	"lfm2",
	"llada-moe",
	"llama3",
	"llama4",
	"llama-bpe",
	"llama-v3",
	"megrez",
	"mellum",
	"mellum2",
	"midm-2.0",
	"minerva-7b",
	"minicpm5",
	"minimax-m2",
	"mmbert",
	"modern-bert",
	"mpt",
	"nemotron-3.5-embed",
	"olmo",
	"phi-2",
	"pixtral",
	"poro-chat",
	"qwen2",
	"qwen35",
	"refact",
	"roberta-bpe",
	"sarvam-moe",
	"seed-coder",
	"smaug-bpe",
	"smollm",
	"solar-open",
	"soprano",
	"spark2_5",
	"stablelm2",
	"starcoder",
	"superbpe",
	"talkie",
	"tekken",
	"tekken-num3",
	"tiny_aya",
	"trillion",
	"ufakzeka",
	"viking",
	"whitespace",
	"youtu",
	"zonos3",
}
