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
	if fam.coder {
		return splitDeepseekCoder(text)
	}
	if fam.falcon {
		return splitFalcon(text)
	}
	if fam.deepseekLLM {
		return splitDeepseekLLM(text)
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

// splitDeepseekCoder applies deepseek-coder's five expressions in order.
//
// Each expression is matched over the pieces the previous one produced, and text a match
// does not cover stays attached to its neighbour rather than becoming a piece of its own:
//
//	[\r\n]
//	\s?\p{L}+
//	\s?\p{P}+
//	[\u4e00-\u9fa5 ...]+
//	\p{N}
//
// The effect is one chunk per run of letters, punctuation or CJK characters, and one chunk
// per digit, with a preceding whitespace character attached to whichever run follows it.
func splitDeepseekCoder(text string) []string {
	// Only two of the five expressions carry an optional \s? prefix -- the letter and
	// punctuation runs. A newline, a CJK run and a digit do not, so a space before them
	// stays a chunk of its own.
	type step struct {
		pred func(string) bool
		// spacePrefix reports whether the expression says \s? before the run.
		spacePrefix bool
		// single makes each matching character its own chunk, which is what an expression
		// without a + quantifier means: the newline alternative takes one CR or LF, so a
		// CRLF pair is two chunks, where the number alternative takes the whole run.
		single bool
	}
	pieces := []string{text}
	for _, st := range []step{
		{pred: isNewlineAt},
		{pred: isLetterAt, spacePrefix: true},
		{pred: isPunctAt, spacePrefix: true},
		{pred: isCJKAt},
		{pred: isDigitAt, single: true},
	} {
		var next []string
		for _, piece := range pieces {
			next = append(next, splitRunsOf(piece, st.pred, st.spacePrefix, st.single)...)
		}
		pieces = next
	}
	return pieces
}

// splitRunsOf cuts one piece at the runs where pred is true, keeping the text between them.
//
// spacePrefix says whether a whitespace character immediately before a run joins that run,
// which is what an optional \s? in the expression means: " x" is one chunk where "x" is
// another. The character has to be taken from the gap, which is why the run start is decided
// before the gap is emitted.
// prevRuneStart returns the byte offset where the character ending at end begins.
func prevRuneStart(text string, end int) (int, bool) {
	if end <= 0 {
		return 0, false
	}
	p := end - 1
	for p > 0 && text[p]&0xC0 == 0x80 {
		p--
	}
	return p, true
}

func splitRunsOf(text string, pred func(string) bool, spacePrefix, single bool) []string {
	var out []string
	start := 0
	i := 0
	for i < len(text) {
		if !pred(text[i:]) {
			i += runeLenAt(text[i:])
			continue
		}
		runStart := i
		// Take the single character that ends at i when it is whitespace, which is what an
		// optional \s? before the run means. It must not cross into text already emitted.
		if spacePrefix && runStart > start {
			if prev, ok := prevRuneStart(text, runStart); ok && prev >= start && isSpaceAt(text[prev:]) {
				runStart = prev
			}
		}
		if runStart > start {
			out = append(out, text[start:runStart])
		}
		runEnd := i
		for runEnd < len(text) && pred(text[runEnd:]) {
			runEnd += runeLenAt(text[runEnd:])
			if single {
				break
			}
		}
		out = append(out, text[runStart:runEnd])
		start = runEnd
		i = runEnd
	}
	if start < len(text) {
		out = append(out, text[start:])
	}
	return out
}

// splitFalcon applies falcon's three expressions in order.
//
//	[\p{P}$+<=>^~|`]+   -- a run of punctuation or one of the listed symbols
//	the GPT-2 expression -- which preGPT2 already describes
//	[0-9][0-9][0-9]      -- groups of exactly three digits
//
// The first expression runs before the second, so an apostrophe is taken as punctuation and
// never forms a contraction: "don't" splits as ["don", "'", "t"] rather than ["don", "'t"].
func splitFalcon(text string) []string {
	gpt2, _ := preFamilyOf(preGPT2)
	var out []string
	// Step 1: punctuation and symbol runs.
	for _, piece := range splitRunsOf(text, isFalconPunctAt, false, false) {
		// Step 2: the GPT-2 expression on what is left.
		out = append(out, splitByRule(piece, gpt2)...)
	}
	// Step 3: three-digit groups, applied to every piece the first two produced.
	var grouped []string
	for _, piece := range out {
		grouped = append(grouped, splitThreeDigitRuns(piece)...)
	}
	return grouped
}

// isFalconPunctAt reports whether s begins with a character falcon's first expression takes:
// Unicode punctuation, or one of the ASCII symbols it lists.
func isFalconPunctAt(s string) bool {
	return classifyAt(s, func(r rune) bool {
		return unicode.IsPunct(r) || strings.ContainsRune("$+<=>^~|`", r)
	})
}

// splitThreeDigitRuns cuts each run of digits into groups of three, keeping any remainder.
func splitThreeDigitRuns(text string) []string {
	var out []string
	i := 0
	for i < len(text) {
		if !isDigitAt(text[i:]) {
			// Everything up to the next digit stays together: the expression only matches
			// digit triples, so text it does not match is left as one piece.
			start := i
			for i < len(text) && !isDigitAt(text[i:]) {
				i += runeLenAt(text[i:])
			}
			out = append(out, text[start:i])
			continue
		}
		start := i
		count := 0
		for i < len(text) && isDigitAt(text[i:]) && count < 3 {
			i += runeLenAt(text[i:])
			count++
		}
		out = append(out, text[start:i])
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
	// coder marks deepseek-coder, whose five expressions are applied one after another to
	// the pieces the previous one left, rather than the earlier-ones-then-main-rule shape
	// preDigits and cjk describe. Keeping it separate avoids a rule type whose fields are
	// only meaningful together.
	coder bool
	// falcon marks falcon's three expressions: a punctuation-and-symbol run, the GPT-2
	// expression, then groups of three digits. The middle one is the rule preGPT2 already
	// describes, so only the two outer steps need code.
	falcon bool
	// punctLetterRun is whether the expression's first alternative is
	// "[!-/:-@\\[-`{-~][A-Za-z]+": one ASCII punctuation character followed by a run of
	// ASCII letters, as a single chunk.
	//
	// It takes precedence over the letter alternative, so ".b" and "(x" are one chunk each.
	// The leading-character alternative does not cover this case -- it excludes punctuation --
	// which is why the rule has to exist separately rather than falling out of it.
	punctLetterRun bool
	// contractionSuffix is whether the letter alternatives may absorb a clitic, which is
	// what GPT-4o and its siblings do. It is a suffix on the run rather than a separate
	// alternative, so a word and its "'s" stay one chunk.
	contractionSuffix bool
	// deepseekLLM marks the five-expression rule whose letter class is spelled out as 87
	// ranges rather than written as \p{L}. The shape matches deepseek-coder's, but none of
	// the terms do, so it has its own splitter.
	deepseekLLM bool
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
	case preDeepseekCoder:
		return preRule{coder: true}, true
	case preDeepseekLLM:
		return preRule{deepseekLLM: true}, true
	case preGPT4O:
		// "[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+(?:'s|'t|...)?"
		// "|...|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
		//
		// Tekken's shape with two differences: a digit group is one to three digits rather
		// than one, and the letter alternatives may absorb a clitic suffix.
		return preRule{
			digits: 3, lead: leadAnyNonLetter, letterMarks: true, caseSplit: true,
			newlinesAfterPunct: true, punctSlashTail: true, newlineChunk: true,
			contractionSuffix: true,
		}, true
	case preFalcon:
		return preRule{falcon: true}, true
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
	// (?i:'s|'t|'re|'ve|'m|'ll|'d)? -- a clitic is absorbed into the word it follows, so
	// "don't" is one piece where GPT-2 gives "don" and "'t". Only the letter alternatives
	// carry the suffix; punctuation and digit runs have no such tail.
	if fam.contractionSuffix {
		i += cliticLen(s[i:])
	}
	return i
}

// cliticLen returns the length of a clitic at the start of s, or 0.
//
// The set and the case folding are the same ones the Qwen2 rule matches at the start of a
// text, so the two share the table rather than each keeping a copy.
func cliticLen(s string) int {
	if s == "" {
		return 0
	}
	return qwen2ContractionLen(s)
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

// isPunct is the stricter class deepseek-coder's \p{P}+ uses: punctuation only, no symbols.
func isPunct(r rune) bool { return unicode.IsPunct(r) }

func isPunctAt(s string) bool { return classifyAt(s, isPunct) }

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

// deepseek-llm's rule is five expressions applied in sequence, which is the same shape as
// deepseek-coder's but with different terms throughout -- so it needs its own splitter
// rather than a flag on that one.
//
// The letter class is the unusual part: the expression spells out 87 ranges and 2729
// characters instead of writing \p{L}. It is a strict subset of the letters -- Latin, Greek,
// Cyrillic, Armenian, Georgian and a few others, with CJK, Arabic and the Indic scripts left
// out -- so reusing isLetterAt would group text this rule splits apart.
var deepseekLLMLetterRanges = [][2]rune{
	{0x0041, 0x005A}, {0x0061, 0x007A}, {0x00B5, 0x00B5}, {0x00C0, 0x00D6},
	{0x00D8, 0x00F6}, {0x00F8, 0x01BA}, {0x01BC, 0x01BF}, {0x01C4, 0x0293},
	{0x0295, 0x02AF}, {0x0370, 0x0373}, {0x0376, 0x0377}, {0x037B, 0x037D},
	{0x037F, 0x037F}, {0x0386, 0x0386}, {0x0388, 0x038A}, {0x038C, 0x038C},
	{0x038E, 0x03A1}, {0x03A3, 0x03F5}, {0x03F7, 0x0481}, {0x048A, 0x052F},
	{0x0531, 0x0556}, {0x10A0, 0x10C5}, {0x13A0, 0x13F5}, {0x13F8, 0x13FD},
	{0x1C90, 0x1CBA}, {0x1CBD, 0x1CBF}, {0x1D00, 0x1D2B}, {0x1D6B, 0x1D77},
	{0x1D79, 0x1D9A}, {0x1E00, 0x1F15}, {0x1F18, 0x1F1D}, {0x1F20, 0x1F45},
	{0x1F48, 0x1F4D}, {0x1F50, 0x1F57}, {0x1F59, 0x1F59}, {0x1F5B, 0x1F5B},
	{0x1F5D, 0x1F5D}, {0x1F5F, 0x1F7D}, {0x1F80, 0x1FB4}, {0x1FB6, 0x1FBC},
	{0x1FBE, 0x1FBE}, {0x1FC2, 0x1FC4}, {0x1FC6, 0x1FCC}, {0x1FD0, 0x1FD3},
	{0x1FD6, 0x1FDB}, {0x1FE0, 0x1FEC}, {0x1FF2, 0x1FF4}, {0x1FF6, 0x1FFC},
	{0x2102, 0x2102}, {0x2107, 0x2107}, {0x210A, 0x2113}, {0x2115, 0x2115},
	{0x2119, 0x211D}, {0x2124, 0x2124}, {0x2126, 0x2126}, {0x2128, 0x2128},
	{0x212A, 0x212D}, {0x212F, 0x2134}, {0x2139, 0x2139}, {0x213C, 0x213F},
	{0x2145, 0x2149}, {0x214E, 0x214E}, {0x2183, 0x2184}, {0x2C00, 0x2C7B},
	{0x2C7E, 0x2CE4}, {0x2CEB, 0x2CEE}, {0x2CF2, 0x2CF3}, {0xA640, 0xA66D},
	{0xA680, 0xA69B}, {0xA722, 0xA76F}, {0xA771, 0xA787}, {0xA78B, 0xA78E},
	{0xAB70, 0xABBF}, {0xFB00, 0xFB06}, {0xFB13, 0xFB17}, {0xFF21, 0xFF3A},
	{0xFF41, 0xFF5A}, {0x10400, 0x1044F}, {0x104B0, 0x104D3}, {0x104D8, 0x104FB},
	{0x10C80, 0x10CB2}, {0x10CC0, 0x10CF2}, {0x118A0, 0x118DF}, {0x1E900, 0x1E943},
}

// deepseekLLMPunctRanges is the punctuation class: ASCII printable punctuation and its
// fullwidth forms, curly quotes, and the CJK space and stop. It is a list rather than \p{P}
// because the expression names these ranges, and a category would take in punctuation the
// rule leaves to the letter pass.
var deepseekLLMPunctRanges = [][2]rune{
	{0x0021, 0x002F}, {0x003A, 0x007E}, {0x2018, 0x201F}, {0x3000, 0x3002},
	{0xFF01, 0xFF0F}, {0xFF1A, 0xFF5E},
}

// inRanges reports whether r falls in one of the ranges, which must be sorted and disjoint.
func inRanges(ranges [][2]rune, r rune) bool {
	lo, hi := 0, len(ranges)-1
	for lo <= hi {
		mid := (lo + hi) / 2
		switch {
		case r < ranges[mid][0]:
			hi = mid - 1
		case r > ranges[mid][1]:
			lo = mid + 1
		default:
			return true
		}
	}
	return false
}

func isDeepseekLLMLetter(r rune) bool { return inRanges(deepseekLLMLetterRanges, r) }
func isDeepseekLLMPunct(r rune) bool  { return inRanges(deepseekLLMPunctRanges, r) }

// isDeepseekLLMCJK is the family's CJK alternative. The expression lists three ranges that
// overlap at U+4E00, which leaves U+0800-U+9FA5 and U+AC00-U+D7FF; the first starts at
// Samaritan rather than at a Han character, so it also takes the scripts between -- that is
// what the expression says, and it is why those scripts stay one chunk instead of falling to
// the letter pass, which does not cover them.
func isDeepseekLLMCJK(r rune) bool {
	return (r >= 0x0800 && r <= 0x9FA5) || (r >= 0xAC00 && r <= 0xD7FF)
}

// isNumber reports whether r is \p{N}: a digit, a letter number, or another numeric
// character. It is wider than isDigit, which is \p{Nd} only, and this family's expression
// writes the wider class.
func isNumber(r rune) bool { return unicode.IsNumber(r) }

func isDeepseekLLMLetterAt(s string) bool { return classifyAt(s, isDeepseekLLMLetter) }
func isDeepseekLLMPunctAt(s string) bool  { return classifyAt(s, isDeepseekLLMPunct) }
func isDeepseekLLMCJKAt(s string) bool    { return classifyAt(s, isDeepseekLLMCJK) }
func isNumberAt(s string) bool            { return classifyAt(s, isNumber) }

// splitDeepseekLLM applies the family's five expressions in order: a newline on its own, then
// letter runs, punctuation runs, CJK runs, and number runs. The first two take one preceding
// whitespace character as part of the run (\s?), the rest do not.
//
// Text that none of them match is left in place, which is how a run of a script outside the
// letter class -- Arabic, say -- stays whole instead of being cut apart.
func splitDeepseekLLM(text string) []string {
	pieces := []string{text}
	for _, st := range []struct {
		pred        func(string) bool
		spacePrefix bool
		// single makes each matching character its own chunk, which is what an expression
		// without a + quantifier means: the newline alternative takes one CR or LF, so a
		// CRLF pair is two chunks, where the number alternative takes the whole run.
		single bool
	}{
		{pred: isNewlineAt, single: true},
		{pred: isDeepseekLLMLetterAt, spacePrefix: true},
		{pred: isDeepseekLLMPunctAt, spacePrefix: true},
		{pred: isDeepseekLLMCJKAt},
		{pred: isNumberAt},
	} {
		var next []string
		for _, piece := range pieces {
			next = append(next, splitRunsOf(piece, st.pred, st.spacePrefix, st.single)...)
		}
		pieces = next
	}
	return pieces
}
