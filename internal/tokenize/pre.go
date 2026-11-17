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
		pieces := splitAtStopSet(text, fam.stopSet)
		if fam.stopSetDigits {
			return splitDigitGroups(pieces, 1)
		}
		return pieces
	}
	if fam.coder {
		return splitDeepseekCoder(text)
	}
	if fam.falcon {
		return splitFalcon(text)
	}
	if fam.chameleon {
		return splitChameleon(text)
	}
	if fam.kimiK2 {
		return splitKimiK2(text)
	}
	if fam.deepseekLLM {
		return splitDeepseekLLM(text)
	}
	if fam.newlineRuns {
		return splitNewlineRuns(text)
	}
	if fam.zonos3 {
		return splitZonos3(text)
	}
	if fam.digitsFromRight && !fam.caseSplit && !fam.letterMarks {
		return splitDigitsFromRight(text, fam.digitsRightBoundary)
	}
	if fam.partition {
		// The digit cut runs first, as llama.cpp applies the expressions in order: this
		// family's first expression is a bare \p{N}, so "a1b2" is cut at the digits before the
		// partition groups what is left.
		return splitDigitGroups(splitPartition(text), 1)
	}
	// The family's earlier expressions cut first, in the order llama.cpp lists them, and the
	// main rule then runs on every piece that leaves.
	pieces := []string{text}
	if fam.leadingNewlinePass {
		pieces = splitNewlineRuns(text)
	}
	if fam.digitsFromRight {
		pieces = splitDigitsFromRight(text, fam.digitsRightBoundary)
	}
	if fam.preDigits > 0 {
		pieces = splitDigitGroups(pieces, fam.preDigits)
	}
	if fam.cjk {
		pieces = splitCJK(pieces)
	}
	if fam.asian {
		pieces = splitEach(pieces, splitAsianRuns)
	}
	if fam.han {
		pieces = splitEach(pieces, splitHanRuns)
	}
	if fam.youtuScripts {
		pieces = splitYoutuScripts(text)
	}
	if len(pieces) == 1 && pieces[0] == text {
		out := splitByRule(text, fam)
		if fam.postDigits > 0 {
			return splitDigitGroups(out, fam.postDigits)
		}
		return out
	}
	var out []string
	for _, piece := range pieces {
		out = append(out, splitByRule(piece, fam)...)
	}
	if fam.postDigits > 0 {
		// A trailing \p{N} runs after the main rule, so digits the first pass grouped are cut
		// again. Only spark2_5 has one.
		out = splitDigitGroups(out, fam.postDigits)
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

// splitKimiK2 applies llama.cpp's hand-written matcher for this family.
//
// The comment above that matcher describes a regex -- "\p{Han}+" and two letter
// alternatives whose halves form a case boundary -- but the matcher departs from it in three
// ways, and the matcher is what runs:
//
//   - The word alternative consumes every non-Han letter in one run. There is no case
//     boundary, so "camelCase" and "HTTPServer" are one chunk each.
//   - A combining mark ends a word instead of joining it. The leading character may still be
//     a mark, which is why "\u0301a" is one chunk while "a\u0301" is two.
//   - The leading character is taken before a clitic is looked for, so a chunk that starts
//     with an apostrophe is that apostrophe plus the word: "'scamelCase" is one chunk where
//     a prefix-contraction rule would cut "'s" off the front. A clitic after letters is
//     absorbed, so "don't" stays one chunk.
//
// Reproducing the matcher is the reason this family is a function rather than more fields on
// preRule: three of its behaviours contradict the flags the other families share, and folding
// them in would mean flags that exist only to be turned off again.
func splitKimiK2(text string) []string {
	var out []string
	pos := 0
	n := len(text)
	for pos < n {
		// \p{Han}+ -- a Han run is its own chunk and never joins a neighbour.
		if isHanAt(text[pos:]) {
			start := pos
			for pos < n && isHanAt(text[pos:]) {
				pos += runeLenAt(text[pos:])
			}
			out = append(out, text[start:pos])
			continue
		}

		// The word alternative: an optional leading character that is not a newline, letter
		// or number, then a run of non-Han letters, then an optional clitic.
		c, ok := runeAt(text, pos)
		if ok && (isLetter(c) && !isHan(c) ||
			!isNewline(c) && !isLetter(c) && !isNumber(c) && restsOnLetter(text, pos)) {
			start := pos
			leading := false
			if !isNewline(c) && !isLetter(c) && !isNumber(c) {
				leading = true
				pos += runeLenAt(text[pos:])
			}
			letters := false
			for pos < n {
				r, ok := runeAt(text, pos)
				if !ok || !isLetter(r) || isHan(r) {
					break
				}
				letters = true
				pos += runeLenAt(text[pos:])
			}
			if letters {
				pos += cliticLen(text[pos:])
				out = append(out, text[start:pos])
				continue
			}
			if leading {
				// The leading character was consumed but no word followed, so it is gap
				// text: put it back and let the alternatives below decide.
				pos = start
			}
		}

		// \p{N}{1,3}
		if r, ok := runeAt(text, pos); ok && isNumber(r) {
			start := pos
			count := 0
			for pos < n && count < 3 {
				r, ok := runeAt(text, pos)
				if !ok || !isNumber(r) {
					break
				}
				pos += runeLenAt(text[pos:])
				count++
			}
			out = append(out, text[start:pos])
			continue
		}

		// " ?[^\s\p{L}\p{N}]+[\r\n]*"
		start := pos
		i := pos
		if i < n && text[i] == ' ' {
			i += 1
		}
		runStart := i
		for i < n {
			r, ok := runeAt(text, i)
			if !ok || isSpace(r) || isLetter(r) || isNumber(r) {
				break
			}
			i += runeLenAt(text[i:])
		}
		if i > runStart || i > pos && text[pos] == ' ' && i > pos+1 {
			pos = i
			for pos < n && isNewlineAt(text[pos:]) {
				pos += runeLenAt(text[pos:])
			}
			out = append(out, text[start:pos])
			continue
		}

		// The whitespace alternatives, in the expression's order: "\\s*[\\r\\n]+" first, then
		// "\\s+(?!\\S)", then "\\s+".
		//
		// The lookahead makes the second alternative drop the last character of a run that is
		// followed by something, so that character leads the next chunk: "a  b" is
		// ["a", " ", " b"], not ["a", "  ", "b"]. A run that ends the text has nothing after it,
		// so it is consumed whole.
		if r, ok := runeAt(text, pos); ok && isSpace(r) {
			start := pos
			newlineEnd, spaceEnd := newlineRunEnd(text, pos)
			switch {
			case newlineEnd > pos:
				pos = newlineEnd
			case spaceEnd < len(text):
				last, ok := prevRuneStart(text, spaceEnd)
				if ok && last > pos {
					pos = last
				} else {
					pos = spaceEnd
				}
			default:
				pos = spaceEnd
			}
			out = append(out, text[start:pos])
			continue
		}

		// Nothing matches: one character is gap text.
		sz := runeLenAt(text[pos:])
		if sz <= 0 {
			sz = 1
		}
		out = append(out, text[pos:pos+sz])
		pos += sz
	}
	return out
}

// newlineRunEnd returns the end of a whitespace run that ends on a newline, and separately the
// end of the whole whitespace run, for the "\s*[\r\n]+" and "\s+" alternatives.
func newlineRunEnd(text string, pos int) (newlineEnd, spaceEnd int) {
	end := pos
	for end < len(text) && isSpaceAt(text[end:]) {
		n := runeLenAt(text[end:])
		if isNewlineAt(text[end:]) {
			newlineEnd = end + n
		}
		end += n
	}
	spaceEnd = end
	return newlineEnd, spaceEnd
}

// runeAt decodes the rune at byte offset pos, reporting false past the end.
func runeAt(text string, pos int) (rune, bool) {
	if pos < 0 || pos >= len(text) {
		return 0, false
	}
	r, _ := utf8.DecodeRuneInString(text[pos:])
	return r, true
}

// restsOnLetter reports whether the character after the one at pos is a non-Han letter.
func restsOnLetter(text string, pos int) bool {
	next, ok := runeAt(text, pos+runeLenAt(text[pos:]))
	return ok && isLetter(next) && !isHan(next)
}

// splitChameleon applies chameleon's expressions in order.
//
// The family is unusual in that its expressions are not alternatives of one main rule but
// five separate passes, each cutting what the last one left. Written as one expression they
// would not agree with the reference, and the punctuation pass is why: it cuts *single*
// characters and runs before the main rule, so "a--b" is ["a", "-", "-", "b"] rather than
// ["a", "--", "b"], and a contraction's apostrophe is cut too, leaving "don't" as
// ["don", "'", "t"] instead of ["don", "'t"]. The space pass has the same shape: two and
// four spaces are alternatives matched at each position, so eight spaces are two chunks of
// four and three spaces are two-plus-one.
//
// The first of llama.cpp's six expressions is left out. It matches "<sentinel:N>", and it
// cannot change the output: every later pass runs over the fragments the earlier ones left,
// so a sentinel matched whole is then cut by the punctuation and digit passes exactly as the
// text alone would be. Checked both ways over a corpus of sentinel-bearing strings -- the
// final partition is identical with the pass and without it, which is why implementing it
// would add code no test could distinguish. llama.cpp's own comment makes a weaker version
// of the same point, that its special-token partition has already split these pieces.
func splitChameleon(text string) []string {
	// Image pass first: "IMGIMG[A-I]{1,4}Z" is one chunk, and it is the only pass whose
	// chunks can contain digits and punctuation the later passes would otherwise cut.
	pieces := splitEach([]string{text}, splitChameleonImages)
	// Space pass runs before the digit and punctuation ones, and the order is observable:
	// the space alternatives are tried at each position, so the run they leave is cut
	// differently depending on what has already been separated from it.
	pieces = splitEach(pieces, splitChameleonSpaces)
	// Digit pass: one number per chunk, over the full \p{N} category.
	pieces = splitEach(pieces, splitChameleonNumbers)
	// Punctuation pass: [\p{P}!-/:-@\[-`{-~], one character at a time.
	pieces = splitEach(pieces, splitChameleonPunct)
	// Main rule: the GPT-2 expression, on what the passes above left.
	gpt2, _ := preFamilyOf(preGPT2)
	return splitEach(pieces, func(s string) []string { return splitByRule(s, gpt2) })
}

// splitChameleonImages applies "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z".
//
// The expression is not anchored to a token boundary, so it matches inside ordinary text
// too: "IMGIMGABZX" is ["IMGIMGABZ", "X"] and "xIMGIMGABZ" is ["x", "IMGIMGABZ"]. That is
// what the reference does, and the shape cannot be folded into a later pass because its
// letters and the Z would otherwise run together with their neighbours.
func splitChameleonImages(text string) []string {
	const prefix, suffix = "IMGIMG", "Z"
	var out []string
	start := 0
	for i := 0; i < len(text); {
		if !strings.HasPrefix(text[i:], prefix) {
			i += runeLenAt(text[i:])
			continue
		}
		j := i + len(prefix)
		n := 0
		for j < len(text) && n < 4 && text[j] >= 'A' && text[j] <= 'I' {
			j++
			n++
		}
		if n == 0 || j >= len(text) || text[j] != suffix[0] {
			i += runeLenAt(text[i:])
			continue
		}
		if i > start {
			out = append(out, text[start:i])
		}
		out = append(out, text[i:j+1])
		i = j + 1
		start = i
	}
	if start < len(text) {
		out = append(out, text[start:])
	}
	return out
}

// splitChameleonNumbers applies chameleon's bare "\p{N}", one number per chunk.
//
// The class is the whole number category, not the decimal digits: "\p{N}" also holds
// superscripts, vulgar fractions and Roman numerals, and the reference cuts those one by
// one the same way. Using unicode.IsDigit would leave "\u00b2\u00b3" whole, which is the
// behaviour of \p{Nd} rather than \p{N}.
func splitChameleonNumbers(text string) []string {
	return splitSingles(text, isNumber)
}

// splitEach applies one pass to every piece and flattens the result.
func splitEach(pieces []string, pass func(string) []string) []string {
	out := make([]string, 0, len(pieces))
	for _, piece := range pieces {
		out = append(out, pass(piece)...)
	}
	return out
}

// splitChameleonPunct applies "[\p{P}!-/:-@\[-`{-~]", which is one character per chunk.
//
// The class is Unicode punctuation plus four ASCII ranges. The ranges cover the ASCII
// symbols as well -- "<", ">", "+", "$" and the rest -- which \p{P} does not, so the class
// is not simply the category; leaving the symbols out would let them run together in the
// main rule where the reference cuts them one by one.
func splitChameleonPunct(text string) []string {
	return splitSingles(text, isChameleonPunct)
}

// isChameleonPunct reports whether r is in chameleon's punctuation class: Unicode
// punctuation, or one of the ASCII symbols the four ranges name.
func isChameleonPunct(r rune) bool {
	if unicode.IsPunct(r) {
		return true
	}
	return r >= '!' && r <= '/' || r >= ':' && r <= '@' ||
		r >= '[' && r <= '`' || r >= '{' && r <= '~'
}

// splitSingles cuts each character that pred accepts into a chunk of its own, leaving
// everything else as gap text.
func splitSingles(text string, pred func(rune) bool) []string {
	var out []string
	start := 0
	for i := 0; i < len(text); {
		n := runeLenAt(text[i:])
		if n == 0 {
			n = 1
		}
		r, _ := utf8.DecodeRuneInString(text[i:])
		if !pred(r) {
			i += n
			continue
		}
		if i > start {
			out = append(out, text[start:i])
		}
		out = append(out, text[i:i+n])
		i += n
		start = i
	}
	if start < len(text) {
		out = append(out, text[start:])
	}
	return out
}

// splitChameleonSpaces applies "([\t\n]|    |  )", one match per chunk.
//
// The alternatives are tried at each position rather than once per run, and the four-space
// one is written first, so eight spaces are two four-space chunks and a run of three is two
// plus one. Carriage return is not in the class, so a CRLF pair is left to the main rule,
// whose whitespace alternative takes the CR together with the newline.
func splitChameleonSpaces(text string) []string {
	var out []string
	start := 0
	for i := 0; i < len(text); {
		n := chameleonSpaceLen(text[i:])
		if n == 0 {
			i += runeLenAt(text[i:])
			continue
		}
		if i > start {
			out = append(out, text[start:i])
		}
		out = append(out, text[i:i+n])
		i += n
		start = i
	}
	if start < len(text) {
		out = append(out, text[start:])
	}
	return out
}

// splitThreeDigitRuns cuts each run of digits into groups of three, keeping any remainder.
func splitThreeDigitRuns(text string) []string {
	// The expression is [0-9][0-9][0-9]: only a run of at least three digits matches. Text
	// with fewer -- a lone "5", or "45" -- is gap text, and gap text joins the piece before
	// it rather than standing alone, which is why " 5" stays whole where " 1234" gives
	// [" ", "123", "4"] and "1234" gives ["123", "4"].
	var out []string
	// gap is text carried forward to attach to the next thing emitted.
	gap := ""
	i := 0
	for i < len(text) {
		if !isDigitAt(text[i:]) {
			start := i
			for i < len(text) && !isDigitAt(text[i:]) {
				i += runeLenAt(text[i:])
			}
			gap += text[start:i]
			continue
		}
		j, n := i, 0
		for j < len(text) && isDigitAt(text[j:]) {
			j += runeLenAt(text[j:])
			n++
		}
		if n < 3 {
			gap += text[i:j]
			i = j
			continue
		}
		// A leading gap that is only whitespace stays separate: the reference emits " " on
		// its own before "123" for " 1234". Anything else joins the first triple.
		if gap != "" && strings.TrimSpace(gap) == "" {
			out = append(out, gap)
			gap = ""
		}
		for k := 0; k < n; k += 3 {
			next := min(k+3, n)
			chunk := text[i+k : i+next]
			if gap != "" {
				chunk = gap + chunk
				gap = ""
			}
			out = append(out, chunk)
		}
		i = j
	}
	if gap != "" {
		out = append(out, gap)
	}
	return out
}

// splitWhitespaceRuns cuts text into runs of non-space characters, the \S+ rule.
//
// The spaces between runs are not emitted: llama.cpp's whitespace pre-tokenizer
// (jina-embeddings-v2-base-zh) produces only the runs, and the vocabulary has no separate
// space piece to give them.
func splitWhitespaceRuns(text string) []string {
	// The expression is \S+: runs of non-space characters. The spaces between them are not
	// deleted from the text -- llama.cpp's splitter emits the gap before each match and the
	// tail after the last as pieces of their own -- so they come out as chunks here too.
	// Dropping them put text into the vocabulary's hands that the model never saw in
	// training: "a  b" became ["a", "b"], losing the two spaces and the word boundary they
	// encode.
	var out []string
	start := 0
	inSpace := false
	for i, r := range text {
		space := unicode.IsSpace(r)
		// A change of class ends the current run; consecutive spaces stay together, so a
		// double space is one chunk, as it is in the reference.
		if i > 0 && space != inSpace {
			out = append(out, text[start:i])
			start = i
		}
		inSpace = space
	}
	if start < len(text) {
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
	// The expression is " ?[^(...)]+": an optional space, then a run that stops at the listed
	// characters. Everything the expression does not match is still a chunk -- llama.cpp's
	// splitter carries the text between matches through as one offset -- so the loop
	// alternates between matches and the gaps between them.
	matchAt := func(s string) int {
		// Returns the byte length of a match starting at s, or 0.
		i := 0
		if s[0] == ' ' && len(s) > 1 {
			i = 1
		}
		start := i
		for i < len(s) {
			r, n := utf8.DecodeRuneInString(s[i:])
			if r == utf8.RuneError || strings.ContainsRune(stop, r) {
				break
			}
			i += n
		}
		if i == start {
			return 0
		}
		return i
	}

	var out []string
	for i := 0; i < len(text); {
		if n := matchAt(text[i:]); n > 0 {
			out = append(out, text[i:i+n])
			i += n
			continue
		}
		// Gap text: extend it until a match can start again. A space that begins a match
		// belongs to that match, not to the gap, which is why the test is a match attempt
		// rather than a membership check.
		j := i
		for j < len(text) && matchAt(text[j:]) == 0 {
			_, n := utf8.DecodeRuneInString(text[j:])
			j += n
		}
		out = append(out, text[i:j])
		i = j
	}
	return out
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
			// No alternative matches here, so this is gap text: characters the expression
			// does not cover. llama.cpp's splitter emits the gap before each match as *one*
			// fragment, so a run of unmatchable characters is one chunk rather than one
			// chunk per character -- deepseek-v3's "\u200c\u200c" is one chunk, and
			// splitting it per character would be a boundary the vocabulary never saw.
			//
			// The gap is measured in characters, not bytes: a byte here would cut a
			// multi-byte rune in two.
			j := i
			for j < len(text) {
				n = runeLenAt(text[j:])
				if n <= 0 {
					n = 1
				}
				j += n
				if chunkLen(text[j:], fam) > 0 {
					break
				}
			}
			chunks = append(chunks, text[i:j])
			i = j
			continue
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
	// punctExcludesMarks is whether the punctuation complement also excludes combining marks,
	// which is qwen35's "[^\s\p{L}\p{M}\p{N}]". The other families write
	// "[^\s\p{L}\p{N}]", where a mark is neither a letter nor a number and so joins the
	// punctuation run -- a difference that shows on a mark after punctuation.
	punctExcludesMarks bool
	// joiners is whether U+200C and U+200D also join a letter run, which is k2-horizon's
	// "(?:\p{L}|\p{M}|\u200C|\u200D)+". Neither character is a letter or a mark, so
	// without the flag they fall to the punctuation alternative and split the word.
	joiners bool
	// newlinesAfterPunct is whether a punctuation run may absorb the newlines that follow
	// it. qwen2 does; gpt-2 ends the chunk at the punctuation.
	newlinesAfterPunct bool
	// punctSlashTail is whether the punctuation run also absorbs following slashes, which is
	// tekken's "[\r\n/]*" tail. It only applies together with newlinesAfterPunct, or with
	// singleTail for the family whose tail is "[\r\n/]?".
	punctSlashTail bool
	// singleTail makes the punctuation tail "[\r\n/]?" rather than a run: at most one
	// character, taken only when it is a newline or a slash. It is exaone-moe's spelling, and
	// it differs from tekken's "*" on a pair: ",\r\n" is [",\r", "\n"] here and [",\r\n"]
	// there.
	singleTail bool
	// newlineChunk is whether the expression has a "\s*[\r\n]+" alternative: a newline
	// with the spaces before it as one chunk. The GPT-2 expression has no such rule, so a
	// newline there falls through to the whitespace alternative and groups differently.
	newlineChunk bool
	// newlineSingle is whether the newline alternative is a bare "[\r\n]" rather than a run,
	// so a CRLF pair is two chunks. Only spark2_5 has that form.
	newlineSingle bool
	// preDigits is the length of each digit group the family cuts out *before* the main
	// expression, or 0 for none.
	//
	// llama.cpp applies a family's expressions in sequence, each cutting the previous round's
	// chunks further. A leading "\p{N}" therefore isolates digits before the main rule runs,
	// so the main rule's digit alternative has nothing left to join: starcoder's bare
	// "\p{N}" gives one chunk per digit while deepseek's "\p{N}{1,3}" gives groups of up to
	// three, and neither is the same as setting the main rule's digit count.
	preDigits int
	// postDigits is the length of each digit group cut out *after* the main rule, or 0 for
	// none. Only spark2_5 uses it: a trailing bare \p{N} runs after the letter rule, so a
	// number the first pass grouped in threes is cut again into single digits.
	postDigits int
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
	// kimiK2 marks a rule whose matcher is hand-written in llama.cpp rather than expressed
	// as a regex, and whose behaviour differs from the comment above it in three ways: the
	// word alternative does not split on case, a combining mark ends a word rather than
	// joining it, and the leading character is taken before a clitic is looked for. The
	// matcher is what runs, so it is what this splitter reproduces.
	kimiK2 bool
	// chameleon marks its six expressions, applied one after another to the pieces the
	// previous one left. They do not compose into one expression: each is a pass over the
	// whole text, and the punctuation and space passes cut single characters the main rule
	// would otherwise run together. The sentinel pass is the one left out, and the splitter
	// explains why it cannot change the output.
	chameleon bool
	// punctLetterRun is whether the expression's first alternative is
	// "[!-/:-@\\[-`{-~][A-Za-z]+": one ASCII punctuation character followed by a run of
	// ASCII letters, as a single chunk.
	//
	// It takes precedence over the letter alternative, so ".b" and "(x" are one chunk each.
	// The leading-character alternative does not cover this case -- it excludes punctuation --
	// which is why the rule has to exist separately rather than falling out of it.
	punctLetterRun bool
	// noContraction removes the expression's leading contraction alternative. ufakzeka is the
	// only family without it, and the removal is what keeps a Turkish apostrophe suffix
	// whole: "'da" is one chunk there and "'d" plus "a" with the alternative present.
	noContraction bool
	// noContractionOnceMore marks a family whose expression has *no* contraction alternative
	// even though it also has a punctuation-letter run. That combination is deepseek-v3's
	// and afmoe's: the first alternative is "[!..~][A-Za-z]+", so an apostrophe is taken as
	// the punctuation character and the letters after it follow, giving "'ta" as one chunk
	// where the contraction reading gives ["'t", "a"].
	//
	// It is a separate field from noContraction because that one names ufakzeka's spelling,
	// where the alternative is absent but the leading-character one may still take the
	// apostrophe as a prefix.
	noContractionAlt bool
	// contractionSuffix is whether the letter alternatives may absorb a clitic, which is
	// what GPT-4o and its siblings do. It is a suffix on the run rather than a separate
	// alternative, so a word and its "'s" stay one chunk.
	contractionSuffix bool
	// partition marks a main rule that is three disjoint classes -- whitespace, word
	// characters, and the rest -- rather than a run with a leading character. Every chunk is
	// one class, so "Hi, there" is ["Hi", ",", " ", "there"] with no leading-character pass.
	partition bool
	// digitsFromRight is whether a digit run is cut into threes counting from its end, so
	// the remainder -- if any -- is the leftmost chunk. The superbpe lookahead asserts a
	// multiple of three digits remain, which puts the boundary there.
	digitsFromRight bool
	// digitsRightBoundary restricts digitsFromRight to runs whose right neighbour is not a
	// word character. tiny_aya's expression ends in \b, so a run touching a letter is gap
	// text rather than a match.
	digitsRightBoundary bool
	// stopSetDigits is BLOOM's expression followed by a single-digit cut. Both steps are
	// needed in that order, which is why the stop-set dispatch cannot return early for this
	// family the way it does for the others.
	stopSetDigits bool
	// newlineRuns marks "[^\n]+|[\n]+": the text alternates between runs that are newlines
	// and runs that are not, and nothing else separates a chunk. It is the only rule whose
	// chunks can contain a space, a comma and a word at once.
	newlineRuns bool
	// zonos3 marks an expression whose alternatives cover only digits, punctuation runs of
	// two or more, and trailing whitespace. Letters match nothing, so they stay gap text.
	zonos3 bool
	// youtuScripts marks a leading pass that cuts four scripts out of the text before the main
	// rule runs: Hangul, CJK punctuation, bopomofo, and CJK. Each is its own run; nothing else
	// is touched by the pass.
	youtuScripts bool
	// spaceTiers replaces the whitespace tail with a set of power-of-two runs, tried largest
	// first. A run of n spaces is cut into tiers until one is left, and that last space is
	// then the leading character of whatever follows. It is jais-2's expression; every other
	// family uses the single "\s+(?!\S)|\s+" tail.
	spaceTiers bool
	// wordSpaces is whether a letter run may continue past a *single* space into the letters
	// after it. exaone-moe's expression is "(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+", so "hi there"
	// is one chunk while "a  b" is not -- two spaces fall outside that alternative.
	wordSpaces bool
	// leadingNewlinePass runs newlineRuns before the main rule instead of as the whole rule.
	// Laguna pairs the newline expression with bailingmoe's, and the pass is not redundant:
	// it separates a newline from whitespace before it, which the main rule would otherwise
	// take into one chunk.
	leadingNewlinePass bool
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
	// han is whether a run of Han ideographs is its own chunk, from kimi-k2's "\p{Han}+".
	// It is separate from cjk and asian because it names the script: the family's letter
	// alternatives then exclude Han, so the two must not overlap.
	han bool
	// asian selects afmoe's wider CJK class in place of the three-range cjk one. The two are
	// separate because the sets are different, not nested: afmoe's class has no Hiragana
	// range beyond U+309F either, but it reaches Thai, Lao, Khmer, Myanmar, Cham, Javanese,
	// Hangul and the halfwidth katakana block, none of which "cjk" covers.
	asian bool
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
	case preBailingMoE:
		// "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}"
		// "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"
		//
		// llama3's shape, except digits are a bare \p{N}: every digit is its own chunk where
		// llama3 takes up to three. The newline alternative lacks llma3's `+`, but both forms
		// consume the whole run, so nothing separates the families there.
		return preRule{
			digits: 1, lead: leadAnyNonLetter,
			newlinesAfterPunct: true, newlineChunk: true,
		}, true
	case preLaguna:
		// "[^\n]+|[\n]+" then bailingmoe's expression. The pass separates newlines from any
		// whitespace before them, so " x \n y " gives [" x", " ", "\n", " y", " "] rather than
		// [" x", " \n", " y", " "]. Everything else the main rule decides alone.
		return preRule{
			leadingNewlinePass: true,
			digits:             1, lead: leadAnyNonLetter,
			newlinesAfterPunct: true, newlineChunk: true,
		}, true
	case preSoprano:
		return preRule{partition: true, preDigits: 1}, true
	case preSuperBPE:
		return preRule{digitsFromRight: true}, true
	case preTinyAya:
		// "\d{1,3}(?=(?:\d{3})*\b)" then a case-splitting letter rule: digits group three
		// at a time from the right, but a run touching a letter is left whole, and letters
		// are cut where a lower-case run meets an upper-case one.
		return preRule{
			digitsFromRight: true, digitsRightBoundary: true,
			digits: 3, lead: leadAnyNonLetter, letterMarks: true, caseSplit: true,
			newlinesAfterPunct: true, punctSlashTail: true, newlineChunk: true,
			contractionSuffix: true,
		}, true
	case preFlint:
		// Four expressions: \p{N}{1,3}, the CJK block, a case-splitting letter rule with no
		// digit alternative, then a punctuation run that names the categories rather than a
		// range. Digits and CJK are cut before the main rule, which is why it has no digit
		// alternative of its own.
		return preRule{
			digits: 0, lead: leadAnyNonLetter, letterMarks: true, caseSplit: true,
			punctIsCategory: true, newlinesAfterPunct: true, newlineChunk: true,
			preDigits: 3, cjk: true, contractionSuffix: true,
		}, true
	case preMiniCPM5:
		// "\\p{N}{1,3}" then a llama3-shaped rule with "\\p{N}+" where llama3 has
		// "\\p{N}{1,3}". The first expression leaves only remainders for the second, so the
		// digit alternative there is a plain run rather than another grouping.
		return preRule{
			digits: 0, lead: leadAnyNonLetter,
			newlinesAfterPunct: true, newlineChunk: true, preDigits: 3,
		}, true
	case preZonos3:
		return preRule{zonos3: true}, true
	case preJais2:
		// "(?:'[sS]|...)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
		// "|\\s*[\\r\\n]+|\\s{512}(?!\\S)|...|\\s{4}(?!\\S)|\\s{1,2}(?!\\S)|\\s{1}"
		//
		// llama3's expression with the whitespace tail replaced by power-of-two tiers, tried
		// largest first. A run of n spaces is cut into tiers until one remains, and that last
		// space leads whatever follows: "a   b" gives ["a", "  ", " b"].
		return preRule{
			digits: 3, lead: leadAnyNonLetter,
			newlinesAfterPunct: true, newlineChunk: true, spaceTiers: true,
		}, true
	case preExaoneMoE:
		// "(?:'[sS]|...)|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}\\p{M}*(?: \\p{L}\\p{M}*)*)+|\\p{N}"
		// "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]?|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"
		//
		// llama3's shape with two changes to the letter alternative: marks join the run, and
		// a single space may continue it, so "hi there" is one chunk while "a  b" is not. The
		// punctuation tail takes one newline or slash rather than any run.
		return preRule{
			digits: 1, lead: leadAnyNonLetter, letterMarks: true, wordSpaces: true,
			newlineChunk: true, punctSlashTail: true, singleTail: true,
		}, true
	case preTekkenNum3:
		// tekken's expression with the digit alternative replaced by
		// "\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))|\p{N}{1,2}": digits group three at a time
		// from the right, so "1234" is ["1", "234"] where tekken gives one digit per chunk.
		return preRule{
			digitsFromRight: true,
			digits:          3, lead: leadAnyNonLetter, letterMarks: true, caseSplit: true,
			newlinesAfterPunct: true, punctSlashTail: true, newlineChunk: true,
		}, true
	case preYoutu:
		// A leading pass cuts four scripts out -- Hangul, CJK punctuation, bopomofo, and CJK --
		// then the main rule runs. That main rule is tiny_aya's: case-split letter runs with
		// marks and a clitic suffix.
		return preRule{
			youtuScripts: true,
			digits:       1, lead: leadAnyNonLetter, letterMarks: true, caseSplit: true,
			newlinesAfterPunct: true, punctSlashTail: true, newlineChunk: true,
			contractionSuffix: true,
		}, true
	case preSpark25:
		// Four expressions: \p{N}{1,3}, the CJK block, a letter rule, then a bare \p{N}.
		//
		// The letter rule's leading character is any non-letter, so a step attaches to the
		// word after it: "a.b" is ["a", ".b"]. The punctuation alternative names the two
		// categories and has no newline tail, and the newline alternative is a single
		// [\r\n], so a pair of newlines is two chunks. The trailing \p{N} cuts every digit
		// the first pass left grouped, so a number ends up one digit per chunk.
		return preRule{
			digits: 0, lead: leadAnyNonLetter, letterMarks: true,
			punctIsCategory: true, newlineSingle: true,
			preDigits: 3, cjk: true, postDigits: 1,
		}, true
	case preChameleon:
		return preRule{chameleon: true}, true
	case preSeedCoder:
		// "(?:'[sS]|...)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\\r\\n]+"
		// "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
		//
		// qwen2's shape with the punctuation alternative excluding newlines and carrying no
		// newline tail. The "\\s*[\\r\\n]+" alternative still groups a run of newlines, but a
		// punctuation run stops before them instead of absorbing them.
		return preRule{
			digits: 1, lead: leadAnyNonLetter, newlineChunk: true,
		}, true
	case preQwen35:
		// "(?:'[sS]|...)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}"
		// "| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
		//
		// qwen2's shape with combining marks in the letter class, so a base letter and the
		// marks after it stay in one run.
		return preRule{
			digits: 1, lead: leadAnyNonLetter, letterMarks: true, punctExcludesMarks: true,
			newlinesAfterPunct: true, newlineChunk: true,
		}, true
	case preAfMoE:
		// "\p{AFMoE_digits}",
		// "[\u4e00-\u9fff\u3400-\u4dbf ... \u0e40-\u0e7f ... \uac00-\ud7af \u1100-\u11ff]+",
		// the deepseek-v3 main expression.
		//
		// The first step is not a regex in llama.cpp but a custom matcher (see
		// unicode_regex_split_custom_afmoe): a digit run is cut into groups of three counted
		// from the right, with the one- or two-digit remainder leading, which is exactly what
		// digitsFromRight describes. The second is a much wider CJK class than the other
		// families name. The main expression is deepseek-v3's, so the rest is fields those
		// families already set.
		return preRule{
			digitsFromRight: true, asian: true,
			digits: 0, lead: leadNotLetterPunctSymbol, letterMarks: true,
			punctIsCategory: true, newlinesAfterPunct: true, newlineChunk: true,
			punctLetterRun: true, noContractionAlt: true,
		}, true
	case preKimiK2:
		return preRule{kimiK2: true}, true
	case preK2Horizon:
		// "(?:'[sS]|...)|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+"
		// "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
		//
		// qwen35's shape with two changes to the letter alternative: the joining controls
		// U+200C and U+200D join the run, and digits group up to three rather than one at a
		// time. The controls matter for scripts that spell with a zero-width joiner -- a word
		// split at one is two chunks the vocabulary never saw.
		return preRule{
			digits: 3, lead: leadAnyNonLetter, letterMarks: true, joiners: true,
			newlinesAfterPunct: true, newlineChunk: true,
		}, true
	case preUfakzeka:
		// "[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
		//
		// qwen2's shape with the contraction alternatives removed. That removal is the whole
		// point: without it the leading character takes the apostrophe and the suffix is one
		// run, so "İstanbul'da" keeps "'da" rather than splitting at "'d".
		return preRule{
			digits: 1, lead: leadAnyNonLetter,
			newlinesAfterPunct: true, newlineChunk: true, noContraction: true,
		}, true
	case preViking:
		// The first expression is BLOOM's verbatim, so the stop set is the same string; the
		// second cuts every digit out on its own.
		return preRule{stopSet: " \t\r\n|()" + ".,!?…。，、।۔،", stopSetDigits: true}, true
	case preMMBert:
		return preRule{newlineRuns: true}, true
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
			preDigits: 3, cjk: true, punctLetterRun: true, noContractionAlt: true,
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
	if s == "" {
		return 0
	}
	// The contraction alternative is the expression's first, so it is tried first -- but only
	// for families that have it. ufakzeka is qwen2 with that alternative removed precisely so
	// an apostrophe suffix stays whole.
	if !fam.noContraction && !fam.noContractionAlt {
		if n := qwen2ContractionLen(s); n > 0 {
			return n
		}
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
	//
	// jais-2 replaces both with power-of-two tiers, which cut a run into largest-first
	// pieces and leave the final space to the next chunk.
	if n := runeLenAt(s); n > 0 && isSpaceAt(s) {
		if fam.spaceTiers {
			return spaceTierLen(s)
		}
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
	if len(s) < 2 || s[0] != '\'' {
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
	n, ok := letterRunFrom(s, fam, 0)
	if ok {
		return n
	}
	// The optional prefix is greedy in the expression, but a match that needs it and then
	// finds no run does not happen: the engine backtracks to the empty prefix and tries
	// again. That case is real -- "\u0301\u200d" under deepseek-v3 matches the mark as the
	// run and leaves the joiner to the next alternative, where taking the mark as a prefix
	// would have consumed it and then failed.
	if i := leadLen(s, fam); i > 0 {
		if n, ok := letterRunFrom(s, fam, i); ok {
			return n
		}
	}
	return 0
}

// leadLen returns the byte length of the family's optional leading character at the start of
// s, or 0 when the character is absent or may not be taken as a prefix.
func leadLen(s string, fam preRule) int {
	switch fam.lead {
	case leadSpace:
		if strings.HasPrefix(s, " ") {
			return 1
		}
	case leadAnyNonLetter:
		// "[^\r\n\p{L}\p{N}]?" -- any one character that is not a letter, digit, or
		// newline. A mark or joiner is none of those, but the families that admit them admit
		// them as members of the run, and a prefix leaves the run empty; those families are
		// handled by trying the empty prefix first.
		//
		// The families whose run requires a letter from the start also require the character
		// after the prefix to be a letter, so a run of spaces followed by a tab cannot have
		// the tab taken as a prefix and then stop: "\t " with nothing after it is a chunk of
		// its own under the whitespace alternative, not a one-character prefix.
		if !startsLetterRun(s, fam) && !isNewlineAt(s) && !isDigitAt(s) && runeLenAt(s) > 0 {
			rest := s[runeLenAt(s):]
			if !fam.wordSpaces || letterRunStartsAt(rest) {
				return runeLenAt(s)
			}
		}
	case leadNotLetterPunctSymbol:
		if !isLetterAt(s) && !isNewlineAt(s) && !isPunctOrSymbolAt(s) && runeLenAt(s) > 0 {
			return runeLenAt(s)
		}
	}
	return 0
}

// letterRunFrom matches "\p{L}..." starting at offset start of s, returning the length of
// the whole chunk (prefix included) or false when no run begins there.
func letterRunFrom(s string, fam preRule, start int) (int, bool) {
	// The families whose run is "(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+" begin with a letter and
	// take marks only after one, so a mark at this offset is not a run. The channels that
	// spell the class "[\p{L}\p{M}]+" accept a mark here.
	if fam.wordSpaces && !isLetterAt(s[start:]) {
		return 0, false
	}
	i := start
	// prevLower records whether the previous letter was lower case, which is what the
	// boundary test needs.
	prevLower := false
	for i < len(s) {
		if fam.joiners && isJoinerAt(s[i:]) {
			// U+200C and U+200D are members of the class rather than a suffix on it, so
			// they neither end the run nor reset its case state.
			i += runeLenAt(s[i:])
			continue
		}
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
		if fam.wordSpaces && s[i] == ' ' {
			// "(?: \p{L}\p{M}*)*": a single space followed by a letter continues the run.
			// Two spaces do not, because the alternative consumes exactly one.
			rest := s[i+1:]
			if isLetterAt(rest) {
				i++
				continue
			}
		}
		break
	}
	if i == start {
		return 0, false
	}
	// (?i:'s|'t|'re|'ve|'m|'ll|'d)? -- a clitic is absorbed into the word it follows, so
	// "don't" is one piece where GPT-2 gives "don" and "'t". Only the letter alternatives
	// carry the suffix; punctuation and digit runs have no such tail.
	if fam.contractionSuffix {
		i += cliticLen(s[i:])
	}
	return i, true
}

// startsLetterRun reports whether s begins with a character the family's letter class
// accepts: a letter, and for the families that open the class, a combining mark or a
// joining control.
//
// The distinction matters at the *start* of a run. "\p{L}\p{M}*" would need a letter first,
// but the expression is "[\p{L}\p{M}]+", so a mark can begin a run: "\u0301a" is one chunk
// where a stricter reading gives the mark its own.
func startsLetterRun(s string, fam preRule) bool {
	if isLetterAt(s) {
		return true
	}
	// The families that spell the run "(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+" need a letter
	// first: there the marks are a suffix on each word rather than members of one class, so
	// a mark on its own is not a run.
	if fam.wordSpaces {
		return false
	}
	if fam.letterMarks && isMarkAt(s) {
		return true
	}
	return fam.joiners && isJoinerAt(s)
}

// letterRunStartsAt reports whether a letter-run can begin at the start of s for the
// families whose run is "(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+". Those must begin with a letter;
// the marks come after one.
func letterRunStartsAt(s string) bool { return isLetterAt(s) }

// isJoinerAt reports whether s begins with the zero-width non-joiner or joiner, the two
// controls k2-horizon includes in its letter class.
func isJoinerAt(s string) bool {
	if s == "" {
		return false
	}
	r, _ := utf8.DecodeRuneInString(s)
	return r == '\u200c' || r == '\u200d'
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
			// qwen35 spells the class "[^\s\p{L}\p{M}\p{N}]", so a combining mark ends the
			// run and the letter alternative can take it. Without the exclusion "#\u0301" is
			// one chunk where the reference gives ["#", "\u0301"].
			if fam.punctExcludesMarks && isMarkAt(s[i:]) {
				break
			}
		}
		i += n
	}
	if i == start {
		return 0
	}
	if fam.singleTail {
		// "[\r\n/]?" -- one character at most, and only when it is a newline or a slash.
		if i < len(s) && (isNewlineAt(s[i:]) || s[i] == '/') {
			i += runeLenAt(s[i:])
		}
		return i
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
	// The expression is "\s*[\r\n]+", greedy and anchored on a newline. A match therefore
	// runs to the farthest newline the leading "\s*" can reach: "\n  \n" is one chunk, while
	// "\n\t" is "\n" with the tab left to the "\s+" alternatives, because the run cannot end
	// on a tab.
	end := 0
	for i := 0; i < len(s) && isSpaceAt(s[i:]); {
		n := runeLenAt(s[i:])
		if isNewlineAt(s[i:]) {
			end = i + n
		}
		i += n
	}
	return end
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
	//
	// The last character has to be measured from its own start, not taken as the length of
	// the first one: a run may mix widths, and "a \u2003b" would otherwise be cut one byte
	// into the em space.
	if i < len(s) {
		last, ok := prevRuneStart(s, i)
		if ok && last > 0 {
			return last
		}
	}
	return i
}

// spaceTierLen matches jais-2's whitespace tail: the alternatives \s{512}(?!\S), \s{256}(?!\S),
// down to \s{4}(?!\S), \s{1,2}(?!\S), then \s{1}.
//
// The tiers are tried largest first, and each carries the same lookahead as qwen2's tail, so a
// tier only claims its run when the character after it is not a non-space. A run followed by a
// word therefore yields tiers summing to one less than its length -- the leftover space is the
// next chunk's leading character -- and a run ending the text yields tiers summing to exactly
// its length. The final \s{1} has no lookahead, so a single space always matches.
func spaceTierLen(s string) int {
	i := 0
	for i < len(s) && isSpaceAt(s[i:]) {
		i += runeLenAt(s[i:])
	}
	if i == 0 {
		return 0
	}
	// The run's length is in characters; the tiers count characters, not bytes.
	n := utf8.RuneCountInString(s[:i])
	// A run followed by a non-space gives up its last space to the following chunk, so the
	// tiers may only consume n-1. A run that ends the text consumes all of it.
	budget := n
	if i < len(s) {
		budget = n - 1
	}
	// Largest tier that fits the budget. Tiers are powers of two from 512 down to 4, then
	// {1,2}, then 1.
	for _, tier := range []int{512, 256, 128, 64, 32, 16, 8, 4, 2} {
		if budget >= tier {
			// Byte length of that many spaces.
			j, k := 0, 0
			for k < tier && j < i {
				j += runeLenAt(s[j:])
				k++
			}
			return j
		}
	}
	// A single space, which the final alternative matches without a lookahead.
	if budget >= 1 {
		j := 0
		j += runeLenAt(s[j:])
		return j
	}
	return 0
}

// Character classification, matching the Unicode categories the Qwen2 expression uses.
//
// \p{L} is any letter, \p{N} any number, and \s the whitespace set. Go's unicode package
// exposes these as tables, so the classification is the standard one rather than an
// approximation; the category names are in the comments so a reader can check them
// against the expression.

// isLetter reports whether r is a letter (\p{L}).
func isLetter(r rune) bool { return unicode.IsLetter(r) }

// isDigit reports whether r is a number (\p{N}): a decimal digit, a letter number such as
// a Roman numeral, or another numeric character such as a superscript or a vulgar fraction.
//
// The class is the whole category rather than \p{Nd}, because that is what the expressions
// write. The distinction is not academic: "\u00b2\u00b3" is two chunks under qwen2 and one
// under llama-bpe, and a decimal-digit reading gets both wrong -- the run would stay whole
// under qwen2 and would not be grouped in threes under llama-bpe.
func isDigit(r rune) bool { return unicode.IsNumber(r) }

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

// isAsianAt reports whether s begins with a character in afmoe's much wider class.
//
// The family lists fifteen ranges rather than the three the other families name: CJK
// extension A, the Kangxi radicals and compatibility ideographs, halfwidth katakana, and the
// Thai, Lao, Khmer, Myanmar, Cham, Javanese, Hangul and Hangul-jamo runs. Those scripts are
// matched by no other alternative in the expression, so without this class they would fall
// through to the letter run and group differently.
func isAsianAt(s string) bool { return classifyAt(s, isAsian) }

// asianRanges is afmoe's CJK expression as inclusive codepoint ranges, in the order the
// expression lists them.
var asianRanges = [...][2]rune{
	{0x4E00, 0x9FFF}, // CJK Unified Ideographs
	{0x3400, 0x4DBF}, // CJK Extension A
	{0x8C48, 0xFAFF}, // CJK Radicals, Kangxi Radicals, compatibility ideographs
	{0x3040, 0x309F}, // Hiragana
	{0x30A0, 0x30FF}, // Katakana
	{0xFF65, 0xFF9F}, // Halfwidth katakana
	{0x2F00, 0x2FDF}, // Kangxi Radicals
	{0x0E40, 0x0E7F}, // Thai
	{0x0E80, 0x0EFF}, // Lao
	{0x1780, 0x17FF}, // Khmer
	{0x1000, 0x109F}, // Myanmar
	{0xAA60, 0xAA7F}, // Cham
	{0xA9E0, 0xA9FF}, // Javanese
	{0xAC00, 0xD7AF}, // Hangul syllables
	{0x1100, 0x11FF}, // Hangul jamo
}

// isAsian reports whether r is in afmoe's CJK class.
//
// The ranges are listed rather than taken from unicode.Is(unicode.Han) for the same reason
// isCJK lists its three: the expression names codepoint ranges, and the scripts it covers
// are not all Han.
func isAsian(r rune) bool {
	for _, rg := range asianRanges {
		if r >= rg[0] && r <= rg[1] {
			return true
		}
	}
	return false
}

// splitAsianRuns cuts each run of afmoe's wider CJK class into its own chunk.
//
// It mirrors splitCJKRuns with the broader predicate, and the two stay separate because the
// classes are different sets rather than one containing the other.
func splitAsianRuns(text string) []string {
	var pieces []string
	start := 0
	i := 0
	for i < len(text) {
		if !isAsianAt(text[i:]) {
			i += runeLenAt(text[i:])
			continue
		}
		if i > start {
			pieces = append(pieces, text[start:i])
		}
		runStart := i
		for i < len(text) && isAsianAt(text[i:]) {
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

// isHan reports whether r is a Han ideograph (\p{Han}).
//
// kimi-k2 is the only family that names the script rather than a set of ranges: its first
// alternative is "\p{Han}+", and its letter alternatives explicitly exclude Han. That
// exclusion is the reason the family cannot reuse the cjk or asian classes -- those name
// codepoint ranges, and Han covers far more of the ideograph blocks.
func isHan(r rune) bool { return unicode.Is(unicode.Han, r) }

func isHanAt(s string) bool { return classifyAt(s, isHan) }

// splitHanRuns cuts each run of Han ideographs into its own chunk.
func splitHanRuns(text string) []string {
	var pieces []string
	start := 0
	i := 0
	for i < len(text) {
		if !isHanAt(text[i:]) {
			i += runeLenAt(text[i:])
			continue
		}
		if i > start {
			pieces = append(pieces, text[start:i])
		}
		runStart := i
		for i < len(text) && isHanAt(text[i:]) {
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

// splitNewlineRuns applies "[^\n]+|[\n]+": alternating runs of newlines and of everything
// else. A run of newlines is one chunk however long it is, and a run of anything else is one
// chunk however much punctuation or whitespace it holds -- "Hello, world!" is a single chunk
// under this rule.
func splitNewlineRuns(text string) []string {
	var out []string
	i := 0
	for i < len(text) {
		newline := text[i] == '\n'
		start := i
		for i < len(text) && (text[i] == '\n') == newline {
			i++
		}
		out = append(out, text[start:i])
	}
	return out
}

// splitDigitsFromRight cuts each digit run into groups of three counted from the end, leaving
// any remainder as the leftmost chunk. Everything that is not a digit is kept in place.
//
// This is superbpe's lookahead: the expression asserts that a multiple of three digits follow,
// which places the first boundary where the leftover digits end. Grouping from the left, as
// the other digit-taking families do, would put the remainder on the right instead.
//
// When needWordBoundary is set the expression ends in \b, as tiny_aya's does: a run whose right
// neighbour is a letter is not matched at all and stays gap text, so "123a" is one chunk where
// superbpe would cut it. The left side is unconstrained -- "a123456" still groups.
func splitDigitsFromRight(text string, needWordBoundary bool) []string {
	var out []string
	i := 0
	for i < len(text) {
		if !isDigitAt(text[i:]) {
			start := i
			for i < len(text) && !isDigitAt(text[i:]) {
				i += runeLenAt(text[i:])
			}
			out = append(out, text[start:i])
			continue
		}
		// Count the run, then walk it emitting from the right.
		j, n := i, 0
		for j < len(text) && isDigitAt(text[j:]) {
			j += runeLenAt(text[j:])
			n++
		}
		if needWordBoundary && j < len(text) {
			r, _ := utf8.DecodeRuneInString(text[j:])
			if unicode.IsLetter(r) || unicode.IsNumber(r) || r == '_' {
				out = append(out, text[i:j])
				i = j
				continue
			}
		}
		// Byte offsets of each digit so multi-byte digits are handled.
		offs := make([]int, n+1)
		offs[0] = i
		for k := 0; k < n; k++ {
			offs[k+1] = offs[k] + runeLenAt(text[offs[k]:])
		}
		// The remainder leads: n mod 3, or three if it divides evenly.
		lead := n % 3
		if lead == 0 {
			lead = 3
		}
		out = append(out, text[offs[0]:offs[lead]])
		for k := lead; k < n; k += 3 {
			out = append(out, text[offs[k]:offs[k+3]])
		}
		i = j
	}
	return out
}

// splitPartition applies a rule whose main expression is three disjoint classes: whitespace,
// word characters, and everything else. Consecutive characters of the same class form one
// chunk, and no class takes a leading character from another -- which is what separates this
// shape from the run-with-a-prefix rules most families use.
func splitPartition(text string) []string {
	kind := func(r rune) int {
		switch {
		case unicode.IsSpace(r):
			return 0
		case unicode.IsLetter(r) || unicode.IsNumber(r) || r == '_':
			return 1
		default:
			return 2
		}
	}
	var out []string
	i := 0
	for i < len(text) {
		r, n := utf8.DecodeRuneInString(text[i:])
		k := kind(r)
		start := i
		i += n
		for i < len(text) {
			r2, n2 := utf8.DecodeRuneInString(text[i:])
			if kind(r2) != k {
				break
			}
			i += n2
		}
		out = append(out, text[start:i])
	}
	return out
}

// splitZonos3 applies "\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)".
//
// The alternatives cover digits, punctuation runs of two or more, and whitespace that ends the
// text. Letters are in none of them, so a letter run is gap text and stays whole: "Hello,
// world!" is one chunk. A single punctuation mark is also too short for the {2,} run, so
// "x=42" keeps "x=" together.
func splitZonos3(text string) []string {
	var out []string
	i := 0
	gap := 0
	for i < len(text) {
		n := zonos3Len(text[i:])
		if n <= 0 {
			// Gap text: a letter. Consume one rune and keep looking, so the gap is flushed
			// as one chunk when the next alternative matches.
			i += runeLenAt(text[i:])
			continue
		}
		if i > gap {
			out = append(out, text[gap:i])
		}
		out = append(out, text[i:i+n])
		i += n
		gap = i
	}
	if gap < len(text) {
		out = append(out, text[gap:])
	}
	return out
}

// zonos3Len returns the length of the longest zonos3 alternative at the start of s, or 0.
func zonos3Len(s string) int {
	// \p{N}{1,3}
	if isDigitAt(s) {
		i, n := 0, 0
		for i < len(s) && n < 3 && isDigitAt(s[i:]) {
			i += runeLenAt(s[i:])
			n++
		}
		return i
	}
	// ?[^\s\p{L}\p{N}]{2,}[\r\n/]* -- an optional space, two or more characters that are
	// not space, letter, or digit, then any newlines or slashes.
	i := 0
	if strings.HasPrefix(s, " ") {
		i = 1
	}
	start := i
	for i < len(s) {
		if isSpaceAt(s[i:]) || isLetterAt(s[i:]) || isDigitAt(s[i:]) {
			break
		}
		i += runeLenAt(s[i:])
	}
	if i-start >= 2 {
		for i < len(s) && (isNewlineAt(s[i:]) || s[i] == '/') {
			i += runeLenAt(s[i:])
		}
		return i
	}
	// +(?!\S) -- a greedy run of plain spaces (the class is a literal space, not \s) that
	// backtracks until the next character is whitespace or the text ends. It takes all but
	// the last space of a run followed by a non-space: "a   b" matches the first two spaces
	// and leaves the third, which becomes the next chunk's leading character. A newline is
	// not a space here, so a run of newlines matches nothing.
	if s[0] == ' ' {
		j, prev := 0, 0
		for j < len(s) && s[j] == ' ' {
			prev = j
			j++
		}
		if j == len(s) {
			return j
		}
		// The run is followed by a non-space. If that neighbour is whitespace the assertion
		// already holds; otherwise give the last space back. A single space leaves nothing.
		if r, _ := utf8.DecodeRuneInString(s[j:]); unicode.IsSpace(r) {
			return j
		}
		if prev > 0 {
			return prev
		}
	}
	return 0
}

// youtuScript reports which of the four leading alternatives a rune belongs to: 1 Hangul,
// 2 CJK punctuation, 3 bopomofo, 4 CJK, 0 none. Each is a separate alternative, so a change
// of class ends a run even though every class is a script.
func youtuScript(r rune) int {
	switch {
	case r >= 0xAC00 && r <= 0xD7A3, r >= 0x3131 && r <= 0x318E:
		return 1
	case r == 0xFF01, r == 0x2026, r >= 0x201C && r <= 0x201D, r >= 0x2018 && r <= 0x2019,
		r == 0x2014, r >= 0xFF1A && r <= 0xFF1B, r == 0xFF0C, r >= 0x3001 && r <= 0x303F,
		r >= 0xFE30 && r <= 0xFE4F:
		return 2
	case r >= 0x3105 && r <= 0x312F:
		return 3
	case r >= 0x4E00 && r <= 0x9FA5, r >= 0x3040 && r <= 0x309F, r >= 0x30A0 && r <= 0x30FF:
		return 4
	}
	return 0
}

// splitYoutuScripts applies youtu's first expression: four script alternatives, each matching
// a run of its own class. Everything else is left where it is for the main rule to cut.
func splitYoutuScripts(text string) []string {
	var out []string
	start := 0
	i := 0
	for i < len(text) {
		r, n := utf8.DecodeRuneInString(text[i:])
		k := youtuScript(r)
		if k == 0 {
			i += n
			continue
		}
		// Flush the text before this run.
		if i > start {
			out = append(out, text[start:i])
		}
		runStart := i
		for i < len(text) {
			r2, n2 := utf8.DecodeRuneInString(text[i:])
			if youtuScript(r2) != k {
				break
			}
			i += n2
		}
		out = append(out, text[runStart:i])
		start = i
	}
	if start < len(text) {
		out = append(out, text[start:])
	}
	return out
}

// chameleonSpaceLen matches chameleon's "([\t\n]|    |  )" expression.
//
// It is a whole pass rather than a whitespace-tail alternative, and that difference is
// observable: the alternatives are tried at each position, eight spaces match as two
// four-space chunks, and a run of three matches two and leaves one for the next pass's
// "\s+(?!\S)". A run that ends the text is not this expression's business either -- the
// space pass leaves it alone and the final GPT-2 expression takes it.
func chameleonSpaceLen(s string) int {
	// Four spaces before two: the alternatives are ordered longest first in the expression,
	// and a four-space run would otherwise be cut as two twos.
	if strings.HasPrefix(s, "    ") {
		return 4
	}
	if strings.HasPrefix(s, "  ") {
		return 2
	}
	if s[0] == '\t' || s[0] == '\n' {
		return 1
	}
	return 0
}
