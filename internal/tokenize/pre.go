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
	if !fam.preSplitDigits {
		return splitByRule(text, fam)
	}
	// The family's first expression is a bare "\p{N}": it cuts each digit out, and the main
	// rule then runs on every piece that leaves. A digit is already alone when the main rule
	// sees it, so its " ?\p{N}+" alternative has nothing to join.
	var out []string
	for _, piece := range splitDigits(text) {
		out = append(out, splitByRule(piece, fam)...)
	}
	return out
}

// splitDigits cuts text at every digit, leaving each digit as its own piece and keeping
// the runs between them intact.
func splitDigits(text string) []string {
	var pieces []string
	start := 0
	for i := 0; i < len(text); {
		n := runeLenAt(text[i:])
		if n == 0 {
			n = 1
		}
		if isDigitAt(text[i:]) {
			if i > start {
				pieces = append(pieces, text[start:i])
			}
			pieces = append(pieces, text[i:i+n])
			start = i + n
		}
		i += n
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
	// newlineChunk is whether the expression has a "\s*[\r\n]+" alternative: a newline
	// with the spaces before it as one chunk. The GPT-2 expression has no such rule, so a
	// newline there falls through to the whitespace alternative and groups differently.
	newlineChunk bool
	// preSplitDigits is whether a bare "\p{N}" runs before the main expression, cutting
	// every digit into its own chunk regardless of what the main rule would do with it.
	//
	// llama.cpp applies a family's expressions in sequence, each cutting the previous
	// round's chunks further, so this is not a different digit count: the digit is already
	// alone before the main rule sees it. That is why a number becomes one chunk per digit
	// here while the same rule with a " ?" prefix would have kept it whole.
	preSplitDigits bool
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
		return preRule{digits: 0, lead: leadSpace, preSplitDigits: true}, true
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
	}
	if i > 0 && !isLetterAt(s[i:]) {
		return 0
	}
	start := i
	for i < len(s) {
		if isLetterAt(s[i:]) {
			i += runeLenAt(s[i:])
			continue
		}
		if fam.letterMarks && isMarkAt(s[i:]) {
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
		if n == 0 || isSpaceAt(s[i:]) || isLetterAt(s[i:]) || isDigitAt(s[i:]) || isMarkAt(s[i:]) {
			break
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
