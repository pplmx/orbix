package tokenize

import (
	"fmt"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/pplmx/orbix/internal/gguf"
)

// TokenType is the role a vocabulary entry plays.
//
// The numbering is GGUF's, and it decides behaviour: control tokens are never
// produced by encoding ordinary text, byte tokens are the fallback for
// characters the vocabulary cannot spell, and unknown tokens mark the one
// entry that represents everything else.
type TokenType int32

// Vocabulary entry roles, in GGUF's numbering.
const (
	Undefined   TokenType = 0
	Normal      TokenType = 1
	Unknown     TokenType = 2
	Control     TokenType = 3
	UserDefined TokenType = 4
	Unused      TokenType = 5
	Byte        TokenType = 6
)

// spacePrefix is the marker SentencePiece uses for a leading space (U+2581,
// "▁"). Models are trained with spaces folded into the following token, so
// " hello" is one token while "hello" may be several.
const spacePrefix = "\u2581"

// bpeBytes is GPT-2's byte-to-character table, the inverse of the escaping its
// byte-level BPE applies before training: byte b is rendered as the character with
// code points[b] so that a vocabulary can be valid UTF-8 text. Every byte outside
// the printable ASCII range, plus space, is mapped.
//
// Qwen-family models use this vocabulary instead of SentencePiece's, and the table
// is what makes them decode: space becomes U+0120 ("Ġ") and newline U+010A ("Ċ").
// A decoder that only knows the SentencePiece marker leaves every word after the
// first joined to the previous one and renders newlines as literal "Ċ", which is
// legible enough to be mistaken for a sampling problem rather than a bug.
//
// Decoding has to undo it. Without this the newline token renders as "Ċ" and the
// tab token as "ĉ", which is what a Qwen checkpoint produces out of the box: the
// text is legible enough to be mistaken for a sampling problem.
var bpeBytes = func() map[rune]byte {
	table := make(map[rune]byte, 256)
	next := 0
	for b := range 256 {
		var r rune
		if bpeKeep[b] {
			r = rune(b)
		} else {
			r = rune(256 + next)
			next++
		}
		table[r] = byte(b)
	}
	return table
}()

// bpeKeep marks the bytes GPT-2's table renders as themselves.
//
// It is not "printable ASCII except space", which is the part that is easy to get wrong.
// The original list is three ranges: '!'..'~', 0xA1..0xAC, and 0xAE..0xFF. Those last two
// are the printable Latin-1 characters, and leaving them out shifts every later byte by 94
// -- so 0xCC, which should be itself, becomes U+016E. A vocabulary trained with the right
// table then cannot find the piece at all, and the encoder drops it: "café" written with a
// combining mark lost the mark instead of reporting anything.
var bpeKeep = func() [256]bool {
	var keep [256]bool
	for b := '!'; b <= '~'; b++ {
		keep[b] = true
	}
	for b := 0xA1; b <= 0xAC; b++ {
		keep[b] = true
	}
	for b := 0xAE; b <= 0xFF; b++ {
		keep[b] = true
	}
	return keep
}()

// preType is tokenizer.ggml.pre: which pre-tokenizer the vocabulary was trained with.
//
// A byte-level BPE model does not merge over the whole text. Text is first cut into
// chunks, and a merge never crosses a boundary -- which is why a vocabulary can contain a
// rule for " "+"=" that never fires in practice. The chunking rule differs per model
// family, so the name is read rather than guessed; an unknown name falls back to no
// splitting, which is what this package did before it read the key at all.
type preType int

const (
	preNone preType = iota
	// preQwen2 is the Qwen2 family rule: contractions, then letter runs, then single
	// digits, then the remaining punctuation runs, then whitespace runs.
	preQwen2
	// preLlama3 is the same rule with digit groups of up to three. It also carries two
	// behaviours that are not part of the expression: a chunk that is itself a whole
	// vocabulary piece skips the merges, and BOS is added by default.
	preLlama3
	// preGPT2 is the original GPT-2 rule: no leading-character alternative, digits
	// grouped as a run, and a punctuation run that stops at the punctuation. A large
	// number of vocabularies name it -- olmo, mpt, jais, trillion, granite-docling -- and
	// most of them mean exactly this expression.
	preGPT2
	// preStarCoder is the GPT-2 expression preceded by a bare \p{N}, so every digit is cut
	// out before the main rule runs. llama.cpp groups starcoder, refact, command-r, smollm,
	// codeshell, exaone, minerva, and mellum2 under it.
	preStarCoder
	// preDeepseek3 is the three-expression rule deepseek-v3 uses: digit groups of up to
	// three, a CJK run, then a main rule whose letter class includes combining marks and
	// whose punctuation alternative names the categories rather than their complement.
	preDeepseek3
	// preTekken is the GPT-4o rule: a letter run cut at a lower-to-upper case boundary, single
	// digits, and a punctuation tail that absorbs slashes as well as newlines.
	preTekken
	// preWhitespace is \S+: the whole rule is where runs of non-space characters end.
	preWhitespace
	// preBloom is " ?[^(\s|.,!?…。，、।۔،)]+": an optional space, then a run that stops at
	// those characters.
	preBloom
	// preDeepseekCoder applies five expressions in order: newline, \s?letters,
	// \s?punctuation, a CJK run, then a single digit.
	preDeepseekCoder
	// preFalcon applies a punctuation-and-symbol run, the GPT-2 expression, then groups of
	// three digits.
	preFalcon
	// preGPT4O is tekken's shape with one-to-three digit groups and an optional clitic
	// suffix on the letter alternatives.
	preGPT4O
	// preDeepseekLLM is deepseek-llm's five-expression rule. It looks like deepseek-coder's
	// but differs in every term that matters: the letter class is spelled out rather than
	// written \p{L}, the punctuation is a range list rather than \p{P}, and numbers group as
	// a run rather than singly.
	preDeepseekLLM
	// preMMBert is "[^\n]+|[\n]+": the text is cut into runs, separated only by whether the
	// run is newlines or not. Everything else -- spaces, punctuation, words -- stays in one
	// chunk, which is why this family splits so much less than the others.
	preMMBert
	// preViking is BLOOM's stop-set run followed by a single-digit cut: the same first
	// expression, then every digit becomes a chunk of its own.
	preViking
	// preSuperBPE is "\p{N}+" then a zero-width (?=(\d{3})+(?!\d)): digit runs are cut from
	// the right into threes, so a remainder stays at the left. "1234" is ["1", "234"], not
	// ["123", "4"].
	preSuperBPE
	// preSoprano cuts each digit out first, then groups the rest into three mutually
	// exclusive classes: whitespace, word characters (letters, digits, underscore), and
	// everything else. It is the only family whose main expression is a partition rather than
	// a run with a leading character.
	preSoprano
	// preBailingMoE is llama3's expression with a bare \p{N} where llama3 has \p{N}{1,3}, so
	// every digit is its own chunk. The newline alternatives differ on paper -- \s*[\r\n]
	// against \s*[\r\n]+ -- but both take the run in one chunk on real text.
	preBailingMoE
	// preTinyAya is a \d{1,3} lookahead that groups digits three at a time from the right and
	// keeps a run touching a letter whole, followed by a case-splitting letter rule.
	preTinyAya
	// preLaguna is a newline pass followed by bailingmoe's expression. The pass is not
	// redundant: it separates a newline from whitespace before it, which the main rule would
	// otherwise take into one chunk -- " x \n y " splits as [" x", " ", "\n", " y", " "] here
	// and [" x", " \n", " y", " "] without it.
	preLaguna
	// preFlint cuts digit groups and the CJK block first, then applies a case-splitting letter
	// rule, then a punctuation run named by category.
	preFlint
	// preUfakzeka is qwen2's expression without the contraction alternative, so a Turkish
	// apostrophe suffix stays whole: "İstanbul'da" is ["İstanbul", "'da"] where bailingmoe's
	// contraction group matches "'d" and leaves "a" behind.
	preUfakzeka
	// preQwen35 is qwen2 with combining marks in the letter class, so a base letter and the
	// marks after it stay together: "e" followed by a combining acute is one chunk.
	preQwen35
	// preSeedCoder is qwen2's expression with two changes to the punctuation alternative:
	// it excludes newlines from the run, and it has no newline tail. Newlines reach the
	// "\s*[\r\n]+" alternative instead, so a run of them is still one chunk -- but a
	// punctuation run no longer absorbs the newlines after it.
	preSeedCoder
	// preMiniCPM5 cuts digit groups of up to three first, then applies llama3's expression with
	// a digit run where llama3 has \p{N}{1,3}. The first pass leaves only remainders for the
	// second, so the digit alternative there is a plain run.
	preMiniCPM5
	// preZonos3 cuts only digits, runs of two or more punctuation, and trailing whitespace.
	// Letters are in no alternative, so a letter run is gap text -- "Hello, world!" is one
	// chunk, and "x=42" keeps "x=" together because a single "=" is shorter than the
	// punctuation run's {2,}.
	preZonos3
	// preJais2 is llama3's expression with a whitespace tail built from power-of-two tiers
	// instead of \s+(?!\S). A run of n spaces is cut greedily into tiers until one is left,
	// which the following letter or punctuation chunk absorbs as its leading character.
	preJais2
	// preExaoneMoE is llama3's shape whose letter alternative continues past a single space:
	// "(?:\p{L}\p{M}*(?: \p{L}\p{M}*)*)+", so "hi there" is one chunk and "a  b" is not.
	preExaoneMoE
	// preTekkenNum3 is tekken's expression with its digit alternative replaced by a lookahead
	// that groups three at a time from the right, so "1234" is ["1", "234"] where tekken gives
	// one digit per chunk.
	preTekkenNum3
	// preYoutu cuts four scripts out first -- Hangul, CJK punctuation, bopomofo, and CJK --
	// then applies a case-splitting letter rule to what remains.
	preYoutu
	// preSpark25 cuts digit groups and the CJK block first, then a letter rule that does not
	// split on case, then a bare \p{N} that cuts every remaining digit on its own -- so a
	// number ends up one digit per chunk despite the first pass grouping in threes.
	preSpark25
	// preChameleon applies six expressions in sequence rather than as alternatives of one
	// rule. Its punctuation pass cuts single characters ahead of the main rule, so an
	// apostrophe is a chunk of its own and no contraction forms; its space pass gives two and
	// four spaces their own chunks; and its digit pass runs over the whole \p{N} category.
	preChameleon
	// preK2Horizon is qwen35's shape whose letter run also takes the joining controls
	// U+200C and U+200D, with digits grouped up to three rather than one at a time.
	preK2Horizon
	// preAfMoE cuts three-at-a-time digit groups from the right, then a wide class of Asian
	// scripts, then deepseek-v3's main expression.
	preAfMoE
	// preKimiK2 cuts Han runs first, then applies letter alternatives that exclude Han and
	// split on a lower-to-upper case boundary.
	preKimiK2
)

// bpeRunes is the encoding direction of the GPT-2 byte table: one escaped rune per byte.
//
// Both directions are needed and they are not the same map. Decoding walks a piece and
// turns each escaped rune back into its byte; encoding walks the input bytes and turns
// each into the rune the vocabulary spells it with. Encoding only ever spelled the space,
// so a newline or a tab reached the merge step as itself -- a rune no piece contains --
// and the character either vanished from the token stream or split the word around it.
var bpeRunes = func() [256]rune {
	var table [256]rune
	next := 0
	for b := range 256 {
		if bpeKeep[b] {
			table[b] = rune(b)
		} else {
			table[b] = rune(256 + next)
			next++
		}
	}
	return table
}()

// bpeEscape renders text as the vocabulary spells it: one rune per byte, with every
// byte outside printable ASCII replaced by its assigned code point.
func bpeEscape(text string) string {
	var b strings.Builder
	b.Grow(len(text))
	for i := range len(text) {
		b.WriteRune(bpeRunes[text[i]])
	}
	return b.String()
}

// unknownSurface is how SentencePiece renders an unknown token on decode.
const unknownSurface = " \u2047 "

// Tokenizer is a SentencePiece-style vocabulary.
type Tokenizer struct {
	tokens []string
	scores []float32
	types  []TokenType

	// index maps a piece to its ID, for the merge search.
	index map[string]int32

	// byteID maps a byte value to the ID of its <0xNN> token, or -1 when the
	// vocabulary has no byte token for that value.
	byteID [256]int32

	unknownID int32
	bosID     int32
	eosID     int32

	// addSpacePrefix prepends a space to encoded text. SentencePiece models
	// are trained on text that starts with a space, and some GGUF files state
	// this explicitly with tokenizer.ggml.add_space_prefix.
	addSpacePrefix bool

	// specials lists the control and user-defined pieces, longest first, so
	// encoding can take whole markers instead of spelling them out.
	//
	// A chat template is not text: "<|im_start|>" is one token the model was
	// trained to see, and encoding it as the six ordinary pieces it happens to be
	// spelled from gives the model a prompt it has never seen -- it then answers
	// as though the marker were prose, re-stating the system prompt and never
	// emitting its end-of-turn token. Matching longest first matters because
	// markers share prefixes.

	// pre is which pre-tokenizer the vocabulary was trained with, or preNone when the
	// model carries no pre-tokenizer name.
	pre preType

	// ignoreMerges takes a chunk that is already one vocabulary piece as that piece,
	// skipping the merge loop. Some pre-tokenizers imply it; see preLlama3.
	ignoreMerges bool

	// preName is tokenizer.ggml.pre verbatim, kept so a caller can report which family the
	// file asked for. The mapping below is a closed list, and a file naming a family this
	// package does not implement would otherwise be indistinguishable from one that named
	// nothing at all -- including from a typo in the key.
	preName string

	// addBOS is whether Encode prepends the beginning-of-sequence token for ordinary
	// text.
	//
	// Its default is true, which is what this package has always done for a file that
	// names a BOS token: every GGUF llama.cpp writes is trained to expect one, and a
	// file that does not want it says so. tokenizer.ggml.add_bos_token overrides it in
	// both directions.
	//
	// It is a pointer rather than a bool because the zero value of a bool cannot mean
	// "true": a Tokenizer built as a struct literal -- which the tests do -- would
	// silently lose the default.
	addBOS *bool

	// merges maps a pair of pieces to its rank in the vocabulary's merge list, for a
	// byte-level BPE vocabulary that ships one. Rank order is the definition of the
	// tokenization, not a preference: a model with 151387 rules cannot be reproduced by
	// any scoring heuristic over its pieces, and a wrong application of them produces
	// valid token ids for text the model was never trained on.
	merges map[string]int

	// byteLevel reports that the vocabulary is GPT-2 byte-level BPE rather than
	// SentencePiece, which changes how a space is represented before merging.
	//
	// SentencePiece folds a space into a U+2581 marker on the following piece, so
	// " a" is one symbol. Byte-level BPE escapes the space as the character 'Ġ'
	// and keeps it as its own symbol, so " a" is two symbols that the merge can
	// join into a "Ġa" piece or leave apart. Using the wrong one does not fail:
	// the encoder produces valid ids for every word and silently drops every
	// space, which is the bug this field exists to prevent.
	byteLevel bool

	// specials lists the control and user-defined pieces, longest first, so
	// encoding can take whole markers instead of spelling them out.
	//
	// A chat template is not text: "<|im_start|>" is one token the model was
	// trained to see, and encoding it as the six ordinary pieces it happens to be
	// spelled from gives the model a prompt it has never seen -- it then answers
	// as though the marker were prose, re-stating the system prompt and never
	// emitting its end-of-turn token. Matching longest first matters because
	// markers share prefixes.
	specials []specialToken
}

// specialToken is one piece that must be matched as a unit.
type specialToken struct {
	text string
	id   int32
}

// New builds a Tokenizer from a parsed GGUF file.
func New(f *gguf.File) (*Tokenizer, error) {
	tokens, ok := f.Strings("tokenizer.ggml.tokens")
	if !ok {
		return nil, fmt.Errorf("tokenize: missing tokenizer.ggml.tokens")
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("tokenize: empty vocabulary")
	}

	t := &Tokenizer{
		tokens: tokens,
		index:  make(map[string]int32, len(tokens)),
		// Defaults are -1 so a model that omits an ID does not silently point
		// at token 0, which is usually <unk> and would corrupt output.
		unknownID: -1,
		bosID:     -1,
		eosID:     -1,
		byteID:    fillNeg1(),
	}

	for i, tok := range tokens {
		t.index[tok] = int32(i)
	}
	// Scores are optional: a vocabulary without them still encodes, using
	// insertion order as the tie-break, which is what llama.cpp does.
	if scores, ok := f.Float32s("tokenizer.ggml.scores"); ok {
		if len(scores) != len(tokens) {
			return nil, fmt.Errorf("tokenize: %d scores for %d tokens", len(scores), len(tokens))
		}
		t.scores = scores
	} else {
		t.scores = make([]float32, len(tokens))
	}

	if types, ok := f.Int32s("tokenizer.ggml.token_type"); ok {
		if len(types) != len(tokens) {
			return nil, fmt.Errorf("tokenize: %d token types for %d tokens", len(types), len(tokens))
		}
		t.types = make([]TokenType, len(types))
		for i, v := range types {
			t.types[i] = TokenType(v)
		}
	} else {
		t.types = make([]TokenType, len(tokens))
		for i := range t.types {
			t.types[i] = Normal
		}
	}

	// Index the byte tokens so encoding can fall back to them.
	for i, tok := range tokens {
		if b, ok := parseByteToken(tok); ok {
			t.byteID[b] = int32(i)
		}
	}

	// Index the control and user-defined pieces so encoding can match them whole.
	// User-defined is included because that is what a chat template's markers are in
	// most checkpoints; control covers the ones a model marks as its own.
	for i, tok := range tokens {
		if t.types[i] == Control || t.types[i] == UserDefined {
			t.specials = append(t.specials, specialToken{text: tok, id: int32(i)})
		}
	}
	// Longest first: "<|im_start|>" and "<|im_end|>" share a prefix, so a shortest-first
	// scan would match the shorter marker inside the longer one.
	slices.SortStableFunc(t.specials, func(a, b specialToken) int {
		return len(b.text) - len(a.text)
	})

	if id, ok := f.Uint32("tokenizer.ggml.unknown_token_id"); ok {
		t.unknownID = int32(id)
	}
	if id, ok := f.Uint32("tokenizer.ggml.bos_token_id"); ok {
		t.bosID = int32(id)
	}
	// The family default is the pre-tokenizer's, so the file's explicit answer is read
	// after it. A false here must not only stop the default; it has to suppress a BOS the
	// family would otherwise have added.
	if v, ok := f.Get[bool]("tokenizer.ggml.add_bos_token"); ok {
		t.addBOS = &v
	}
	if id, ok := f.Uint32("tokenizer.ggml.eos_token_id"); ok {
		t.eosID = int32(id)
	}
	// The vocab family decides both the space marker and whether a leading space is
	// added at all, so it is read before either is decided.
	if model, ok := f.Get[string]("tokenizer.ggml.model"); ok {
		switch model {
		case "gpt2":
			t.byteLevel = true
		case "llama":
			// SentencePiece; the default below applies.
		default:
			return nil, fmt.Errorf("tokenize: unsupported tokenizer.ggml.model %q (want llama or gpt2)", model)
		}
	}

	if name, ok := f.Get[string]("tokenizer.ggml.pre"); ok {
		t.preName = name
		switch name {
		case "qwen2", "deepseek-r1-qwen", "kormo", "f2llmv2", "hunyuan", "solar-open",
			"megrez",
			// llama.cpp puts stablelm2 in the Qwen2 branch: same expression, so the
			// only difference from the names above is the spelling.
			"stablelm2":
			t.pre = preQwen2
		case "gpt-2", "phi-2", "jina-es", "jina-de", "gigachat", "jina-v2-es",
			"jina-v2-de", "a.x-4.0", "mellum", "modern-bert", "olmo", "mpt", "jais",
			"trillion", "granite-docling",
			// llama.cpp puts these on GPT2 as well. roberta-bpe, jina-v1-en and
			// jina-v2-code additionally set add_sep, which is a caller-facing flag for
			// joining sequences and not part of the chunk boundaries; exaone4 sets
			// clean_spaces, which only affects how decoded text is tidied. Neither
			// changes where a chunk ends, so the chunking aliases are exact.
			"roberta-bpe", "jina-v1-en", "jina-v2-code", "exaone4":
			t.pre = preGPT2
		case "starcoder", "refact", "command-r", "smollm", "codeshell", "exaone",
			"minerva-7b", "mellum2":
			t.pre = preStarCoder
		case "deepseek-v3", "hunyuan-dense", "joyai-llm", "hy_v4":
			t.pre = preDeepseek3
		case "tekken", "nemotron-3.5-embed", "pixtral":
			t.pre = preTekken
		case "granite-embed-multi-97m", "gpt-4o", "llama4", "inkling2", "kanana2", "talkie", "minimax-m2":
			t.pre = preGPT4O
		// dbrx and smaug-bpe share llama3's expression verbatim: llama.cpp's DBRX and SMAUG
		// cases fall through to the same regex, and the source says so in a comment. What
		// differs in llama.cpp is "clean_up_tokenization_spaces", a decoding-side setting
		// this package does not model, so the split is the same.
		// chatglm-bpe and glm5 are in llama.cpp's CHATGLM4 branch, whose expression is
		// llama3's verbatim. The branch differs only in settings this package does not model
		// -- a null BOS id, ignore_merges, and clean_up_tokenization_spaces.
		case "chatglm-bpe", "glm4", "glm5",
			"dbrx", "smaug-bpe",
			"llama-bpe", "llama3", "llama-v3", "falcon3", "falcon-h1", "midm-2.0", "lfm2",
			// jina-v5-nano is in llama.cpp's llama3 branch, ignore_merges and all.
			"jina-v5-nano":
			t.pre = preLlama3
			// A whole chunk that the vocabulary already spells as one piece is taken as
			// that piece without merging. Without it a word the vocabulary contains can
			// still come out as several tokens, because the merge loop has no reason to
			// prefer the single piece.
			t.ignoreMerges = true

		case "whitespace":
			t.pre = preWhitespace
		case "bloom", "gpt3-finnish", "poro-chat":
			t.pre = preBloom
		// bailingmoe, bailingmoe2 and llada-moe are llama.cpp's BAILINGMOE branch.
		// grok-2 differs from bailingmoe only in writing \s*[\r\n]+ where bailingmoe has
		// \s*[\r\n]. Both consume the whole run -- checked over seven newline inputs,
		// including CRLF pairs -- so the rules are the same.
		case "grok-2",
			"bailingmoe", "bailingmoe2", "llada-moe":
			t.pre = preBailingMoE
		// cohere2moe and tiny_aya are llama.cpp's TINY_AYA branch.
		case "cohere2moe", "tiny_aya":
			t.pre = preTinyAya
		// laguna runs bailingmoe's expression after a newline pass. The newline pass separates
		// runs of newlines from everything else, which cannot change what the main rule does
		// to a chunk that already contains no newline.
		case "exaone-moe":
			t.pre = preExaoneMoE
		case "tekken-num3":
			t.pre = preTekkenNum3
		case "youtu":
			t.pre = preYoutu
		case "spark2_5":
			t.pre = preSpark25
		case "chameleon":
			t.pre = preChameleon
		case "k2-horizon":
			t.pre = preK2Horizon
		case "afmoe":
			t.pre = preAfMoE
		case "kimi-k2":
			t.pre = preKimiK2
		case "jais-2":
			t.pre = preJais2
		case "zonos3":
			t.pre = preZonos3
		case "minicpm5":
			t.pre = preMiniCPM5
		case "seed-coder":
			t.pre = preSeedCoder
		case "qwen35":
			t.pre = preQwen35
		case "ufakzeka":
			t.pre = preUfakzeka
		case "flint":
			t.pre = preFlint
		case "laguna":
			t.pre = preLaguna
		case "soprano":
			t.pre = preSoprano
		case "superbpe":
			t.pre = preSuperBPE
		case "viking":
			t.pre = preViking
		// gemma4 and granite-embed-multi-311m are in llama.cpp's GEMMA4 branch, whose
		// expression is "[^\n]+|[\n]+" -- the same one mmbert uses. The comment there says
		// the model relies on BPE over the whole text and only needs newlines split because the
		// merge lookup asserts no newline inside a token.
		case "sarvam-moe", "domyn-small", "gemma4", "granite-embed-multi-311m", "mmbert":
			t.pre = preMMBert
		case "deepseek-llm":
			t.pre = preDeepseekLLM
		case "deepseek-coder":
			t.pre = preDeepseekCoder
		case "falcon":
			t.pre = preFalcon

		case "", "default":
			t.pre = preNone
		default:
			// An unknown pre-tokenizer is not a reason to refuse the file: the merge list
			// still tokenizes, just without the chunk boundaries. Reporting it would be
			// better than silence, but this path has no logger, so the choice is
			// documented here and in README.
			t.pre = preNone
		}
	}

	if raw, ok := f.GetArray[string]("tokenizer.ggml.merges"); ok && len(raw) > 0 {
		// Each entry is "left right". The pieces can themselves contain spaces -- "Ġ Ġ"
		// is the real first merge of a byte-level vocabulary -- so the split is on the
		// first space: the left piece is an escaped single character or a merged run with
		// no space of its own until the separator.
		t.merges = make(map[string]int, len(raw))
		for rank, entry := range raw {
			left, right, found := strings.Cut(entry, " ")
			if !found || left == "" || right == "" {
				// Reported rather than skipped: a vocabulary missing one rule tokenizes
				// differently, and ignoring the entry would make the model produce subtly
				// wrong text with no error anywhere.
				return nil, fmt.Errorf("tokenize: merge %d is not a pair: %q", rank, entry)
			}
			t.merges[left+"\x00"+right] = rank
		}
	}

	if v, ok := f.Lookup("tokenizer.ggml.add_space_prefix"); ok && v.Type == gguf.TypeBool {
		t.addSpacePrefix = v.Bool
	} else {
		// SentencePiece always prepends a space during training, and a model
		// that omits the key is following that convention. A byte-level BPE
		// vocabulary does not: text arrives exactly as written, and the space is
		// part of the text rather than a training-time convention.
		t.addSpacePrefix = !t.byteLevel
	}

	return t, nil
}

func fillNeg1() (out [256]int32) {
	for i := range out {
		out[i] = -1
	}
	return out
}

// VocabSize returns the number of entries in the vocabulary.
func (t *Tokenizer) VocabSize() int { return len(t.tokens) }

// UnknownID returns the ID of the unknown token, or -1 when the model has
// none.
func (t *Tokenizer) UnknownID() int32 { return t.unknownID }

// BOSID returns the beginning-of-sequence token ID, or -1 when unset.
func (t *Tokenizer) BOSID() int32 { return t.bosID }

// PreName returns tokenizer.ggml.pre verbatim, or "" when the file names no pre-tokenizer.
func (t *Tokenizer) PreName() string { return t.preName }

// PreImplemented reports whether this package reproduces the file's pre-tokenizer.
//
// It is false in three cases that all mean "no chunking here": the file names nothing, the
// file names a family this package implements nothing for, or the name is not a family at
// all. The distinction matters to a caller only in that a *known* family left unimplemented
// silently tokenizes differently from the model's training data -- text the model never saw,
// with nothing to report it. PreKnown separates that case from a typo.
func (t *Tokenizer) PreImplemented() bool {
	// "default" and an absent key both mean llama.cpp applies no regex to this file, which
	// is exactly what this package does; both are implemented by not splitting.
	if t.preName == "" || t.preName == "default" {
		return true
	}
	_, ok := preFamilyOf(t.pre)
	return ok
}

// PreKnown reports whether the file's pre-tokenizer name is one llama.cpp recognises, so a
// caller can tell "this family is not implemented yet" from "this is not a family name".
func (t *Tokenizer) PreKnown() bool {
	if t.preName == "" {
		return true
	}
	if _, ok := preFamilyOf(t.pre); ok {
		return true
	}
	return slices.Contains(knownPreNames, t.preName)
}

// addsBOS reports whether Encode prepends BOS for ordinary text. An unset field means the
// default, which is yes.
func (t *Tokenizer) addsBOS() bool {
	return t.addBOS == nil || *t.addBOS
}

// EOSID returns the end-of-sequence token ID, or -1 when unset.
func (t *Tokenizer) EOSID() int32 { return t.eosID }

// Token returns the piece stored at id.
func (t *Tokenizer) Token(id int32) (string, bool) {
	if id < 0 || int(id) >= len(t.tokens) {
		return "", false
	}
	return t.tokens[id], true
}

// TokenType returns the role of the token at id.
func (t *Tokenizer) TokenType(id int32) TokenType {
	if id < 0 || int(id) >= len(t.types) {
		return Undefined
	}
	return t.types[id]
}

// symbol is one span of the normalized text during segmentation.
//
// start and len delimit the text slice the symbol covers; prev and next form
// a doubly linked list over the symbols so a merge can unlink a neighbour in
// constant time. len == 0 marks a symbol that has been merged away, which is
// how a stale queue entry is detected.
type symbol struct {
	start, len int
	prev, next int
}

// bigram is a merge candidate: two adjacent symbols and the score of the piece
// they would form.
type bigram struct {
	left, right int
	// rank orders the merge. With a merge list, lower rank is applied first, which is
	// what the list defines. Without one, the vocabulary's own score is used and the
	// direction is reversed by the queue, because a unigram vocabulary scores longer
	// pieces higher.
	rank  int
	score float32
	// byRank reports which of the two orderings this entry uses, so a vocabulary with a
	// merge list cannot be sorted by a score it does not define.
	byRank bool
	size   int // combined byte length, used to detect stale entries
}

// bigramQueue is a max-heap on (score, then leftmost first).
//
// The tie-break on the left index is not cosmetic: it makes the resulting
// segmentation deterministic when several pairs score equally, and it matches
// the reference implementation's comparator exactly, which is what lets a
// GGUF model produce the same tokens llama.cpp would.
type bigramQueue []bigram

func (q *bigramQueue) push(b bigram) {
	*q = append(*q, b)
	i := len(*q) - 1
	for i > 0 {
		parent := (i - 1) / 2
		if !q.less(i, parent) {
			break
		}
		(*q)[i], (*q)[parent] = (*q)[parent], (*q)[i]
		i = parent
	}
}

func (q *bigramQueue) pop() (bigram, bool) {
	old := *q
	if len(old) == 0 {
		return bigram{}, false
	}
	top := old[0]
	last := len(old) - 1
	old[0] = old[last]
	*q = old[:last]
	if len(*q) > 0 {
		i := 0
		for {
			l, r := 2*i+1, 2*i+2
			best := i
			if l < len(*q) && q.less(l, best) {
				best = l
			}
			if r < len(*q) && q.less(r, best) {
				best = r
			}
			if best == i {
				break
			}
			(*q)[i], (*q)[best] = (*q)[best], (*q)[i]
			i = best
		}
	}
	return top, true
}

// less reports whether entry i outranks entry j, i.e. whether i should come out
// of the heap first.
//
// With a merge list the lower rank wins: the list is ordered by when the merge was
// learned, and applying a later rule before an earlier one tokenizes text the model never
// saw. Without one, the vocabulary's own score decides and the larger score wins, which
// is what a unigram vocabulary means by it. Ties go to the leftmost pair, which is what
// makes the result deterministic rather than heap-order dependent.
func (q bigramQueue) less(i, j int) bool {
	if q[i].byRank != q[j].byRank {
		// Two orderings in one heap would be meaningless; it cannot happen because
		// byRank comes from the tokenizer, but ordering by rank first keeps the
		// comparison total if it ever does.
		return q[i].byRank
	}
	if q[i].byRank {
		if q[i].rank != q[j].rank {
			return q[i].rank < q[j].rank
		}
		return q[i].left < q[j].left
	}
	if q[i].score != q[j].score {
		return q[i].score > q[j].score
	}
	return q[i].left < q[j].left
}

// Encode converts text into token IDs.
//
// When addSpecial is true and the model defines a beginning-of-sequence
// token, it is prepended. Ordinary text never produces control tokens: a
// token like "<s>" typed by a user is encoded as characters, not as the
// control token, which is what a chat template expects.
func (t *Tokenizer) Encode(text string, addSpecial bool) []int32 {
	var out []int32
	if addSpecial && t.addsBOS() && t.bosID >= 0 {
		out = append(out, t.bosID)
	}
	out = append(out, t.encodeText(text)...)
	return out
}

// EncodeWithSpecials encodes text, recognising the vocabulary's control and
// user-defined pieces wherever they appear.
//
// This is the entry point for a rendered chat template. Encode treats text as text, so
// a marker like "<|im_start|>" becomes the characters it is spelled from -- correct for
// a user who typed those characters, wrong for a template that means the token. The two
// cannot be one function: only the caller knows whether a marker is markup or prose.
func (t *Tokenizer) EncodeWithSpecials(text string, addSpecial bool) []int32 {
	var out []int32
	if addSpecial && t.addsBOS() && t.bosID >= 0 {
		out = append(out, t.bosID)
	}
	if len(t.specials) == 0 {
		return append(out, t.encodeText(text)...)
	}

	// Split the text at each special piece and encode the gaps as ordinary text, so a
	// marker in the middle of a string does not disturb the pieces around it. The
	// segment boundaries are what keeps SentencePiece's leading-space rule intact: an
	// empty gap between two adjacent markers produces no tokens at all.
	// The dummy space prefix is added once, to the piece at the start of the string. A gap
	// after a marker is not the start of the sequence, and adding a prefix there is what
	// turned "the<|x|> garden" into "the  garden": the space in the text became a second
	// leading space marker.
	first := true
	start := 0
	for i := 0; i < len(text); {
		if sp, ok := t.matchSpecial(text[i:]); ok {
			if start < i {
				out = append(out, t.encodeTextPart(text[start:i], first)...)
				first = false
			}
			out = append(out, sp.id)
			i += len(sp.text)
			start = i
			continue
		}
		_, size := utf8.DecodeRuneInString(text[i:])
		if size <= 0 {
			size = 1
		}
		i += size
	}
	if start < len(text) {
		out = append(out, t.encodeTextPart(text[start:], first)...)
	}
	return out
}

// matchSpecial returns the special piece text starts with, if any.
//
// The scan is linear because the list is short (a handful of markers per vocabulary)
// and text is searched once per encoded string; an index over first bytes would save
// little and hide the prefix rule.
func (t *Tokenizer) matchSpecial(text string) (specialToken, bool) {
	for _, sp := range t.specials {
		if sp.text != "" && strings.HasPrefix(text, sp.text) {
			return sp, true
		}
	}
	return specialToken{}, false
}

// encodeText is the SentencePiece segmentation of one string.
//
// It follows llama.cpp's llm_tokenizer_spm_session: seed a priority queue with
// every adjacent pair that is itself a vocabulary piece, then repeatedly merge
// the highest-scoring pair, pushing the two new pairs the merge creates. The
// score is the piece's unigram log-probability, so a long, likely piece wins
// over a sequence of shorter ones.
func (t *Tokenizer) encodeText(text string) []int32 {
	return t.encodeTextPart(text, true)
}

// encodeTextPart is encodeText with control over the dummy space prefix.
//
// The prefix belongs to the sequence, not to each piece of it. EncodeWithSpecials cuts a
// string at its special markers and encodes the gaps separately, so passing first=false for
// a gap that follows a marker is what keeps a leading space from being doubled: the prefix a
// real tokenizer adds once has already been accounted for by the gap before, or by the
// marker itself having consumed the start of the text.
func (t *Tokenizer) encodeTextPart(text string, first bool) []int32 {
	if text == "" {
		return nil
	}
	if t.addSpacePrefix && first {
		text = " " + text
	}
	// Represent each space the way the vocabulary does before merging. The two
	// families differ in when the space becomes part of a piece: SentencePiece
	// folds it into the marker on the following character, so " a" is one symbol,
	// while byte-level BPE escapes it as its own symbol and lets the merge decide
	// whether "Ġa" exists as a piece.
	if t.byteLevel {
		// Every byte, not just the space. The vocabulary is UTF-8 text whose pieces are
		// spelled with the escaped runes, so a byte left as itself is a rune no piece
		// contains: it cannot merge, and the character is lost or the word is split
		// around it.
		var out []int32
		for _, chunk := range t.preSplit(text) {
			escaped := bpeEscape(chunk)
			// A chunk the vocabulary spells as one piece is emitted as that piece. The
			// merge loop would usually find it anyway, but not always: a shorter merge
			// with a better rank can win first and leave a split the model was not
			// trained to see.
			if t.ignoreMerges {
				if id, ok := t.index[escaped]; ok {
					out = append(out, id)
					continue
				}
			}
			out = append(out, t.encodeChunk(escaped)...)
		}
		return out
	} else {
		text = strings.ReplaceAll(text, " ", spacePrefix)
	}
	return t.encodeChunk(text)
}

// encodeChunk runs the merge loop over one already-escaped run of text.
//
// Pre-tokenization and escaping happen in encodeText; this is the part both families
// share, and it must not merge across a chunk boundary, which is why the boundary is
// drawn by the caller.
func (t *Tokenizer) encodeChunk(text string) []int32 {
	symbols := make([]symbol, 0, len(text))
	for off := 0; off < len(text); {
		_, size := utf8.DecodeRuneInString(text[off:])
		i := len(symbols)
		symbols = append(symbols, symbol{start: off, len: size, prev: i - 1, next: -1})
		if i > 0 {
			symbols[i-1].next = i
		}
		off += size
	}

	// rev records how a merged span was formed, so a span that is not itself a
	// token can be split back into the two pieces it came from. A single left
	// reduction would lose the intermediate pieces -- e.g. "abcdef" formed from
	// "abc"+"def" where neither "abc" nor "def" is a whole token.
	rev := make(map[string][2]string, len(symbols))

	var queue bigramQueue
	tryAddBigram := func(left, right int) {
		if left < 0 || right < 0 {
			return
		}
		span := text[symbols[left].start : symbols[right].start+symbols[right].len]
		id, ok := t.index[span]
		if !ok {
			return
		}
		entry := bigram{left: left, right: right, size: len(span)}
		if t.merges != nil {
			// The pair is keyed by the two pieces it merges, which is what the list
			// records. A pair absent from the list never merges, whatever its score.
			key := text[symbols[left].start : symbols[left].start+symbols[left].len]
			key += "\x00"
			key += text[symbols[right].start : symbols[right].start+symbols[right].len]
			rank, ok := t.merges[key]
			if !ok {
				return
			}
			entry.rank, entry.byRank = rank, true
		} else {
			entry.score = t.scores[id]
		}
		queue.push(entry)
		rev[span] = [2]string{
			text[symbols[left].start : symbols[left].start+symbols[left].len],
			text[symbols[right].start : symbols[right].start+symbols[right].len],
		}
	}

	for i := 1; i < len(symbols); i++ {
		tryAddBigram(i-1, i)
	}

	for {
		b, ok := queue.pop()
		if !ok {
			break
		}
		l, r := &symbols[b.left], &symbols[b.right]
		// A stale entry: one side was already merged into something else, or
		// the pair no longer spans what it did when it was queued.
		if l.len == 0 || r.len == 0 || l.len+r.len != b.size {
			continue
		}
		l.len += r.len
		r.len = 0
		l.next = r.next
		if r.next >= 0 {
			symbols[r.next].prev = b.left
		}
		tryAddBigram(l.prev, b.left)
		tryAddBigram(b.left, l.next)
	}

	var out []int32
	for i := 0; i >= 0 && i < len(symbols); i = symbols[i].next {
		span := text[symbols[i].start : symbols[i].start+symbols[i].len]
		out = t.resegment(span, rev, out)
	}
	return out
}

// resegment turns one merged span into tokens.
//
// A span that is a vocabulary piece becomes that token. Otherwise it is split
// back into the pair that formed it and each half is examined. A span with no
// recorded split is emitted as byte tokens, which is the fallback for
// characters the vocabulary cannot spell.
func (t *Tokenizer) resegment(span string, rev map[string][2]string, out []int32) []int32 {
	todo := []string{span}
	for len(todo) > 0 {
		cur := todo[len(todo)-1]
		todo = todo[:len(todo)-1]

		if id, ok := t.index[cur]; ok {
			out = append(out, id)
			continue
		}
		if halves, ok := rev[cur]; ok {
			// Push in reverse so the left half is processed first and the
			// output order matches the text.
			todo = append(todo, halves[1], halves[0])
			continue
		}
		out = t.encodeBytes(cur, out)
	}
	return out
}

// encodeBytes appends the byte-token sequence for a piece the vocabulary
// cannot represent directly.
func (t *Tokenizer) encodeBytes(piece string, out []int32) []int32 {
	for i := range len(piece) {
		if id := t.byteID[piece[i]]; id >= 0 {
			out = append(out, id)
		} else if t.unknownID >= 0 {
			out = append(out, t.unknownID)
		}
		// With neither a byte token nor an unknown token there is nothing
		// honest to emit, so the byte is dropped rather than guessed at.
	}
	return out
}

// Decode converts token IDs back into text.
//
// The piece-to-text mapping mirrors SentencePiece's decoder exactly, because
// a mismatch shows up as a stray leading space or a lost character in
// generated output:
//
//   - the word marker U+2581 becomes a space,
//   - a single leading space is then dropped, so "▁Once" decodes to "Once",
//   - an unknown token renders as U+2047, the reference's "⁇" surface form,
//   - control tokens are skipped: they are metadata, not text,
//   - byte tokens are accumulated and decoded as one UTF-8 sequence, since a
//     single non-ASCII character spans several of them.
func (t *Tokenizer) Decode(ids []int32) string {
	var b strings.Builder
	var pending []byte

	flushBytes := func() {
		if len(pending) == 0 {
			return
		}
		b.WriteString(strings.ToValidUTF8(string(pending), "\uFFFD"))
		pending = pending[:0]
	}

	for _, id := range ids {
		if id < 0 || int(id) >= len(t.tokens) {
			continue
		}
		if t.types[id] == Control {
			continue
		}
		if tok, ok := parseByteToken(t.tokens[id]); ok {
			pending = append(pending, tok)
			continue
		}
		if t.types[id] == Unknown {
			flushBytes()
			b.WriteString(unknownSurface)
			continue
		}
		// A byte-level piece is bytes too, not text: "Ġ" is the escaped space byte, and two
		// adjacent pieces can hold the halves of one character. Rendering each one
		// independently would emit an incomplete sequence as-is, which is invalid UTF-8 --
		// and a JSON body cannot carry that. Buffering them like <0xNN> pieces lets an
		// incomplete run become U+FFFD instead.
		if t.byteLevel {
			pending = append(pending, bpeUnescapeBytes(t.tokens[id])...)
			continue
		}
		flushBytes()
		b.WriteString(t.renderPiece(t.tokens[id]))
	}
	flushBytes()

	out := b.String()
	// A SentencePiece vocabulary prepends a space to the text it encodes, so the marker on
	// the first piece stands for a space the caller never wrote and decoding takes it back
	// off. A byte-level vocabulary does not: the pieces already carry the caller's own
	// spaces, and both forms of a word exist ("the" and "Ġthe"), so a leading space in the
	// output is input and trimming it loses data. The flag that decided whether to prepend
	// is the same one that says whether to trim.
	if t.addSpacePrefix {
		out = strings.TrimPrefix(out, " ")
	}
	return out
}

// DecodeCompletion renders generated tokens as text to show a user.
//
// It differs from Decode in one respect: a byte-level vocabulary spells a word two ways
// ("the" and "Ġthe"), and a generation that starts a word picks the spaced form because it
// is the one that continues running text. Shown as-is, that space is a stray leading
// character the user did not ask for. llama.cpp handles this with an lstrip argument to
// token_to_piece rather than inside its detokenize, and the same split is kept here: Decode
// stays the exact inverse of Encode so it can be used to check round trips, and this method
// is what a caller showing a completion wants.
//
// A SentencePiece vocabulary needs no special case here, because Decode has already removed
// its marker space.
func (t *Tokenizer) DecodeCompletion(ids []int32) string {
	return strings.TrimPrefix(t.Decode(ids), " ")
}

// parseByteToken recognizes a SentencePiece byte token of the form <0xNN>.
func parseByteToken(s string) (byte, bool) {
	if len(s) != 6 || s[0] != '<' || s[1] != '0' || s[2] != 'x' || s[5] != '>' {
		return 0, false
	}
	hi, ok := hexVal(s[3])
	if !ok {
		return 0, false
	}
	lo, ok := hexVal(s[4])
	if !ok {
		return 0, false
	}
	return hi<<4 | lo, true
}

func hexVal(c byte) (byte, bool) {
	switch {
	case c >= '0' && c <= '9':
		return c - '0', true
	case c >= 'A' && c <= 'F':
		return c - 'A' + 10, true
	case c >= 'a' && c <= 'f':
		return c - 'a' + 10, true
	default:
		return 0, false
	}
}

// StreamDecoder decodes tokens incrementally as a sequence generates them.
//
// Decoding one token at a time with Decode is wrong for streaming, and both
// reasons are visible in Decode's own contract:
//
//   - It drops the single leading space that the word marker contributed, so
//     decoding each token alone would strip the space before every word after the
//     first, turning "Once upon" into "Onceupon".
//   - It renders an incomplete byte sequence with a replacement character, so a
//     character whose bytes span several tokens would arrive broken in the middle
//     and could not be repaired.
//
// A StreamDecoder therefore keeps the two pieces of state a stream needs: whether
// any text has been emitted (so only the first one drops a leading space), and the
// bytes of a character still waiting for the rest of its tokens. Token returns the
// text that became available, which is empty when the token only completed part of
// a character.
//
// The zero value is not usable; call Seq to start one. A StreamDecoder is not safe
// for concurrent use, which matches the one-writer stream it serves.
type StreamDecoder struct {
	tok *Tokenizer

	// pending holds bytes of a character whose remaining byte tokens have not
	// arrived yet. Decode cannot render them, so they wait rather than becoming a
	// replacement character.
	pending []byte

	// started reports whether any text has been emitted. Only the first text drops
	// the leading space the word marker contributes, which is what Decode does to
	// a whole sequence.
	started bool
}

// Seq starts a stream decoder.
func (t *Tokenizer) Seq() *StreamDecoder {
	return &StreamDecoder{tok: t}
}

// Token returns the text token id contributes, given everything already emitted.
//
// The result is empty for a control token, and also for a byte token that only
// completes part of a character: nothing can be shown yet.
func (d *StreamDecoder) Token(id int32) string {
	if id < 0 || int(id) >= len(d.tok.tokens) {
		return ""
	}
	if d.tok.types[id] == Control {
		return ""
	}

	if b, ok := parseByteToken(d.tok.tokens[id]); ok {
		d.pending = append(d.pending, b)
		// The leading-space rule applies here too. A leading space can arrive as a byte
		// token rather than as a word marker -- a byte-level vocabulary spells it <0x20>
		// -- and routing this path around the rule made the streamed text differ from
		// Decode's by one space.
		return d.emit(d.drainPending(false))
	}

	// A non-byte token ends any byte run: whatever is left is a character that will
	// never be completed, so it is rendered the way Decode renders it.
	var b strings.Builder
	b.WriteString(d.drainPending(true))
	if d.tok.types[id] == Unknown {
		b.WriteString(unknownSurface)
	} else {
		b.WriteString(d.tok.renderPiece(d.tok.tokens[id]))
	}
	return d.emit(b.String())
}

// emit applies the leading-space rule and records that text has been produced.
//
// Every path that can produce text goes through here, which is what keeps the streamed
// output equal to DecodeCompletion's: both drop exactly one space at the start, and neither
// can forget to.
//
// The space is dropped here rather than in Emit's callers because a stream has no whole
// sequence to post-process: the first token is gone by the time the last arrives. That is
// also why Decode itself does not drop it -- a caller checking Encode/Decode round trips
// needs the space preserved, and DecodeCompletion is the wrapper that takes it off.
func (d *StreamDecoder) emit(out string) string {
	if !d.started && strings.HasPrefix(out, " ") {
		out = out[1:]
		// The drop happens once, whether or not text remains afterwards. Leaving
		// started false when the piece was nothing but a space would drop the next
		// space too, and a sequence of word markers would lose all of them while
		// Decode, which trims after joining, loses exactly one.
		d.started = true
		return out
	}
	if out != "" {
		d.started = true
	}
	return out
}

// drainPending emits every complete character in pending, keeping a trailing
// incomplete sequence for the next token.
//
// When force is set the caller knows no more bytes are coming, so an incomplete
// tail is emitted with a replacement character instead of being kept.
func (d *StreamDecoder) drainPending(force bool) string {
	if len(d.pending) == 0 {
		return ""
	}
	// Find the longest prefix that is valid UTF-8, stopping at the first invalid
	// byte: that is where a character begins that the next token must complete.
	cut := len(d.pending)
	for i := 0; i < len(d.pending); {
		if !utf8.FullRune(d.pending[i:]) {
			cut = i
			break
		}
		_, size := utf8.DecodeRune(d.pending[i:])
		i += size
	}
	if !force && cut < len(d.pending) {
		out := string(d.pending[:cut])
		d.pending = append(d.pending[:0], d.pending[cut:]...)
		return out
	}
	// Every pending byte is converted at once, matching Decode: a run of bytes that never
	// forms a character becomes one replacement character, not one per byte.
	out := strings.ToValidUTF8(string(d.pending), "\uFFFD")
	d.pending = d.pending[:0]
	return out
}

// renderPiece renders one vocabulary piece as text.
//
// The two vocabulary families spell their pieces differently and must not share a decoder.
// A SentencePiece piece is plain text plus U+2581 for a space; only the marker needs undoing,
// and a character that happens to be one of GPT-2's escaped runes is still just that
// character. A byte-level piece is the opposite: every rune in it stands for a byte, so
// "Ġ¡" is two bytes, not a space marker and an exclamation.
//
// Applying the byte table to a SentencePiece piece is not a harmless no-op. Once the table
// covered the whole byte range, a Spanish "¡" in a SentencePiece vocabulary decoded as the
// single byte 0xA1, because the piece's own character looked like an escape. That is what a
// shared decode path costs, and the fix is to keep the two apart rather than to make the
// table narrower.
func (t *Tokenizer) renderPiece(piece string) string {
	if !t.byteLevel {
		return strings.ReplaceAll(piece, spacePrefix, " ")
	}
	return bpeUnescape(piece)
}

// bpeUnescapeBytes renders one byte-level piece as the bytes it stands for.
//
// It is bpeUnescape without the string conversion, so the result can be appended to a byte
// buffer. The two must agree: a piece that is not an escaped rune is passed through as its
// own UTF-8 bytes, which is what the vocabulary means by it.
func bpeUnescapeBytes(piece string) []byte {
	out := make([]byte, 0, len(piece))
	for _, r := range piece {
		if r == ' ' || r == 0x2581 {
			out = append(out, ' ')
			continue
		}
		if c, ok := bpeBytes[r]; ok {
			out = append(out, c)
			continue
		}
		out = utf8.AppendRune(out, r)
	}
	return out
}

// bpeUnescape renders one vocabulary piece as text, undoing both space markers and
// GPT-2's byte escaping.
//
// A SentencePiece piece carries U+2581 for a space and is otherwise plain text; a
// GPT-2 byte-level piece carries one escaped character per byte. Both are handled
// here so the two vocabulary families decode through the same call site, which is
// what keeps the incremental decoder from needing to know which one it has.
func bpeUnescape(piece string) string {
	if !strings.ContainsFunc(piece, func(r rune) bool {
		_, ok := bpeBytes[r]
		return ok && r > 0x7e
	}) {
		// Fast path: no escaped bytes, so only the SentencePiece marker can appear.
		return strings.ReplaceAll(piece, spacePrefix, " ")
	}
	var b strings.Builder
	for _, r := range piece {
		if r == ' ' || r == 0x2581 {
			b.WriteByte(' ')
			continue
		}
		if c, ok := bpeBytes[r]; ok {
			b.WriteByte(c)
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}
