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

// bpeSpace is how a byte-level BPE vocabulary escapes a space: the byte 0x20 is
// rendered as the character at code point 256+offset, which for space is U+0120 'Ġ'.
// It is spelled here rather than looked up in bpeBytes because encoding needs the
// rune, and the table maps runes to bytes.
const bpeSpace = "\u0120"

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
	// The reversible table GPT-2 uses: printable ASCII (except space) maps to
	// itself, and everything else is assigned the next unused code point from 256.
	var keep [256]bool
	for b := '!'; b <= '~'; b++ {
		keep[b] = true
	}
	table := make(map[rune]byte, 256)
	next := 0
	for b := range 256 {
		var r rune
		if keep[b] {
			r = rune(b)
		} else {
			r = rune(256 + next)
			next++
		}
		table[r] = byte(b)
	}
	return table
}()

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
	if addSpecial && t.bosID >= 0 {
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
	if addSpecial && t.bosID >= 0 {
		out = append(out, t.bosID)
	}
	if len(t.specials) == 0 {
		return append(out, t.encodeText(text)...)
	}

	// Split the text at each special piece and encode the gaps as ordinary text, so a
	// marker in the middle of a string does not disturb the pieces around it. The
	// segment boundaries are what keeps SentencePiece's leading-space rule intact: an
	// empty gap between two adjacent markers produces no tokens at all.
	start := 0
	for i := 0; i < len(text); {
		if sp, ok := t.matchSpecial(text[i:]); ok {
			if start < i {
				out = append(out, t.encodeText(text[start:i])...)
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
		out = append(out, t.encodeText(text[start:])...)
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
	if text == "" {
		return nil
	}
	if t.addSpacePrefix {
		text = " " + text
	}
	// Represent each space the way the vocabulary does before merging. The two
	// families differ in when the space becomes part of a piece: SentencePiece
	// folds it into the marker on the following character, so " a" is one symbol,
	// while byte-level BPE escapes it as its own symbol and lets the merge decide
	// whether "Ġa" exists as a piece.
	if t.byteLevel {
		text = strings.ReplaceAll(text, " ", bpeSpace)
	} else {
		text = strings.ReplaceAll(text, " ", spacePrefix)
	}

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
		flushBytes()
		if t.types[id] == Unknown {
			b.WriteString(unknownSurface)
			continue
		}
		b.WriteString(bpeUnescape(t.tokens[id]))
	}
	flushBytes()

	// Drop the single leading space that the word marker contributed.
	out := b.String()
	return strings.TrimPrefix(out, " ")
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
		b.WriteString(bpeUnescape(d.tok.tokens[id]))
	}
	return d.emit(b.String())
}

// emit applies the leading-space rule and records that text has been produced.
//
// Every path that can produce text goes through here, which is what keeps the streamed
// output equal to Decode's: both drop exactly one space at the start, and neither can
// forget to.
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
