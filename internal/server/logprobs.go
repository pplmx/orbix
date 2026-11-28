package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/pplmx/orbix/internal/engine"
)

// sampledToken is one generated token with the distribution it came from.
//
// The id is kept rather than the text because the text depends on the tokens around it: a
// byte-level vocabulary spells one character across several tokens, and the only way to
// render them is to walk the sequence in order through one decoder. So the ids are collected
// as they arrive and turned into text in a single pass at the end.
type sampledToken struct {
	id      int32
	logprob float32
	top     []engine.Alternative
}

// logprobText is one entry of a response's logprob list.
// orderedLogprobs is one position's alternatives, kept as a list so the encoded object preserves
// their order. See MarshalJSON for why that matters.
type orderedLogprobs []logprobEntry

// logprobEntry is one alternative before encoding.
type logprobEntry struct {
	text    string
	logprob float32
}

type logprobText struct {
	// text is the token's text, and raw is its UTF-8 bytes.
	text string
	raw  []byte
	// logprob is the natural logarithm of the probability the model gave this token.
	logprob float32
	// top holds the alternatives, already rendered to text.
	top []logprobText
}

// describeTokens renders a sequence of sampled tokens, plus the alternatives at each position.
//
// Each entry is described by the bytes its token contributes, not by decoding it on its own.
// The difference is not cosmetic: a byte-level vocabulary spells one byte per token, so a
// character can span several of them and decoding one token alone yields either nothing (the
// bytes do not yet form a character) or a replacement character. Holding the bytes keeps the
// concatenation exact, and the text is that byte run made JSON-safe, which is the best effort
// the schema's separate "bytes" field exists to supplement.
//
// Leading spaces are dropped so that the entries describe exactly the text DecodeCompletion
// returns. Doing it here rather than on the chunk's text keeps the offsets indexing the string the
// caller receives.
//
// How many is not a constant. A vocabulary that spells a space as its own token makes the encoder
// emit a leading space token whether or not the caller wrote one, and the decoder drops that token's
// space again -- so the pieces can run one space longer than the decoded text, or two, depending on
// what was encoded. Measured on this fixture: Encode(" Hi") is [space space H i], whose pieces join
// to "  Hi" while Decode returns " Hi"; the rule "drop the leading space of the first token with
// bytes" left both tokens' spaces in place once the first was a bare space, so the entries joined
// to " Hi" against a content of "Hi" -- every entry shifted by one byte.
//
// The count is therefore taken from the tokenizer rather than assumed: pieces minus decoded is
// exactly the run of leading spaces to remove.
func (s *Server) describeTokens(tokens []sampledToken) []logprobText {
	out := make([]logprobText, len(tokens))
	ids := make([]int32, len(tokens))
	for i, t := range tokens {
		ids[i] = t.id
	}
	// dropLeft is how many leading bytes the decoder removed: the pieces' own length minus what
	// DecodeCompletion renders. Counting is safe because every dropped byte is a space, which is
	// one byte in UTF-8 and renders as itself.
	pieces := 0
	for _, t := range tokens {
		pieces += len(s.tok.PieceBytes(t.id))
	}
	dropLeft := max(0, pieces-len(s.tok.DecodeCompletion(ids)))
	for i, t := range tokens {
		raw := s.tok.PieceBytes(t.id)
		if dropLeft > 0 && len(raw) > 0 {
			drop := min(dropLeft, len(raw))
			dropLeft -= drop
			raw = raw[drop:]
		}
		out[i] = logprobText{
			text:    strings.ToValidUTF8(string(raw), "\uFFFD"),
			raw:     raw,
			logprob: t.logprob,
		}
		if len(t.top) > 0 {
			out[i].top = make([]logprobText, len(t.top))
			for j, alt := range t.top {
				// An alternative never took part in this sequence, so its bytes are just its
				// own piece -- and no leading space is trimmed from it, because the trim
				// belongs to the first token of the response and this is not one.
				altRaw := s.tok.PieceBytes(alt.ID)
				out[i].top[j] = logprobText{
					text:    strings.ToValidUTF8(string(altRaw), "\uFFFD"),
					raw:     altRaw,
					logprob: alt.Logprob,
				}
			}
		}
	}
	return out
}

// describeStreamedToken renders one token whose text the stream decoder already produced,
// plus its alternatives.
//
// The text is passed in rather than re-decoded because the stream's decoder carries the byte
// state a token's text depends on; a fresh one would render a continuation byte as "". The
// alternatives get their own fresh decoder for the opposite reason: they were not emitted, so
// they never took part in that state.
func (s *Server) describeStreamedToken(id int32, shown string, logprob float32, top []engine.Alternative, first bool) []logprobText {
	raw := s.tok.PieceBytes(id)
	text := strings.ToValidUTF8(string(raw), "\uFFFD")
	// With logprobs the schema still wants the chunk's own text, which the stream decoder has
	// already produced with the leading-space rule applied. The byte run is what the offset is
	// counted over, so the space is removed here whenever the chunk's text came out shorter.
	if first && len(raw) > 0 && raw[0] == ' ' && !strings.HasPrefix(shown, " ") {
		raw = raw[1:]
		text = strings.ToValidUTF8(string(raw), "\uFFFD")
	}
	out := []logprobText{{text: text, raw: raw, logprob: logprob}}
	if len(top) > 0 {
		out[0].top = make([]logprobText, len(top))
		for j, a := range top {
			altRaw := s.tok.PieceBytes(a.ID)
			out[0].top[j] = logprobText{
				text:    strings.ToValidUTF8(string(altRaw), "\uFFFD"),
				raw:     altRaw,
				logprob: a.Logprob,
			}
		}
	}
	return out
}

// keptTokens cuts a described sequence down to what the response actually shows.
//
// A stop string removes text from the end, so the tokens after the cut have no text in the
// response and a logprob for them would describe something the caller cannot see. A token
// that straddles the boundary goes too: half its text survived, and neither "include it" nor
// "include half of it" describes the logprob of the thing that did.
//
// It returns the number of leading entries to keep and the retained text's length in bytes,
// which is what the completions schema's text_offset needs.
func keptTokens(described []logprobText, retained int) (keep, used int) {
	// Nothing retained means nothing is shown, so no entry describes it. Without this the loop
	// below keeps every entry whose text is empty -- including the leading one when the whole
	// response was cut away -- because their length adds nothing and so never exceeds the
	// budget. Measured on a six-token choice with a stop string equal to its entire text:
	// shownTokens returned 6 where the response shows nothing, and the same call feeds
	// choiceObject's logprob list, so both reported entries for text the caller never sees.
	if retained == 0 {
		return 0, 0
	}
	for keep < len(described) {
		// Counted over the rendered text rather than the raw bytes: retained is the length of
		// the text the caller receives, which renders an invalid byte run as replacement
		// characters. For a well-formed vocabulary -- every real checkpoint -- the two are the
		// same number.
		next := used + len(described[keep].text)
		if next > retained {
			break
		}
		used = next
		keep++
	}
	return keep, used
}

// keptTokensInSpans keeps the described entries whose text falls inside one of the given byte
// ranges of the reply.
//
// This is what a parsed tool call needs and what keptTokens cannot express: removing a call cuts
// a hole in the middle of the reply, and a length budget can only keep a prefix. Measured before
// this existed: "Hi. <tool_call>...</tool_call> Done." reported eleven logprob entries joining to
// "Hi. <tool_" while the message's content was "Hi.  Done." -- the list described the call's own
// bytes, which is exactly what keptTokens' comment says must not happen.
//
// An entry that straddles a boundary is dropped for the reason keptTokens gives: half its text
// survived, and neither including it nor including half describes the token the caller can see.
func keptTokensInSpans(described []logprobText, spans []proseSpan) []logprobText {
	var out []logprobText
	if len(spans) == 0 {
		// Nothing of the reply is prose -- an all-call reply -- so no token describes anything
		// the caller sees.
		return nil
	}
	at := 0
	next := 0
	for _, d := range described {
		start := at
		at += len(d.text)
		// Skip spans that end before this token starts; they cannot hold any of it.
		for next < len(spans) && spans[next].end <= start {
			next++
		}
		if next == len(spans) {
			break
		}
		// The token has to sit wholly inside one span: a token that only partly overlaps was
		// partly removed by the parse.
		if start >= spans[next].start && at <= spans[next].end {
			out = append(out, d)
		}
	}
	return out
}

// completionLogprobs builds the completions endpoint's logprobs object for one choice.
//
// The schema is a set of parallel arrays rather than a list of objects -- tokens, their log
// probabilities, the byte offset each begins at, and one map of alternatives per position.
// The map's key is the alternative's text, which is the schema's shape and the reason this
// endpoint describes fewer alternatives than chat does for the same request.
//
// Two alternatives whose bytes render to the same string collapse into one entry, and the
// surviving value is whichever the encoder wrote last -- so the object can come back with fewer
// keys than were asked for, and the ordering the type exists to preserve no longer holds past
// that point.
//
// This is not the rare vocabulary defect it was first documented as. Any token whose bytes are
// not valid UTF-8 on its own renders as U+FFFD, and a byte-level vocabulary has one of those per
// continuation byte: measured, Qwen3-0.6B has 1138 token ids that render as a single U+FFFD, and
// stories15M has 129. A top-k list containing two of them cannot be expressed in this object.
//
// Kept rather than worked around because the alternative is a non-standard extension -- a suffix
// on the key, or a bytes field beside it -- and a client parsing OpenAI's schema would then have
// to special-case this server. chat's shape needs neither: its alternatives are a list of
// objects, each carrying `bytes`, so nothing collapses there.
func completionLogprobs(described []logprobText, offset int) map[string]any {
	count := len(described)
	tokens := make([]string, count)
	logprobs := make([]float32, count)
	offsets := make([]int, count)
	top := make([]orderedLogprobs, count)
	at := offset
	for i := range count {
		d := described[i]
		tokens[i] = d.text
		logprobs[i] = d.logprob
		offsets[i] = at
		if d.top != nil {
			// Built as an ordered list rather than a map: the schema is an object, but the order
			// carries meaning. The alternatives arrive in descending probability and a client
			// reading the first entry as the top-1 prediction is reading what llama.cpp's own
			// servers return -- while encoding a Go map sorts its keys, which silently reordered
			// the alternatives alphabetically. Measured on Gemma 3 before this: the most likely
			// token "Paris" at -0.0007 came third, after " **" and " the".
			top[i] = orderedLogprobs(alternatives(d.top))
		}
		// Rendered length, for the same reason keptTokens counts it: text_offset indexes the
		// text the caller has, and an invalid byte run is longer once rendered. Identical for
		// a well-formed vocabulary.
		at += len(d.text)
	}
	return map[string]any{
		"tokens":         tokens,
		"token_logprobs": logprobs,
		"text_offset":    offsets,
		"top_logprobs":   top,
	}
}

// chatLogprobs builds the chat endpoint's logprobs object for one choice.
//
// The schema nests differently from the completions one: a list of per-token objects, each
// carrying its own text, bytes and alternatives. refusal is always null, because this server
// has no separate refusal channel -- a reply is either generated normally or reported as an
// error.
func chatLogprobs(described []logprobText) map[string]any {
	count := len(described)
	content := make([]map[string]any, count)
	for i := range count {
		d := described[i]
		entry := map[string]any{
			"token":   d.text,
			"bytes":   bytesOf(d.raw),
			"logprob": d.logprob,
		}
		alternatives := make([]map[string]any, len(d.top))
		for j, a := range d.top {
			alternatives[j] = map[string]any{
				"token":   a.text,
				"bytes":   bytesOf(a.raw),
				"logprob": a.logprob,
			}
		}
		entry["top_logprobs"] = alternatives
		content[i] = entry
	}
	return map[string]any{
		"content": content,
		"refusal": nil,
	}
}

// bytesOf renders a token's bytes the way the schema wants them: a list of integers, not a
// string. An empty token's bytes are an empty list rather than null -- null means "no byte
// representation", which is a different claim from "no bytes".
func bytesOf(raw []byte) []int {
	out := make([]int, len(raw))
	for i, b := range raw {
		out[i] = int(b)
	}
	return out
}

// orderedLogprobs is the alternatives object for one position, encoded in the order given.
//
// The schema calls this field an object -- a JSON map of token text to log probability -- and a Go
// map would be the obvious way to write it. It is the wrong way: encoding/json sorts a map's keys,
// so the object would come out in alphabetical order and the sort would lose the only information
// the field carries besides the numbers. The alternatives are computed in descending probability
// order, and a client that reads the first entry is reading the model's top prediction.
//
// Measured on Gemma 3 with the prompt "The capital of France is" before this type existed, the
// most likely token "Paris" (-0.0007) came third in the encoded object, behind " **" and " the".
func (o orderedLogprobs) MarshalJSON() ([]byte, error) {
	// A nil list is the "no alternatives requested" case, and the schema says null for it rather
	// than an empty object. Encoding it as {} would tell a client the model was offered zero
	// candidates, which is a different claim.
	if o == nil {
		return []byte("null"), nil
	}
	var b bytes.Buffer
	b.WriteByte('{')
	for i, a := range o {
		if i > 0 {
			b.WriteByte(',')
		}
		key, err := json.Marshal(a.text)
		if err != nil {
			return nil, fmt.Errorf("server: encode token text %q: %w", a.text, err)
		}
		b.Write(key)
		b.WriteByte(':')
		value, err := json.Marshal(a.logprob)
		if err != nil {
			return nil, fmt.Errorf("server: encode logprob for %q: %w", a.text, err)
		}
		b.Write(value)
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

// alternatives converts the rendered alternatives to the pair a position needs.
func alternatives(tops []logprobText) orderedLogprobs {
	out := make(orderedLogprobs, len(tops))
	for i, a := range tops {
		out[i] = logprobEntry{text: a.text, logprob: a.logprob}
	}
	return out
}

// shownLogprobTokens cuts the described tokens down to the ones that produced text the response
// shows.
//
// Two cuts are possible and they are not the same shape. A stop string removes the end of the
// reply, so a prefix is what survives and keptTokens' length budget is right. A parsed tool call
// removes a hole in the middle, so the surviving tokens are not contiguous and only the spans the
// parser reported can say which ones they are -- a length budget keeps the reply's first N bytes,
// which for "Hi. <call> Done." is mostly the call.
//
// calls decides which cut applies: when the parser found none, the whole of text is what the
// response shows, and when it found one, spans is what is left of it -- possibly nothing, which is
// why this is keyed on calls rather than on spans being non-empty.
func shownLogprobTokens(described []logprobText, text string, calls []toolCall, spans []proseSpan) []logprobText {
	if calls != nil {
		return keptTokensInSpans(described, spans)
	}
	// The budget is the content's length, but keptTokens counts the entries' rendered text, and for
	// a byte-level vocabulary those are different units: three tokens contribute one character, and
	// each token's bytes render on their own as a replacement character. Passing the content length
	// through cut the list short -- measured on "\u4e2d\u6587", 3 of 7 entries survived a reply that
	// was never truncated.
	if keep, _ := keptTokens(described, renderedCovering(described, len(text))); keep < len(described) {
		return described[:keep]
	}
	return described
}

// heldToken is one described token waiting for the streaming tool gate to decide whether its bytes
// are prose a client will see.
//
// The description is kept rather than the rendered chunk because the chunk's shape depends on the
// offset at which the run is finally released, and that is not known until the gate lets go.
type heldToken struct {
	described []logprobText
	offset    int
}

// appendHeld prepends the token descriptions the gate was holding to the one it just released.
//
// The runs are contiguous in the reply, so the first held description's offset is where the
// combined run starts. A held run is empty for a token that carried no log probability, which is
// why the offsets are stored per run rather than assumed.
func appendHeld(held []heldToken, released []logprobText, offset int) ([]logprobText, int) {
	var all []logprobText
	for _, h := range held {
		if h.described == nil {
			continue
		}
		if all == nil {
			offset = h.offset
		}
		all = append(all, h.described...)
	}
	if released != nil {
		if all == nil {
			// Nothing was held: the released description starts the run.
			return released, offset
		}
		all = append(all, released...)
	}
	return all, offset
}

// stopsHeldWith reports whether the text the stop gate released ends with this token's own text,
// which is the condition for the token's entry to belong in the chunk rather than in the withheld
// list.
//
// A release can be longer than the token -- it usually carries text withheld from earlier pushes --
// and it can be exactly the token, which is the case where the token was withheld before and its
// entry is already in held. Any other shape means the release stopped before this token and the
// entry waits.
func stopsHeldWith(text, release string, described []logprobText, held []heldToken) bool {
	if text == "" || described == nil {
		return false
	}
	if !strings.HasSuffix(release, text) {
		return false
	}
	// The release is this token and nothing else: its entry is already withheld.
	if len(release) == len(text) && len(held) > 0 {
		return false
	}
	return true
}
