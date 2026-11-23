package server

import (
	"strings"

	"github.com/pplmx/orbix/internal/llama"
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
	top     []llama.Alternative
}

// logprobText is one entry of a response's logprob list.
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
// One leading space is dropped, from the first token that has one, because DecodeCompletion
// trims exactly that from a completion. Doing it here rather than on the chunk's text keeps
// the offsets indexing the string the caller receives.
func (s *Server) describeTokens(tokens []sampledToken) []logprobText {
	out := make([]logprobText, len(tokens))
	trimmed := false
	for i, t := range tokens {
		raw := s.tok.PieceBytes(t.id)
		if !trimmed && len(raw) > 0 {
			trimmed = true
			if raw[0] == ' ' {
				raw = raw[1:]
			}
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
func (s *Server) describeStreamedToken(id int32, shown string, logprob float32, top []llama.Alternative, first bool) []logprobText {
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

// completionLogprobs builds the completions endpoint's logprobs object for one choice.
//
// The schema is a set of parallel arrays rather than a list of objects -- tokens, their log
// probabilities, the byte offset each begins at, and one map of alternatives per position.
// The map's key is the alternative's text, which is the schema's shape and also its weakness:
// two alternatives that render identically collapse into one entry. That can only happen with
// a vocabulary holding duplicate pieces, and there is no way to express both in the object the
// schema asks for.
func completionLogprobs(described []logprobText, count int, offset int) map[string]any {
	tokens := make([]string, count)
	logprobs := make([]float32, count)
	offsets := make([]int, count)
	top := make([]map[string]float32, count)
	at := offset
	for i := range count {
		d := described[i]
		tokens[i] = d.text
		logprobs[i] = d.logprob
		offsets[i] = at
		if d.top != nil {
			alternatives := make(map[string]float32, len(d.top))
			for _, a := range d.top {
				alternatives[a.text] = a.logprob
			}
			top[i] = alternatives
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
func chatLogprobs(described []logprobText, count int) map[string]any {
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
