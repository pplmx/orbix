package server

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguftest"
)

// TestDescribeStreamedTokenStripsTheFirstLeadingSpace pins the branch that removes a leading
// space from the token the leading-space rule applies to.
//
// The schema wants the chunk's own text, which the stream decoder produced with that rule
// already applied, but the offset is counted over the byte run -- so when the chunk's text came
// out shorter than the piece, the piece loses its space to match. Deleting the branch left the
// whole package green: every existing streaming-logprobs test goes through the fixture's
// decoder, and its first emitted token does not begin with a space.
func TestDescribeStreamedTokenStripsTheFirstLeadingSpace(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 512})

	// id 3's piece is a bare space in this fixture.
	const spaceID = 3
	if raw := srv.tok.PieceBytes(spaceID); len(raw) == 0 || raw[0] != ' ' {
		t.Fatalf("token %d is not a space-led piece any more (%q); pick another id",
			spaceID, string(raw))
	}

	// First, and the shown text has no leading space: the space is removed so the offset and
	// the text agree.
	got := srv.describeStreamedToken(spaceID, "", 0, nil, true)
	if len(got[0].raw) != 0 || got[0].text != "" {
		t.Errorf("first token: raw %q text %q, want both empty after stripping the space",
			string(got[0].raw), got[0].text)
	}

	// Not first: the rule does not apply, so the space stays.
	got = srv.describeStreamedToken(spaceID, "", 0, nil, false)
	if len(got[0].raw) != 1 || got[0].raw[0] != ' ' {
		t.Errorf("non-first token: raw %q, want the space kept", string(got[0].raw))
	}

	// First, but the shown text already begins with a space: the piece keeps its own.
	got = srv.describeStreamedToken(spaceID, " x", 0, nil, true)
	if len(got[0].raw) != 1 {
		t.Errorf("shown text starts with a space: raw %q, want it kept", string(got[0].raw))
	}
}

// TestCompletionAlternativesKeepTheMostLikelyOfATiedText pins which of two alternatives that
// render to the same string survives into the response.
//
// The completions schema keys its alternatives by token text, so two tokens whose bytes render
// the same can only contribute one key. Which one matters: the list arrives in descending
// probability, and a JSON parser keeps the last duplicate -- so emitting both meant the value a
// client read back was the *least* likely of the clashing tokens.
//
// Measured before the fix: {"\uFFFD":-0.1,"\uFFFD":-2.5,"ok":-3} parsed to -2.5, so a request
// asking for the top candidates reported a token 2.4 nats less likely than the one it displaced.
// A byte-level vocabulary is where this happens -- Qwen3-0.6B has 1138 ids that render as a bare
// U+FFFD -- so it is reachable by any non-ASCII top-k list, not a contrived input.
//
// The assertion is on the parsed value rather than on the encoded bytes, because the bytes
// containing a duplicate key is only a problem insofar as a client loses the right number.
func TestCompletionAlternativesKeepTheMostLikelyOfATiedText(t *testing.T) {
	described := []logprobText{{
		text:    "a",
		raw:     []byte("a"),
		logprob: -0.2,
		top: []logprobText{
			{text: "\uFFFD", raw: []byte{0x80}, logprob: -0.1},
			{text: "\uFFFD", raw: []byte{0x81}, logprob: -2.5},
			{text: "ok", raw: []byte("ok"), logprob: -3.0},
		},
	}}

	out := completionLogprobs(described, 0)
	tops, ok := out["top_logprobs"].([]orderedLogprobs)
	if !ok || len(tops) != 1 {
		t.Fatalf("top_logprobs = %#v, want one position", out["top_logprobs"])
	}

	raw, err := json.Marshal(tops[0])
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var parsed map[string]float32
	if err := json.Unmarshal(raw, &parsed); err != nil {
		t.Fatalf("unmarshal %s: %v", raw, err)
	}
	if got := parsed["\uFFFD"]; got != -0.1 {
		t.Errorf("the clashing key reads back %v, want -0.1 -- the most likely of the two, not "+
			"the last one written (the parsed object had %d keys from 3 alternatives: %s)",
			got, len(parsed), raw)
	}
	if parsed["ok"] != -3.0 {
		t.Errorf("the alternative that does not clash reads back %v, want -3", parsed["ok"])
	}
	if len(parsed) != 2 {
		t.Errorf("parsed %d keys, want 2: the duplicate text has to cost exactly one entry",
			len(parsed))
	}

	// The encoded object must not carry a repeated key at all. Asserting only on the parsed value
	// would pass an encoder that wrote both and happened to put the likely one last, which a
	// later ordering change would silently invert.
	//
	// Counted on the literal character, not on a `\uFFFD` escape: json.Marshal writes the
	// replacement character as itself, so counting escapes finds zero and a wrong needle would
	// make this assertion vacuous.
	if n := strings.Count(string(raw), "\uFFFD"); n != 1 {
		t.Errorf("the encoded object mentions the clashing text %d times: %s", n, raw)
	}
}
