package server

import (
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
