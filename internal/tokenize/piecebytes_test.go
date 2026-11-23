package tokenize

import (
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/pplmx/orbix/internal/gguftest"
)

// TestPieceBytesConcatenateToDecode pins the contract PieceBytes documents: concatenating one
// entry per token reproduces the text Decode builds from the same ids.
//
// It is the property the response schema depends on. A token's own text is not always valid
// UTF-8 -- a SentencePiece vocabulary spells an arbitrary byte as <0xNN>, and a byte-level
// vocabulary spells each byte as an escaped rune -- so a client reassembling a completion from
// per-token text alone can be left with fragments, which is what the schema's separate bytes
// field exists to avoid. Concatenating the byte runs is the authoritative reconstruction, and
// this test is what keeps it equal to what Decode already returns.
//
// The comparison allows Decode's one leading space: for a SentencePiece vocabulary the marker
// on the first piece stands for a space the caller never wrote, and Decode removes it.
func TestPieceBytesConcatenateToDecode(t *testing.T) {
	tok := bpeFixture(t, gguftest.Default())

	// Every id in the vocabulary, which for the fixture includes control tokens, byte tokens
	// and ordinary pieces -- the three cases PieceBytes treats differently.
	ids := make([]int32, 0, 256)
	for i := int32(0); i < 256; i++ {
		ids = append(ids, i)
	}

	var joined strings.Builder
	for _, id := range ids {
		joined.Write(tok.PieceBytes(id))
	}
	want := tok.Decode(ids)
	// ToValidUTF8, because the two describe the same run at different levels: PieceBytes is
	// the bytes and Decode is their rendering, and a vocabulary holding an invalid byte --
	// which this fixture's does -- renders it as a replacement character. Two further
	// differences are allowed and expected: Decode drops one leading space, and it validates
	// each contiguous byte run separately, which can only differ from validating the whole
	// concatenation where a run is split by text.
	got := strings.ToValidUTF8(joined.String(), "\uFFFD")
	if got != want && got != " "+want && strings.TrimPrefix(got, " ") != want {
		t.Errorf("ToValidUTF8(concatenated bytes) = %q, Decode = %q", got, want)
	}
}

// TestPieceBytesOfAByteToken covers the family that needs it most: a SentencePiece vocabulary
// spells a raw byte as <0xNN>, and that token's bytes are the single byte it names rather than
// the characters of the marker.
//
// Getting this wrong is invisible in most of a response -- the marker happens to be printable
// text -- and shows up as a completion containing a literal "<0xFF>" where the model meant one
// byte. It is also what a byte run split across tokens needs in order to be reassembled.
func TestPieceBytesOfAByteToken(t *testing.T) {
	vocab := []string{"<unk>", "<s>", "</s>", "<0x2F>", "<0xFF>", "hi"}
	tok := bpeFixture(t, gguftest.Model{Vocab: vocab})

	// The byte tokens are found by name rather than assumed to be at a fixed index, since the
	// fixture assigns ids in the order the vocabulary lists them but that is not this package's
	// contract.
	index := make(map[string]int32, len(vocab))
	for i, piece := range vocab {
		index[piece] = int32(i)
	}

	for _, c := range []struct {
		piece string
		want  []byte
	}{
		{"<0x2F>", []byte{'/'}},
		{"<0xFF>", []byte{0xFF}},
		{"hi", []byte("hi")},
	} {
		if got := tok.PieceBytes(index[c.piece]); string(got) != string(c.want) {
			t.Errorf("PieceBytes(%q) = %v, want %v", c.piece, got, c.want)
		}
	}

	// A control token contributes nothing, which is what Decode does with it as well.
	if got := tok.PieceBytes(index["<s>"]); len(got) != 0 {
		t.Errorf("PieceBytes(<s>) = %v, want no bytes: it is markup, not text", got)
	}

	// An id outside the vocabulary is a nil run rather than a panic: the caller is describing a
	// sequence the engine produced, and a response should not be lost to one hostile id.
	// A far-out id rather than len(vocab)+10: the fixture appends a run of byte tokens to
	// whatever the caller listed, so just past the declared vocabulary is still in range.
	if got := tok.PieceBytes(1 << 20); got != nil {
		t.Errorf("PieceBytes(out of range) = %v, want nil", got)
	}
	if got := tok.PieceBytes(-1); got != nil {
		t.Errorf("PieceBytes(-1) = %v, want nil", got)
	}
}

// TestPieceBytesOfInvalidUTF8IsBytesNotRunes records the difference between a byte run and a
// rendering.
//
// The two halves of a UTF-8 character can arrive as separate tokens, and the byte half on its
// own is not valid UTF-8. PieceBytes returns the byte, which is what lets a caller concatenate
// and only then decide how to render; a method that returned a string would have to substitute
// a replacement character per token and could never be reassembled.
func TestPieceBytesOfInvalidUTF8IsBytesNotRunes(t *testing.T) {
	tok := bpeFixture(t, gguftest.Model{
		Vocab: []string{"<unk>", "<s>", "</s>", "<0xC3>", "<0xA9>"},
	})
	// "é" is C3 A9. Split across two tokens, neither half is valid on its own.
	c3 := tok.PieceBytes(3)
	a9 := tok.PieceBytes(4)
	if len(c3) != 1 || c3[0] != 0xC3 {
		t.Fatalf("PieceBytes(<0xC3>) = %v, want the single byte 0xC3", c3)
	}
	if len(a9) != 1 || a9[0] != 0xA9 {
		t.Fatalf("PieceBytes(<0xA9>) = %v, want the single byte 0xA9", a9)
	}
	if utf8.Valid(c3) || utf8.Valid(a9) {
		t.Skip("this assertion only means something when the halves are invalid on their own")
	}
	// Joined, they are the character -- which is the whole point of returning bytes.
	if joined := append(append([]byte{}, c3...), a9...); string(joined) != "é" {
		t.Errorf("joined halves = %q, want \"é\"", joined)
	}
}
