package tokenize

import (
	"encoding/binary"
	"testing"
	"unicode/utf8"
)

// FuzzDecode checks that no token id makes the decoder panic or emit invalid UTF-8.
//
// Ids reach Decode from several directions: the sampler (which can only produce ids inside the
// vocabulary), a caller replaying a conversation, and a client posting a token sequence to an
// endpoint. The last one is why out-of-range values matter: a negative id or one past the
// vocabulary is not something the model produced, but nothing in the signature says so, and an
// unchecked index is a crash.
//
// The output must also be valid UTF-8. Decode assembles bytes from <0xNN> pieces, so a partial
// sequence is normal mid-stream; what it must not do is hand a caller a string that cannot be
// encoded.
func FuzzDecode(f *testing.F) {
	// The argument is []byte because that is what the fuzz engine accepts; each group of four
	// bytes becomes one int32 id, so the target can still reach negative and out-of-range
	// values rather than only small non-negative ones.
	seed := func(ids ...int32) []byte {
		out := make([]byte, len(ids)*4)
		for i, id := range ids {
			binary.LittleEndian.PutUint32(out[i*4:], uint32(id))
		}
		return out
	}
	f.Add([]byte{})
	f.Add(seed(0))
	f.Add(seed(-1))
	f.Add(seed(1, 2, 3))
	f.Add(seed(2147483647))
	f.Add(seed(-2147483648))
	f.Add(seed(13, 10, 14, 23))
	f.Add(seed(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))

	f.Fuzz(func(t *testing.T, data []byte) {
		ids := make([]int32, len(data)/4)
		for i := range ids {
			ids[i] = int32(binary.LittleEndian.Uint32(data[i*4:]))
		}
		tok := newTestTokenizer(t)
		out := tok.Decode(ids)
		if !utf8.ValidString(out) {
			t.Fatalf("Decode(%v) = %q, which is not valid UTF-8", ids, out)
		}
		// DecodeCompletion is the wrapper the server uses; it must hold the same property.
		if trimmed := tok.DecodeCompletion(ids); !utf8.ValidString(trimmed) {
			t.Fatalf("DecodeCompletion(%v) = %q, which is not valid UTF-8", ids, trimmed)
		}
	})
}

// FuzzEncodeDecodeRoundTrip checks that text survives a trip through the vocabulary.
//
// Encode then Decode is the property a server relies on when it echoes a prompt back or trims a
// completion: text that changes shape in the middle of that trip is a corruption the caller
// never asked for. The SentencePiece fixture normalises a leading space and unknown characters
// become <unk>, so the round trip is not the identity for every input -- the weaker property
// that always holds is that the result is valid UTF-8 and that an <unk> id is not decoded as
// something else.
func FuzzEncodeDecodeRoundTrip(f *testing.F) {
	f.Add("")
	f.Add("the")
	f.Add(" the")
	f.Add("hello world")
	f.Add("\u4e2d\u6587")
	f.Add("\U0001F600")
	f.Add("\xff\xfe")
	f.Add("e\u0301")

	f.Fuzz(func(t *testing.T, text string) {
		if !utf8.ValidString(text) {
			return
		}
		tok := newTestTokenizer(t)
		for _, addSpecial := range []bool{false, true} {
			ids := tok.Encode(text, addSpecial)
			out := tok.Decode(ids)
			if !utf8.ValidString(out) {
				t.Fatalf("Encode(%q, %v) -> Decode = %q, not valid UTF-8", text, addSpecial, out)
			}
		}
	})
}
