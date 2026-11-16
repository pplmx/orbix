package tokenize

import (
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
)

// newTestTokenizer builds a Tokenizer directly from the fixture vocabulary,
// bypassing GGUF since the fixture is already in the shape New expects.
func newTestTokenizer(tb testing.TB) *Tokenizer {
	tb.Helper()
	tok := &Tokenizer{
		tokens:         spmTokens,
		scores:         spmScores,
		types:          make([]TokenType, len(spmTokens)),
		index:          make(map[string]int32, len(spmTokens)),
		byteID:         fillNeg1(),
		unknownID:      0,
		bosID:          1,
		eosID:          2,
		addSpacePrefix: true,
	}
	for i, p := range spmTokens {
		tok.index[p] = int32(i)
	}
	for i := range tok.types {
		tok.types[i] = Normal
	}
	tok.types[0] = Unknown
	tok.types[1] = Control
	tok.types[2] = Control
	return tok
}

func TestEncodeMatchesReference(t *testing.T) {
	tok := newTestTokenizer(t)
	for _, tc := range spmCases {
		got := tok.Encode(tc.text, false)
		if len(got) != len(tc.ids) {
			t.Errorf("Encode(%q) = %v (%d tokens), want %v (%d tokens)",
				tc.text, got, len(got), tc.ids, len(tc.ids))
			continue
		}
		for i := range got {
			if got[i] != tc.ids[i] {
				t.Errorf("Encode(%q)[%d] = %d, want %d (got %v, want %v)",
					tc.text, i, got[i], tc.ids[i], got, tc.ids)
				break
			}
		}
	}
}

func TestDecodeRoundTrip(t *testing.T) {
	// Decoding what Encode produced gives back the original text, for text
	// the vocabulary can represent.
	tok := newTestTokenizer(t)
	for _, text := range []string{
		"Once upon a time",
		"The cat sat on the mat.",
		"Tom went to the park with his dog.",
		"a",
	} {
		if got := tok.Decode(tok.Encode(text, false)); got != text {
			t.Errorf("round trip %q -> %q", text, got)
		}
	}
}

func TestDecodeUnknownTokens(t *testing.T) {
	// Characters outside the vocabulary become <unk> and decode to the
	// reference's "⁇" surface form. This is lossy by construction: the
	// model genuinely cannot represent '!' or ',' with this vocabulary, and
	// pretending otherwise would hide the data loss.
	tok := newTestTokenizer(t)
	got := tok.Decode(tok.Encode("Hello, world!", false))
	const want = "Hello \u2047  world \u2047 "
	if got != want {
		t.Errorf("Decode(Hello, world!) = %q, want %q", got, want)
	}
}

func TestDecodeSkipsControlAndByteTokens(t *testing.T) {
	tok := newTestTokenizer(t)
	ids := append([]int32{1}, tok.Encode("Once upon a time", false)...)
	ids = append(ids, 2)
	if got, want := tok.Decode(ids), "Once upon a time"; got != want {
		t.Errorf("Decode with BOS/EOS = %q, want %q", got, want)
	}
}

func TestDecodeAfterEncodeRoundTrips(t *testing.T) {
	// Encoding then decoding must reproduce the input for text the vocabulary
	// can represent, which is the property a chat REPL depends on.
	tok := newTestTokenizer(t)
	for _, text := range []string{
		"Once upon a time there was a little girl named Lily.",
		"The cat sat on the mat.",
		"Tom went to the park with his dog.",
	} {
		if got := tok.Decode(tok.Encode(text, false)); got != text {
			t.Errorf("round trip %q -> %q", text, got)
		}
	}
}

func TestEncodeBOS(t *testing.T) {
	tok := newTestTokenizer(t)
	got := tok.Encode("a", true)
	if len(got) == 0 || got[0] != 1 {
		t.Fatalf("Encode(with special) = %v, want a leading BOS (1)", got)
	}
	// Without the flag there must be no BOS, or a chat template that adds its
	// own would emit it twice.
	for _, id := range tok.Encode("a", false) {
		if id == 1 {
			t.Errorf("Encode(without special) = %v, want no BOS", got)
		}
	}
}

func TestEncodeEmpty(t *testing.T) {
	tok := newTestTokenizer(t)
	if got := tok.Encode("", false); len(got) != 0 {
		t.Errorf("Encode(\"\") = %v, want empty", got)
	}
	// With the special flag an empty prompt is still just BOS.
	tok.addSpacePrefix = false
	if got := tok.Encode("", true); len(got) != 1 || got[0] != 1 {
		t.Errorf("Encode(\"\", true) = %v, want [1]", got)
	}
}

func TestEncodeControlTokensAreNotProduced(t *testing.T) {
	// A user typing "<s>" must not be able to inject the BOS control token.
	tok := newTestTokenizer(t)
	for _, id := range tok.Encode("<s>", false) {
		if id == 1 || id == 2 {
			t.Fatalf("Encode(\"<s>\") produced control token %d", id)
		}
	}
}

func TestByteFallback(t *testing.T) {
	// A byte token vocabulary lets any input be encoded, including characters
	// the model never saw during training. The byte tokens are decoded back
	// as one UTF-8 sequence, not byte by byte.
	tokens := []string{"<unk>", "<s>", "</s>", "a", "▁a"}
	for i := 0; i < 256; i++ {
		tokens = append(tokens, byteTokenName(i))
	}
	tok := &Tokenizer{
		tokens:         tokens,
		scores:         make([]float32, len(tokens)),
		types:          make([]TokenType, len(tokens)),
		index:          make(map[string]int32, len(tokens)),
		byteID:         fillNeg1(),
		unknownID:      0,
		bosID:          1,
		eosID:          2,
		addSpacePrefix: false,
	}
	for i, p := range tokens {
		tok.index[p] = int32(i)
		tok.types[i] = Normal
		if b, ok := parseByteToken(p); ok {
			tok.byteID[b] = int32(i)
			tok.types[i] = Byte
		}
	}

	// "é" is two UTF-8 bytes and is not in the vocabulary.
	const text = "é"
	ids := tok.Encode(text, false)
	if len(ids) != 2 {
		t.Fatalf("Encode(%q) = %v, want 2 byte tokens", text, ids)
	}
	if got := tok.Decode(ids); got != text {
		t.Errorf("Decode = %q, want %q", got, text)
	}
}

func TestDecodeInvalidUTF8(t *testing.T) {
	// A truncated multi-byte sequence must decode to the replacement
	// character rather than panicking or producing invalid UTF-8.
	tokens := []string{"<unk>", "<s>", "</s>", byteTokenName(0xE4), byteTokenName(0xB8)}
	tok := &Tokenizer{
		tokens:         tokens,
		scores:         make([]float32, len(tokens)),
		types:          []TokenType{Unknown, Control, Control, Byte, Byte},
		index:          map[string]int32{},
		byteID:         fillNeg1(),
		addSpacePrefix: false,
	}
	for i, p := range tokens {
		tok.index[p] = int32(i)
		if b, ok := parseByteToken(p); ok {
			tok.byteID[b] = int32(i)
		}
	}
	got := tok.Decode([]int32{3, 4})
	if got != "\uFFFD" {
		t.Errorf("Decode(truncated UTF-8) = %q, want the replacement character", got)
	}
}

func TestParseByteToken(t *testing.T) {
	tests := map[string]struct {
		b  byte
		ok bool
	}{
		"<0x00>": {0, true},
		"<0xFF>": {0xff, true},
		"<0xff>": {0xff, true},
		"<0x0A>": {0x0a, true},
		"<0x>":   {0, false},
		"<0xGG>": {0, false},
		"<s>":    {0, false},
		"hello":  {0, false},
	}
	for in, want := range tests {
		b, ok := parseByteToken(in)
		if ok != want.ok || (ok && b != want.b) {
			t.Errorf("parseByteToken(%q) = %#x, %v; want %#x, %v", in, b, ok, want.b, want.ok)
		}
	}
}

func byteTokenName(b int) string {
	const hex = "0123456789ABCDEF"
	return "<0x" + string(hex[b>>4]) + string(hex[b&0xf]) + ">"
}

func TestNewFromGGUF(t *testing.T) {
	// New must reject a file without a vocabulary rather than building a
	// tokenizer that silently maps everything to nothing.
	if _, err := New(&gguf.File{Metadata: map[string]gguf.Value{}}); err == nil {
		t.Fatal("New(no tokens) succeeded, want error")
	}
}

func BenchmarkEncode(b *testing.B) {
	tok := newTestTokenizer(b)
	const text = "Once upon a time there was a little girl named Lily."
	b.SetBytes(int64(len(text)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.Encode(text, true)
	}
}
