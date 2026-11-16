package quant

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/pplmx/orbix/internal/f16"
	"github.com/pplmx/orbix/internal/gguf"
)

// encodeF16 is a test-side helper: it writes a float as a binary16 word.
func encodeF16(t *testing.T, f float32) []byte {
	t.Helper()
	h := f16.FromFloat32(f)
	var b [2]byte
	binary.LittleEndian.PutUint16(b[:], h)
	// Guard against a test that accidentally uses a value the half type
	// cannot represent, which would make the expectation wrong rather than
	// the decoder.
	if got := f16.ToFloat32(h); got != f {
		t.Fatalf("test value %v is not representable in binary16 (got %v)", f, got)
	}
	return b[:]
}

// TestCodecsMatchGGUF pins every decoder to the size table gguf declares.
// The two tables are maintained separately, so a mismatch here means one of
// them is wrong and the other would read garbage.
func TestCodecsMatchGGUF(t *testing.T) {
	for typ, c := range codecs {
		block, ok := typ.BlockSize()
		if !ok {
			t.Errorf("%s: decoder exists but gguf does not know the type", typ)
			continue
		}
		if c.block != block {
			t.Errorf("%s: decoder block = %d, gguf block = %d", typ, c.block, block)
		}
		size, ok := typ.Size(int64(c.block))
		if !ok {
			t.Errorf("%s: gguf size lookup failed", typ)
			continue
		}
		if int64(c.size) != size {
			t.Errorf("%s: decoder block size = %d bytes, gguf = %d", typ, c.size, size)
		}
	}
}

func TestDecodeF32(t *testing.T) {
	v := []float32{1, -2.5, math.MaxFloat32, math.SmallestNonzeroFloat32}
	src := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(src[i*4:], math.Float32bits(f))
	}
	dst := make([]float32, len(v))
	if err := Decode(gguf.F32, src, dst); err != nil {
		t.Fatal(err)
	}
	for i := range v {
		if dst[i] != v[i] {
			t.Errorf("[%d] = %v, want %v", i, dst[i], v[i])
		}
	}
}

func TestDecodeF16(t *testing.T) {
	v := []float32{0, 1, -1, 0.5, 65504}
	src := make([]byte, 0, len(v)*2)
	for _, f := range v {
		src = append(src, encodeF16(t, f)...)
	}
	dst := make([]float32, len(v))
	if err := Decode(gguf.F16, src, dst); err != nil {
		t.Fatal(err)
	}
	for i := range v {
		if dst[i] != v[i] {
			t.Errorf("[%d] = %v, want %v", i, dst[i], v[i])
		}
	}
}

func TestDecodeBF16(t *testing.T) {
	// bfloat16 keeps the top 16 bits of a float32. 1.0 and -2.0 are exactly
	// representable; 1.00390625 shows that the low mantissa bits are dropped.
	words := []uint16{0x3f80, 0xc000, 0x3f81}
	want := []float32{1, -2, 1.0078125}
	src := make([]byte, len(words)*2)
	for i, w := range words {
		binary.LittleEndian.PutUint16(src[i*2:], w)
	}
	dst := make([]float32, len(words))
	if err := Decode(gguf.BF16, src, dst); err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if dst[i] != want[i] {
			t.Errorf("[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

// q4_0Block builds one Q4_0 block from codes (values 0..15) and a scale.
func q4_0Block(t *testing.T, codes []int, scale float32) []byte {
	t.Helper()
	if len(codes) != 32 {
		t.Fatalf("q4_0Block needs 32 codes, got %d", len(codes))
	}
	out := append([]byte{}, encodeF16(t, scale)...)
	for j := 0; j < 16; j++ {
		out = append(out, byte(codes[j]&0xf|(codes[j+16]&0xf)<<4))
	}
	return out
}

func TestDecodeQ4_0Layout(t *testing.T) {
	// The nibble interleaving is the easiest part of this format to get
	// wrong, and a wrong layout silently transposes halves of a matrix.
	//
	// Use codes that identify their index: code j = j % 16. Then weight j
	// must decode to (j%16 - 8) * scale.
	codes := make([]int, 32)
	for j := range codes {
		codes[j] = j % 16
	}
	const scale = 0.5
	src := q4_0Block(t, codes, scale)

	dst := make([]float32, 32)
	if err := Decode(gguf.Q4_0, src, dst); err != nil {
		t.Fatal(err)
	}
	for j := 0; j < 32; j++ {
		want := float32(codes[j]-8) * scale
		if dst[j] != want {
			t.Errorf("weight %d = %v, want %v", j, dst[j], want)
		}
	}
	// Spot-check the range: codes 0..15 decode to -8..7 before scaling.
	if dst[0] != -4 || dst[8] != 0 || dst[15] != 3.5 {
		t.Errorf("range check: [0]=%v [8]=%v [15]=%v", dst[0], dst[8], dst[15])
	}
}

func TestDecodeQ4_0MultipleBlocks(t *testing.T) {
	// Two blocks with different scales must not bleed into each other.
	src := append(q4_0Block(t, make([]int, 32), 1), q4_0Block(t, fill(32, 15), 2)...)
	dst := make([]float32, 64)
	if err := Decode(gguf.Q4_0, src, dst); err != nil {
		t.Fatal(err)
	}
	if dst[0] != -8 || dst[31] != -8 {
		t.Errorf("block 0 = %v .. %v, want -8", dst[0], dst[31])
	}
	if dst[32] != 14 || dst[63] != 14 {
		t.Errorf("block 1 = %v .. %v, want 14", dst[32], dst[63])
	}
}

func fill(n, v int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = v
	}
	return out
}

func TestDecodeQ8_0(t *testing.T) {
	// Q8_0 is one signed byte per weight in order, with no interleaving.
	src := append([]byte{}, encodeF16(t, 0.25)...)
	for j := 0; j < 32; j++ {
		src = append(src, byte(int8(j-16)))
	}
	dst := make([]float32, 32)
	if err := Decode(gguf.Q8_0, src, dst); err != nil {
		t.Fatal(err)
	}
	for j := 0; j < 32; j++ {
		want := float32(j-16) * 0.25
		if dst[j] != want {
			t.Errorf("weight %d = %v, want %v", j, dst[j], want)
		}
	}
}

func TestDecodeRejectsMismatchedSizes(t *testing.T) {
	tests := []struct {
		name string
		typ  gguf.QuantType
		src  int // bytes
		n    int // elements
	}{
		{"q4_0 short source", gguf.Q4_0, 17, 32},
		{"q4_0 long source", gguf.Q4_0, 19, 32},
		{"q4_0 partial block", gguf.Q4_0, 18, 16},
		{"q8_0 short source", gguf.Q8_0, 33, 32},
		{"f32 short source", gguf.F32, 7, 2},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dst := make([]float32, tc.n)
			if err := Decode(tc.typ, make([]byte, tc.src), dst); err == nil {
				t.Fatal("Decode succeeded, want a size error")
			}
		})
	}
}

func TestDecodeRejectsUnsupportedType(t *testing.T) {
	// Q4_K is a real ggml type that this package does not decode yet. It must
	// be reported as unsupported rather than decoded with the wrong layout.
	if err := Decode(gguf.Q4_K, make([]byte, 144), make([]float32, 256)); err == nil {
		t.Fatal("Decode(Q4_K) succeeded, want an unsupported-type error")
	}
	if Supported(gguf.Q4_K) {
		t.Error("Supported(Q4_K) = true")
	}
	if !Supported(gguf.Q4_0) {
		t.Error("Supported(Q4_0) = false")
	}
}

func BenchmarkDecodeQ4_0(b *testing.B) {
	// One 4096x4096 tensor is 16 Mi elements, the size of a 7B model's
	// attention projection.
	const n = 4096 * 4096
	src := make([]byte, n/32*18)
	dst := make([]float32, n)
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Decode(gguf.Q4_0, src, dst)
	}
}

func BenchmarkDecodeQ8_0(b *testing.B) {
	const n = 4096 * 4096
	src := make([]byte, n/32*34)
	dst := make([]float32, n)
	b.SetBytes(int64(len(src)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Decode(gguf.Q8_0, src, dst)
	}
}
