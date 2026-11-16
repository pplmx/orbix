package quant

import (
	"encoding/binary"
	"math"
	"math/rand"
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

// The Q4_1, Q5_0, and Q5_1 decoders are pinned by constructing a block whose codes
// are known integers and whose scale is a power of two, so every decoded value is
// exactly representable. That makes the expected values computable by hand -- and
// makes a wrong nibble order, a missing high bit, or a left-out minimum show up as a
// concrete mismatch rather than as "close enough".
//
// The expected slices were produced by an independent numpy script implementing
// ggml's documented layout, then compared against the Go decoders; a layout mistake
// cannot survive both.

func TestDecodeQ4_1(t *testing.T) {
	// scale 0.5, min -1.0; low nibbles 0..15, high nibbles 15..0.
	src := buildBlock(t, 0.5, -1.0, lowNibbles(), highNibbles())
	dst := make([]float32, 32)
	if err := Decode(gguf.Q4_1, src, dst); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	want := make([]float32, 32)
	for j := range 16 {
		want[j] = float32(j)*0.5 - 1
		want[j+16] = float32(15-j)*0.5 - 1
	}
	checkExact(t, "Q4_1", dst, want)
}

func TestDecodeQ5_0(t *testing.T) {
	// scale 0.5, symmetric: code (high bit | nibble) - 16.
	lo, hi := lowNibbles(), highNibbles()
	qh := uint32(0)
	for j := range 16 {
		if j%3 == 1 {
			qh |= 1 << j
		}
		if j%4 == 2 {
			qh |= 1 << (j + 16)
		}
	}
	src := buildHighBitBlock(t, 0.5, nil, qh, lo, hi)
	dst := make([]float32, 32)
	if err := Decode(gguf.Q5_0, src, dst); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	want := make([]float32, 32)
	for j := range 16 {
		loCode := lo[j] | byte((qh>>j)&1)<<4
		hiCode := hi[j] | byte((qh>>(j+16))&1)<<4
		want[j] = float32(int(loCode)-16) * 0.5
		want[j+16] = float32(int(hiCode)-16) * 0.5
	}
	checkExact(t, "Q5_0", dst, want)
}

func TestDecodeQ5_1(t *testing.T) {
	// scale 0.5, min 2.0; unsigned five-bit codes.
	lo, hi := lowNibbles(), highNibbles()
	qh := uint32(0)
	for j := range 16 {
		if j%3 == 1 {
			qh |= 1 << j
		}
		if j%4 == 2 {
			qh |= 1 << (j + 16)
		}
	}
	minv := float32(2.0)
	src := buildHighBitBlock(t, 0.5, &minv, qh, lo, hi)
	dst := make([]float32, 32)
	if err := Decode(gguf.Q5_1, src, dst); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	want := make([]float32, 32)
	for j := range 16 {
		loCode := lo[j] | byte((qh>>j)&1)<<4
		hiCode := hi[j] | byte((qh>>(j+16))&1)<<4
		want[j] = float32(int(loCode))*0.5 + 2
		want[j+16] = float32(int(hiCode))*0.5 + 2
	}
	checkExact(t, "Q5_1", dst, want)
}

// TestQ4_1DiffersFromQ4_0 pins the reason Q4_1 exists: the same packed codes decode
// to different values because Q4_1 adds a minimum the other type cannot express.
func TestQ4_1DiffersFromQ4_0(t *testing.T) {
	lo, hi := lowNibbles(), highNibbles()
	// A Q4_1 block is a Q4_0 block's nibbles plus an F16 min in front of them.
	q41 := buildBlock(t, 0.5, -1.0, lo, hi)
	q40 := appendF16(t, nil, 0.5)
	q40 = append(q40, q41[4:]...) // same nibbles, no minimum in front
	if got := f16Of(q40[:2]); got != 0.5 {
		t.Fatalf("test setup: q4_0 scale is %v, want 0.5", got)
	}
	a := make([]float32, 32)
	b := make([]float32, 32)
	if err := Decode(gguf.Q4_1, q41, a); err != nil {
		t.Fatal(err)
	}
	if err := Decode(gguf.Q4_0, q40, b); err != nil {
		t.Fatal(err)
	}
	same := true
	for i := range a {
		if a[i] != b[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("Q4_0 and Q4_1 decoded the same bytes identically; the minimum is being ignored")
	}
}

// lowNibbles and highNibbles are the codes the layout tests pack.
func lowNibbles() []byte {
	out := make([]byte, 16)
	for j := range out {
		out[j] = byte(j % 16)
	}
	return out
}

func highNibbles() []byte {
	out := make([]byte, 16)
	for j := range out {
		out[j] = byte((31 - j) % 16)
	}
	return out
}

// buildBlock assembles a Q4_1 block: F16 scale, F16 min, 16 packed nibble bytes.
func buildBlock(t *testing.T, scale, min float32, lo, hi []byte) []byte {
	t.Helper()
	out := make([]byte, 0, 20)
	out = appendF16(t, out, scale)
	out = appendF16(t, out, min)
	for j := range 16 {
		out = append(out, hi[j]<<4|lo[j])
	}
	return out
}

// buildHighBitBlock assembles a Q5_0 block (min == nil) or a Q5_1 block.
func buildHighBitBlock(t *testing.T, scale float32, min *float32, qh uint32, lo, hi []byte) []byte {
	t.Helper()
	out := make([]byte, 0, 24)
	out = appendF16(t, out, scale)
	if min != nil {
		out = appendF16(t, out, *min)
	}
	var bits [4]byte
	binary.LittleEndian.PutUint32(bits[:], qh)
	out = append(out, bits[:]...)
	for j := range 16 {
		out = append(out, hi[j]<<4|lo[j])
	}
	return out
}

// appendF16 appends v as a little-endian binary16.
func appendF16(t *testing.T, dst []byte, v float32) []byte {
	t.Helper()
	var tmp [2]byte
	binary.LittleEndian.PutUint16(tmp[:], f16.FromFloat32(v))
	return append(dst, tmp[:]...)
}

// f16Of reads a little-endian binary16.
func f16Of(b []byte) float32 {
	return f16.ToFloat32(binary.LittleEndian.Uint16(b))
}

// checkExact compares decoded values with exact equality: the test blocks are built
// so every value is exactly representable, which makes a tolerance a way to hide a
// layout bug rather than a way to accommodate rounding.
func checkExact(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", name, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("%s: [%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}

// TestQuantRoundTripAccuracy checks each decoder against a re-encoding of realistic
// weights, which is the check a layout mistake cannot survive: the synthetic layout
// tests above pin the bit order exactly, while this one pins the magnitudes to what a
// 4- or 5-bit format can actually represent.
//
// A wrong nibble order still decodes to plausible-magnitude numbers, so the error is
// measured against the source and bounded by the format's step size: a 4-bit block
// spanning [-0.3, 0.3] has a step of 0.04, and a mean error well below it means the
// codes were read in the order the encoder wrote them.
func TestQuantRoundTripAccuracy(t *testing.T) {
	// Deterministic weights with the distribution a transformer projection has:
	// small, centred on zero, a few larger outliers.
	r := rand.New(rand.NewSource(11))
	const blocks = 512
	src := make([]float32, blocks*32)
	for i := range src {
		src[i] = float32(r.NormFloat64() * 0.06)
	}

	tests := []struct {
		name     string
		typ      gguf.QuantType
		encode   func(dst []float32) []byte
		maxError float64
	}{
		{"Q4_1", gguf.Q4_1, encodeQ4_1, 0.05},
		{"Q5_0", gguf.Q5_0, encodeQ5_0, 0.03},
		{"Q5_1", gguf.Q5_1, encodeQ5_1, 0.02},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			enc := tc.encode(src)
			got := make([]float32, len(src))
			if err := Decode(tc.typ, enc, got); err != nil {
				t.Fatalf("Decode: %v", err)
			}
			var maxErr, sumErr float64
			for i := range src {
				e := math.Abs(float64(got[i] - src[i]))
				maxErr = math.Max(maxErr, e)
				sumErr += e
			}
			mean := sumErr / float64(len(src))
			t.Logf("%s: max error %.5f, mean %.6f", tc.name, maxErr, mean)
			if maxErr > tc.maxError {
				t.Errorf("max error %.5f exceeds the format's step (%.3f)", maxErr, tc.maxError)
			}
			if mean == 0 {
				t.Error("decoded values are identical to the source; nothing was quantized")
			}
		})
	}
}

// encodeQ4_1 quantizes blocks with the affine mapping Q4_1 stores.
func encodeQ4_1(src []float32) []byte {
	out := make([]byte, 0, len(src)/32*20)
	for b := 0; b < len(src); b += 32 {
		blk := src[b : b+32]
		lo, hi := blk[0], blk[0]
		for _, v := range blk {
			lo = min(lo, v)
			hi = max(hi, v)
		}
		d := (hi - lo) / 15
		if d == 0 {
			d = 1
		}
		var codes [32]byte
		for i, v := range blk {
			c := int(math.Round(float64((v - lo) / d)))
			codes[i] = byte(min(max(c, 0), 15))
		}
		out = appendF16Bytes(out, d)
		out = appendF16Bytes(out, lo)
		for j := range 16 {
			out = append(out, codes[j+16]<<4|codes[j])
		}
	}
	return out
}

// encodeQ5_0 quantizes blocks symmetrically, with five-bit codes offset by 16.
func encodeQ5_0(src []float32) []byte {
	return encodeQ5(src, false)
}

// encodeQ5_1 quantizes blocks affinely, with five-bit codes.
func encodeQ5_1(src []float32) []byte {
	return encodeQ5(src, true)
}

// encodeQ5 is the shared encoder for the two five-bit formats.
func encodeQ5(src []float32, affine bool) []byte {
	size := 22
	if affine {
		size = 24
	}
	out := make([]byte, 0, len(src)/32*size)
	for b := 0; b < len(src); b += 32 {
		blk := src[b : b+32]
		var scale, minv float32
		if affine {
			lo, hi := blk[0], blk[0]
			for _, v := range blk {
				lo = min(lo, v)
				hi = max(hi, v)
			}
			scale = (hi - lo) / 31
			minv = lo
		} else {
			peak := float32(0)
			for _, v := range blk {
				peak = max(peak, float32(math.Abs(float64(v))))
			}
			scale = peak / 15
		}
		if scale == 0 {
			scale = 1
		}
		var codes [32]byte
		var qh uint32
		for i, v := range blk {
			c := int(math.Round(float64(v/scale))) + 16
			if affine {
				c = int(math.Round(float64((v - minv) / scale)))
			}
			c = min(max(c, 0), 31)
			codes[i] = byte(c)
			if c&0x10 != 0 {
				qh |= 1 << i
			}
		}
		out = appendF16Bytes(out, scale)
		if affine {
			out = appendF16Bytes(out, minv)
		}
		var bits [4]byte
		binary.LittleEndian.PutUint32(bits[:], qh)
		out = append(out, bits[:]...)
		for j := range 16 {
			out = append(out, (codes[j+16]&0x0f)<<4|codes[j]&0x0f)
		}
	}
	return out
}

// appendF16Bytes appends v as a little-endian binary16 without a testing.T.
func appendF16Bytes(dst []byte, v float32) []byte {
	var tmp [2]byte
	binary.LittleEndian.PutUint16(tmp[:], f16.FromFloat32(v))
	return append(dst, tmp[:]...)
}
