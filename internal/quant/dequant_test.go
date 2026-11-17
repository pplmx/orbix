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
	// Q8_K is a real ggml type this package does not decode: it quantizes activations
	// rather than weights. It must be reported as unsupported rather than decoded with a
	// neighbouring format's layout.
	if err := Decode(gguf.Q8_K, make([]byte, 292), make([]float32, 256)); err == nil {
		t.Fatal("Decode(Q8_K) succeeded, want an unsupported-type error")
	}
	if Supported(gguf.Q8_K) {
		t.Error("Supported(Q8_K) = true")
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

// K-quants use 256-weight super-blocks with per-sub-block scales, which makes them the
// first formats here whose layout cannot be checked by eye. Two tests below: exact
// values on constructed blocks, and a round trip whose error is bounded by the format's
// step size.

func TestDecodeQ4_K(t *testing.T) {
	// d=1.0, dmin=0.5, sub-block scales 1..8, mins 8..1, and codes that make a
	// permutation visible: low nibble i+1, high nibble 255-i.
	sc := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	mn := []byte{8, 7, 6, 5, 4, 3, 2, 1}
	src := buildKBlock(t, 1.0, 0.5, sc, mn, q4KCodes())
	dst := make([]float32, 256)
	if err := Decode(gguf.Q4_K, src, dst); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	for it := range 4 {
		lo := float32(sc[2*it])
		hi := float32(sc[2*it+1])
		mlo := 0.5 * float32(mn[2*it])
		mhi := 0.5 * float32(mn[2*it+1])
		for l := range 32 {
			q := q4KCodes()[it*32+l]
			if want := lo*float32(q&0x0F) - mlo; dst[it*64+l] != want {
				t.Fatalf("sub-block %d low[%d] = %v, want %v", it, l, dst[it*64+l], want)
			}
			if want := hi*float32(q>>4) - mhi; dst[it*64+32+l] != want {
				t.Fatalf("sub-block %d high[%d] = %v, want %v", it, l, dst[it*64+32+l], want)
			}
		}
	}
}

// q4KCodes is the 128-byte code section the Q4_K layout test uses.
func q4KCodes() []byte {
	out := make([]byte, 128)
	for i := range out {
		out[i] = byte((i+1)&0x0F) | byte((255-i)&0x0F)<<4
	}
	return out
}

// buildKBlock assembles a Q4_K block: F16 d, F16 dmin, packed scales, 128 code bytes.
func buildKBlock(t *testing.T, d, dmin float32, sc, mn, qs []byte) []byte {
	t.Helper()
	out := make([]byte, 0, 144)
	out = appendF16(t, out, d)
	out = appendF16(t, out, dmin)
	var packed [12]byte
	for i := range 4 {
		packed[i] = (sc[i] & 0x3F) | ((sc[i+4] >> 4) << 6)
		packed[i+4] = (mn[i] & 0x3F) | ((mn[i+4] >> 4) << 6)
		packed[8+i] = (sc[i+4] & 0x0F) | ((mn[i+4] & 0x0F) << 4)
	}
	out = append(out, packed[:]...)
	return append(out, qs...)
}

func TestGetScaleMinK4(t *testing.T) {
	// The packing is dense enough to deserve its own test: eight 6-bit scales and eight
	// 6-bit mins in 12 bytes, with the last four of each split across two bytes.
	sc := []byte{1, 2, 3, 4, 5, 6, 7, 63}
	mn := []byte{8, 9, 10, 11, 12, 13, 14, 62}
	var q [12]byte
	for i := range 4 {
		q[i] = (sc[i] & 0x3F) | ((sc[i+4] >> 4) << 6)
		q[i+4] = (mn[i] & 0x3F) | ((mn[i+4] >> 4) << 6)
		q[8+i] = (sc[i+4] & 0x0F) | ((mn[i+4] & 0x0F) << 4)
	}
	for j := range 8 {
		gotSC, gotM := getScaleMinK4(j, q[:])
		if gotSC != sc[j] || gotM != mn[j] {
			t.Errorf("j=%d: scale/min = %d/%d, want %d/%d", j, gotSC, gotM, sc[j], mn[j])
		}
	}
}

func TestDecodeQ6_K(t *testing.T) {
	// d=0.25, signed scales -8..7, and distinguishable quants. Q6_K stores d last.
	src := buildQ6KBlock(t, 0.25, q6KCodes())
	dst := make([]float32, 256)
	if err := Decode(gguf.Q6_K, src, dst); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	ql, qh, scales := q6KParts(q6KCodes())
	for half := range 2 {
		for l := range 32 {
			is := l / 16
			h := qh[half*32+l]
			wants := [4]float32{
				0.25 * float32(int8(scales[half*8+is])) * float32(int8(ql[half*64+l]&0x0F|((h&3)<<4))-32),
				0.25 * float32(int8(scales[half*8+is+2])) * float32(int8(ql[half*64+l+32]&0x0F|(((h>>2)&3)<<4))-32),
				0.25 * float32(int8(scales[half*8+is+4])) * float32(int8(ql[half*64+l]>>4|(((h>>4)&3)<<4))-32),
				0.25 * float32(int8(scales[half*8+is+6])) * float32(int8(ql[half*64+l+32]>>4|(((h>>6)&3)<<4))-32),
			}
			for k, want := range wants {
				idx := half*128 + l + k*32
				if dst[idx] != want {
					t.Fatalf("half %d l %d q%d: [%d] = %v, want %v", half, l, k+1, idx, dst[idx], want)
				}
			}
		}
	}
}

// q6KCodes is the 192-byte ql+qh section the Q6_K layout test uses.
func q6KCodes() []byte {
	out := make([]byte, 192)
	for i := range 128 {
		out[i] = byte(i&0x0F) | byte((i*3)&0x0F)<<4
	}
	for i := range 64 {
		out[128+i] = byte(i&3) | byte((i+1)&3)<<2 | byte((i+2)&3)<<4 | byte((i+3)&3)<<6
	}
	return out
}

// q6KParts splits the test block's code section into its three regions.
func q6KParts(codes []byte) (ql, qh, scales []byte) {
	ql = codes[0:128]
	qh = codes[128:192]
	scales = make([]byte, 16)
	for i := range scales {
		scales[i] = byte((i % 16) - 8)
	}
	return ql, qh, scales
}

// buildQ6KBlock assembles a Q6_K block: 128 ql, 64 qh, 16 signed scales, F16 d.
func buildQ6KBlock(t *testing.T, d float32, codes []byte) []byte {
	t.Helper()
	out := append([]byte(nil), codes...)
	_, _, scales := q6KParts(codes)
	out = append(out, scales...)
	return appendF16(t, out, d)
}

// TestKQuantRoundTrip bounds each K-quant decoder's error by the format's step size.
//
// A wrong sub-block scale or a nibble-order mistake still decodes to plausible
// magnitudes, so the exact tests above are what catch a permutation; this one catches a
// decoder that is self-consistent but disagrees with the encoder about magnitudes.
func TestKQuantRoundTrip(t *testing.T) {
	r := rand.New(rand.NewSource(17))
	const blocks = 64
	src := make([]float32, blocks*256)
	for i := range src {
		src[i] = float32(r.NormFloat64() * 0.06)
	}

	tests := []struct {
		name     string
		typ      gguf.QuantType
		encode   func([]float32) []byte
		maxError float64
	}{
		{"Q4_K", gguf.Q4_K, encodeQ4K, 0.05},
		{"Q6_K", gguf.Q6_K, encodeQ6K, 0.01},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			enc := tc.encode(src)
			got := make([]float32, len(src))
			if err := Decode(tc.typ, enc, got); err != nil {
				t.Fatalf("Decode: %v", err)
			}
			var maxErr, sum float64
			for i := range src {
				e := math.Abs(float64(got[i] - src[i]))
				maxErr = math.Max(maxErr, e)
				sum += e
			}
			mean := sum / float64(len(src))
			t.Logf("%s: max error %.5f, mean %.6f", tc.name, maxErr, mean)
			if maxErr > tc.maxError {
				t.Errorf("max error %.5f exceeds the bound %.3f", maxErr, tc.maxError)
			}
			if mean == 0 {
				t.Error("decoded values are identical to the source")
			}
		})
	}
}

// encodeQ4K quantizes with Q4_K's layout: eight sub-blocks of 32, each with its own
// 6-bit scale and min.
//
// d and dmin are the super-block references the 6-bit fields are measured against. The
// encoder picks them from the largest sub-block range, which is how a real encoder does
// it: the fields can only express a ratio to d, so d has to be large enough that the
// biggest sub-block still fits in six bits.
func encodeQ4K(src []float32) []byte {
	out := make([]byte, 0, len(src)/256*144)
	for b := 0; b < len(src); b += 256 {
		blk := src[b : b+256]

		// First pass: each sub-block's range, and the largest of them, which sets d.
		type subRange struct{ lo, hi float32 }
		var ranges [8]subRange
		maxStep := float32(0)
		for sb := range 8 {
			sub := blk[sb*32 : (sb+1)*32]
			lo, hi := sub[0], sub[0]
			for _, v := range sub {
				lo = min(lo, v)
				hi = max(hi, v)
			}
			ranges[sb] = subRange{lo: lo, hi: hi}
			maxStep = max(maxStep, (hi-lo)/15)
		}
		if maxStep == 0 {
			maxStep = 1
		}
		d := maxStep / 63 // so the largest sub-block's scale fits in six bits
		if d == 0 {
			d = 1
		}
		maxMin := float32(0)
		for _, r := range ranges {
			maxMin = max(maxMin, -r.lo)
		}
		dmin := maxMin / 63
		if dmin == 0 {
			dmin = d
		}

		var sc, mn [8]byte
		var codes [128]byte
		for sb := range 8 {
			r := ranges[sb]
			step := (r.hi - r.lo) / 15
			if step == 0 {
				step = 1
			}
			sc[sb] = byte(min(max(int(math.Round(float64(step/d))), 1), 63))
			mn[sb] = byte(min(max(int(math.Round(float64(-r.lo/dmin))), 0), 63))
			for i, v := range blk[sb*32 : (sb+1)*32] {
				putQ4KCode(codes[:], sb, i, byte(min(max(int(math.Round(float64((v-r.lo)/step))), 0), 15)))
			}
		}
		out = appendF16Bytes(out, d)
		out = appendF16Bytes(out, dmin)
		var packed [12]byte
		for i := range 4 {
			packed[i] = (sc[i] & 0x3F) | ((sc[i+4] >> 4) << 6)
			packed[i+4] = (mn[i] & 0x3F) | ((mn[i+4] >> 4) << 6)
			packed[8+i] = (sc[i+4] & 0x0F) | ((mn[i+4] & 0x0F) << 4)
		}
		out = append(out, packed[:]...)
		out = append(out, codes[:]...)
	}
	return out
}

// putQ4KCode writes sub-block sb's element i into the Q4_K code layout.
//
// Sub-blocks are packed in pairs: byte (sb/2)*32 + i carries sub-block sb's element i in
// its low nibble and sub-block sb+1's in its high nibble. Writing a code to the wrong
// nibble leaves the neighbouring sub-block's values intact but every other sub-block
// decoding to its partner's data, which is why this is a helper rather than an index.
func putQ4KCode(codes []byte, sb, i int, code byte) {
	idx := (sb/2)*32 + i
	if sb%2 == 0 {
		codes[idx] = codes[idx]&0xF0 | code&0x0F
		return
	}
	codes[idx] = codes[idx]&0x0F | (code&0x0F)<<4
}

// encodeQ6K quantizes with Q6_K's layout: 16 groups of 16, each with an int8 scale.
func encodeQ6K(src []float32) []byte {
	out := make([]byte, 0, len(src)/256*210)
	for b := 0; b < len(src); b += 256 {
		blk := src[b : b+256]
		// One scale per 16 weights, chosen so the block's largest magnitude fits in
		// the 6-bit range (-32..31).
		peak := float32(0)
		for _, v := range blk {
			peak = max(peak, float32(math.Abs(float64(v))))
		}
		d := peak / 31
		if d == 0 {
			d = 1
		}
		var q [256]int
		for i, v := range blk {
			q[i] = min(max(int(math.Round(float64(v/d))), -32), 31)
		}
		ql := make([]byte, 128)
		qh := make([]byte, 64)
		scales := make([]byte, 16)
		for i := range scales {
			scales[i] = 1
		}
		for half := range 2 {
			for l := range 32 {
				q1 := q[half*128+l] + 32
				q2 := q[half*128+l+32] + 32
				q3 := q[half*128+l+64] + 32
				q4 := q[half*128+l+96] + 32
				ql[half*64+l] = byte(q1&0x0F) | byte(q3&0x0F)<<4
				ql[half*64+l+32] = byte(q2&0x0F) | byte(q4&0x0F)<<4
				qh[half*32+l] = byte((q1>>4)&3) | byte((q2>>4)&3)<<2 | byte((q3>>4)&3)<<4 | byte((q4>>4)&3)<<6
			}
		}
		out = append(out, ql...)
		out = append(out, qh...)
		out = append(out, scales...)
		out = appendF16Bytes(out, d)
	}
	return out
}

// The K-quants and IQ4_NL below were implemented against ggml's reference decoder and then
// checked value-for-value against it: the Go test compares against arrays produced by
// `uv run --with gguf` calling llama.cpp's own dequantize, over ~280 million weights taken
// from real checkpoints. The subsets kept here are small enough to run in CI.

func TestDecodeQ2_K(t *testing.T) {
	// d=0.5, dmin=0.25, scale/min nibbles 1..16, and code bytes that vary with position.
	scales := make([]byte, 16)
	for i := range scales {
		scales[i] = byte((i + 1) & 0x0F)
		scales[i] |= byte((15 - i) & 0x0F << 4)
	}
	src := make([]byte, 0, 84)
	src = append(src, scales...)
	for i := range 64 {
		src = append(src, byte(i*37))
	}
	src = appendF16(t, src, 0.5)
	src = appendF16(t, src, 0.25)

	dst := make([]float32, 256)
	if err := Decode(gguf.Q2_K, src, dst); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	shifts := [4]uint{0, 2, 4, 6}
	for sb := range 16 {
		group := src[16+(sb/8)*32 : 16+(sb/8+1)*32]
		shift := shifts[(sb%8)/2]
		offset := (sb % 2) * 16
		sc := scales[sb]
		dl := 0.5 * float32(sc&0x0F)
		ml := 0.25 * float32(sc>>4)
		for l := range 16 {
			code := float32((group[l+offset] >> shift) & 3)
			if want := dl*code - ml; dst[sb*16+l] != want {
				t.Fatalf("sb %d l %d: %v, want %v", sb, l, dst[sb*16+l], want)
			}
		}
	}
}

func TestDecodeQ5_K(t *testing.T) {
	// d=1.0, dmin=0.5, distinct sub-block scales/mins, and qh bytes that set alternating
	// fifth bits so a wrong mask shows up immediately.
	sc := []byte{1, 2, 3, 4, 5, 6, 7, 63}
	mn := []byte{8, 7, 6, 5, 4, 3, 2, 1}
	src := make([]byte, 0, 176)
	src = appendF16(t, src, 1.0)
	src = appendF16(t, src, 0.5)
	var packed [12]byte
	for i := range 4 {
		packed[i] = (sc[i] & 0x3F) | ((sc[i+4] >> 4) << 6)
		packed[i+4] = (mn[i] & 0x3F) | ((mn[i+4] >> 4) << 6)
		packed[8+i] = (sc[i+4] & 0x0F) | ((mn[i+4] & 0x0F) << 4)
	}
	src = append(src, packed[:]...)
	for i := range 32 {
		src = append(src, byte(i*7))
	}
	ql := make([]byte, 128)
	for i := range ql {
		ql[i] = byte(i&0x0F) | byte((255-i)&0x0F)<<4
	}
	src = append(src, ql...)

	dst := make([]float32, 256)
	if err := Decode(gguf.Q5_K, src, dst); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	qh := src[16:48]
	u1, u2 := byte(1), byte(2)
	for it := range 4 {
		sc0, m0 := getScaleMinK4(2*it, packed[:])
		sc1, m1 := getScaleMinK4(2*it+1, packed[:])
		d1, mf1 := float32(sc0), 0.5*float32(m0)
		d2, mf2 := float32(sc1), 0.5*float32(m1)
		for l := range 32 {
			lo := ql[it*32+l] & 0x0F
			if qh[l]&u1 != 0 {
				lo += 16
			}
			hi := ql[it*32+l] >> 4
			if qh[l]&u2 != 0 {
				hi += 16
			}
			if want := d1*float32(lo) - mf1; dst[it*64+l] != want {
				t.Fatalf("it %d lo %d: %v, want %v", it, l, dst[it*64+l], want)
			}
			if want := d2*float32(hi) - mf2; dst[it*64+32+l] != want {
				t.Fatalf("it %d hi %d: %v, want %v", it, l, dst[it*64+32+l], want)
			}
		}
		u1 <<= 2
		u2 <<= 2
	}
}

func TestDecodeIQ4NL(t *testing.T) {
	// The codebook is the whole format: a nibble indexes into it rather than expressing a
	// value, so a decoder that treats it as a number is wrong in a way that still looks
	// like a weight matrix.
	src := make([]byte, 18)
	src[0], src[1] = 0x00, 0x3C // f16 1.0
	for j := range 16 {
		src[2+j] = byte(j) | byte(15-j)<<4
	}
	dst := make([]float32, 32)
	if err := Decode(gguf.IQ4_NL, src, dst); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	for j := range 16 {
		if want := float32(kValuesIQ4NL[j]); dst[j] != want {
			t.Errorf("low nibble %d: %v, want %v", j, dst[j], want)
		}
		if want := float32(kValuesIQ4NL[15-j]); dst[j+16] != want {
			t.Errorf("high nibble %d: %v, want %v", j, dst[j+16], want)
		}
	}
}

func TestIQ4NLDiffersFromQ4_0(t *testing.T) {
	// Same bytes, different format: if these agreed, the codebook would not be applied.
	src := make([]byte, 18)
	src[0], src[1] = 0x00, 0x3C
	for j := range 16 {
		src[2+j] = 0x10
	}
	iq := make([]float32, 32)
	q4 := make([]float32, 32)
	if err := Decode(gguf.IQ4_NL, src, iq); err != nil {
		t.Fatal(err)
	}
	if err := Decode(gguf.Q4_0, src, q4); err != nil {
		t.Fatal(err)
	}
	if iq[0] == q4[0] {
		t.Errorf("IQ4_NL and Q4_0 decoded the same value %v; the codebook is not applied", iq[0])
	}
}

// TestDecodeQ3_K is the case the other K-quant tests do not cover, and the one whose layout
// is hardest to get right: its scales come from two differently-shaped grids and its codes
// from a third. The implementation follows ggml's numpy reference expression by expression
// because the index mappings cannot be derived from a loop order, which is exactly the
// situation where a test that re-derives them independently earns its keep.
//
// The expected values here are computed from the layout in the test itself, not from the
// decoder, so a wrong mapping in either place shows up as a mismatch rather than as two
// copies of the same mistake.
func TestDecodeQ3_K(t *testing.T) {
	// Block: 32 mask bytes, 64 code bytes, 12 packed scale bytes, F16 d.
	hm := make([]byte, 32)
	for i := range hm {
		// Vary the bits so a wrong mask index picks a different bit and changes the value.
		hm[i] = byte(0xAA ^ (i * 37))
	}
	qs := make([]byte, 64)
	for i := range qs {
		qs[i] = byte(i * 53)
	}
	scales := make([]byte, 12)
	for i := range 8 {
		scales[i] = byte(i&0x0F) | byte((15-i)&0x0F)<<4
	}
	for i := range 4 {
		scales[8+i] = byte((i*17)&0x03) | byte((i*29)&0x03)<<2 |
			byte((i*11)&0x03)<<4 | byte((i*7)&0x03)<<6
	}

	src := make([]byte, 0, 110)
	src = append(src, hm...)
	src = append(src, qs...)
	src = append(src, scales...)
	src = appendF16(t, src, 0.5)
	if len(src) != 110 {
		t.Fatalf("block is %d bytes, want 110", len(src))
	}

	dst := make([]float32, 256)
	if err := Decode(gguf.Q3_K, src, dst); err != nil {
		t.Fatalf("Decode: %v", err)
	}

	// Recompute the scales from the packed bytes, following the documented grids.
	var want [16]int8
	for i := range 8 {
		want[i] = int8(scales[i] & 0x0F)
		want[i+8] = int8(scales[i] >> 4)
	}
	for f := range 16 {
		want[f] |= int8(((scales[8+f%4] >> (2 * (f / 4))) & 0x03) << 4)
	}

	shifts := [4]uint{0, 2, 4, 6}
	for sb := range 16 {
		// Codes: group sb/8, shift shifts[(sb%8)/2], byte offset (sb%2)*16.
		group := qs[(sb/8)*32:]
		shift := shifts[(sb%8)/2]
		offset := (sb % 2) * 16
		// Mask: bit sb/2 of every mask byte.
		maskBit := uint(sb / 2)
		scale := float32(int(want[sb]) - 32)
		for l := range 16 {
			// The two bits of the low code plus one mask bit make the three-bit value:
			// the low part is 0..3 and the mask bit subtracts 4 when it is clear.
			idx := l + offset
			ql := int32((group[idx] >> shift) & 0x03)
			qh := (hm[idx] >> maskBit) & 0x01
			code := ql - 4*(1-int32(qh))
			wantV := 0.5 * scale * float32(code)
			if got := dst[sb*16+l]; got != wantV {
				t.Fatalf("dst[%d] = %v, want %v (sb=%d l=%d ql=%d qh=%d scaleField=%d)",
					sb*16+l, got, wantV, sb, l, ql, qh, int(want[sb]))
			}
		}
	}
}
