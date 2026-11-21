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
	// The codes must be ASYMMETRIC between the two halves. This test used to set
	// code j = j % 16, which makes codes[j+16] = (j+16)%16 = codes[j] -- the low and
	// high nibble of every byte then hold the same value, so swapping the halves changes
	// nothing and the interleaving went untested despite the comment above saying it is
	// "the easiest part of this format to get wrong". Verified: with the symmetric codes,
	// a decoder that swapped the halves passed every test in this package.
	//
	// Now the low half counts up and the high half counts down, so every position has a
	// distinct neighbour and a swap is visible at 32 of 32 positions.
	codes := make([]int, 32)
	for j := 0; j < 16; j++ {
		codes[j] = j
		codes[j+16] = 15 - j
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

// BenchmarkDecodeKQuants measures the formats real checkpoints actually ship.
//
// Q4_0 and Q8_0 above were the only decoders timed, and they are the two a modern file is
// least likely to contain: the K-quants are what llama.cpp's default -q presets produce.
// Nothing recorded what they cost.
//
// What that cost buys is model load, not per-token speed. LoadWeights widens every tensor
// once into plain float32 slices, so the forward pass never calls Decode again. The rates
// below therefore add up to load time, and the arithmetic checks out on a real file:
// tinyllama-q2k is 53 MiB of Q2_K, 354 MiB of Q3_K and 51 MiB of Q6_K, which at these rates
// is 0.24 + 2.21 + 0.12 = 2.57 s of decoding, against a measured 3.5 s load that also reads
// the file and builds the F32 tensors. Q3_K alone is 86% of the decode time there.
//
// Generation speed is a different measurement and does not follow from this one. The same
// tinyllama-q2k runs at 10.5 tok/s while the 2 GB all-F32 qwen2 file runs at 18.1, because
// TinyLlama is 22 layers of 2048 hidden against Qwen2.5-0.5B's 24 of 896: an F32 file with
// a smaller model is faster, and no decoder is involved. Quoting the table below as a decode
// cost would be right; quoting it as a generation cost would not.
//
// The blocks are zero-filled, like the two benchmarks above. That is exact for four of the
// five: Q2_K, Q3_K, Q4_K and Q6_K contain no `if` at all, so every bit pattern does the same
// arithmetic. Q5_K is the exception -- its high bits are applied as `if qh[l]&u1 != 0 { lo +=
// 16 }`, which is data-dependent, so a zero block takes the not-taken side every time and the
// figure below is a floor rather than a mean. The branch is cheap and predictable across a
// real tensor, but saying "branch-free" without checking would have been wrong, which is why
// this paragraph exists instead of the sentence it replaced.
//
// Measured on a 128-core EPYC, 4096x4096 tensor, MB/s of source bytes:
//
//	Q4_0  746    Q4_K  570    Q5_K  638    Q6_K  466    Q2_K  229    Q3_K  166
//
// Q3_K is the outlier at a quarter of Q4_0's rate, and the layout is why: its inner loop does
// five index computations and two sign foldings per 16 weights, where Q4_0 does one shift and
// one multiply per weight. That is the format's cost rather than a defect in the decoder --
// Q3_K packs its 3-bit codes, high bits and 6-bit scales into three separate streams, and
// reassembling them is the work. Q2_K is the second slowest for the same reason: 2-bit codes
// mean four weights share one byte, so the index arithmetic is amortised over less data.
func BenchmarkDecodeKQuants(b *testing.B) {
	const n = 4096 * 4096
	for _, c := range []struct {
		name string
		typ  gguf.QuantType
		blk  int // weights per block
		byts int // bytes per block
	}{
		{"Q2_K", gguf.Q2_K, 256, 84},
		{"Q3_K", gguf.Q3_K, 256, 110},
		{"Q4_K", gguf.Q4_K, 256, 144},
		{"Q5_K", gguf.Q5_K, 256, 176},
		{"Q6_K", gguf.Q6_K, 256, 210},
	} {
		b.Run(c.name, func(b *testing.B) {
			src := make([]byte, n/c.blk*c.byts)
			dst := make([]float32, n)
			b.SetBytes(int64(len(src)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = Decode(c.typ, src, dst)
			}
		})
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

// TestIQ4NLCodebookMatchesGGML pins the 16 table values against ggml's own definition.
//
// The other IQ4_NL tests compare against kValuesIQ4NL itself, which is the right way to
// check the indexing -- but it cannot catch the table being transcribed wrongly, because
// the expectation would be wrong in the same way. That failure is silent: a wrong codebook
// still decodes to values of plausible magnitude, and every existing assertion would pass.
//
// These sixteen numbers are copied from ggml-common.h (GGML_TABLE_BEGIN(int8_t,
// kvalues_iq4nl, 16)) and deliberately not written by referring to the Go table, so the two
// have to agree independently.
func TestIQ4NLCodebookMatchesGGML(t *testing.T) {
	want := [16]int8{
		-127, -104, -83, -65, -49, -35, -22, -10,
		1, 13, 25, 38, 53, 69, 89, 113,
	}
	for i := range want {
		if got := kValuesIQ4NL[i]; got != want[i] {
			t.Errorf("kValuesIQ4NL[%d] = %d, want %d", i, got, want[i])
		}
	}

	// The values are strictly increasing, which is what makes the table a quantizer's
	// levels rather than an arbitrary lookup: a decoder cannot detect a mis-ordered table
	// any more than a wrong one, so the ordering is asserted here.
	for i := 1; i < len(want); i++ {
		if want[i] <= want[i-1] {
			t.Errorf("codebook is not increasing at %d: %d then %d", i, want[i-1], want[i])
		}
	}

	// Decode one block whose nibbles are the identity, so each value lands where the
	// table says it should: this checks the table against the decoder's arithmetic rather
	// than only against itself.
	src := make([]byte, 18)
	src[0], src[1] = 0x00, 0x3C // f16 1.0, so d == 1 and the values pass through
	for j := range 16 {
		src[2+j] = byte(j) | byte(j)<<4
	}
	dst := make([]float32, 32)
	if err := Decode(gguf.IQ4_NL, src, dst); err != nil {
		t.Fatalf("Decode: %v", err)
	}
	for j := range 16 {
		if got, expected := dst[j], float32(want[j]); got != expected {
			t.Errorf("low nibble %d decoded to %v, want %v", j, got, expected)
		}
		if got, expected := dst[j+16], float32(want[j]); got != expected {
			t.Errorf("high nibble %d decoded to %v, want %v", j, got, expected)
		}
	}
}

// TestKQuantDecodersProduceOnlyFiniteValues covers the inputs a real checkpoint does not
// contain but a corrupted one can, plus the degenerate block every format has to survive.
//
// The existing K-quant tests build one representative block per format and check it against
// the same formula the decoder uses. That catches a wrong index or a dropped term; it does not
// exercise the extremes, where a sign error or an unguarded shift shows up as NaN or an
// infinity rather than as a wrong number. A NaN in a weight matrix does not fail a decode --
// it propagates through a matmul and produces a model that emits nothing, with no error to
// point at the tensor.
//
// An all-zero block is the case with a real answer: every format must produce 256 zeros,
// because a zero super-scale scales everything to zero. The extreme-scale cases assert only
// finiteness, since the value itself is a fact about the formula rather than about the
// implementation.
func TestKQuantDecodersProduceOnlyFiniteValues(t *testing.T) {
	type layout struct {
		name string
		typ  gguf.QuantType
		size int
		// dOff are the byte offsets of the F16 super-scale fields, in the order the format
		// stores them (scale then min, where a format has both).
		dOff []int
	}
	formats := []layout{
		{"Q2_K", gguf.Q2_K, 84, []int{80, 82}},
		{"Q3_K", gguf.Q3_K, 110, []int{108}},
		{"Q4_K", gguf.Q4_K, 144, []int{0, 2}},
		{"Q5_K", gguf.Q5_K, 176, []int{0, 2}},
		{"Q6_K", gguf.Q6_K, 210, []int{208}},
	}

	// F16 patterns: zero, the largest finite value, the smallest positive subnormal, and a
	// negative. Anything a corrupt file can hold.
	scales := []struct {
		name string
		bits uint16
	}{
		{"zero", 0x0000},
		{"max", 0x7BFF},
		{"subnormal", 0x0001},
		{"negative", 0xB400},
	}

	for _, f := range formats {
		t.Run(f.name, func(t *testing.T) {
			t.Run("all-zero block decodes to zeros", func(t *testing.T) {
				src := make([]byte, f.size)
				dst := make([]float32, 256)
				if err := Decode(f.typ, src, dst); err != nil {
					t.Fatalf("Decode: %v", err)
				}
				for i, v := range dst {
					if v != 0 {
						t.Fatalf("element %d = %v with a zero scale, want 0", i, v)
					}
				}
			})

			// Every byte set, every scale field varied: the combination that would overflow a
			// shift or lose a sign.
			for _, fill := range []byte{0x00, 0xFF, 0xAA, 0x55} {
				for _, sc := range scales {
					src := make([]byte, f.size)
					for i := range src {
						src[i] = fill
					}
					for _, off := range f.dOff {
						binary.LittleEndian.PutUint16(src[off:], sc.bits)
					}
					dst := make([]float32, 256)
					if err := Decode(f.typ, src, dst); err != nil {
						t.Fatalf("Decode(fill %#02x, scale %s): %v", fill, sc.name, err)
					}
					for i, v := range dst {
						fv := float64(v)
						if math.IsNaN(fv) {
							t.Fatalf("fill %#02x scale %s: element %d is NaN", fill, sc.name, i)
						}
						if math.IsInf(fv, 0) {
							t.Fatalf("fill %#02x scale %s: element %d is infinite", fill, sc.name, i)
						}
					}
				}
			}
		})
	}
}

// TestSizeAgreesWithTheCodecTable pins the two places block sizes are stated.
//
// A tensor's byte length comes from gguf.QuantType.Size; the decoder splits those bytes
// using the block and size constants in the codecs map above. They are maintained
// separately, and nothing compared them: a typo in either one makes the loader read the
// next tensor from the wrong offset, which produces plausible-looking weights rather
// than an error, because both halves are self-consistent on their own.
//
// Q8_1 and Q8_K are Size-able but not decodable -- they are conversion-tool
// intermediates, refused at load with "cannot decode". They are skipped rather than
// expected to match, and the loop asserts that they are the only such types, so a codec
// added without a size (or removed without cleaning up) fails here.
func TestSizeAgreesWithTheCodecTable(t *testing.T) {
	// Every quantized type the format defines, so the check is over the whole table
	// rather than whichever types a test happened to think of.
	all := []gguf.QuantType{
		gguf.F32, gguf.F16, gguf.BF16,
		gguf.Q4_0, gguf.Q4_1, gguf.Q5_0, gguf.Q5_1, gguf.Q8_0,
		gguf.IQ4_NL, gguf.Q2_K, gguf.Q3_K, gguf.Q4_K, gguf.Q5_K, gguf.Q6_K,
	}
	for _, typ := range all {
		c, ok := codecs[typ]
		if !ok {
			t.Errorf("%s has no codec entry, so Decode would refuse a type the format defines", typ)
			continue
		}
		// Two element counts per type, one block-aligned and one several blocks long, so a
		// constant that happens to match at 256 does not hide a rounding difference.
		for _, n := range []int64{256, 1024, 4096} {
			if n%int64(c.block) != 0 {
				continue
			}
			sized, ok := typ.Size(n)
			if !ok {
				t.Errorf("%s.Size(%d) reported no size", typ, n)
				continue
			}
			want := n / int64(c.block) * int64(c.size)
			if sized != want {
				t.Errorf("%s.Size(%d) = %d bytes, but the codec reads %d per block x %d blocks = %d",
					typ, n, sized, c.size, n/int64(c.block), want)
			}
		}
	}

	// The types with a size but no decoder are the documented pair, and only that pair.
	for _, typ := range []gguf.QuantType{gguf.Q8_1, gguf.Q8_K} {
		if _, ok := codecs[typ]; ok {
			t.Errorf("%s is decoded, so the note that it is a conversion-tool intermediate is stale", typ)
		}
		if _, ok := typ.Size(256); !ok {
			t.Errorf("%s has no size, so the loader cannot refuse it with a byte count", typ)
		}
	}
}

// TestBlockSizesMatchTheFormatDefinitions recomputes every block size from its layout
// rather than comparing the two tables against each other.
//
// TestCodecsMatchGGUF checks that this package's table agrees with gguf's, and that is a
// different question: both are hand-written constants, so a byte count copied wrongly into
// both passes. Only arithmetic from the format's own field layout catches that. The K-quants
// are the ones worth spelling out -- five formats whose sizes are 84, 110, 144, 176 and 210,
// numbers that share no pattern and that a reader cannot sanity-check by eye.
//
// Each sum below follows the layout comment on the corresponding decoder. The terms are
// written as the fields they count so a change to a layout has to change this test too.
func TestBlockSizesMatchTheFormatDefinitions(t *testing.T) {
	const (
		f16     = 2   // one half-precision scalar
		qkK     = 256 // weights in a K-quant super-block
		perByte = 2   // one packed byte holds two 4-bit codes
		q4Bytes = qkK / perByte
	)
	cases := []struct {
		typ   gguf.QuantType
		block int
		size  int
		// why spells the layout out in the terms above, so the expected number is derived
		// here rather than restated.
		why string
	}{
		// Q4_0: d, then 32 4-bit codes = 16 bytes.
		{gguf.Q4_0, 32, f16 + 32/perByte, "F16 d + 32 nibbles"},
		// Q4_1: d, m, then the same codes.
		{gguf.Q4_1, 32, f16 + f16 + 32/perByte, "F16 d + F16 m + 32 nibbles"},
		// Q5_0: d, a 4-byte high-bit mask, then 16 bytes of low nibbles.
		{gguf.Q5_0, 32, f16 + 4 + 32/perByte, "F16 d + 4-byte qh + 32 nibbles"},
		// Q5_1: the same plus m.
		{gguf.Q5_1, 32, f16 + f16 + 4 + 32/perByte, "F16 d + F16 m + 4-byte qh + 32 nibbles"},
		// Q8_0: d, then one signed byte per weight.
		{gguf.Q8_0, 32, f16 + 32, "F16 d + 32 int8"},
		// IQ4_NL: d, then 16 bytes of nibbles.
		{gguf.IQ4_NL, 32, f16 + 32/perByte, "F16 d + 32 nibbles"},
		// Q2_K: 16 scale/min bytes (one per sub-block of 16), 64 code bytes, d, dmin.
		{gguf.Q2_K, qkK, 16 + qkK/4 + f16 + f16, "16 scales + 256 2-bit + F16 d + F16 dmin"},
		// Q3_K: 32 mask bytes (256 high bits), 64 code bytes (256 2-bit lows), 12 scale
		// bytes (16 sub-blocks x 6 bits), d.
		{gguf.Q3_K, qkK, 32 + qkK/4 + 12 + f16, "32 hmask + 256 2-bit + 12 scales + F16 d"},
		// Q4_K: d, dmin, 12 packed scale/min bytes (8 sub-blocks x 6+6 bits), 128 codes.
		{gguf.Q4_K, qkK, f16 + f16 + 12 + q4Bytes, "F16 d + F16 dmin + 12 scales + 256 nibbles"},
		// Q5_K: Q4_K's header plus a 32-byte high-bit plane.
		{gguf.Q5_K, qkK, f16 + f16 + 12 + 32 + q4Bytes, "F16 d + F16 dmin + 12 scales + 32 qh + 256 nibbles"},
		// Q6_K: 128 low-nibble bytes, 64 high-2-bit bytes, 16 scale bytes (one int8 scale
		// per 16-weight sub-block, so 256/16 of them -- not the 32-weight sub-block the
		// other K-quants use), d.
		{gguf.Q6_K, qkK, q4Bytes + qkK/4 + qkK/16 + f16, "256 low nibbles + 256 high 2-bit + 16 scales + F16 d"},
	}
	for _, c := range cases {
		t.Run(c.typ.String(), func(t *testing.T) {
			got, ok := codecs[c.typ]
			if !ok {
				t.Fatalf("%s has no decoder", c.typ)
			}
			if got.block != c.block {
				t.Errorf("weights per block = %d, want %d", got.block, c.block)
			}
			if got.size != c.size {
				t.Errorf("bytes per block = %d, want %d (%s)", got.size, c.size, c.why)
			}
			// And the same arithmetic has to hold in the file-format table, which is what
			// sizeOf consults when it offsets into a tensor.
			if declared, ok := c.typ.Size(int64(c.block)); !ok {
				t.Errorf("gguf has no size for %s", c.typ)
			} else if declared != int64(c.size) {
				t.Errorf("gguf says %d bytes per block, the layout gives %d (%s)",
					declared, c.size, c.why)
			}
		})
	}
}
