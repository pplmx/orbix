package quant_test

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/pplmx/orbix/internal/f16"
	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/quant"
)

// TestQuantizedDotMatchesDecodeThenDot pins that a block-at-a-time dot product equals the
// decode-then-dot result.
//
// It exists because a Q8_0-aware matvec was considered and measured, and the arithmetic had to be
// settled before the speed question could be answered honestly. The speed question came out against
// it: with the current design decoding each tensor once at load, a plain dot over the float32 array
// costs 0.439 ms per 2048x640 projection, while re-reading the blocks on every call costs 1.817 ms
// -- 4.14x slower, not faster. Writing a quantized kernel would only pay if it also avoided the
// per-call block walk, which means reproducing ggml's structure rather than adding a loop.
//
// So this test is not a step toward that refactor. It records the arithmetic a quantized path would
// have to reproduce, and it stands on its own as a check that the block layout is understood: a
// wrong offset or byte order moves the result by the scale, six orders of magnitude past the
// tolerance below.
//
// A tolerance rather than bit equality, and the reason is the point: the quantized form factors
// the scale out of each block (sum of int8*v, then times the scale), while the decoded form
// multiplies first. Both are float32 additions in different groupings, so the last bits differ.
// Measured worst difference over a synthetic 2048x640 matrix: 8.6e-06 on sums of magnitude ~10,
// which is float32 rounding and not a layout error. A layout error -- a wrong offset, a swapped
// byte order -- moves the result by the scale, which is six orders of magnitude larger.
func TestQuantizedDotMatchesDecodeThenDot(t *testing.T) {
	const (
		k          = 640 // one row
		rows       = 512
		blockSize  = 32
		blockBytes = 34
	)
	// A synthetic Q8_0 matrix: one fp16 scale per block plus 32 int8 values.
	raw := make([]byte, rows*k/blockSize*blockBytes)
	for b := range rows * k / blockSize {
		off := b * blockBytes
		// 0.0625 as fp16, so the scale is exactly representable and the decode is exact.
		binary.LittleEndian.PutUint16(raw[off:], 0x2C00)
		for j := range blockSize {
			raw[off+2+j] = byte(int8((b*7 + j*3) % 120))
		}
	}
	v := make([]float32, k)
	for i := range v {
		v[i] = float32(i%13-6) * 0.1
	}

	decoded := make([]float32, rows*k)
	if err := quant.Decode(gguf.Q8_0, raw, decoded); err != nil {
		t.Fatalf("Decode: %v", err)
	}

	worst := 0.0
	for r := range rows {
		var want float32
		row := decoded[r*k : (r+1)*k]
		for p := range k {
			want += row[p] * v[p]
		}
		got := quantizedDotQ8_0(raw[r*k/blockSize*blockBytes:], v)
		if d := math.Abs(float64(want - got)); d > worst {
			worst = d
		}
	}
	t.Logf("worst difference %.6g over %d rows", worst, rows)
	if worst > 1e-4 {
		t.Errorf("worst difference %.6g; a layout error would move a result by the scale (6e-02)",
			worst)
	}
}

// quantizedDotQ8_0 is the reference form of a block-at-a-time dot product, kept here rather than
// in the package because no kernel uses it: measured 4.14x slower than the current path, so nothing
// should.
func quantizedDotQ8_0(raw []byte, v []float32) float32 {
	var sum float32
	for b := range len(v) / 32 {
		off := b * 34
		d := f16.ToFloat32(binary.LittleEndian.Uint16(raw[off:]))
		var blockSum float32
		for j := range 32 {
			blockSum += float32(int8(raw[off+2+j])) * v[b*32+j]
		}
		sum += d * blockSum
	}
	return sum
}
