package quant

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/pplmx/orbix/internal/f16"
	"github.com/pplmx/orbix/internal/gguf"
)

// codec decodes one quantized type.
//
// Each function decodes whole blocks and writes exactly block elements into
// dst, so the block boundaries are the caller's responsibility and the
// function only has to deal with the layout inside one block.
type codec struct {
	// block is the number of weights per encoded block.
	block int
	// size is the encoded size of one block in bytes.
	size int
	// decode writes block*len(dst)/block floats from src into dst.
	decode func(dst []float32, src []byte)
}

// codecs holds the types this package can decode.
//
// Adding a type is a two-step change: implement its block layout here, and add
// it to gguf's size table. The two must agree, which TestCodecsMatchGGUF
// enforces by comparing against the declared block size.
var codecs = map[gguf.QuantType]codec{
	gguf.F32:    {1, 4, decodeF32},
	gguf.F16:    {1, 2, decodeF16},
	gguf.BF16:   {1, 2, decodeBF16},
	gguf.Q4_0:   {32, 18, decodeQ4_0},
	gguf.Q4_1:   {32, 20, decodeQ4_1},
	gguf.Q5_0:   {32, 22, decodeQ5_0},
	gguf.Q5_1:   {32, 24, decodeQ5_1},
	gguf.Q8_0:   {32, 34, decodeQ8_0},
	gguf.IQ4_NL: {32, 18, decodeIQ4NL},
	// K-quants: 256-weight super-blocks with per-sub-block scales.
	gguf.Q2_K: {256, 84, decodeQ2_K},
	gguf.Q3_K: {256, 110, decodeQ3_K},
	gguf.Q4_K: {256, 144, decodeQ4_K},
	gguf.Q5_K: {256, 176, decodeQ5_K},
	gguf.Q6_K: {256, 210, decodeQ6_K},
}

// Decode dequantizes len(dst) weights of type t from src into dst.
//
// The number of elements is taken from len(dst); src must hold exactly the
// number of bytes those elements encode to. Decoding into a caller-provided
// slice keeps the allocation policy with the caller, which matters because a
// model's largest tensor can be hundreds of megabytes.
func Decode(t gguf.QuantType, src []byte, dst []float32) error {
	c, ok := codecs[t]
	if !ok {
		return fmt.Errorf("quant: cannot decode %s", t)
	}
	n := len(dst)
	if n%c.block != 0 {
		return fmt.Errorf("quant: %d elements is not a multiple of %s's %d-element block", n, t, c.block)
	}
	want := n / c.block * c.size
	if len(src) != want {
		return fmt.Errorf("quant: %s needs %d bytes for %d elements, got %d", t, want, n, len(src))
	}
	c.decode(dst, src)
	return nil
}

// Supported reports whether Decode can handle t.
func Supported(t gguf.QuantType) bool {
	_, ok := codecs[t]
	return ok
}

// decodeF32 reads little-endian float32 words.
func decodeF32(dst []float32, src []byte) {
	for i := range dst {
		dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
	}
}

// decodeF16 widens half-precision words.
func decodeF16(dst []float32, src []byte) {
	for i := range dst {
		dst[i] = f16.ToFloat32(binary.LittleEndian.Uint16(src[i*2:]))
	}
}

// decodeBF16 widens bfloat16 words.
//
// bfloat16 keeps the exponent range of float32 and truncates the mantissa to
// 8 bits, so the encoded bits are exactly the high half of a float32 and
// widening is a shift with no rounding.
func decodeBF16(dst []float32, src []byte) {
	for i := range dst {
		dst[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(src[i*2:])) << 16)
	}
}

// q4_0BlockSize is the number of weights in one Q4_0 block.
const q4_0BlockSize = 32

// decodeQ4_0 expands a Q4_0 tensor.
//
// Block layout: an F16 scale, then 16 bytes of packed 4-bit codes. ggml
// interleaves the halves: the low nibble of byte j is weight j and the high
// nibble is weight j+16, not weight 2j and 2j+1. Getting that wrong decodes a
// tensor into a plausible-looking but completely scrambled matrix, so the
// indexing here follows ggml's dequantize_row_q4_0 literally.
//
// Codes are offset by 8, giving the range -8..7 before scaling.
func decodeQ4_0(dst []float32, src []byte) {
	const blockBytes = 18
	for b := range len(dst) / q4_0BlockSize {
		blk := src[b*blockBytes:]
		d := f16.ToFloat32(binary.LittleEndian.Uint16(blk[0:]))
		y := dst[b*q4_0BlockSize : (b+1)*q4_0BlockSize]
		for j := range q4_0BlockSize / 2 {
			qs := blk[2+j]
			y[j] = float32(int(qs&0x0f)-8) * d
			y[j+q4_0BlockSize/2] = float32(int(qs>>4)-8) * d
		}
	}
}

// q8_0BlockSize is the number of weights in one Q8_0 block.
const q8_0BlockSize = 32

// decodeQ8_0 expands a Q8_0 tensor.
//
// Block layout: an F16 scale followed by 32 signed bytes, one per weight, in
// order. Unlike Q4_0 there is no interleaving and no offset.
func decodeQ8_0(dst []float32, src []byte) {
	const blockBytes = 34
	for b := range len(dst) / q8_0BlockSize {
		blk := src[b*blockBytes:]
		d := f16.ToFloat32(binary.LittleEndian.Uint16(blk[0:]))
		y := dst[b*q8_0BlockSize : (b+1)*q8_0BlockSize]
		for j := range q8_0BlockSize {
			y[j] = float32(int8(blk[2+j])) * d
		}
	}
}

// q4_1BlockSize is the number of weights in one Q4_1 block.
const q4_1BlockSize = 32

// decodeQ4_1 expands a Q4_1 tensor.
//
// Q4_1 is Q4_0 with an affine mapping instead of a symmetric one: the block stores
// a minimum as well as a scale, and decodes x = code*scale + min. That is what lets
// it represent a weight distribution that is not centred on zero, which is why it
// costs two extra bytes per block (20 against 18) and is preferred for tensors with a
// large mean.
//
// Layout: F16 scale, F16 min, then 16 bytes of packed nibbles. The nibble
// interleaving matches Q4_0 -- byte j's low nibble is weight j, its high nibble is
// weight j+16 -- and the codes are unsigned (0..15) because the offset moved into
// min.
func decodeQ4_1(dst []float32, src []byte) {
	const blockBytes = 20
	for b := range len(dst) / q4_1BlockSize {
		blk := src[b*blockBytes:]
		d := f16.ToFloat32(binary.LittleEndian.Uint16(blk[0:]))
		m := f16.ToFloat32(binary.LittleEndian.Uint16(blk[2:]))
		y := dst[b*q4_1BlockSize : (b+1)*q4_1BlockSize]
		for j := range q4_1BlockSize / 2 {
			qs := blk[4+j]
			y[j] = float32(qs&0x0f)*d + m
			y[j+q4_1BlockSize/2] = float32(qs>>4)*d + m
		}
	}
}

// q5_0BlockSize is the number of weights in one Q5_0 block.
const q5_0BlockSize = 32

// decodeQ5_0 expands a Q5_0 tensor.
//
// Q5_0 is symmetric like Q4_0 but stores five bits per weight: a block has an F16
// scale, 4 bytes of high bits, then 16 bytes of low nibbles. The high bits are one
// bit per weight, packed in order: the first 8 weights' high bits are in byte 0 from
// the least significant bit up, and weight i reads bit i%8 of byte i/8.
//
// The code is (high bit << 4 | nibble) - 16, giving -16..15 before scaling. Reading
// the high bits as a mask over the wrong nibble order produces a tensor that is
// subtly wrong everywhere rather than obviously broken, so this follows ggml's
// dequantize_row_q5_0 exactly.
func decodeQ5_0(dst []float32, src []byte) {
	const blockBytes = 22
	for b := range len(dst) / q5_0BlockSize {
		blk := src[b*blockBytes:]
		d := f16.ToFloat32(binary.LittleEndian.Uint16(blk[0:]))
		// qh holds four bytes of high bits, little-endian, one bit per weight.
		qh := binary.LittleEndian.Uint32(blk[2:])
		y := dst[b*q5_0BlockSize : (b+1)*q5_0BlockSize]
		for j := range q5_0BlockSize / 2 {
			// Weight j: low nibble of byte j, high bit j.
			lo := int(blk[6+j] & 0x0f)
			hi := int(qh>>j) & 1
			y[j] = float32((lo|hi<<4)-16) * d

			// Weight j+16: high nibble of byte j, high bit j+16.
			lo = int(blk[6+j] >> 4)
			hi = int(qh>>(j+16)) & 1
			y[j+q5_0BlockSize/2] = float32((lo|hi<<4)-16) * d
		}
	}
}

// q5_1BlockSize is the number of weights in one Q5_1 block.
const q5_1BlockSize = 32

// decodeQ5_1 expands a Q5_1 tensor.
//
// Q5_1 combines both ideas: five bits per weight like Q5_0, and an affine mapping
// like Q4_1. Layout is F16 scale, F16 min, 4 bytes of high bits, then 16 bytes of low
// nibbles -- Q5_0's packing with Q4_1's two extra bytes in front.
//
// Codes are 0..31, and x = code*scale + min.
func decodeQ5_1(dst []float32, src []byte) {
	const blockBytes = 24
	for b := range len(dst) / q5_1BlockSize {
		blk := src[b*blockBytes:]
		d := f16.ToFloat32(binary.LittleEndian.Uint16(blk[0:]))
		m := f16.ToFloat32(binary.LittleEndian.Uint16(blk[2:]))
		qh := binary.LittleEndian.Uint32(blk[4:])
		y := dst[b*q5_1BlockSize : (b+1)*q5_1BlockSize]
		for j := range q5_1BlockSize / 2 {
			lo := int(blk[8+j] & 0x0f)
			hi := int(qh>>j) & 1
			y[j] = float32(lo|hi<<4)*d + m

			lo = int(blk[8+j] >> 4)
			hi = int(qh>>(j+16)) & 1
			y[j+q5_1BlockSize/2] = float32(lo|hi<<4)*d + m
		}
	}
}

// K-quant super-block size: every K-quant covers 256 weights, with sub-blocks of 32
// sharing a scale.
const qkK = 256

// getScaleMinK4 unpacks sub-block j's scale and minimum from a Q4_K/Q5_K block's
// 12-byte packed header.
//
// The packing is dense: eight 6-bit scales and eight 6-bit mins in 12 bytes. Bytes 0-3
// hold scales 0-3 and bytes 4-7 hold mins 0-3, each using six bits and leaving two bits
// at the top; bytes 8-11 hold the low four bits of scales 4-7 in their low nibble and of
// mins 4-7 in their high nibble. The high bits of the first eight bytes complete them.
//
// Reading this wrong is not a visible failure -- the result is a matrix of plausible
// magnitudes with the wrong sub-block scales, which produces fluent text at the wrong
// probabilities. The four branches below are the whole function, so they are spelled out
// rather than folded into an index expression.
func getScaleMinK4(j int, q []byte) (scale, minv uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4),
		(q[j+4] >> 4) | ((q[j] >> 6) << 4)
}

// decodeQ3_K expands a Q3_K tensor.
//
// The implementation mirrors ggml's own numpy reference decode expression by expression,
// because Q3_K's two index mappings are not derivable from a loop order: both are defined
// by how 2-D grids flatten back to 16 sub-blocks.
//
// Block layout (110 bytes): 32 mask bytes, 64 code bytes, 12 packed scale bytes, F16 d.
//
// Scales: the first eight bytes are split into low and high nibbles as a (2, 8) grid, so
// scales 0-7 take bytes 0-7's low nibbles in order and scales 8-15 their high nibbles. The
// last four bytes form a (4, 4) grid -- byte 8+i shifted by 0/2/4/6 -- whose flattening
// puts scale 4*i+j's high two bits in bit pair j of byte 8+i.
//
// Codes: the 64 code bytes form a (2 groups, 4 shifts, 32 bytes) grid and the 32 mask bytes
// a (1, 8 masks, 32 bytes) grid. Both flatten to (16 sub-blocks, 16 values), so sub-block
// sb takes its codes from group sb/8 shifted by shifts[(sb%8)/2] and its mask from bit sb/2,
// at byte offset (sb%2)*16.
//
// x = d * (scale - 32) * (ql - 4*(1 - qh))
func decodeQ3_K(dst []float32, src []byte) {
	const blockBytes = 110
	shifts := [4]uint{0, 2, 4, 6}
	for b := range len(dst) / qkK {
		blk := src[b*blockBytes:]
		hm := blk[0:32]
		qs := blk[32:96]
		d := f16.ToFloat32(binary.LittleEndian.Uint16(blk[108:]))
		y := dst[b*qkK : (b+1)*qkK]

		var scales [16]int8
		unpackQ3KScales(blk[96:108], &scales)

		for sb := range 16 {
			group := qs[(sb/8)*32 : (sb/8+1)*32]
			shift := shifts[(sb%8)/2]
			offset := (sb % 2) * 16
			maskBit := uint(sb / 2)
			dl := d * float32(scales[sb]-32)
			base := sb * 16
			for l := range 16 {
				idx := l + offset
				ql := int8((group[idx] >> shift) & 3)
				qh := int8((hm[idx] >> maskBit) & 1)
				y[base+l] = dl * float32(ql-4*(1-qh))
			}
		}
	}
}

// unpackQ3KScales expands Q3_K's 12 packed bytes into 16 six-bit scales.
//
// Low nibbles: bytes 0-7 give scales 0-7 (low nibble) and scales 8-15 (high nibble).
// High bits: byte 8+i gives scale 4*i+j's high two bits in bits 2j..2j+1.
func unpackQ3KScales(packed []byte, out *[16]int8) {
	for i := range 8 {
		out[i] = int8(packed[i] & 0x0F)
		out[i+8] = int8(packed[i] >> 4)
	}
	// The high two bits come from bytes 8-11 as a (shift, byte) grid: index f of the
	// flattened 16 takes shift f/4 and byte f%4, so scale f's high bits are bit pair
	// f/4 of byte 8 + f%4.
	for f := range 16 {
		out[f] |= int8(((packed[8+f%4] >> (2 * (f / 4))) & 0x03) << 4)
	}
}

// decodeQ4_K expands a Q4_K tensor.
//
// Block layout (144 bytes): F16 super-scale d, F16 super-min dmin, 12 packed scale/min
// bytes, then 128 bytes of 4-bit codes. The 256 weights are eight sub-blocks of 32, each
// with its own 6-bit scale and min:
//
//	x = d * scale[j] * code - dmin * min[j]
//
// The codes run over four iterations of 64 weights: each iteration's first 32 come from
// the low nibbles of its 32 bytes, the next 32 from the high nibbles of the same bytes.
// That means the nibble halves are not interleaved per byte as in Q4_0 -- they are
// separated by 32 positions -- which is the detail most likely to be gotten wrong.
func decodeQ4_K(dst []float32, src []byte) {
	const blockBytes = 144
	for b := range len(dst) / qkK {
		blk := src[b*blockBytes:]
		d := f16.ToFloat32(binary.LittleEndian.Uint16(blk[0:]))
		dmin := f16.ToFloat32(binary.LittleEndian.Uint16(blk[2:]))
		scales := blk[4:16]
		qs := blk[16:144]
		y := dst[b*qkK : (b+1)*qkK]

		for it := range 4 {
			sc0, m0 := getScaleMinK4(2*it, scales)
			sc1, m1 := getScaleMinK4(2*it+1, scales)
			d1, m1f := d*float32(sc0), dmin*float32(m0)
			d2, m2f := d*float32(sc1), dmin*float32(m1)

			qOff := it * 32
			yOff := it * 64
			for l := range 32 {
				q := qs[qOff+l]
				y[yOff+l] = d1*float32(q&0x0F) - m1f
				y[yOff+32+l] = d2*float32(q>>4) - m2f
			}
		}
	}
}

// decodeQ6_K expands a Q6_K tensor.
//
// Block layout (210 bytes): 128 bytes of low nibbles, 64 bytes of high 2-bit codes, 16
// signed 8-bit sub-block scales, then the F16 super-scale -- note that d comes *last*,
// unlike every other format here, which is the detail that makes this decoder different
// from the rest rather than a variation on Q4_K.
//
//	x = d * scales[is] * (code - 32)
//
// The 256 weights are processed as two halves of 128. Within a half, each of 32 byte
// positions yields four weights: low nibble of ql[l] and ql[l+32], then their high
// nibbles, with the high two bits taken from successive bit pairs of qh[l].
func decodeQ6_K(dst []float32, src []byte) {
	const blockBytes = 210
	for b := range len(dst) / qkK {
		blk := src[b*blockBytes:]
		ql := blk[0:128]
		qh := blk[128:192]
		scales := blk[192:208]
		d := f16.ToFloat32(binary.LittleEndian.Uint16(blk[208:]))
		y := dst[b*qkK : (b+1)*qkK]

		// Two halves of 128 weights, each using 64 bytes of ql, 32 of qh, 8 scales.
		for half := range 2 {
			yOff := half * 128
			qlOff := half * 64
			qhOff := half * 32
			scOff := half * 8
			for l := range 32 {
				is := l / 16
				h := qh[qhOff+l]
				q1 := int8((ql[qlOff+l]&0x0F)|((h&0x03)<<4)) - 32
				q2 := int8((ql[qlOff+l+32]&0x0F)|(((h>>2)&0x03)<<4)) - 32
				q3 := int8((ql[qlOff+l]>>4)|(((h>>4)&0x03)<<4)) - 32
				q4 := int8((ql[qlOff+l+32]>>4)|(((h>>6)&0x03)<<4)) - 32

				y[yOff+l] = d * float32(int8(scales[scOff+is])) * float32(q1)
				y[yOff+l+32] = d * float32(int8(scales[scOff+is+2])) * float32(q2)
				y[yOff+l+64] = d * float32(int8(scales[scOff+is+4])) * float32(q3)
				y[yOff+l+96] = d * float32(int8(scales[scOff+is+6])) * float32(q4)
			}
		}
	}
}

// iq4nlBlockSize is the number of weights in one IQ4_NL block.
const iq4nlBlockSize = 32

// kValuesIQ4NL is IQ4_NL's non-linear 4-bit codebook.
//
// A nibble is an index into this table rather than an integer offset, which is what
// separates IQ4_NL from Q4_0: the spacing is denser near zero, matching a weight
// distribution with many small values. Treating the nibble as a number produces output of
// the right shape and entirely the wrong values.
var kValuesIQ4NL = [16]int8{
	-127, -104, -83, -65, -49, -35, -22, -10,
	1, 13, 25, 38, 53, 69, 89, 113,
}

// decodeIQ4NL expands an IQ4_NL tensor.
//
// Block layout (18 bytes): an F16 scale, then 16 packed nibble bytes in Q4_0's split-half
// order -- byte j holds weight j in its low nibble and weight j+16 in its high nibble.
//
//	x = d * kValuesIQ4NL[nibble]
func decodeIQ4NL(dst []float32, src []byte) {
	const blockBytes = 18
	for b := range len(dst) / iq4nlBlockSize {
		blk := src[b*blockBytes:]
		d := f16.ToFloat32(binary.LittleEndian.Uint16(blk[0:]))
		y := dst[b*iq4nlBlockSize : (b+1)*iq4nlBlockSize]
		for j := range iq4nlBlockSize / 2 {
			qs := blk[2+j]
			y[j] = d * float32(kValuesIQ4NL[qs&0x0F])
			y[j+iq4nlBlockSize/2] = d * float32(kValuesIQ4NL[qs>>4])
		}
	}
}

// decodeQ2_K expands a Q2_K tensor.
//
// Block layout (84 bytes): 16 scale/min bytes, 64 code bytes, then F16 d and F16 dmin.
// The 256 weights are 16 sub-blocks of 16, each with a 4-bit scale in its byte's low
// nibble and a 4-bit minimum in the high nibble:
//
//	x = d * (sc & 0xF) * code - dmin * (sc >> 4)
//
// The code grid is the same shape Q3_K uses for its codes: two groups of 32 bytes, each
// shifted by 0, 2, 4, and 6, flattening to 16 sub-blocks -- so sub-block sb reads group
// sb/8 at shift (sb%8)/2 and byte offset (sb%2)*16.
func decodeQ2_K(dst []float32, src []byte) {
	const blockBytes = 84
	shifts := [4]uint{0, 2, 4, 6}
	for b := range len(dst) / qkK {
		blk := src[b*blockBytes:]
		scales := blk[0:16]
		d := f16.ToFloat32(binary.LittleEndian.Uint16(blk[80:]))
		dmin := f16.ToFloat32(binary.LittleEndian.Uint16(blk[82:]))
		y := dst[b*qkK : (b+1)*qkK]

		for sb := range 16 {
			group := blk[16+(sb/8)*32 : 16+(sb/8+1)*32]
			shift := shifts[(sb%8)/2]
			offset := (sb % 2) * 16
			sc := scales[sb]
			dl := d * float32(sc&0x0F)
			ml := dmin * float32(sc>>4)
			base := sb * 16
			for l := range 16 {
				code := int8((group[l+offset] >> shift) & 3)
				y[base+l] = dl*float32(code) - ml
			}
		}
	}
}

// decodeQ5_K expands a Q5_K tensor.
//
// Block layout (176 bytes): F16 d, F16 dmin, 12 packed scale/min bytes in Q4_K's format
// read by the same getScaleMinK4, 32 high-bit bytes, then 128 code bytes.
//
//	x = d * scale[j] * (code + 16*highbit) - dmin * min[j]
//
// The high-bit mask steps two bits per 64-weight iteration: u1 for the low nibbles and u2
// for the high, both shifted left by two each time.
func decodeQ5_K(dst []float32, src []byte) {
	const blockBytes = 176
	for b := range len(dst) / qkK {
		blk := src[b*blockBytes:]
		d := f16.ToFloat32(binary.LittleEndian.Uint16(blk[0:]))
		dmin := f16.ToFloat32(binary.LittleEndian.Uint16(blk[2:]))
		scales := blk[4:16]
		qh := blk[16:48]
		ql := blk[48:176]
		y := dst[b*qkK : (b+1)*qkK]

		u1, u2 := byte(1), byte(2)
		for it := range 4 {
			sc0, m0 := getScaleMinK4(2*it, scales)
			sc1, m1 := getScaleMinK4(2*it+1, scales)
			d1, m1f := d*float32(sc0), dmin*float32(m0)
			d2, m2f := d*float32(sc1), dmin*float32(m1)
			qOff := it * 32
			yOff := it * 64
			for l := range 32 {
				lo := ql[qOff+l] & 0x0F
				if qh[l]&u1 != 0 {
					lo += 16
				}
				hi := ql[qOff+l] >> 4
				if qh[l]&u2 != 0 {
					hi += 16
				}
				y[yOff+l] = d1*float32(lo) - m1f
				y[yOff+l+32] = d2*float32(hi) - m2f
			}
			u1 <<= 2
			u2 <<= 2
		}
	}
}
