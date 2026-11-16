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
	gguf.F32:  {1, 4, decodeF32},
	gguf.F16:  {1, 2, decodeF16},
	gguf.BF16: {1, 2, decodeBF16},
	gguf.Q4_0: {32, 18, decodeQ4_0},
	gguf.Q4_1: {32, 20, decodeQ4_1},
	gguf.Q5_0: {32, 22, decodeQ5_0},
	gguf.Q5_1: {32, 24, decodeQ5_1},
	gguf.Q8_0: {32, 34, decodeQ8_0},
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
