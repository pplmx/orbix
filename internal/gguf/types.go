package gguf

import (
	"fmt"
	"math"
)

// Magic is the four-byte file signature that starts every GGUF file.
const Magic = "GGUF"

// Version is the newest GGUF format revision this package parses.
const Version = 3

// MinVersion is the oldest revision it parses.
//
// Version 2 is still what a large part of the published corpus is: TheBloke's Mistral and Llama 2
// conversions, among others, were written before version 3 existed and were never re-converted.
// Accepting them is safe for the same reason it is necessary -- little-endian version 2 and
// version 3 have the same header, the same metadata and the same tensor directory, because the
// only thing version 3 changed was to drop big-endian support. A big-endian version 2 file does
// not look like version 2 here: read little-endian its version field is 0x02000000, which is
// outside [MinVersion, Version] and is refused.
//
// Version 1 used 32-bit counts for the tensor and metadata directories, so parsing it as version 2
// would read the second count from the wrong bytes and produce a plausible, wrong directory. It
// stays refused.
const MinVersion = 2

// ValueType identifies the type of a metadata value.
type ValueType uint32

// Metadata value types, in the order defined by the GGUF specification.
// The numeric values are part of the on-disk encoding.
const (
	TypeUint8   ValueType = 0
	TypeInt8    ValueType = 1
	TypeUint16  ValueType = 2
	TypeInt16   ValueType = 3
	TypeUint32  ValueType = 4
	TypeInt32   ValueType = 5
	TypeFloat32 ValueType = 6
	TypeBool    ValueType = 7
	TypeString  ValueType = 8
	TypeArray   ValueType = 9
	TypeUint64  ValueType = 10
	TypeInt64   ValueType = 11
	TypeFloat64 ValueType = 12
)

// String returns the specification name of the value type.
func (t ValueType) String() string {
	switch t {
	case TypeUint8:
		return "uint8"
	case TypeInt8:
		return "int8"
	case TypeUint16:
		return "uint16"
	case TypeInt16:
		return "int16"
	case TypeUint32:
		return "uint32"
	case TypeInt32:
		return "int32"
	case TypeFloat32:
		return "float32"
	case TypeBool:
		return "bool"
	case TypeString:
		return "string"
	case TypeArray:
		return "array"
	case TypeUint64:
		return "uint64"
	case TypeInt64:
		return "int64"
	case TypeFloat64:
		return "float64"
	default:
		return fmt.Sprintf("ValueType(%d)", uint32(t))
	}
}

// QuantType is a ggml tensor data type.
//
// The numbering is ggml's, not Orbix's: it comes from the enum in ggml.h and
// is shared by every GGUF file in the wild. Values that ggml removed or
// never assigned are skipped, which is why the sequence has gaps.
type QuantType uint32

// Tensor data types orbix can name. Parsing a file that uses a type outside
// this list is not an error; Size() and String() report it honestly instead.
const (
	// Floating point.
	F32  QuantType = 0
	F16  QuantType = 1
	BF16 QuantType = 30
	F64  QuantType = 28

	// Legacy block quants: 32 weights per block.
	Q4_0   QuantType = 2
	Q4_1   QuantType = 3
	Q5_0   QuantType = 6
	Q5_1   QuantType = 7
	Q8_0   QuantType = 8
	Q8_1   QuantType = 9
	IQ4_NL QuantType = 20

	// K-quants: 256 weights per super-block.
	Q2_K QuantType = 10
	Q3_K QuantType = 11
	Q4_K QuantType = 12
	Q5_K QuantType = 13
	Q6_K QuantType = 14
	Q8_K QuantType = 15

	// I-quants: 256 weights per super-block, lookup-table based.
	IQ2_XXS QuantType = 16
	IQ2_XS  QuantType = 17
	IQ3_XXS QuantType = 18
	IQ1_S   QuantType = 19
	IQ3_S   QuantType = 21
	IQ2_S   QuantType = 22
	IQ4_XS  QuantType = 23
	IQ1_M   QuantType = 29

	// Ternary and very low-bit quants.
	TQ1_0 QuantType = 34
	TQ2_0 QuantType = 35
	Q1_0  QuantType = 41

	// FP4 variants.
	MXFP4 QuantType = 39
	NVFP4 QuantType = 40

	// Plain integers, used for some quantized types' metadata tensors.
	I8  QuantType = 24
	I16 QuantType = 25
	I32 QuantType = 26
	I64 QuantType = 27
)

// quantSizes maps each type to (weights per block, bytes per block).
//
// The byte counts are ggml's block layouts, not approximations: a Q4_0 block
// stores 32 weights in 18 bytes (one F16 scale plus 16 packed nibbles). Every
// value here is copied from ggml's GGML_QUANT_SIZES; see quant_test.go for
// the arithmetic that pins them down.
var quantSizes = map[QuantType]struct {
	block int // number of weights covered by one block
	size  int // encoded size of one block, in bytes
}{
	F32:     {1, 4},
	F16:     {1, 2},
	BF16:    {1, 2},
	F64:     {1, 8},
	I8:      {1, 1},
	I16:     {1, 2},
	I32:     {1, 4},
	I64:     {1, 8},
	Q4_0:    {32, 18},
	Q4_1:    {32, 20},
	Q5_0:    {32, 22},
	Q5_1:    {32, 24},
	Q8_0:    {32, 34},
	Q8_1:    {32, 40},
	IQ4_NL:  {32, 18},
	Q2_K:    {256, 84},
	Q3_K:    {256, 110},
	Q4_K:    {256, 144},
	Q5_K:    {256, 176},
	Q6_K:    {256, 210},
	Q8_K:    {256, 292},
	IQ2_XXS: {256, 66},
	IQ2_XS:  {256, 74},
	IQ3_XXS: {256, 98},
	IQ1_S:   {256, 50},
	IQ3_S:   {256, 110},
	IQ2_S:   {256, 82},
	IQ4_XS:  {256, 136},
	IQ1_M:   {256, 56},
	TQ1_0:   {256, 54},
	TQ2_0:   {256, 66},
	Q1_0:    {128, 18},
	MXFP4:   {32, 17},
	NVFP4:   {64, 36},
}

var quantNames = map[QuantType]string{
	F32: "F32", F16: "F16", BF16: "BF16", F64: "F64",
	I8: "I8", I16: "I16", I32: "I32", I64: "I64",
	Q4_0: "Q4_0", Q4_1: "Q4_1", Q5_0: "Q5_0", Q5_1: "Q5_1",
	Q8_0: "Q8_0", Q8_1: "Q8_1", IQ4_NL: "IQ4_NL",
	Q2_K: "Q2_K", Q3_K: "Q3_K", Q4_K: "Q4_K", Q5_K: "Q5_K",
	Q6_K: "Q6_K", Q8_K: "Q8_K",
	IQ2_XXS: "IQ2_XXS", IQ2_XS: "IQ2_XS", IQ3_XXS: "IQ3_XXS",
	IQ1_S: "IQ1_S", IQ3_S: "IQ3_S", IQ2_S: "IQ2_S", IQ4_XS: "IQ4_XS",
	IQ1_M: "IQ1_M", TQ1_0: "TQ1_0", TQ2_0: "TQ2_0", Q1_0: "Q1_0",
	MXFP4: "MXFP4", NVFP4: "NVFP4",
}

// String returns the ggml name of the type, or "QuantType(n)" when the type
// is one this package does not know.
func (q QuantType) String() string {
	if name, ok := quantNames[q]; ok {
		return name
	}
	return fmt.Sprintf("QuantType(%d)", uint32(q))
}

// Size returns the number of bytes needed to store n weights of this type.
//
// n must be a multiple of the type's block size, which ggml guarantees for
// every tensor in a valid file. The second return value is false for unknown
// types, so callers can report the type instead of inventing a size.
func (q QuantType) Size(n int64) (int64, bool) {
	info, ok := quantSizes[q]
	if !ok {
		return 0, false
	}
	// A quantized tensor is a whole number of blocks: the encoded representation has no way to
	// store a partial one. An element count that is not a multiple of the block size therefore
	// describes a tensor this package cannot size, and the integer division below would answer
	// with the floor -- understating the bytes and letting a bounds check pass for a reader
	// that runs out halfway through. Reported as unknown, like an unrecognised type.
	//
	// The dequantizer already refuses this case (quant.Decode's "is not a multiple of" check);
	// this is the same rule at the layer that computes the length, so a file is rejected where
	// its numbers can still be named rather than after a short read.
	if n%int64(info.block) != 0 {
		return 0, false
	}
	blocks := n / int64(info.block)
	// n comes from a product of dimensions read out of the file, so it can be enormous. The
	// multiplication below would wrap to a negative byte count, and a negative length is worse
	// than an unavailable one: a bounds check like "start + nbytes > file size" passes for it.
	// Reporting the size as unknown is what the second return value is for.
	if blocks > math.MaxInt64/int64(info.size) {
		return 0, false
	}
	return blocks * int64(info.size), true
}

// BlockSize returns the number of weights in one encoded block.
func (q QuantType) BlockSize() (int, bool) {
	info, ok := quantSizes[q]
	if !ok {
		return 1, false
	}
	return info.block, true
}
