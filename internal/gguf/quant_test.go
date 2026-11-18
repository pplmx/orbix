package gguf

import (
	"math"
	"strconv"
	"testing"
)

func TestQuantSizeTable(t *testing.T) {
	// The size table is the one place where a typo silently produces a size
	// mismatch that only shows up as garbage weights much later. These
	// expectations are the documented block layouts, written out independently
	// of the table so the two have to agree.
	want := map[QuantType]struct {
		name  string
		block int
		size  int
	}{
		F32:     {"F32", 1, 4},
		F16:     {"F16", 1, 2},
		BF16:    {"BF16", 1, 2},
		F64:     {"F64", 1, 8},
		Q4_0:    {"Q4_0", 32, 18}, // F16 scale + 16 packed nibbles
		Q4_1:    {"Q4_1", 32, 20}, // F16 scale + F16 min + 16 packed nibbles
		Q5_0:    {"Q5_0", 32, 22}, // F16 scale + uint32 high bits + 16 bytes
		Q5_1:    {"Q5_1", 32, 24}, // F16 scale + F16 min + uint32 high bits + 16 bytes
		Q8_0:    {"Q8_0", 32, 34}, // F16 scale + 32 int8
		Q8_1:    {"Q8_1", 32, 40}, // F32 scale + F32 sum + 32 int8
		IQ4_NL:  {"IQ4_NL", 32, 18},
		Q2_K:    {"Q2_K", 256, 84},
		Q3_K:    {"Q3_K", 256, 110},
		Q4_K:    {"Q4_K", 256, 144},
		Q5_K:    {"Q5_K", 256, 176},
		Q6_K:    {"Q6_K", 256, 210},
		Q8_K:    {"Q8_K", 256, 292},
		IQ2_XXS: {"IQ2_XXS", 256, 66},
		IQ2_XS:  {"IQ2_XS", 256, 74},
		IQ3_XXS: {"IQ3_XXS", 256, 98},
		IQ1_S:   {"IQ1_S", 256, 50},
		IQ3_S:   {"IQ3_S", 256, 110},
		IQ2_S:   {"IQ2_S", 256, 82},
		IQ4_XS:  {"IQ4_XS", 256, 136},
		IQ1_M:   {"IQ1_M", 256, 56},
		TQ1_0:   {"TQ1_0", 256, 54},
		TQ2_0:   {"TQ2_0", 256, 66},
		Q1_0:    {"Q1_0", 128, 18},
		MXFP4:   {"MXFP4", 32, 17},
		NVFP4:   {"NVFP4", 64, 36},
		I8:      {"I8", 1, 1},
		I16:     {"I16", 1, 2},
		I32:     {"I32", 1, 4},
		I64:     {"I64", 1, 8},
	}

	for q, w := range want {
		if got := q.String(); got != w.name {
			t.Errorf("%d.String() = %q, want %q", uint32(q), got, w.name)
		}
		block, ok := q.BlockSize()
		if !ok {
			t.Errorf("%s.BlockSize() reported unknown", w.name)
			continue
		}
		if block != w.block {
			t.Errorf("%s.BlockSize() = %d, want %d", w.name, block, w.block)
		}
		got, ok := q.Size(int64(w.block))
		if !ok {
			t.Errorf("%s.Size() reported unknown", w.name)
			continue
		}
		if got != int64(w.size) {
			t.Errorf("%s.Size(%d) = %d bytes, want %d", w.name, w.block, got, w.size)
		}
	}
}

func TestQuantSizeScalesWithElementCount(t *testing.T) {
	// 4096 Q4_0 weights are 128 blocks of 18 bytes.
	if got, ok := Q4_0.Size(4096); !ok || got != 128*18 {
		t.Errorf("Q4_0.Size(4096) = %d, %v, want %d", got, ok, 128*18)
	}
	// A 7B-ish FFN tensor: 4096 x 11008 F16.
	const n = 4096 * 11008
	if got, ok := F16.Size(n); !ok || got != n*2 {
		t.Errorf("F16.Size(%d) = %d, %v, want %d", n, got, ok, n*2)
	}
}

func TestUnknownQuantType(t *testing.T) {
	// Type 4 was Q4_2, removed from ggml; a file using it must be reported,
	// not guessed at.
	unknown := QuantType(4)
	if got, want := unknown.String(), "QuantType(4)"; got != want {
		t.Errorf("String() = %q, want %q", got, want)
	}
	if _, ok := unknown.Size(32); ok {
		t.Error("Size() reported a size for an unknown type")
	}
	if _, ok := unknown.BlockSize(); ok {
		t.Error("BlockSize() reported a block for an unknown type")
	}
}

func TestValueTypeStrings(t *testing.T) {
	want := map[ValueType]string{
		TypeUint8: "uint8", TypeInt8: "int8",
		TypeUint16: "uint16", TypeInt16: "int16",
		TypeUint32: "uint32", TypeInt32: "int32",
		TypeFloat32: "float32", TypeBool: "bool",
		TypeString: "string", TypeArray: "array",
		TypeUint64: "uint64", TypeInt64: "int64",
		TypeFloat64: "float64",
	}
	for vt, w := range want {
		if got := vt.String(); got != w {
			t.Errorf("%d.String() = %q, want %q", uint32(vt), got, w)
		}
	}
	if got, want := ValueType(99).String(), "ValueType(99)"; got != want {
		t.Errorf("unknown String() = %q, want %q", got, want)
	}
}

// TestValueString covers the rendering `orbix model inspect` prints for every metadata
// value.
//
// The method has a branch per GGUF value type, and the CLI depends on all of them: the
// human-readable dump prints these strings, and the --json path uses the same method for its
// metadata map. Nothing tested it, so a missed case -- a bool printed as "0"/"1", an int64
// losing its sign, an array reported without its element type -- would appear only in the
// tool's output.
//
// The float cases are the ones worth looking at: a float32 stored as 1e-5 does not survive a
// float64 round trip exactly, so FormatFloat renders it at the precision the file's own type
// carries. Printing the float64 expansion would suggest the file holds digits it cannot.
func TestValueString(t *testing.T) {
	cases := []struct {
		name string
		v    Value
		want string
	}{
		{"string", Value{Type: TypeString, Str: "llama"}, "llama"},
		{"empty string", Value{Type: TypeString, Str: ""}, ""},
		{"bool true", Value{Type: TypeBool, Bool: true}, "true"},
		{"bool false", Value{Type: TypeBool, Bool: false}, "false"},
		{"uint8", Value{Type: TypeUint8, Uint: 42}, "42"},
		{"uint64 max", Value{Type: TypeUint64, Uint: math.MaxUint64}, strconv.FormatUint(math.MaxUint64, 10)},
		{"int8 negative", Value{Type: TypeInt8, Int: -7}, "-7"},
		{"int64 min", Value{Type: TypeInt64, Int: math.MinInt64}, strconv.FormatInt(math.MinInt64, 10)},
		// A float32 carried in a float64 shows its float32 digits, not the expansion.
		{"float32", Value{Type: TypeFloat32, Float: float64(float32(0.1))}, "0.1"},
		{"float64", Value{Type: TypeFloat64, Float: 0.1}, "0.1"},
		{"array", Value{Type: TypeArray, ElemType: TypeString,
			Array: []Value{{Type: TypeString, Str: "a"}}}, "[1 string]"},
		{"empty array", Value{Type: TypeArray, ElemType: TypeInt32}, "[0 int32]"},
		// An unrecognized type reports its number rather than guessing a rendering: a value
		// this build does not understand is a fact a reader should see.
		{"unknown type", Value{Type: ValueType(99)}, "ValueType(99)"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.v.String(); got != c.want {
				t.Errorf("String() = %q, want %q", got, c.want)
			}
		})
	}
}

// TestFormatFloatPrecision pins the precision decision FormatFloat makes from the on-disk
// type.
//
// A float32 stored as 1e-5 is 9.999999747378752e-06 as a float64, and printing that in a dump
// claims a precision the file does not have. The two types therefore format differently, and
// the difference is the whole point of passing the type in.
func TestFormatFloatPrecision(t *testing.T) {
	// The same float64 value, formatted as each source type. The float32 rendering is
	// shorter because the value is what a float32 can hold.
	f := float64(float32(1e-5))
	if got, want := FormatFloat(TypeFloat32, f), strconv.FormatFloat(f, 'g', -1, 32); got != want {
		t.Errorf("FormatFloat(float32) = %q, want %q", got, want)
	}
	// A value that is exactly representable formats the same either way, so this pins that
	// the float32 path is not truncating indiscriminately.
	if got, want := FormatFloat(TypeFloat32, 8), "8"; got != want {
		t.Errorf("FormatFloat(float32, 8) = %q, want %q", got, want)
	}
	// A float64 keeps its own digits.
	if got, want := FormatFloat(TypeFloat64, 0.1), "0.1"; got != want {
		t.Errorf("FormatFloat(float64, 0.1) = %q, want %q", got, want)
	}
	// Any non-float32 type takes the float64 path, which is what a caller reaching a
	// float64 metadata value expects.
	if got, want := FormatFloat(TypeFloat64, 1.0/3), strconv.FormatFloat(1.0/3, 'g', -1, 64); got != want {
		t.Errorf("FormatFloat(float64, 1/3) = %q, want %q", got, want)
	}
}
