package gguf

import (
	"bytes"
	"math"
	"testing"
)

// TestUint64ValuesSurviveAReadAtTheirOwnType covers the top half of the uint64 range.
//
// Every unsigned source used to be routed through int64 before the destination was
// consulted, so a value above MaxInt64 was rejected with ok=false even when the caller asked
// for uint64 and the value was exactly representable. Measured: Get[uint64] on stored values
// of 2^63 and 2^64-1 both returned (0, false), and GetArray[uint64] over an array containing
// 2^63 returned (nil, false) -- the whole array lost to one element.
//
// The rejection looked like a bounds check and was documented as one ("Narrowing conversions
// are bounds-checked rather than truncated"), which is why it survived: the check was correct
// for a signed destination and was being applied before anyone knew which destination it was.
//
// The signed half is asserted too, because the fix must not widen those: uint64 max still
// has to be reported unrepresentable as int64, int32 and int8.
func TestUint64ValuesSurviveAReadAtTheirOwnType(t *testing.T) {
	store := func(v Value) *File {
		t.Helper()
		data := buildFile([]kv{{"k", v}}, nil, 32, nil)
		f, err := Read(bytes.NewReader(data))
		if err != nil {
			t.Fatalf("Read: %v", err)
		}
		return f
	}
	const aboveMaxInt64 = uint64(math.MaxInt64) + 1

	t.Run("scalar", func(t *testing.T) {
		for _, want := range []uint64{0, math.MaxInt64, aboveMaxInt64, math.MaxUint64} {
			f := store(Value{Type: TypeUint64, Uint: want})
			got, ok := f.Get[uint64]("k")
			if !ok {
				t.Errorf("Get[uint64] rejected the stored value %d", want)
				continue
			}
			if got != want {
				t.Errorf("Get[uint64] = %d, want %d", got, want)
			}
		}
	})

	t.Run("array", func(t *testing.T) {
		elems := []Value{
			{Type: TypeUint64, Uint: 1},
			{Type: TypeUint64, Uint: aboveMaxInt64},
			{Type: TypeUint64, Uint: math.MaxUint64},
		}
		f := store(Value{Type: TypeArray, ElemType: TypeUint64, Array: elems})
		got, ok := f.GetArray[uint64]("k")
		if !ok {
			t.Fatal("GetArray[uint64] rejected an array holding 2^63 and 2^64-1")
		}
		want := []uint64{1, aboveMaxInt64, math.MaxUint64}
		if len(got) != len(want) {
			t.Fatalf("got %d elements, want %d", len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("element %d = %d, want %d", i, got[i], want[i])
			}
		}
	})

	t.Run("signed destinations still reject it", func(t *testing.T) {
		f := store(Value{Type: TypeUint64, Uint: math.MaxUint64})
		if v, ok := f.Get[int64]("k"); ok {
			t.Errorf("int64 accepted uint64 max as %d", v)
		}
		if v, ok := f.Get[int32]("k"); ok {
			t.Errorf("int32 accepted uint64 max as %d", v)
		}
		if v, ok := f.Get[int8]("k"); ok {
			t.Errorf("int8 accepted uint64 max as %d", v)
		}
		// And a value that does fit must still convert.
		f2 := store(Value{Type: TypeUint64, Uint: 42})
		if v, ok := f2.Get[int8]("k"); !ok || v != 42 {
			t.Errorf("Get[int8] on 42 = (%d, %v), want (42, true)", v, ok)
		}
	})

	t.Run("narrow unsigned destinations", func(t *testing.T) {
		for _, c := range []struct {
			v    uint64
			fits bool
		}{
			// uint16 holds up to 65535, so 256 is not a boundary for it -- an earlier
			// version of this table said it was, confusing the uint8 edge with this one.
			{255, true},
			{256, true},
			{math.MaxUint16, true},
			{math.MaxUint16 + 1, false},
		} {
			f := store(Value{Type: TypeUint64, Uint: c.v})
			got, ok := f.Get[uint16]("k")
			if ok != c.fits {
				t.Errorf("Get[uint16] on %d: ok=%v, want %v", c.v, ok, c.fits)
			}
			if c.fits && got != uint16(c.v) {
				t.Errorf("Get[uint16] on %d = %d", c.v, got)
			}
		}
	})
}
