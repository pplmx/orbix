package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
)

type kv struct {
	key string
	val Value
}

// buildFile renders a complete GGUF file: magic, version, counts, metadata,
// tensor directory, alignment padding, then tensor data.
func buildFile(kvs []kv, tensors []TensorInfo, alignment uint64, data []byte) []byte {
	var b bytes.Buffer
	b.WriteString(Magic)
	u32 := func(v uint32) {
		var tmp [4]byte
		binary.LittleEndian.PutUint32(tmp[:], v)
		b.Write(tmp[:])
	}
	u64 := func(v uint64) {
		var tmp [8]byte
		binary.LittleEndian.PutUint64(tmp[:], v)
		b.Write(tmp[:])
	}
	str := func(s string) {
		u64(uint64(len(s)))
		b.WriteString(s)
	}

	u32(Version)
	u64(uint64(len(tensors)))
	u64(uint64(len(kvs)))

	for _, e := range kvs {
		str(e.key)
		u32(uint32(e.val.Type))
		encodeValue(&b, e.val)
	}
	for _, ti := range tensors {
		str(ti.Name)
		u32(uint32(len(ti.Dims)))
		for _, d := range ti.Dims {
			u64(d)
		}
		u32(uint32(ti.Type))
		u64(ti.Offset)
	}

	if alignment > 0 {
		if pad := align(int64(b.Len()), int64(alignment)); pad > int64(b.Len()) {
			b.Write(make([]byte, pad-int64(b.Len())))
		}
	}
	b.Write(data)
	return b.Bytes()
}

func encodeValue(b *bytes.Buffer, v Value) {
	u8 := func(x uint8) { b.WriteByte(x) }
	u16 := func(x uint16) {
		var tmp [2]byte
		binary.LittleEndian.PutUint16(tmp[:], x)
		b.Write(tmp[:])
	}
	u32 := func(x uint32) {
		var tmp [4]byte
		binary.LittleEndian.PutUint32(tmp[:], x)
		b.Write(tmp[:])
	}
	u64 := func(x uint64) {
		var tmp [8]byte
		binary.LittleEndian.PutUint64(tmp[:], x)
		b.Write(tmp[:])
	}
	str := func(s string) {
		u64(uint64(len(s)))
		b.WriteString(s)
	}

	switch v.Type {
	case TypeUint8:
		u8(uint8(v.Uint))
	case TypeInt8:
		u8(uint8(int8(v.Int)))
	case TypeUint16:
		u16(uint16(v.Uint))
	case TypeInt16:
		u16(uint16(int16(v.Int)))
	case TypeUint32:
		u32(uint32(v.Uint))
	case TypeInt32:
		u32(uint32(int32(v.Int)))
	case TypeUint64:
		u64(v.Uint)
	case TypeInt64:
		u64(uint64(v.Int))
	case TypeFloat32:
		u32(math.Float32bits(float32(v.Float)))
	case TypeFloat64:
		u64(math.Float64bits(v.Float))
	case TypeBool:
		if v.Bool {
			u8(1)
		} else {
			u8(0)
		}
	case TypeString:
		str(v.Str)
	case TypeArray:
		u32(uint32(v.ElemType))
		u64(uint64(len(v.Array)))
		for _, e := range v.Array {
			encodeValue(b, e)
		}
	default:
		panic("encodeValue: unsupported value type")
	}
}

func strVal(s string) Value { return Value{Type: TypeString, Str: s} }
func u32Val(v uint32) Value { return Value{Type: TypeUint32, Uint: uint64(v)} }
func i32Val(v int32) Value  { return Value{Type: TypeInt32, Int: int64(v)} }
func u64Val(v uint64) Value { return Value{Type: TypeUint64, Uint: v} }
func f32Val(v float32) Value {
	return Value{Type: TypeFloat32, Float: float64(v)}
}
func boolVal(v bool) Value { return Value{Type: TypeBool, Bool: v} }

func strArray(vals ...string) Value {
	arr := make([]Value, len(vals))
	for i, s := range vals {
		arr[i] = strVal(s)
	}
	return Value{Type: TypeArray, ElemType: TypeString, Array: arr}
}

func i32Array(vals ...int32) Value {
	arr := make([]Value, len(vals))
	for i, v := range vals {
		arr[i] = i32Val(v)
	}
	return Value{Type: TypeArray, ElemType: TypeInt32, Array: arr}
}
