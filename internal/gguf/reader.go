package gguf

import (
	"fmt"
	"io"
	"math"
)

// reader wraps an io.Reader with the little-endian primitives every GGUF
// field is encoded with.
//
// GGUF is unconditionally little-endian, so the host's byte order never
// matters; binary.LittleEndian does the right thing on every platform. The
// reader tracks how many bytes it has consumed so callers can locate the
// tensor data section, which starts right after the directory.
type reader struct {
	r   io.Reader
	off int64 // bytes consumed so far
}

// readFull fills b, returning the underlying error (including io.ErrUnexpectedEOF).
func (r *reader) readFull(b []byte) error {
	n, err := io.ReadFull(r.r, b)
	r.off += int64(n)
	return err
}

func (r *reader) bytes(n uint64) ([]byte, error) {
	if n > uint64(math.MaxInt32) {
		return nil, fmt.Errorf("gguf: refusing to allocate %d bytes", n)
	}
	b := make([]byte, n)
	if err := r.readFull(b); err != nil {
		return nil, err
	}
	return b, nil
}

func (r *reader) u32() (uint32, error) {
	var b [4]byte
	if err := r.readFull(b[:]); err != nil {
		return 0, err
	}
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24, nil
}

func (r *reader) u64() (uint64, error) {
	var b [8]byte
	if err := r.readFull(b[:]); err != nil {
		return 0, err
	}
	var v uint64
	for i := 7; i >= 0; i-- {
		v = v<<8 | uint64(b[i])
	}
	return v, nil
}

// str reads a GGUF string: a uint64 length followed by that many UTF-8 bytes.
func (r *reader) str() (string, error) {
	n, err := r.u64()
	if err != nil {
		return "", err
	}
	b, err := r.bytes(n)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// value reads one metadata value of the given type.
//
// Arrays are decoded into []Value recursively. Scalars are decoded into the
// narrowest Go type that holds them losslessly, except that booleans become
// bool and strings become string. Every numeric type keeps its signedness,
// which matters for keys like tokenizer.ggml.token_type.
func (r *reader) value(t ValueType) (Value, error) {
	switch t {
	case TypeUint8:
		b, err := r.bytes(1)
		if err != nil {
			return Value{}, err
		}
		return Value{Type: t, Uint: uint64(b[0]), Num: float64(b[0])}, nil
	case TypeInt8:
		b, err := r.bytes(1)
		if err != nil {
			return Value{}, err
		}
		v := int64(int8(b[0]))
		return Value{Type: t, Int: v, Num: float64(v)}, nil
	case TypeUint16:
		b, err := r.bytes(2)
		if err != nil {
			return Value{}, err
		}
		v := uint64(b[0]) | uint64(b[1])<<8
		return Value{Type: t, Uint: v, Num: float64(v)}, nil
	case TypeInt16:
		b, err := r.bytes(2)
		if err != nil {
			return Value{}, err
		}
		v := int64(int16(uint16(b[0]) | uint16(b[1])<<8))
		return Value{Type: t, Int: v, Num: float64(v)}, nil
	case TypeUint32:
		v, err := r.u32()
		if err != nil {
			return Value{}, err
		}
		return Value{Type: t, Uint: uint64(v), Num: float64(v)}, nil
	case TypeInt32:
		v, err := r.u32()
		if err != nil {
			return Value{}, err
		}
		i := int64(int32(v))
		return Value{Type: t, Int: i, Num: float64(i)}, nil
	case TypeUint64:
		v, err := r.u64()
		if err != nil {
			return Value{}, err
		}
		return Value{Type: t, Uint: v, Num: float64(v)}, nil
	case TypeInt64:
		v, err := r.u64()
		if err != nil {
			return Value{}, err
		}
		i := int64(v)
		return Value{Type: t, Int: i, Num: float64(i)}, nil
	case TypeFloat32:
		v, err := r.u32()
		if err != nil {
			return Value{}, err
		}
		f := math.Float32frombits(v)
		return Value{Type: t, Float: float64(f), Num: float64(f)}, nil
	case TypeFloat64:
		v, err := r.u64()
		if err != nil {
			return Value{}, err
		}
		f := math.Float64frombits(v)
		return Value{Type: t, Float: f, Num: f}, nil
	case TypeBool:
		b, err := r.bytes(1)
		if err != nil {
			return Value{}, err
		}
		return Value{Type: t, Bool: b[0] != 0}, nil
	case TypeString:
		s, err := r.str()
		if err != nil {
			return Value{}, err
		}
		return Value{Type: t, Str: s}, nil
	case TypeArray:
		return r.array()
	default:
		return Value{}, fmt.Errorf("gguf: unknown metadata value type %d", uint32(t))
	}
}

// array reads a GGUF array: element type, element count, then the elements.
func (r *reader) array() (Value, error) {
	elemType, err := r.u32()
	if err != nil {
		return Value{}, err
	}
	n, err := r.u64()
	if err != nil {
		return Value{}, err
	}
	et := ValueType(elemType)
	// GGUF's metadata section forbids an array of arrays: "Arrays can contain other types
	// but not other arrays" (gguf-spec.md, section III). Accepting one would give the value
	// a shape the rest of this package has no meaning for -- GetArray compares each element's
	// type against T, so an array element inside an array reads as "the key is absent", which
	// is the silent skip this package's parser rules exist to prevent.
	//
	// Refused here rather than by depth counting: one level is the rule, so there is nothing
	// to count. The recursion in value() cannot run away because it can no longer be entered
	// for an array element.
	if et == TypeArray {
		return Value{}, fmt.Errorf("gguf: array of arrays is not allowed by the format")
	}
	v := Value{Type: TypeArray, ElemType: et}

	// An array element is at least one byte on disk, so n can never exceed
	// the remaining file size. Guard against absurd counts before allocating
	// the slice, which is what a trivially corrupt header would otherwise
	// trigger.
	if n > math.MaxInt32 {
		return Value{}, fmt.Errorf("gguf: array of %d elements is too large", n)
	}
	v.Array = make([]Value, 0, n)
	for i := uint64(0); i < n; i++ {
		elem, err := r.value(et)
		if err != nil {
			return Value{}, fmt.Errorf("gguf: array element %d: %w", i, err)
		}
		v.Array = append(v.Array, elem)
	}
	return v, nil
}
