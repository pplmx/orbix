package gguf

import (
	"fmt"
	"io"
	"math"
	"unicode/utf8"
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

// maxArrayElements bounds one metadata array's declared length.
//
// An array element is at least one byte on disk, so this is also the largest count a file of
// that many bytes could honestly carry. Real files are far below it: a 128k vocabulary is an
// array of 128000 short strings.
//
// It replaces a MaxInt32 check, which bounded the count at a number the allocation could not
// survive -- 2^31 Values is roughly 200 GiB -- while still letting a small file ask for 6 GiB.
const maxArrayElements = 64 << 20

// readFull fills b, returning the underlying error (including io.ErrUnexpectedEOF).
func (r *reader) readFull(b []byte) error {
	n, err := io.ReadFull(r.r, b)
	r.off += int64(n)
	return err
}

func (r *reader) bytes(n uint64) ([]byte, error) {
	// The length comes from the file and is not corroborated by anything before the bytes
	// are read, so it must not decide an allocation on its own. Measured: a file of a few
	// hundred bytes claiming a 1 GiB string made Read allocate 1024 MiB and then report
	// io.ErrUnexpectedEOF -- eight bytes of input costing a gigabyte, seven times over for a
	// 64-bit length field. The cap below is the first half of the answer and the chunked read
	// is the second: with the cap alone, every value just under it is still a free 2 GiB.
	//
	// maxMetadataBytes bounds one string or byte run. Real GGUF metadata is small -- the
	// largest strings in a 580 MB checkpoint are a few hundred bytes of chat template, and
	// the vocabulary arrives as an array of short strings -- so this is orders of magnitude
	// above anything a writer produces while still being far below a gigabyte.
	const maxMetadataBytes = 64 << 20
	if n > maxMetadataBytes {
		return nil, fmt.Errorf("gguf: refusing to allocate %d bytes for one value (limit %d)",
			n, maxMetadataBytes)
	}

	// Read in bounded chunks and grow only as bytes actually arrive. A file that claims more
	// than it holds therefore costs the bytes it really has, not the bytes it promised.
	b := make([]byte, 0, min(n, 4096))
	var chunk [4096]byte
	for remaining := n; remaining > 0; {
		size := min(remaining, uint64(len(chunk)))
		if err := r.readFull(chunk[:size]); err != nil {
			return nil, err
		}
		b = append(b, chunk[:size]...)
		remaining -= size
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
	// A GGUF string is declared UTF-8 (doc.go, and the format spec's metadata section), and
	// nothing downstream is written for a string that is not. The damage is quiet rather than
	// loud: encoding/json replaces an invalid byte with U+FFFD, so `orbix model inspect --json`
	// reports a value that is not in the file, and a vocabulary piece with a bad byte splits
	// into RuneError symbols that can never merge into the token they should be.
	//
	// Refused here rather than tolerated because a file that fails this is corrupt, and the
	// alternative is a tool that silently shows a different file from the one on disk.
	if !utf8.Valid(b) {
		return "", fmt.Errorf("gguf: string value is not valid UTF-8")
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
	if n > maxArrayElements {
		return Value{}, fmt.Errorf("gguf: array of %d elements is too large", n)
	}
	// Grown as elements arrive, not reserved up front: the count is the file's claim and is
	// not corroborated by anything until the elements have been read. Measured before this:
	// a file of a few dozen bytes claiming 67M elements made Read allocate 6 GiB before
	// reporting EOF, because a Value is about 96 bytes and the capacity was taken from the
	// claim. An element occupies at least one byte on disk, so a file cannot honestly hold
	// more elements than it has bytes -- but Read is given a plain io.Reader and cannot know
	// that length, which is why the growth has to be driven by what arrives.
	v.Array = make([]Value, 0, min(n, 64))
	for i := uint64(0); i < n; i++ {
		elem, err := r.value(et)
		if err != nil {
			return Value{}, fmt.Errorf("gguf: array element %d: %w", i, err)
		}
		v.Array = append(v.Array, elem)
	}
	return v, nil
}
