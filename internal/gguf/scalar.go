package gguf

import "math"

// Scalar is the set of Go types a single GGUF metadata value can be converted
// to.
//
// The constraint is what makes Get and GetArray one method each instead of a
// family of per-type accessors. It covers exactly the types GGUF can store -- a
// string, a bool, and the fixed-width integers and floats -- so there is no
// combination a caller can ask for that the file format cannot express.
type Scalar interface {
	~string | ~bool |
		~int8 | ~int16 | ~int32 | ~int64 | ~int |
		~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uint |
		~float32 | ~float64
}

// scalarAs converts one metadata value to T, reporting false when the value
// cannot be represented exactly.
//
// Numeric metadata is converted in both directions because GGUF writers are
// inconsistent about how they store the same hyperparameter; a model may record
// a rope base as uint32 while another uses float32. Narrowing conversions are
// bounds-checked rather than truncated, so a corrupt or unrepresentable value
// shows up as a missing key instead of a wrong number that silently changes
// inference.
func scalarAs[T Scalar](v Value) (T, bool) {
	var zero T
	switch any(zero).(type) {
	case string:
		if v.Type != TypeString {
			return zero, false
		}
		// T is constrained to ~string here, so asserting back to a named string
		// type through any is well-defined; the fallthrough keeps the compiler
		// from having to reason about the union.
		if out, ok := any(v.Str).(T); ok {
			return out, true
		}
		return zero, false
	case bool:
		if v.Type != TypeBool {
			return zero, false
		}
		if out, ok := any(v.Bool).(T); ok {
			return out, true
		}
		return zero, false
	}

	// Everything below is numeric.
	isFloat := v.Type == TypeFloat32 || v.Type == TypeFloat64
	if !isFloat && !v.Type.isInteger() {
		return zero, false
	}

	switch any(zero).(type) {
	case float32, float64:
		// Accept any numeric source, converting integer metadata to a float.
		var f float64
		switch {
		case isFloat:
			f = v.Float
		case v.Type.isUnsigned():
			f = float64(v.Uint)
		default:
			f = float64(v.Int)
		}
		if out, ok := floatAs[T](f); ok {
			return out, true
		}
		return zero, false
	default:
		// Target is an integer type. A float source is accepted only when it is
		// integral, so float32(3) converts to 3 but float32(3.5) is reported as
		// unrepresentable rather than silently truncated.
		if isFloat {
			if v.Float != math.Trunc(v.Float) || math.IsInf(v.Float, 0) || math.IsNaN(v.Float) {
				return zero, false
			}
			return intAs[T](int64(v.Float))
		}
		if v.Type.isUnsigned() {
			if v.Uint > math.MaxInt64 {
				return zero, false
			}
			return intAs[T](int64(v.Uint))
		}
		return intAs[T](v.Int)
	}
}

// intAs converts n to the integer type T, reporting false on overflow.
func intAs[T Scalar](n int64) (T, bool) {
	var zero T
	switch any(zero).(type) {
	case int8:
		if n < math.MinInt8 || n > math.MaxInt8 {
			return zero, false
		}
		return convert[T](int8(n))
	case int16:
		if n < math.MinInt16 || n > math.MaxInt16 {
			return zero, false
		}
		return convert[T](int16(n))
	case int32:
		if n < math.MinInt32 || n > math.MaxInt32 {
			return zero, false
		}
		return convert[T](int32(n))
	case int64:
		return convert[T](n)
	case int:
		if int64(int(n)) != n {
			return zero, false
		}
		// The argument must be an int, not the int64 n: convert asserts to T,
		// and asserting an int64 to int fails even when the value fits.
		return convert[T](int(n))
	case uint8:
		if n < 0 || n > math.MaxUint8 {
			return zero, false
		}
		return convert[T](uint8(n))
	case uint16:
		if n < 0 || n > math.MaxUint16 {
			return zero, false
		}
		return convert[T](uint16(n))
	case uint32:
		if n < 0 || n > math.MaxUint32 {
			return zero, false
		}
		return convert[T](uint32(n))
	case uint64:
		if n < 0 {
			return zero, false
		}
		return convert[T](uint64(n))
	case uint:
		if n < 0 || uint64(n) > uint64(^uint(0)) {
			return zero, false
		}
		return convert[T](uint(n))
	case float32:
		return floatAs[T](float64(n))
	case float64:
		return floatAs[T](float64(n))
	default:
		return zero, false
	}
}

// isInteger reports whether the type holds an integer.
func (t ValueType) isInteger() bool {
	switch t {
	case TypeUint8, TypeInt8, TypeUint16, TypeInt16, TypeUint32, TypeInt32,
		TypeUint64, TypeInt64:
		return true
	default:
		return false
	}
}

// isUnsigned reports whether the type holds an unsigned integer.
func (t ValueType) isUnsigned() bool {
	switch t {
	case TypeUint8, TypeUint16, TypeUint32, TypeUint64:
		return true
	default:
		return false
	}
}

// floatAs converts f to the float type T, reporting false on overflow.
//
// A float32 destination cannot represent every float64, so the conversion is
// checked: a value that becomes +Inf on narrowing is reported as absent rather
// than silently replacing a real number with an infinity.
func floatAs[T Scalar](f float64) (T, bool) {
	var zero T
	switch any(zero).(type) {
	case float32:
		c := float32(f)
		if math.IsInf(float64(c), 0) && !math.IsInf(f, 0) {
			return zero, false
		}
		return any(c).(T), true
	case float64:
		return any(f).(T), true
	default:
		return zero, false
	}
}

// convert narrows a concrete Go value to T through a checked assertion.
//
// The Scalar constraint spans string, bool, and the numeric types, so the
// compiler cannot accept T(x) directly: it would have to prove x's type is in
// the union, which a type switch on `any(zero)` does not narrow for it. The
// assertion is safe because each call site only reaches this after a switch
// that established the matching kind.
func convert[T Scalar](x any) (T, bool) {
	out, ok := x.(T)
	return out, ok
}
