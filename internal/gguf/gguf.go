package gguf

import (
	"errors"
	"fmt"
	"io"
	"iter"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
)

// File is a parsed GGUF file.
//
// Parsing reads the header, metadata, and tensor directory, and leaves the
// tensor data on disk until TensorData is called. That keeps opening a
// multi-gigabyte model cheap, which matters for `orbix model inspect` and for
// any code that only wants the architecture or hyperparameters.
type File struct {
	// Version is the GGUF format revision from the header. Only Version is
	// accepted today.
	Version uint32

	// Metadata holds every key/value pair in the file, keyed by its
	// specification name ("general.architecture", "llama.block_count", ...).
	Metadata map[string]Value

	// TensorInfos lists tensor information in file order. TensorDataLoc is the
	// offset of the tensor data section, not of an individual tensor.
	TensorInfos []TensorInfo

	// TensorDataLoc is the absolute offset in the file where tensor data
	// begins. Individual tensors are addressed relative to it.
	TensorDataLoc int64

	// Alignment is the byte alignment of the tensor data section, taken from
	// general.alignment when present and defaulting to 32.
	Alignment int64
}

// TensorInfo describes one tensor in the file.
type TensorInfo struct {
	// Name is the tensor name, e.g. "blk.0.attn_q.weight".
	Name string
	// Dims holds the tensor shape in ggml order (innermost first), so a
	// 4096x4096 weight matrix is [4096, 4096] and a 1-D norm is [4096].
	Dims []uint64
	// Type is the ggml data type of the tensor elements.
	Type QuantType
	// Offset is the tensor's byte offset relative to File.TensorDataLoc.
	Offset uint64
}

// N returns the number of elements in the tensor.
func (t TensorInfo) N() int64 {
	n := int64(1)
	for _, d := range t.Dims {
		n *= int64(d)
	}
	return n
}

// Bytes returns the encoded size of the tensor in bytes, and false when the
// tensor's data type is not known to this package.
func (t TensorInfo) Bytes() (int64, bool) {
	return t.Type.Size(t.N())
}

// Open parses the GGUF file at path and closes it immediately after reading
// the directory. Tensor data must be requested through TensorData.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	file, err := Read(f)
	// The directory is fully parsed by now, so a close error cannot lose data.
	// It is still reported rather than discarded, because on some platforms a
	// close failure is the first sign of an I/O problem.
	if cerr := f.Close(); cerr != nil && err == nil {
		return nil, cerr
	}
	return file, err
}

// maxPreallocEntries caps how many entries the parser reserves space for before reading them.
//
// The counts in a header are attacker-controlled, and make() with a huge capacity is an
// immediate allocation: a file declaring 2^60 tensors is killed by the allocator before any
// read can fail. Reserving a sane amount and letting append grow the rest keeps a legitimate
// large file working -- the append cost is amortised -- while a lying header fails on the read
// that runs out of bytes.
//
// 4096 is well above any real metadata block and small enough to be free.
const maxPreallocEntries = 4096

// Read parses a GGUF file from r.
//
// Errors are wrapped with enough context to name the field that failed, since
// a truncated or corrupt model file otherwise produces an unhelpful "EOF".
func Read(r io.Reader) (*File, error) {
	var magic [4]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return nil, fmt.Errorf("gguf: read magic: %w", err)
	}
	if string(magic[:]) != Magic {
		return nil, fmt.Errorf("gguf: bad magic %q, want %q", magic[:], Magic)
	}

	rr := &reader{r: r, off: 4}
	version, err := rr.u32()
	if err != nil {
		return nil, fmt.Errorf("gguf: read version: %w", err)
	}
	if version != Version {
		return nil, fmt.Errorf("gguf: unsupported version %d, want %d", version, Version)
	}

	count, err := rr.u64()
	if err != nil {
		return nil, fmt.Errorf("gguf: read tensor count: %w", err)
	}
	kvCount, err := rr.u64()
	if err != nil {
		return nil, fmt.Errorf("gguf: read metadata count: %w", err)
	}
	// Both counts drive an allocation below, and both come from the file. A corrupted or
	// hostile header can declare 2^60 entries, and make() honours that as a request for
	// terabytes -- the process is killed by the allocator before any read can fail, so the
	// error paths below never get a chance to run.
	//
	// The counts themselves are left alone: a legitimate file may declare tens of thousands
	// of tensors, and rejecting a large count would reject it too. What is bounded is the
	// *pre-allocation*. Appending grows the slice on demand, so a file that declares more
	// entries than it holds fails on the read that runs out of data, with the contextual
	// error it already produced.
	file := &File{
		Version:  version,
		Metadata: make(map[string]Value, min(kvCount, maxPreallocEntries)),
	}

	for i := uint64(0); i < kvCount; i++ {
		key, err := rr.str()
		if err != nil {
			return nil, fmt.Errorf("gguf: read metadata key %d: %w", i, err)
		}
		typ, err := rr.u32()
		if err != nil {
			return nil, fmt.Errorf("gguf: read type of %q: %w", key, err)
		}
		val, err := rr.value(ValueType(typ))
		if err != nil {
			return nil, fmt.Errorf("gguf: read %q: %w", key, err)
		}
		file.Metadata[key] = val
	}

	file.TensorInfos = make([]TensorInfo, 0, min(count, maxPreallocEntries))
	for i := uint64(0); i < count; i++ {
		ti, err := readTensorInfo(rr)
		if err != nil {
			return nil, fmt.Errorf("gguf: read tensor info %d: %w", i, err)
		}
		file.TensorInfos = append(file.TensorInfos, ti)
	}

	file.Alignment = int64(file.floatOr("general.alignment", 32))
	if file.Alignment <= 0 {
		return nil, fmt.Errorf("gguf: invalid general.alignment %d", file.Alignment)
	}
	// Tensor data is aligned relative to the start of the file, so round the
	// current directory offset up to the next alignment boundary.
	file.TensorDataLoc = align(rr.off, file.Alignment)
	return file, nil
}

// readTensorInfo reads one tensor directory entry.
func readTensorInfo(r *reader) (TensorInfo, error) {
	name, err := r.str()
	if err != nil {
		return TensorInfo{}, err
	}
	nDims, err := r.u32()
	if err != nil {
		return TensorInfo{}, err
	}
	// A GGUF tensor is at most 4-dimensional in ggml; allow a little headroom
	// but reject counts that would only be seen in a corrupt file.
	if nDims > 8 {
		return TensorInfo{}, fmt.Errorf("gguf: tensor %q has %d dimensions", name, nDims)
	}
	dims := make([]uint64, nDims)
	for i := range dims {
		d, err := r.u64()
		if err != nil {
			return TensorInfo{}, err
		}
		dims[i] = d
	}
	typ, err := r.u32()
	if err != nil {
		return TensorInfo{}, err
	}
	off, err := r.u64()
	if err != nil {
		return TensorInfo{}, err
	}
	return TensorInfo{Name: name, Dims: dims, Type: QuantType(typ), Offset: off}, nil
}

func align(n, a int64) int64 {
	if r := n % a; r != 0 {
		return n - r + a
	}
	return n
}

// openReaderAt returns an io.ReaderAt over the file at path.
func openReaderAt(path string) (*os.File, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	st, err := f.Stat()
	if err != nil {
		// The handle is being discarded because the call already failed; a
		// close error would add nothing to the report.
		_ = f.Close()
		return nil, 0, err
	}
	return f, st.Size(), nil
}

// TensorData returns a reader over the named tensor's bytes. Closing it
// releases the file handle.
//
// This opens the file itself, which makes it convenient for reading one
// tensor. Callers streaming many tensors should keep one *os.File open and
// seek with TensorDataLoc and TensorInfo.Offset instead, so the model is not
// reopened per tensor.
func (f *File) TensorData(path, name string) (io.ReadCloser, error) {
	fh, size, err := openReaderAt(path)
	if err != nil {
		return nil, err
	}
	// Every path below either returns a reader that owns fh or returns an
	// error, in which case the handle must not leak. owned flips to true only
	// when the caller takes responsibility for closing it.
	owned := false
	defer func() {
		if !owned {
			// Discarding the handle alongside an already-failing call, so a
			// close error is not worth reporting.
			_ = fh.Close()
		}
	}()

	ti, ok := f.Tensor(name)
	if !ok {
		return nil, fmt.Errorf("gguf: no tensor named %q", name)
	}
	n, ok := ti.Bytes()
	if !ok {
		return nil, fmt.Errorf("gguf: tensor %q has unsupported type %s", name, ti.Type)
	}
	start := f.TensorDataLoc + int64(ti.Offset)
	if start < 0 || start+n > size {
		return nil, fmt.Errorf("gguf: tensor %q extends past end of file", name)
	}
	if _, err := fh.Seek(start, io.SeekStart); err != nil {
		return nil, err
	}
	// The returned reader must close the file, so wrap it in a closer rather
	// than NopCloser: leaking one descriptor per tensor would exhaust the
	// process limit on a model with hundreds of tensors.
	owned = true
	return &tensorReader{r: io.LimitReader(fh, n), c: fh}, nil
}

// tensorReader is an io.ReadCloser that closes its file when done.
type tensorReader struct {
	r io.Reader
	c io.Closer
}

func (t *tensorReader) Read(p []byte) (int, error) { return t.r.Read(p) }
func (t *tensorReader) Close() error               { return t.c.Close() }

// Tensor returns the tensor info for name.
func (f *File) Tensor(name string) (TensorInfo, bool) {
	for _, t := range f.TensorInfos {
		if t.Name == name {
			return t, true
		}
	}
	return TensorInfo{}, false
}

// -- typed metadata accessors ---------------------------------------------

// Value is one metadata entry.
//
// A single struct covers every GGUF value type instead of an interface, which
// keeps the accessors below simple and makes a metadata dump trivial to print.
// Use the field that matches Value.Type; the Num field is a convenience that
// holds any numeric value as a float64.
type Value struct {
	Type ValueType

	// Uint and Int hold integer values, preserving signedness.
	Uint uint64
	Int  int64

	// Float holds float32 and float64 values.
	Float float64

	// Num holds any numeric value as a float64. It exists so callers that
	// only care about magnitude do not need a type switch.
	Num float64

	// Bool holds bool values.
	Bool bool

	// Str holds string values.
	Str string

	// Array holds array elements. ElemType is the element type, which is
	// needed when the array is empty.
	Array    []Value
	ElemType ValueType
}

// String returns a short, human-readable rendering of the value.
func (v Value) String() string {
	switch v.Type {
	case TypeString:
		return v.Str
	case TypeBool:
		if v.Bool {
			return "true"
		}
		return "false"
	case TypeUint8, TypeUint16, TypeUint32, TypeUint64:
		return fmt.Sprint(v.Uint)
	case TypeInt8, TypeInt16, TypeInt32, TypeInt64:
		return fmt.Sprint(v.Int)
	case TypeFloat32, TypeFloat64:
		return FormatFloat(v.Type, v.Float)
	case TypeArray:
		return fmt.Sprintf("[%d %s]", len(v.Array), v.ElemType)
	default:
		return fmt.Sprintf("ValueType(%d)", uint32(v.Type))
	}
}

// FormatFloat renders a float with the precision its on-disk type carries.
//
// A float32 stored as 1e-5 is 9.999999747378752e-06 as a float64, and printing
// that in full is noise rather than information: the file cannot represent
// anything more precise. Formatting at the type's own precision keeps dumps
// honest and readable.
func FormatFloat(t ValueType, f float64) string {
	if t == TypeFloat32 {
		return strconv.FormatFloat(f, 'g', -1, 32)
	}
	return strconv.FormatFloat(f, 'g', -1, 64)
}

// MetadataCount returns the number of metadata entries.
func (f *File) MetadataCount() int { return len(f.Metadata) }

// Keys returns every metadata key in sorted order.
//
// Sorting is part of the contract: callers print these keys, and map iteration
// order would reorder the output between runs.
func (f *File) Keys() []string {
	return slices.Sorted(maps.Keys(f.Metadata))
}

// All iterates every metadata entry in sorted key order.
//
// This is the streaming counterpart to Keys: a dump can format each value as it
// arrives instead of materialising a slice of keys first.
func (f *File) All() iter.Seq2[string, Value] {
	return func(yield func(string, Value) bool) {
		for _, k := range f.Keys() {
			if !yield(k, f.Metadata[k]) {
				return
			}
		}
	}
}

// TensorCount returns the number of tensors in the directory.
//
// It exists because len() does not apply to the streaming accessor, and the
// count appears in every summary the CLI prints.
func (f *File) TensorCount() int { return len(f.TensorInfos) }

// Tensors iterates the tensor directory in file order.
//
// The directory is exposed both as a slice (TensorInfos) and as a stream: a
// dump or a loader walks it, and either form reads naturally at the call site.
func (f *File) Tensors() iter.Seq[TensorInfo] {
	return slices.Values(f.TensorInfos)
}

// Lookup returns the metadata value for key.
func (f *File) Lookup(key string) (Value, bool) {
	v, ok := f.Metadata[key]
	return v, ok
}

// Get returns the metadata value at key converted to T.
//
// One generic method replaces the per-type accessors (Uint32, Uint64, Float64,
// String, ...): the target type selects the conversion, and a value that cannot
// be represented as T is reported as absent rather than zero. Generic methods
// are a Go 1.27 feature, which is why this package requires that language
// version.
//
// Supported T: string, bool, and the integer and float types. Conversions
// between numeric metadata and Go numbers are allowed in both directions, which
// matches how GGUF writers store hyperparameters inconsistently -- the same key
// may be uint32 in one checkpoint and float32 in another. Narrowing to T is
// checked, so a uint64 that does not fit in a uint32 reports false instead of
// wrapping.
func (f *File) Get[T Scalar](key string) (T, bool) {
	var zero T
	v, ok := f.Metadata[key]
	if !ok {
		return zero, false
	}
	return scalarAs[T](v)
}

// GetArray returns the array stored at key converted elementwise to T.
//
// Element types are not silently mixed: asking for []float32 at a key holding
// an int32 array reports false, because a malformed score table should be
// reported rather than widened into something that looks plausible.
func (f *File) GetArray[T Scalar](key string) ([]T, bool) {
	v, ok := f.Metadata[key]
	if !ok || v.Type != TypeArray {
		return nil, false
	}
	out := make([]T, 0, len(v.Array))
	for _, e := range v.Array {
		converted, ok := scalarAs[T](e)
		if !ok {
			return nil, false
		}
		out = append(out, converted)
	}
	if len(out) == 0 {
		// A zero-length result is indistinguishable from the failure value, so
		// report success explicitly through a non-nil empty slice.
		out = []T{}
	}
	return out, true
}

// floatOr returns the floating-point metadata value for key, or def when the
// key is missing or not numeric.
//
// It exists for the alignment lookup, which happens before the accessors above
// are usable because it needs a value from the file being parsed.
func (f *File) floatOr(key string, def float64) float64 {
	if v, ok := f.Get[float64](key); ok {
		return v
	}
	return def
}

// Uint32 returns the unsigned integer metadata value for key.
func (f *File) Uint32(key string) (uint32, bool) { return f.Get[uint32](key) }

// Uint64 returns the unsigned integer metadata value for key.
func (f *File) Uint64(key string) (uint64, bool) { return f.Get[uint64](key) }

// Float64 returns the floating-point metadata value for key. Integer values
// are converted, which is what GGUF writers do for hyperparameters such as rope
// frequencies that are sometimes stored as integers.
func (f *File) Float64(key string) (float64, bool) { return f.Get[float64](key) }

// String returns the string metadata value for key.
func (f *File) String(key string) (string, bool) { return f.Get[string](key) }

// Strings returns the array of strings stored at key.
func (f *File) Strings(key string) ([]string, bool) { return f.GetArray[string](key) }

// Float32s returns the array of float32 values stored at key.
func (f *File) Float32s(key string) ([]float32, bool) { return f.GetArray[float32](key) }

// Int32s returns the array of int32 values stored at key.
func (f *File) Int32s(key string) ([]int32, bool) { return f.GetArray[int32](key) }

// ErrNoTensor is returned when a required tensor is missing from a file.
var ErrNoTensor = errors.New("gguf: tensor not found")

// Arch returns general.architecture, which names the model family and is the
// prefix for every architecture-specific metadata key.
func (f *File) Arch() (string, error) {
	arch, ok := f.String("general.architecture")
	if !ok {
		return "", fmt.Errorf("gguf: missing general.architecture")
	}
	return arch, nil
}

// BaseName returns the model name without directory or extension. It is the
// fallback display name when general.name is absent.
func BaseName(path string) string {
	base := filepath.Base(path)
	// strings.CutLast drops the final dot and everything after it, which is
	// the extension. A name with no dot is returned unchanged, and a dotfile
	// such as ".gguf" keeps its name rather than becoming empty.
	name, _, found := strings.CutLast(base, ".")
	if !found || name == "" {
		return base
	}
	return name
}
