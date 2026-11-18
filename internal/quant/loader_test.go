package quant

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
)

// writeModel writes a GGUF file holding the given tensors and returns its
// path. It exists so the loader tests exercise the real file path, including
// offset arithmetic and alignment, rather than an in-memory stand-in.
func writeModel(t *testing.T, tensors []gguf.TensorInfo, data []byte) string {
	t.Helper()

	var b bytes.Buffer
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

	b.WriteString("GGUF")
	u32(3)
	u64(uint64(len(tensors)))
	u64(1) // one metadata entry: the architecture

	str("general.architecture")
	u32(8) // string
	str("llama")

	for _, ti := range tensors {
		str(ti.Name)
		u32(uint32(len(ti.Dims)))
		for _, d := range ti.Dims {
			u64(d)
		}
		u32(uint32(ti.Type))
		u64(ti.Offset)
	}
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	b.Write(data)

	path := filepath.Join(t.TempDir(), "model.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestLoaderRejectsWrappingOffset covers the arithmetic that decides whether a tensor's
// bytes are inside the file.
//
// All three entry points computed the start as `TensorDataLoc + int64(ti.Offset)` and then
// tested `start < 0 || start+nbytes > size`. Both halves can be defeated by a value the file
// supplies: an offset above MaxInt64 casts to a negative int64, so the sum is merely shifted
// and stays positive, and a large start plus a large length wraps the sum negative. The
// reader is then handed a position before the data section and reads the wrong bytes with no
// error. This is the same defect gguf.TensorData had, fixed there first; the three copies here
// were missed because the fix went to the site the investigation named.
//
// They now share dataStart, which checks the offset against MaxInt64 before casting and
// rewrites the remaining test so no addition can overflow after that point.
func TestLoaderRejectsWrappingOffset(t *testing.T) {
	l := &Loader{size: 1000}
	l.file = &gguf.File{TensorDataLoc: 100}
	const nbytes = 16

	// Offsets that cast negative: refused as too large, so the message names the offset
	// rather than blaming the file length.
	for _, off := range []uint64{1 << 63, math.MaxUint64, math.MaxUint64 - 8} {
		ti := gguf.TensorInfo{Name: "w", Offset: off, Dims: []uint64{4}, Type: gguf.F32}
		if _, err := l.dataStart(ti, nbytes); err == nil {
			t.Errorf("dataStart accepted offset %d, which wraps to %d as int64", off, int64(off))
		} else if !strings.Contains(err.Error(), "too large") {
			t.Errorf("offset %d: error %q does not say the offset is the problem", off, err)
		}
	}

	// A representable offset that is simply past the end is a different failure, and must be
	// reported as such rather than as a range problem with the offset.
	ti := gguf.TensorInfo{Name: "w", Offset: 5000, Dims: []uint64{4}, Type: gguf.F32}
	if _, err := l.dataStart(ti, nbytes); err == nil {
		t.Error("dataStart accepted an offset past the end of the file")
	} else if !strings.Contains(err.Error(), "past end") {
		t.Errorf("error %q does not say the tensor is past the end", err)
	}

	// A tensor that exactly fills the file must still be accepted, so the new test cannot be
	// satisfied by rejecting everything near the boundary.
	ti = gguf.TensorInfo{Name: "w", Offset: 884, Dims: []uint64{4}, Type: gguf.F32}
	if start, err := l.dataStart(ti, nbytes); err != nil || start != 984 {
		t.Errorf("dataStart at the last whole tensor = %d, %v; want 984, nil", start, err)
	}
}

// TestLoaderRejectsBadTensor covers the loader's refusal paths.
//
// Each is a check that exists so a corrupt or unsupported file produces a message naming
// the tensor, rather than a short read, a slice panic, or a decoded array of garbage. They
// were reachable but unasserted: coverage showed every one of these branches at zero, on a
// package whose decoder functions are all at 100% -- the arithmetic was tested and the
// guards around it were not.
//
// Tensor, TensorInto and TensorReader each perform their own checks rather than sharing
// one, so each is driven separately; the assertions name the message because on these paths
// the message is the entire product.
func TestLoaderRejectsBadTensor(t *testing.T) {
	// A file with one F32 tensor of two elements, so a valid name exists alongside the
	// invalid ones.
	good := writeModel(t, []gguf.TensorInfo{
		{Name: "w", Type: gguf.F32, Dims: []uint64{2}},
	}, make([]byte, 8))

	t.Run("unknown name on every entry point", func(t *testing.T) {
		l, err := OpenLoader(good)
		if err != nil {
			t.Fatalf("OpenLoader: %v", err)
		}
		defer func() { _ = l.Close() }()
		if _, err := l.Tensor("absent"); err == nil {
			t.Error("Tensor(absent) succeeded")
		} else if !strings.Contains(err.Error(), "no tensor named") {
			t.Errorf("Tensor error %q does not name the tensor", err)
		}
		if err := l.TensorInto("absent", make([]float32, 2)); err == nil {
			t.Error("TensorInto(absent) succeeded")
		} else if !strings.Contains(err.Error(), "no tensor named") {
			t.Errorf("TensorInto error %q does not name the tensor", err)
		}
		if _, err := l.TensorReader("absent"); err == nil {
			t.Error("TensorReader(absent) succeeded")
		} else if !strings.Contains(err.Error(), "no tensor named") {
			t.Errorf("TensorReader error %q does not name the tensor", err)
		}
	})

	t.Run("destination size mismatch", func(t *testing.T) {
		l, err := OpenLoader(good)
		if err != nil {
			t.Fatalf("OpenLoader: %v", err)
		}
		defer func() { _ = l.Close() }()
		// The tensor holds two elements; a three-element destination is a caller bug that
		// must be reported rather than half-filled.
		if err := l.TensorInto("w", make([]float32, 3)); err == nil {
			t.Error("TensorInto accepted a mismatched destination")
		} else if !strings.Contains(err.Error(), "destination has") {
			t.Errorf("error %q does not name the destination size", err)
		}
	})

	t.Run("unsupported quantization type", func(t *testing.T) {
		// A type this build cannot decode. The directory records it faithfully and the
		// loader is what has to refuse it.
		unsupported := gguf.QuantType(9999)
		path := writeModel(t, []gguf.TensorInfo{
			{Name: "odd", Type: unsupported, Dims: []uint64{2}},
		}, make([]byte, 8))
		l, err := OpenLoader(path)
		if err != nil {
			// The loader may refuse at open time, which is also a correct answer.
			if !strings.Contains(err.Error(), "unsupported") {
				t.Errorf("OpenLoader error %q does not say the type is unsupported", err)
			}
			return
		}
		defer func() { _ = l.Close() }()
		if _, err := l.Tensor("odd"); err == nil {
			t.Error("Tensor accepted an unsupported quantization type")
		} else if !strings.Contains(err.Error(), "unsupported type") {
			t.Errorf("error %q does not say the type is unsupported", err)
		}
	})
}

func TestLoaderTensor(t *testing.T) {
	// Two tensors: 8 F32 values at offset 0, one Q4_0 block at offset 32.
	data := make([]byte, 0, 32+18)
	for i := 0; i < 8; i++ {
		var tmp [4]byte
		binary.LittleEndian.PutUint32(tmp[:], math.Float32bits(float32(i+1)))
		data = append(data, tmp[:]...)
	}
	q := q4_0Block(t, fill(32, 9), 1) // all codes 9 => +1.0 each
	data = append(data, q...)

	path := writeModel(t, []gguf.TensorInfo{
		{Name: "a", Dims: []uint64{8}, Type: gguf.F32, Offset: 0},
		{Name: "b", Dims: []uint64{32}, Type: gguf.Q4_0, Offset: 32},
	}, data)

	l, err := OpenLoader(path)
	if err != nil {
		t.Fatalf("OpenLoader: %v", err)
	}
	defer closeLoader(t, l)

	got, err := l.Tensor("a")
	if err != nil {
		t.Fatalf("Tensor(a): %v", err)
	}
	for i := 0; i < 8; i++ {
		if want := float32(i + 1); got[i] != want {
			t.Errorf("a[%d] = %v, want %v", i, got[i], want)
		}
	}

	got, err = l.Tensor("b")
	if err != nil {
		t.Fatalf("Tensor(b): %v", err)
	}
	if len(got) != 32 {
		t.Fatalf("len(b) = %d, want 32", len(got))
	}
	for i := range got {
		if got[i] != 1 {
			t.Errorf("b[%d] = %v, want 1", i, got[i])
		}
	}

	if _, err := l.Tensor("missing"); err == nil {
		t.Error("Tensor(missing) succeeded")
	}
}

func TestLoaderTensorInto(t *testing.T) {
	data := make([]byte, 32)
	for i := 0; i < 8; i++ {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(float32(i)))
	}
	path := writeModel(t, []gguf.TensorInfo{
		{Name: "a", Dims: []uint64{8}, Type: gguf.F32, Offset: 0},
	}, data)

	l, err := OpenLoader(path)
	if err != nil {
		t.Fatalf("OpenLoader: %v", err)
	}
	defer closeLoader(t, l)

	dst := make([]float32, 8)
	if err := l.TensorInto("a", dst); err != nil {
		t.Fatalf("TensorInto: %v", err)
	}
	if dst[3] != 3 {
		t.Errorf("dst[3] = %v, want 3", dst[3])
	}

	// A destination of the wrong length must be rejected before any bytes are
	// written, otherwise a caller's buffer would be partially overwritten.
	if err := l.TensorInto("a", make([]float32, 4)); err == nil {
		t.Error("TensorInto with a short destination succeeded")
	}
}

func TestLoaderTensorReader(t *testing.T) {
	data := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	path := writeModel(t, []gguf.TensorInfo{
		{Name: "raw", Dims: []uint64{8}, Type: gguf.I8, Offset: 0},
	}, data)

	l, err := OpenLoader(path)
	if err != nil {
		t.Fatalf("OpenLoader: %v", err)
	}
	defer closeLoader(t, l)

	r, err := l.TensorReader("raw")
	if err != nil {
		t.Fatalf("TensorReader: %v", err)
	}
	var buf bytes.Buffer
	if _, err := buf.ReadFrom(r); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(buf.Bytes(), data) {
		t.Errorf("raw bytes = %v, want %v", buf.Bytes(), data)
	}
}

func TestLoaderRejectsTruncatedTensor(t *testing.T) {
	// The directory claims 1 MiB of F32 data but the file ends early. The
	// loader must report that instead of returning a buffer with zeros.
	path := writeModel(t, []gguf.TensorInfo{
		{Name: "big", Dims: []uint64{262144}, Type: gguf.F32, Offset: 0},
	}, make([]byte, 64))

	l, err := OpenLoader(path)
	if err != nil {
		t.Fatalf("OpenLoader: %v", err)
	}
	defer closeLoader(t, l)
	if _, err := l.Tensor("big"); err == nil {
		t.Fatal("Tensor on a truncated file succeeded")
	}
}

func TestLoaderFile(t *testing.T) {
	path := writeModel(t, nil, nil)
	l, err := OpenLoader(path)
	if err != nil {
		t.Fatalf("OpenLoader: %v", err)
	}
	defer closeLoader(t, l)
	if got := l.File(); got == nil {
		t.Fatal("File() = nil")
	}
	if arch, err := l.File().Arch(); err != nil || arch != "llama" {
		t.Errorf("Arch() = %q, %v", arch, err)
	}
}

// closeLoader closes a loader at the end of a test. A close error on a test
// fixture means the test file itself is broken, which the failure reports.
func closeLoader(t *testing.T, l *Loader) {
	t.Helper()
	if err := l.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
}

func TestPreload(t *testing.T) {
	// Two tensors of different quant types, loaded through the concurrent path,
	// must decode exactly as the sequential one does.
	data := make([]byte, 0, 32+18)
	for i := 0; i < 8; i++ {
		var tmp [4]byte
		binary.LittleEndian.PutUint32(tmp[:], math.Float32bits(float32(i+1)))
		data = append(data, tmp[:]...)
	}
	data = append(data, q4_0Block(t, fill(32, 9), 1)...)

	path := writeModel(t, []gguf.TensorInfo{
		{Name: "a", Dims: []uint64{8}, Type: gguf.F32, Offset: 0},
		{Name: "b", Dims: []uint64{32}, Type: gguf.Q4_0, Offset: 32},
	}, data)

	l, err := OpenLoader(path)
	if err != nil {
		t.Fatalf("OpenLoader: %v", err)
	}
	defer closeLoader(t, l)

	got, err := l.Preload([]string{"a", "b"}, 2)
	if err != nil {
		t.Fatalf("Preload: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("Preload returned %d tensors, want 2", len(got))
	}
	for i := 0; i < 8; i++ {
		if want := float32(i + 1); got["a"][i] != want {
			t.Errorf("a[%d] = %v, want %v", i, got["a"][i], want)
		}
	}
	if got["b"][0] != 1 {
		t.Errorf("b[0] = %v, want 1", got["b"][0])
	}

	// An unknown name must fail the whole call rather than return a partial map
	// the caller might mistake for a complete model.
	if _, err := l.Preload([]string{"a", "missing"}, 2); err == nil {
		t.Error("Preload with a missing tensor succeeded")
	}
	// Zero workers is normalised to one, not treated as "no work".
	if got, err := l.Preload([]string{"a"}, 0); err != nil || len(got) != 1 {
		t.Errorf("Preload(workers=0) = %v, %v", got, err)
	}
	// An empty request is a valid no-op.
	if got, err := l.Preload(nil, 4); err != nil || len(got) != 0 {
		t.Errorf("Preload(nil) = %v, %v", got, err)
	}
}

func TestPreloadConcurrentReads(t *testing.T) {
	// Exercise the shared file descriptor with many tensors at once. ReadAt is
	// position-independent, so this must be race-free; the -race build of this
	// test is what proves the claim.
	const n = 16
	tensors := make([]gguf.TensorInfo, n)
	data := make([]byte, 0, n*32)
	for i := 0; i < n; i++ {
		tensors[i] = gguf.TensorInfo{
			Name:   fmt.Sprintf("t%d", i),
			Dims:   []uint64{8},
			Type:   gguf.F32,
			Offset: uint64(i * 32),
		}
		for j := 0; j < 8; j++ {
			var tmp [4]byte
			binary.LittleEndian.PutUint32(tmp[:], math.Float32bits(float32(i*100+j)))
			data = append(data, tmp[:]...)
		}
	}
	path := writeModel(t, tensors, data)

	l, err := OpenLoader(path)
	if err != nil {
		t.Fatalf("OpenLoader: %v", err)
	}
	defer closeLoader(t, l)

	names := make([]string, n)
	for i := range names {
		names[i] = fmt.Sprintf("t%d", i)
	}
	got, err := l.Preload(names, 4)
	if err != nil {
		t.Fatalf("Preload: %v", err)
	}
	for i := 0; i < n; i++ {
		name := fmt.Sprintf("t%d", i)
		for j := 0; j < 8; j++ {
			if want := float32(i*100 + j); got[name][j] != want {
				t.Errorf("%s[%d] = %v, want %v", name, j, got[name][j], want)
			}
		}
	}
}
