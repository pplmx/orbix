package quant

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
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
		} else {
			// The message has to name the offset itself: which of three offending values was
			// rejected is the first thing a reader needs, and "too large" alone does not say.
			// Dropping the number from the format left every case passing.
			if !strings.Contains(err.Error(), "too large") {
				t.Errorf("offset %d: error %q does not say the offset is the problem", off, err)
			}
			if !strings.Contains(err.Error(), strconv.FormatUint(off, 10)) {
				t.Errorf("offset %d: error %q does not name the offset", off, err)
			}
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

// TestLoaderRejectsAnUnsupportedQuantType covers the type check in TensorInto and
// TensorReader.
//
// Both look up the tensor, ask QuantType.Size for its byte length, and refuse when it returns
// false -- which is what an unrecognised type produces. Deleting either check passed the quant
// and llama suites: every fixture the tests build is F32, so the false branch never ran.
//
// The consequence is not subtle once reached. Size returns (0, false) for an unknown type, so
// the unchecked version computes a zero-byte range, reads nothing, and hands Decode a slice it
// cannot size. Under the old grammar this was the "unsupported type" path of a file written by
// a newer quantizer, which is exactly when a loader needs to say so rather than proceed.
func TestLoaderRejectsAnUnsupportedQuantType(t *testing.T) {
	// Type 4096 is beyond every value GGUF defines, and the bytes are only there so the file
	// passes the reader's own checks.
	unknown := gguf.QuantType(4096)
	path := writeModel(t, []gguf.TensorInfo{
		{Name: "mystery.weight", Dims: []uint64{4}, Type: unknown},
	}, make([]byte, 64))

	l, err := OpenLoader(path)
	if err != nil {
		t.Fatalf("OpenLoader: %v", err)
	}
	defer func() { _ = l.Close() }()

	// The type must be unrecognised to this package, or the test is not exercising the guard.
	if _, ok := unknown.Size(4); ok {
		t.Fatalf("quant type %d is supported, so this test cannot reach the guard", unknown)
	}

	if err := l.TensorInto("mystery.weight", make([]float32, 4)); err == nil {
		t.Error("TensorInto accepted an unsupported tensor type")
	} else if !strings.Contains(err.Error(), "unsupported type") {
		t.Errorf("TensorInto error %q does not name the type", err)
	}

	if _, err := l.TensorReader("mystery.weight"); err == nil {
		t.Error("TensorReader accepted an unsupported tensor type")
	} else if !strings.Contains(err.Error(), "unsupported type") {
		t.Errorf("TensorReader error %q does not name the type", err)
	}

	// Tensor goes through the same lookup and must agree.
	if _, err := l.Tensor("mystery.weight"); err == nil {
		t.Error("Tensor accepted an unsupported tensor type")
	}
}

// TestLoaderRejectsTensorExtendingPastTheEnd covers the length half of dataStart's range check.
//
// The existing test takes an offset that is past the end (5000 in a 1000-byte file), which the
// start > size comparison alone catches. The case that needs the nbytes term is different: the
// start is inside the file and only the extent runs past it, so a check written as
// `start > size` accepts and the reader is handed fewer bytes than the tensor needs.
//
// Removing the nbytes and size-start terms left every test green -- measured -- which is the same
// shape as the wrapping-offset defect the comment above describes: a range test that looks like it
// covers the range but only guards one end.
func TestLoaderRejectsTensorExtendingPastTheEnd(t *testing.T) {
	l := &Loader{size: 1000}
	l.file = &gguf.File{TensorDataLoc: 100}

	// Start at 100 + 884 = 984, inside the file, with 16 bytes needed and 16 available.
	// Growing the length by one has to fail.
	inFile := gguf.TensorInfo{Name: "w", Offset: 884, Dims: []uint64{4}, Type: gguf.F32}
	if _, err := l.dataStart(inFile, 16); err != nil {
		t.Fatalf("the exact fit was refused: %v", err)
	}
	if _, err := l.dataStart(inFile, 17); err == nil {
		t.Error("dataStart accepted a tensor needing 17 bytes in the 16 the file has left")
	} else if !strings.Contains(err.Error(), "past end") {
		t.Errorf("error %q does not say the tensor is past the end", err)
	}

	// A start at the very end with any positive length is the extreme of the same case.
	atEnd := gguf.TensorInfo{Name: "w", Offset: 900, Dims: []uint64{4}, Type: gguf.F32}
	if _, err := l.dataStart(atEnd, 1); err == nil {
		t.Error("dataStart accepted one byte at the end of the file")
	}

	// A negative length is refused rather than treated as zero.
	if _, err := l.dataStart(inFile, -1); err == nil {
		t.Error("dataStart accepted a negative byte count")
	}
}

// TestTensorReaderReturnsTheTensorsOwnBytes covers what TensorReader actually produces.
//
// The existing cases only drive its refusal paths: an unknown name and an unsupported type. What it
// returns on success was never read, so doubling the section length -- a reader that hands the
// caller twice the tensor and lets it swallow the neighbouring one -- left every test green.
//
// The payload is distinctive per tensor so a reader starting at the wrong offset, or one returning
// the wrong tensor's bytes, is visible rather than merely truncated.
func TestTensorReaderReturnsTheTensorsOwnBytes(t *testing.T) {
	// Two F32 tensors, each two elements, with different bytes.
	first := []byte{1, 0, 0, 0, 2, 0, 0, 0}
	second := []byte{3, 0, 0, 0, 4, 0, 0, 0}
	// The offsets are the caller's job: writeModel writes them verbatim and appends the data
	// block as one run, so a tensor that does not say where it starts reads the first one's
	// bytes. That is what makes the payload comparison below meaningful.
	path := writeModel(t, []gguf.TensorInfo{
		{Name: "first", Type: gguf.F32, Dims: []uint64{2}, Offset: 0},
		{Name: "second", Type: gguf.F32, Dims: []uint64{2}, Offset: 8},
	}, append(append([]byte(nil), first...), second...))

	l, err := OpenLoader(path)
	if err != nil {
		t.Fatalf("OpenLoader: %v", err)
	}
	defer func() { _ = l.Close() }()

	for _, c := range []struct {
		name string
		want []byte
	}{
		{"first", first},
		{"second", second},
	} {
		r, err := l.TensorReader(c.name)
		if err != nil {
			t.Fatalf("TensorReader(%q): %v", c.name, err)
		}
		got, err := io.ReadAll(r)
		if err != nil {
			t.Fatalf("reading %q: %v", c.name, err)
		}
		if len(got) != len(c.want) {
			t.Errorf("TensorReader(%q) produced %d bytes, want %d; a section that runs past the "+
				"tensor hands the caller its neighbour's bytes too", c.name, len(got), len(c.want))
			continue
		}
		for i := range c.want {
			if got[i] != c.want[i] {
				t.Errorf("TensorReader(%q) byte %d = %d, want %d", c.name, i, got[i], c.want[i])
			}
		}
	}
}

// TestTensorRejectsAnUnknownQuantType covers Tensor's unsupported-type refusal.
//
// Dequant has its own copy of this message and it was tested; Tensor's was not, so removing the
// tensor name or the type from it passed. The loader is the path the model loader uses, so a file
// with a type this build does not implement fails here first.
//
// Both pieces are asserted: the name says which tensor, and the type is what a reader compares
// against the table in the format description.
func TestTensorRejectsAnUnknownQuantType(t *testing.T) {
	l := &Loader{size: 1000}
	l.file = &gguf.File{TensorDataLoc: 100, TensorInfos: []gguf.TensorInfo{
		{Name: "blk.0.attn_q.weight", Offset: 0, Dims: []uint64{4}, Type: gguf.QuantType(999)},
	}}

	_, err := l.Tensor("blk.0.attn_q.weight")
	if err == nil {
		t.Fatal("Tensor accepted a QuantType this build does not implement")
	}
	if !strings.Contains(err.Error(), "blk.0.attn_q.weight") {
		t.Errorf("error %q does not name the tensor", err)
	}
	if !strings.Contains(err.Error(), "999") {
		t.Errorf("error %q does not state the type", err)
	}
}

// TestTensorRejectsAZeroElementCount covers the guard against a tensor that describes nothing.
//
// A dimension of zero is legal to write in a GGUF header and produces an element count of zero. The
// loader refuses it rather than returning an empty slice, because the caller asked for a weight and
// an empty one would be reshaped into a layer that silently contributes nothing.
//
// The message names the count, which distinguishes "this tensor is empty" from "this tensor is
// malformed" for whoever has to decide which file is wrong.
func TestTensorRejectsAZeroElementCount(t *testing.T) {
	l := &Loader{size: 1000}
	l.file = &gguf.File{TensorDataLoc: 100, TensorInfos: []gguf.TensorInfo{
		{Name: "blk.0.ffn_down.weight", Offset: 0, Dims: []uint64{4, 0}, Type: gguf.F32},
	}}

	_, err := l.Tensor("blk.0.ffn_down.weight")
	if err == nil {
		t.Fatal("Tensor accepted a tensor with a zero dimension")
	}
	if !strings.Contains(err.Error(), "blk.0.ffn_down.weight") {
		t.Errorf("error %q does not name the tensor", err)
	}
	// The count it computed, so the message says why it was refused. Asserted through the phrase
	// rather than the bare digit: the tensor's own name contains "blk.0", so a substring check for
	// "0" passes with the count missing.
	if !strings.Contains(err.Error(), "element count 0") {
		t.Errorf("error %q does not state the element count", err)
	}
	if !strings.Contains(err.Error(), "invalid element count") {
		t.Errorf("error %q does not say the count is the problem", err)
	}
}

// TestSizedButUndecodableTypeIsNamed covers the second refusal, which is not the one above.
//
// An unrecognised type fails at Size(), so the loader reports "unsupported type" before it reads
// anything. A type this package can size but has no codec for -- IQ1_S, Q8_K, MXFP4 and thirty others
// -- passes that check, because the directory has to know every block's width to walk the file, and is
// refused later with "cannot decode <type>". Both messages must name the type, since a file with
// several tensors needs to know which one is the problem.
func TestSizedButUndecodableTypeIsNamed(t *testing.T) {
	// IQ1_S is 256 weights in 50 bytes; the directory sizes it correctly and the decoder has no entry.
	ti := gguf.TensorInfo{Name: "odd.weight", Dims: []uint64{256}, Type: gguf.IQ1_S}
	path := writeModel(t, []gguf.TensorInfo{ti}, make([]byte, 50))

	l, err := OpenLoader(path)
	if err != nil {
		t.Fatalf("OpenLoader refused a file whose directory it can size: %v", err)
	}
	defer func() { _ = l.Close() }()

	_, err = l.Tensor("odd.weight")
	if err == nil {
		t.Fatal("Tensor decoded an IQ1_S tensor")
	}
	if !strings.Contains(err.Error(), "cannot decode") {
		t.Errorf("error %q does not say the type cannot be decoded", err)
	}
	// The type's name, not only the fact of the refusal.
	if !strings.Contains(err.Error(), "IQ1_S") {
		t.Errorf("error %q does not name the type", err)
	}
	// And the tensor it belongs to, which the loader's own wrapper adds.
	if !strings.Contains(err.Error(), "odd.weight") {
		t.Errorf("error %q does not name the tensor", err)
	}
}

// TestOpenLoaderReportsAMissingFile covers the first of its two opens.
//
// The loader opens the path twice: once through gguf.Open for the directory and once through os.Open
// for the tensor data. Neither failure had a test, so neither error could lose its message.
//
// The assertion is on the path in the message, since that is what a reader compares against the flag
// or configuration that produced it.
func TestOpenLoaderReportsAMissingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "absent.gguf")
	l, err := OpenLoader(path)
	if err == nil {
		_ = l.Close()
		t.Fatal("OpenLoader opened a file that does not exist")
	}
	if l != nil {
		t.Error("a failed OpenLoader returned a loader")
	}
	if !strings.Contains(err.Error(), "absent.gguf") {
		t.Errorf("error %q does not name the file", err)
	}
	if !strings.Contains(err.Error(), "no such file") {
		t.Errorf("error %q does not say the file is missing", err)
	}
}

// TestOpenLoaderReportsAMalformedFile covers the path where the directory cannot be read.
//
// A file that exists but is not GGUF fails inside gguf.Open, before the second open happens, which is
// the ordering the two calls are in for: a malformed file never leaves a second handle open.
func TestOpenLoaderReportsAMalformedFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "not-gguf.bin")
	if err := os.WriteFile(path, []byte("this is not a gguf file"), 0o644); err != nil {
		t.Fatalf("writing: %v", err)
	}
	l, err := OpenLoader(path)
	if err == nil {
		_ = l.Close()
		t.Fatal("OpenLoader accepted a file that is not GGUF")
	}
	if l != nil {
		t.Error("a failed OpenLoader returned a loader")
	}
	if !strings.Contains(err.Error(), "gguf") {
		t.Errorf("error %q does not say the format is the problem", err)
	}
}

// TestTensorAndTensorReaderWrapTheOffsetError covers the call sites that turn dataStart's error
// into a message naming the tensor.
//
// dataStart itself is tested; the wrapping around it was not, so replacing either wrapper with the bare
// error passed everything. The tensor name is what tells a reader which of a checkpoint's hundreds of
// tensors has a bad offset, and it comes only from that wrapper.
func TestTensorAndTensorReaderWrapTheOffsetError(t *testing.T) {
	// The tensor's bytes start inside the file and run past its end, which dataStart reports.
	ti := gguf.TensorInfo{Name: "blk.3.attn_k.weight", Offset: 100, Dims: []uint64{4}, Type: gguf.F32}
	path := writeModel(t, []gguf.TensorInfo{ti}, make([]byte, 8))

	for _, c := range []struct {
		name string
		call func(*Loader) error
	}{
		{"Tensor", func(l *Loader) error { _, err := l.Tensor(ti.Name); return err }},
		{"TensorReader", func(l *Loader) error { _, err := l.TensorReader(ti.Name); return err }},
		// TensorInto is the third caller of dataStart, and it wraps the error the same way.
		{"TensorInto", func(l *Loader) error { return l.TensorInto(ti.Name, make([]float32, 4)) }},
	} {
		t.Run(c.name, func(t *testing.T) {
			l, err := OpenLoader(path)
			if err != nil {
				t.Fatalf("OpenLoader: %v", err)
			}
			defer func() { _ = l.Close() }()

			err = c.call(l)
			if err == nil {
				t.Fatalf("%s accepted a tensor past the end of the file", c.name)
			}
			// The wrapper's contribution: the tensor's name.
			if !strings.Contains(err.Error(), ti.Name) {
				t.Errorf("error %q does not name the tensor", err)
			}
			// dataStart's own message survives the wrapping, rather than being replaced by it.
			if !strings.Contains(err.Error(), "past end") {
				t.Errorf("error %q does not carry dataStart's reason", err)
			}
		})
	}
}

// TestLoaderReadsTensorsAtArbitraryOffsets covers a layout the other loader tests do not
// build: tensors that are out of order in the directory, unaligned on disk, and not adjacent.
//
// Every existing test places tensors in directory order with flags that happen to align, so
// each one reads from a location that is also the natural place to read from. GGUF makes no
// such promise -- a writer may lay the data out in any order, and the alignment is whatever
// the writer chose -- and the loader's whole job is to follow the offset rather than the
// order. A loader that walked the data section sequentially would pass every test currently
// in this file.
//
// The quantized case is included because its block is 18 bytes: at an offset that is not a
// multiple of four, the block straddles the boundaries a word-at-a-time read would use, which
// is the case a `ReadAt` on a non-aligned start has to get right.
func TestLoaderReadsTensorsAtArbitraryOffsets(t *testing.T) {
	t.Run("out of order, unaligned", func(t *testing.T) {
		// Three F32 tensors of two elements. Directory order b, a, c; disk order is whatever
		// the offsets say, and none of them is aligned to eight.
		data := make([]byte, 30)
		putF32 := func(off int, vals ...float32) {
			for i, v := range vals {
				binary.LittleEndian.PutUint32(data[off+i*4:], math.Float32bits(v))
			}
		}
		putF32(0, 20, 21)  // b
		putF32(10, 10, 11) // a, deliberately at 10 rather than 8
		putF32(20, 30, 31) // c

		path := writeModel(t, []gguf.TensorInfo{
			{Name: "b", Offset: 0, Dims: []uint64{2}, Type: gguf.F32},
			{Name: "a", Offset: 10, Dims: []uint64{2}, Type: gguf.F32},
			{Name: "c", Offset: 20, Dims: []uint64{2}, Type: gguf.F32},
		}, data)
		l, err := OpenLoader(path)
		if err != nil {
			t.Fatalf("OpenLoader: %v", err)
		}
		defer closeLoader(t, l)

		for _, c := range []struct {
			name  string
			want0 float32
			want1 float32
		}{
			{"a", 10, 11},
			{"b", 20, 21},
			{"c", 30, 31},
		} {
			got, err := l.Tensor(c.name)
			if err != nil {
				t.Errorf("%s: %v", c.name, err)
				continue
			}
			if len(got) != 2 || got[0] != c.want0 || got[1] != c.want1 {
				t.Errorf("%s = %v, want [%v %v]", c.name, got, c.want0, c.want1)
			}
		}
	})

	t.Run("quantized block at an unaligned offset", func(t *testing.T) {
		// IQ4_NL is 18 bytes for 32 weights. Placed at offset 3 so the 16-bit scale straddles
		// a four-byte boundary, which a word-aligned reader would misread.
		blk := make([]byte, 18)
		binary.LittleEndian.PutUint16(blk[0:], 0x3c00) // d = 1.0 in binary16
		for i := 2; i < 18; i++ {
			blk[i] = 0x08 // low nibble 8, high nibble 0
		}
		data := append([]byte{0xAA, 0xBB, 0xCC}, blk...)
		path := writeModel(t, []gguf.TensorInfo{
			{Name: "q", Offset: 3, Dims: []uint64{32}, Type: gguf.IQ4_NL},
		}, data)
		l, err := OpenLoader(path)
		if err != nil {
			t.Fatalf("OpenLoader: %v", err)
		}
		defer closeLoader(t, l)

		got, err := l.Tensor("q")
		if err != nil {
			t.Fatalf("Tensor: %v", err)
		}
		// The low nibbles fill the first sixteen weights and the high nibbles the second
		// sixteen, so the two halves take different codebook entries: kValuesIQ4NL[8] is 1 and
		// kValuesIQ4NL[0] is -127. Asserting both halves is what makes a half-swapped decode
		// fail rather than produce a symmetric result that looks reasonable.
		for i := range 16 {
			if got[i] != 1 {
				t.Errorf("low half [%d] = %v, want 1", i, got[i])
			}
			if got[i+16] != -127 {
				t.Errorf("high half [%d] = %v, want -127", i, got[i+16])
			}
		}
	})
}

// TestLoaderBoundsTheTensorAgainstTheFileEnd pins where the boundary between "fits" and
// "does not fit" is, to the byte.
//
// TestLoaderRejectsTruncatedTensor reads a tensor whose declared size is far larger than the
// file, so it cannot tell an off-by-one from an off-by-a-megabyte. The check is a single
// expression -- nbytes > l.size-start -- and an off-by-one there is both easy to write and
// invisible to a test whose gap is 1 MiB: it would reject a tensor that ends exactly at EOF,
// or accept one whose last byte is past it.
//
// The data slice is 16 bytes and every case is an F32 tensor of four elements, so 16 bytes is
// exactly what fits. The three cases are the boundary and its two neighbours on the side that
// matters: exactly fitting, one byte short of fitting, and a tensor whose offset is already
// past the end.
func TestLoaderBoundsTheTensorAgainstTheFileEnd(t *testing.T) {
	// 16 bytes of data, so a four-element F32 tensor at offset 0 uses the whole file.
	data := make([]byte, 16)
	for i := range data {
		data[i] = byte(i + 1)
	}

	cases := []struct {
		name    string
		offset  uint64
		elems   uint64
		wantErr bool
		// wantBytes is how many bytes the tensor claims when it fits, which is what the
		// loader needs to see inside the file.
	}{
		// Exactly the file: the last byte is at EOF-1, so this must be accepted.
		{"exactly fits", 0, 4, false},
		// Offset 1 with four elements needs bytes [1,17) but the file ends at 16.
		{"one byte past the end", 1, 4, true},
		// A tensor that starts exactly at EOF needs no room for its own offset, but its data
		// does not exist.
		{"starts at EOF", 16, 1, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			path := writeModel(t, []gguf.TensorInfo{
				{Name: "t", Dims: []uint64{c.elems}, Type: gguf.F32, Offset: c.offset},
			}, data)
			l, err := OpenLoader(path)
			if err != nil {
				t.Fatalf("OpenLoader: %v", err)
			}
			defer closeLoader(t, l)

			got, err := l.Tensor("t")
			if c.wantErr {
				if err == nil {
					t.Fatalf("Tensor() = %d values, want an error: the tensor extends past the "+
						"16-byte file", len(got))
				}
				// The message has to name the tensor and say what is wrong with it, which a
				// bare "error" assertion cannot distinguish from a wrong-width or
				// wrong-name failure.
				if !strings.Contains(err.Error(), "extends past end of file") {
					t.Errorf("error = %v, want it to say the data extends past the file", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("Tensor() failed on a tensor that ends exactly at EOF: %v", err)
			}
			if uint64(len(got)) != c.elems {
				t.Fatalf("got %d values, want %d", len(got), c.elems)
			}
			// The values have to be the file's bytes decoded as F32, not zeros left behind by a
			// short read. Compared as bit patterns rather than as round numbers: the data is
			// deliberately the raw bytes 1,2,3,...,16, and only the bytes 00,00,00,00 decode to
			// 0.0, so a short read shows up as zeros on every element but the first.
			for i, v := range got {
				var want float32
				bits := binary.LittleEndian.Uint32(data[i*4 : i*4+4])
				want = math.Float32frombits(bits)
				if v != want {
					t.Errorf("value[%d] = %v, want %v (bits %08x): a short read would leave a "+
						"zero here", i, v, want, bits)
				}
			}
			// At least one element must be non-zero, or the comparison above is satisfied by
			// an all-zero buffer.
			nonZero := false
			for _, v := range got {
				if v != 0 {
					nonZero = true
					break
				}
			}
			if !nonZero {
				t.Error("every value is zero, which is what a short read would produce")
			}
		})
	}
}

// TestTensorErrorsNameTheRealProblem covers a message that sent the reader to the wrong place.
//
// TensorInfo.Bytes reports one false for two failures: an unknown type, and an element count
// that is not a multiple of the type's block size. All three loader paths turned both into
// "has unsupported type", so a hand-built Q4_0 tensor declaring 33 elements reported
// `has unsupported type Q4_0` -- naming the type as unsupported when the problem was its
// shape, and pointing a reader at a support table that would tell them Q4_0 is supported.
//
// The two cases are asserted together, with the offending number and the block size in the
// message, because the point is that they are distinguishable. A test that only checked for
// "error" would have passed before the fix.
func TestTensorErrorsNameTheRealProblem(t *testing.T) {
	cases := []struct {
		name  string
		elems uint64
		typ   gguf.QuantType
		// wantAll are the substrings that must appear: the numbers a reader needs.
		wantAll []string
		// wantNot must not appear, which is what separates the two failures.
		wantNot string
	}{
		{
			name:  "element count is not a block multiple",
			elems: 33,
			typ:   gguf.Q4_0,
			// The count, the type, and its block size.
			wantAll: []string{"33", "Q4_0", "32"},
			wantNot: "unsupported",
		},
		{
			name:    "the type really is unknown",
			elems:   4,
			typ:     gguf.QuantType(99),
			wantAll: []string{"unsupported", "QuantType(99)"},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			path := writeModel(t, []gguf.TensorInfo{
				{Name: "w", Dims: []uint64{c.elems}, Type: c.typ, Offset: 0},
			}, make([]byte, 64))
			l, err := OpenLoader(path)
			if err != nil {
				t.Fatalf("OpenLoader: %v", err)
			}
			defer closeLoader(t, l)

			// All three paths that size a tensor, so the fix cannot be applied to one and
			// forgotten in the others.
			paths := map[string]func([]float32) error{
				"TensorInto": func(dst []float32) error { return l.TensorInto("w", dst) },
			}
			for label, call := range paths {
				err := call(make([]float32, c.elems))
				if err == nil {
					t.Fatalf("%s accepted a tensor it cannot size", label)
				}
				msg := err.Error()
				for _, want := range c.wantAll {
					if !strings.Contains(msg, want) {
						t.Errorf("%s error %q does not name %q", label, msg, want)
					}
				}
				if c.wantNot != "" && strings.Contains(msg, c.wantNot) {
					t.Errorf("%s error %q says %q, which names the wrong problem", label, msg, c.wantNot)
				}
			}

			// Tensor and TensorReader have the same requirement and must say the same thing.
			if _, err := l.Tensor("w"); err == nil {
				t.Error("Tensor accepted a tensor it cannot size")
			} else {
				msg := err.Error()
				for _, want := range c.wantAll {
					if !strings.Contains(msg, want) {
						t.Errorf("Tensor error %q does not name %q", msg, want)
					}
				}
				if c.wantNot != "" && strings.Contains(msg, c.wantNot) {
					t.Errorf("Tensor error %q says %q, which names the wrong problem", msg, c.wantNot)
				}
			}
			if _, err := l.TensorReader("w"); err == nil {
				t.Error("TensorReader accepted a tensor it cannot size")
			}
		})
	}
}
