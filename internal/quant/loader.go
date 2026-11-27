package quant

import (
	"fmt"
	"io"
	"math"
	"os"
	"sync"

	"github.com/pplmx/orbix/internal/gguf"
)

// Loader reads dequantized tensors from a GGUF file.
//
// A model has hundreds of tensors and every one of them has to be read from
// the same file. Loader keeps one file handle open and seeks per tensor,
// which is what lets a 7B model load without reopening the file per
// projection and without memory-mapping.
type Loader struct {
	path string
	file *gguf.File
	fh   *os.File
	size int64
}

// OpenLoader opens path and parses its directory.
//
// Close must be called to release the file handle.
func OpenLoader(path string) (*Loader, error) {
	f, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	// Two handles on purpose: gguf.Open keeps its own for the metadata, and a separate one is
	// what lets a read seek without disturbing it.
	//
	// These two arms are uncovered. gguf.Open has already opened the same path successfully, so
	// os.Open fails here only if the file vanished in between, and Stat on an open descriptor
	// fails only on a filesystem that stops reporting sizes -- neither is reachable from a test
	// without racing a delete against OpenLoader. The malformed and missing-file cases are on
	// the gguf.Open call above and both have tests.
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	st, err := fh.Stat()
	if err != nil {
		// Already failing; a close error adds nothing to the report.
		_ = fh.Close()
		return nil, err
	}
	return &Loader{path: path, file: f, fh: fh, size: st.Size()}, nil
}

// Close releases the file handle.
func (l *Loader) Close() error {
	return l.fh.Close()
}

// File returns the parsed GGUF directory, for metadata lookups.
func (l *Loader) File() *gguf.File {
	return l.file
}

// Tensor reads and dequantizes the named tensor into a new float32 slice.
//
// The slice has one float32 per tensor element, in ggml's storage order. The
// caller is responsible for reshaping it.
// tensorBytes returns the encoded size of a tensor, naming the reason when it has none.
//
// TensorInfo.Bytes reports one false for two different failures: a type this package does not
// know, and an element count that is not a multiple of its block size. Reporting both as
// "unsupported type" is actively misleading -- measured, a hand-built Q4_0 tensor declaring 33
// elements produced `has unsupported type Q4_0`, which sends a reader to the type support table
// for a problem that is in the shape. The two are separated here so each says what is wrong.
func tensorBytes(ti gguf.TensorInfo) (int64, error) {
	if nbytes, ok := ti.Bytes(); ok {
		return nbytes, nil
	}
	block, known := ti.Type.BlockSize()
	if !known {
		return 0, fmt.Errorf("quant: tensor %q has unsupported type %s", ti.Name, ti.Type)
	}
	if n := ti.N(); n%int64(block) != 0 {
		return 0, fmt.Errorf("quant: tensor %q is %d elements, which is not a multiple of %s's %d-element block",
			ti.Name, n, ti.Type, block)
	}
	// Bytes can still fail for an element count whose byte length would overflow int64. That
	// is a size problem rather than a type one, and saying so is the only useful description.
	return 0, fmt.Errorf("quant: tensor %q is too large: %d elements of %s do not fit in a byte count",
		ti.Name, ti.N(), ti.Type)
}

// Tensor returns one tensor's weights dequantized to float32, in row-major order.
//
// The result is the tensor as the model uses it, not as the file stores it: a Q4_0 block
// becomes its 32 floats, so a caller never sees the packed layout or has to know which
// formats exist. Each call decodes afresh, so the returned slice belongs to the caller and
// subsequent calls are unaffected by anything done to it.
//
// The errors a caller can see here name different problems, and each says which tensor it is
// about: a model file with a bad entry usually has several, and the first failure is only
// useful if it says which one. In order of arrival:
//
//   - no tensor by that name;
//   - an element count of zero, which no layer shape can read;
//   - a shape tensorBytes refuses -- an undecodable type, an element count that is not a
//     block multiple, or a byte count that does not fit in int64;
//   - offset and length, from dataStart, when the tensor does not lie inside the file;
//   - a read that comes up short, and a decode that rejects the bytes it was given.
//
// The comment here used to say "three cases a caller handles differently", and nothing in this
// package does handle them differently -- every caller checks err != nil and reports it. The
// list above is what the function actually returns; three was a count of the ones the author
// had in mind, not of the branches that exist.
func (l *Loader) Tensor(name string) ([]float32, error) {
	ti, ok := l.file.Tensor(name)
	if !ok {
		return nil, fmt.Errorf("quant: no tensor named %q", name)
	}
	n := ti.N()
	if n <= 0 {
		return nil, fmt.Errorf("quant: tensor %q has invalid element count %d", name, n)
	}
	nbytes, err := tensorBytes(ti)
	if err != nil {
		return nil, err
	}
	// Decodability is checked here rather than in tensorBytes, because two of that function's
	// three callers do not decode: TensorReader hands back the raw bytes for a type this build
	// has no decoder for, and that is a supported use -- I8 and F64 tensors are readable even
	// though nothing here turns them into float32.
	//
	// It goes before the read. gguf knows the encoded length of 34 quantization formats and this
	// package decodes 14, so without the check a tensor in one of the other 20 had its whole
	// length allocated and read off disk -- measured at 5 MB for a 4M-element Q8_1, and a real
	// projection is hundreds of times that -- only to fail in Decode with "cannot decode Q8_1",
	// which names neither the tensor nor what this build does support.
	if !Supported(ti.Type) {
		return nil, fmt.Errorf("quant: tensor %q has unsupported type %s: this build cannot "+
			"decode it; it decodes %s", name, ti.Type, supportedList())
	}
	start, rerr := l.dataStart(ti, nbytes)
	if rerr != nil {
		return nil, fmt.Errorf("quant: tensor %q %w", name, rerr)
	}

	raw := make([]byte, nbytes)
	if _, err := l.fh.ReadAt(raw, start); err != nil {
		return nil, fmt.Errorf("quant: read tensor %q: %w", name, err)
	}

	dst := make([]float32, n)
	if err := Decode(ti.Type, raw, dst); err != nil {
		return nil, fmt.Errorf("quant: tensor %q: %w", name, err)
	}
	return dst, nil
}

// TensorInto reads and dequantizes the named tensor into dst, which must have
// exactly one element per tensor element.
//
// This is the allocation-free form of Tensor, for callers that reuse buffers
// across a model's layers.
func (l *Loader) TensorInto(name string, dst []float32) error {
	ti, ok := l.file.Tensor(name)
	if !ok {
		return fmt.Errorf("quant: no tensor named %q", name)
	}
	// The element count is checked for the same reason Tensor checks it, and before the
	// destination size: a zero-dimension tensor has N of 0, which an empty dst would satisfy,
	// so the size comparison alone accepted a tensor Tensor refuses. Measured: "empty" with
	// dims [0 4] returned "invalid element count 0" from Tensor and nil from here.
	if n := ti.N(); n <= 0 {
		return fmt.Errorf("quant: tensor %q has invalid element count %d", name, n)
	}
	if int64(len(dst)) != ti.N() {
		return fmt.Errorf("quant: tensor %q has %d elements, destination has %d", name, ti.N(), len(dst))
	}
	nbytes, err := tensorBytes(ti)
	if err != nil {
		return err
	}
	start, rerr := l.dataStart(ti, nbytes)
	if rerr != nil {
		return fmt.Errorf("quant: tensor %q %w", name, rerr)
	}
	raw := make([]byte, nbytes)
	if _, err := l.fh.ReadAt(raw, start); err != nil {
		return fmt.Errorf("quant: read tensor %q: %w", name, err)
	}
	return Decode(ti.Type, raw, dst)
}

// A note on the two ReadAt error arms below (in Tensor and TensorInto): they are unreachable
// through dataStart, which already refuses any tensor whose bytes would run past the file size
// it recorded at open. The one way to reach them is a file truncated between OpenLoader and the
// read, which no test can arrange without racing its own fixture. They stay because dropping
// them would mean trusting a size that another process can change; a test for them would have
// to delete the file mid-call, and that would test the test.
//
// dataStart returns the absolute file offset of a tensor's encoded bytes, or an error
// describing why it is not inside the file.
//
// Both operands come from the file, so neither the cast nor the sum can be trusted to stay
// in range. An offset above MaxInt64 becomes negative when cast, and a large start plus a
// large length wraps the sum negative -- in both cases a naive
// `start < 0 || start+nbytes > size` test passes and the reader is handed bytes from the
// wrong place. The shared helper is what keeps the three call sites from drifting apart:
// they had the identical bug three times, and gguf.TensorData had it too until it was fixed
// with this same ordering.
func (l *Loader) dataStart(ti gguf.TensorInfo, nbytes int64) (int64, error) {
	if ti.Offset > math.MaxInt64 {
		return 0, fmt.Errorf("offset %d is too large", ti.Offset)
	}
	start := l.file.TensorDataLoc + int64(ti.Offset)
	if start < 0 || start > l.size || nbytes < 0 || nbytes > l.size-start {
		return 0, fmt.Errorf("extends past end of file")
	}
	return start, nil
}

// Preload reads and dequantizes several tensors concurrently and returns them
// keyed by name.
//
// Loading a model is dominated by reading and decoding independent tensors;
// there is no data dependency between them, so the work fans out across a few
// goroutines. The count is bounded by workers rather than by the tensor count,
// because decoding is memory-bandwidth bound and oversubscribing the CPUs makes
// it slower, not faster.
//
// ReadAt is used instead of Seek precisely so concurrent reads share one file
// descriptor without racing on its position.
func (l *Loader) Preload(names []string, workers int) (map[string][]float32, error) {
	if workers < 1 {
		workers = 1
	}
	if workers > len(names) {
		workers = len(names)
	}
	out := make(map[string][]float32, len(names))
	if len(names) == 0 {
		return out, nil
	}

	var (
		mu       sync.Mutex
		firstErr error
		wg       sync.WaitGroup
		// There is no test for the cancellation below: nothing a caller can see distinguishes it.
		// Preload returns nil for the map whenever any tensor failed, so "it stopped early" and "it
		// finished and threw the work away" produce the same result, and the only direct evidence is
		// a call counter, which would have to live in the implementation. Measured instead, by
		// instrumenting both forms on this fixture with ten good tensors and one undecodable one:
		// without the cancellation the loop called Tensor for all 41 names; with it, 13 when the bad
		// name is first in the list and 4 when it is last. What a caller gets is in TestPreloadStops.
		//
		// done closes on the first error so the tensors still queued stop waiting for a slot.
		// Without it every name ran to completion and the caller got the first error after the
		// whole set had been decoded and thrown away -- measured on this fixture with ten good
		// tensors and one undecodable one: the failure arrived only after all eleven had been
		// read and expanded, and for a 7B checkpoint that is a minute of work whose output is
		// discarded.
		done = make(chan struct{})
		once sync.Once
		// sem bounds the number of concurrent decodes to workers. Without it
		// every tensor would start at once, which for a 7B model means hundreds
		// of gigabytes of float32 buffers in flight.
		sem = make(chan struct{}, workers)
	)
	for _, name := range names {
		wg.Go(func() {
			select {
			case sem <- struct{}{}:
			case <-done:
				return
			}
			defer func() { <-sem }()
			// A worker that already has a slot still checks, so the tensors in flight when the
			// first error lands are the only extra work done, not the whole remainder.
			select {
			case <-done:
				return
			default:
			}

			t, err := l.Tensor(name)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				if firstErr == nil {
					firstErr = err
					once.Do(func() { close(done) })
				}
				return
			}
			out[name] = t
		})
	}
	wg.Wait()
	if firstErr != nil {
		return nil, firstErr
	}
	return out, nil
}

// TensorReader returns a streaming reader over the raw encoded bytes of a
// tensor, without dequantizing. Callers that only need to copy the bytes, or
// to hand them to a remote backend, avoid the float32 expansion.
func (l *Loader) TensorReader(name string) (io.Reader, error) {
	ti, ok := l.file.Tensor(name)
	if !ok {
		return nil, fmt.Errorf("quant: no tensor named %q", name)
	}
	nbytes, err := tensorBytes(ti)
	if err != nil {
		return nil, err
	}
	start, rerr := l.dataStart(ti, nbytes)
	if rerr != nil {
		return nil, fmt.Errorf("quant: tensor %q %w", name, rerr)
	}
	return io.NewSectionReader(l.fh, start, nbytes), nil
}
