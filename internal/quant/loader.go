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
func (l *Loader) Tensor(name string) ([]float32, error) {
	ti, ok := l.file.Tensor(name)
	if !ok {
		return nil, fmt.Errorf("quant: no tensor named %q", name)
	}
	n := ti.N()
	if n <= 0 {
		return nil, fmt.Errorf("quant: tensor %q has invalid element count %d", name, n)
	}
	nbytes, ok := ti.Bytes()
	if !ok {
		return nil, fmt.Errorf("quant: tensor %q has unsupported type %s", name, ti.Type)
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
	if int64(len(dst)) != ti.N() {
		return fmt.Errorf("quant: tensor %q has %d elements, destination has %d", name, ti.N(), len(dst))
	}
	nbytes, ok := ti.Bytes()
	if !ok {
		return fmt.Errorf("quant: tensor %q has unsupported type %s", name, ti.Type)
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
		// sem bounds the number of concurrent decodes to workers. Without it
		// every tensor would start at once, which for a 7B model means hundreds
		// of gigabytes of float32 buffers in flight.
		sem = make(chan struct{}, workers)
	)
	for _, name := range names {
		wg.Go(func() {
			sem <- struct{}{}
			defer func() { <-sem }()

			t, err := l.Tensor(name)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				if firstErr == nil {
					firstErr = err
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
	nbytes, ok := ti.Bytes()
	if !ok {
		return nil, fmt.Errorf("quant: tensor %q has unsupported type %s", name, ti.Type)
	}
	start, rerr := l.dataStart(ti, nbytes)
	if rerr != nil {
		return nil, fmt.Errorf("quant: tensor %q %w", name, rerr)
	}
	return io.NewSectionReader(l.fh, start, nbytes), nil
}
