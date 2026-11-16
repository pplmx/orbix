package quant

import (
	"fmt"
	"io"
	"os"

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
	start := l.file.TensorDataLoc + int64(ti.Offset)
	if start < 0 || start+nbytes > l.size {
		return nil, fmt.Errorf("quant: tensor %q extends past end of file", name)
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
	start := l.file.TensorDataLoc + int64(ti.Offset)
	if start < 0 || start+nbytes > l.size {
		return fmt.Errorf("quant: tensor %q extends past end of file", name)
	}
	raw := make([]byte, nbytes)
	if _, err := l.fh.ReadAt(raw, start); err != nil {
		return fmt.Errorf("quant: read tensor %q: %w", name, err)
	}
	return Decode(ti.Type, raw, dst)
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
	start := l.file.TensorDataLoc + int64(ti.Offset)
	if start < 0 || start+nbytes > l.size {
		return nil, fmt.Errorf("quant: tensor %q extends past end of file", name)
	}
	return io.NewSectionReader(l.fh, start, nbytes), nil
}
