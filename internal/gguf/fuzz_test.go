package gguf

import (
	"bytes"
	"testing"
)

// FuzzRead checks that no input makes the parser panic.
//
// Read takes a file whose bytes come from wherever the user got it -- a download, a conversion
// script, a shared disk. A panic there does not produce a bad answer, it takes the process
// down, and the server loads files on request. The parser already returns contextual errors
// for everything it has been pointed at, so the property to hold is that it never does anything
// else.
//
// The seeds are the shapes that matter: a valid file, a truncated one, one whose counts are
// large enough to be hostile, and the byte patterns that trip length-prefixed readers.
func FuzzRead(f *testing.F) {
	// A real fixture: the smallest complete model the test package can build.
	f.Add(sampleModel())
	f.Add(sampleModel()[:20])
	f.Add([]byte("GGUF"))
	f.Add([]byte{})
	f.Add([]byte("GGUF\x03\x00\x00\x00"))
	// Declared counts far larger than the data behind them.
	f.Add(append([]byte("GGUF\x03\x00\x00\x00"),
		0, 0, 0, 0, 0, 0, 0, 0, // tensor count 0
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // kv count huge
	))
	// A string length that wraps when 1 is added to it.
	f.Add(append([]byte("GGUF\x03\x00\x00\x00"),
		0, 0, 0, 0, 0, 0, 0, 0,
		1, 0, 0, 0, 0, 0, 0, 0,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	))

	f.Fuzz(func(t *testing.T, data []byte) {
		// The parser must return, with or without an error, and must not panic.
		f2, err := Read(bytes.NewReader(data))
		if err != nil || f2 == nil {
			return
		}
		// A file that parses is still untrusted. Anything it derives from a dimension must be
		// sane, because a wrapped count is the shape of bug that does not panic: it produces a
		// negative length, which a bounds check written as "start + n > size" accepts. The
		// counts are only asserted to be non-negative here -- deciding they are plausible needs
		// the file size, which is the caller's job.
		for ti := range f2.Tensors() {
			if n := ti.N(); n < 0 {
				t.Fatalf("tensor %q: N() = %d from dims %v", ti.Name, n, ti.Dims)
			}
			if b, ok := ti.Bytes(); ok && b < 0 {
				t.Fatalf("tensor %q: Bytes() = %d from dims %v", ti.Name, b, ti.Dims)
			}
		}
	})
}
