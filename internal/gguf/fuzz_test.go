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
		// The parser must return, with or without an error, and must not panic. Nothing is
		// asserted about the result: a hostile file that parses to nonsense is a wrong
		// answer, not a crash, and separating those two is the point of the target.
		_, _ = Read(bytes.NewReader(data))
	})
}
