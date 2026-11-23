package quant_test

import (
	"math"
	"os"
	"testing"

	"github.com/pplmx/orbix/internal/quant"
)

// TestTensorDecodesTheSameTwice pins the property a lazy decoder depends on.
//
// Today every tensor is decoded once, up front, and the model keeps the float32 values. Decoding
// per matmul instead -- the optimization LoadWeights names as the one to make later -- reads the
// same tensor many times over a run, and it is only a safe refactor if decoding is a pure function
// of the file's bytes. If a second call could return different values, the forward pass would
// depend on how many times a weight had been touched.
//
// The check is bit equality rather than a tolerance: the same bytes through the same arithmetic
// have no reason to differ, and allowing a tolerance here would permit exactly the drift a
// stateful decoder would produce.
//
// Gated on MODEL because it needs a checkpoint with its real mix of tensor types. Measured on
// gemma-3-270m-it-Q8_0: 236 tensors, all identical on the second call.
func TestTensorDecodesTheSameTwice(t *testing.T) {
	path := os.Getenv("MODEL")
	if path == "" {
		t.Skip("set MODEL to a GGUF file")
	}
	l, err := quant.OpenLoader(path)
	if err != nil {
		t.Fatalf("OpenLoader: %v", err)
	}
	defer func() { _ = l.Close() }()

	var names []string
	for ti := range l.File().Tensors() {
		names = append(names, ti.Name)
	}
	if len(names) == 0 {
		t.Fatal("the file carries no tensors")
	}

	for _, name := range names {
		first, err := l.Tensor(name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		second, err := l.Tensor(name)
		if err != nil {
			t.Fatalf("%s on the second call: %v", name, err)
		}
		if len(first) != len(second) {
			t.Errorf("%s: %d values then %d", name, len(first), len(second))
			continue
		}
		for i := range first {
			if math.Float32bits(first[i]) != math.Float32bits(second[i]) {
				t.Errorf("%s: value %d is %v on one call and %v on the next", name, i,
					first[i], second[i])
				break
			}
		}
	}
}
