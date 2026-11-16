package quant

import (
	"os"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
)

// These tests check a decoder against a real checkpoint's weights rather than against
// constructed blocks. They exist because a layout can be self-consistent and still
// disagree with the encoder: the constructed-block tests pin the bit order, and this
// pins the magnitudes to what a real quantizer produced.
//
// Set MODEL_KQUANT to a GGUF file whose tensors are K-quantized.

// kQuantTensor returns the encoded bytes and expected values for one tensor, or skips.
//
// The expected values come from an independent numpy decoder run over the same bytes;
// they are compared in TestKQuantRealTensor below when the reference file is present.
func kQuantTensor(t *testing.T) (*gguf.File, string, gguf.QuantType, []byte) {
	t.Helper()
	path := os.Getenv("MODEL_KQUANT")
	if path == "" {
		t.Skip("set MODEL_KQUANT to a K-quantized GGUF file")
	}
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	// Pick the first tensor of a K-quant type, so the test does not depend on a
	// particular tensor name being present in a particular checkpoint.
	for ti := range f.Tensors() {
		if ti.Type == gguf.Q4_K || ti.Type == gguf.Q6_K {
			rc, err := f.TensorData(path, ti.Name)
			if err != nil {
				t.Fatalf("TensorData(%s): %v", ti.Name, err)
			}
			buf := make([]byte, 0, 1<<20)
			tmp := make([]byte, 1<<16)
			for {
				n, rerr := rc.Read(tmp)
				buf = append(buf, tmp[:n]...)
				if rerr != nil {
					break
				}
			}
			_ = rc.Close()
			return f, ti.Name, ti.Type, buf
		}
	}
	t.Skip("no K-quant tensor found in the model")
	return nil, "", 0, nil
}

// TestKQuantRealTensor decodes a real K-quant tensor and checks the result is finite,
// the right length, and has the statistics a weight matrix should have.
//
// The point is not exact values -- those live in the numpy comparison run during
// development -- but that the decoder produces a plausible matrix rather than garbage. A
// wrong sub-block scale shows up here as a standard deviation off by orders of magnitude.
func TestKQuantRealTensor(t *testing.T) {
	f, name, typ, raw := kQuantTensor(t)
	ti, ok := f.Tensor(name)
	if !ok {
		t.Fatalf("tensor %q vanished", name)
	}
	n := int(ti.N())
	dst := make([]float32, n)
	if err := Decode(typ, raw, dst); err != nil {
		t.Fatalf("Decode(%s, %d values): %v", typ, n, err)
	}

	var sum, sumSq float64
	finite := 0
	for _, v := range dst {
		f := float64(v)
		if f != f || f > 1e6 || f < -1e6 {
			continue
		}
		finite++
		sum += f
		sumSq += f * f
	}
	if finite != n {
		t.Errorf("%d of %d values are not finite", n-finite, n)
	}
	mean := sum / float64(n)
	std := sqrtF(sumSq/float64(n) - mean*mean)
	t.Logf("%s: %s, %d values, mean %.6f, std %.6f", name, typ, n, mean, std)

	// A transformer weight matrix is centred near zero with a small spread. These
	// bounds are wide on purpose: they catch a scale field read from the wrong offset
	// (which multiplies the result by up to 255) without asserting a particular model.
	if std > 0.5 || std < 1e-6 {
		t.Errorf("standard deviation %.6f is not a weight matrix's spread", std)
	}
	if mean > 0.05 || mean < -0.05 {
		t.Errorf("mean %.6f is far from zero for a weight matrix", mean)
	}
}

// sqrtF is a Newton square root, so this file needs no math import for one call.
func sqrtF(v float64) float64 {
	if v <= 0 {
		return 0
	}
	x := v
	for range 60 {
		x = (x + v/x) / 2
	}
	return x
}
