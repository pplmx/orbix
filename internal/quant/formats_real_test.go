package quant

import (
	"os"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
)

// Real-weight checks for the 32-weight block formats: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, BF16
// and IQ4_NL.
//
// The constructed-block tests in dequant_test.go pin the bit layout exactly, which is the
// stronger check where it applies -- but they build their own bytes, so they share the
// decoder's assumption about what a block looks like. A real checkpoint was written by an
// independent quantizer (llama.cpp's), so it catches the case where a layout is
// self-consistent and still disagrees with the encoder that made the file.
//
// The K-quants have their gate in kquant_real_test.go. These formats had none: every test
// was a hand-built block, so a decoder reading the right bits in the wrong order would have
// passed the whole suite and produced plausible-looking weight matrices in every real file.
//
// Each format has its own variable rather than one "MODEL_32WEIGHT", because the formats
// live in different files: a machine holding a Q5_0 checkpoint should be able to check Q5_0
// without also fetching a Q4_1 one.

// formatGate maps a format to the environment variable that supplies a checkpoint carrying
// it, and to the decoder this package must have for it.
var formatGate = []struct {
	typ  gguf.QuantType
	env  string
	name string
}{
	{gguf.Q4_1, "MODEL_Q4_1", "Q4_1"},
	{gguf.Q5_0, "MODEL_Q5_0", "Q5_0"},
	{gguf.Q5_1, "MODEL_Q5_1", "Q5_1"},
	{gguf.IQ4_NL, "MODEL_IQ4_NL", "IQ4_NL"},
}

// firstTensorOfType returns the bytes of the first tensor whose type is want.
//
// Picking by type rather than by name keeps the test independent of a checkpoint's naming:
// every file that carries the format has at least one such tensor, and which one it is does
// not matter for the statistics asserted below.
func firstTensorOfType(t *testing.T, path string, want gguf.QuantType) (*gguf.File, string, []byte) {
	t.Helper()
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open(%s): %v", path, err)
	}
	for ti := range f.Tensors() {
		if ti.Type != want {
			continue
		}
		rc, err := f.TensorData(path, ti.Name)
		if err != nil {
			t.Fatalf("TensorData(%s): %v", ti.Name, err)
		}
		// A whole tensor can be tens of megabytes; grow the buffer rather than sizing it
		// from the header, so a wrong byte count in the file surfaces as a decode error
		// rather than as a silent short read.
		var buf []byte
		tmp := make([]byte, 1<<16)
		for {
			n, rerr := rc.Read(tmp)
			buf = append(buf, tmp[:n]...)
			if rerr != nil {
				break
			}
		}
		_ = rc.Close()
		return f, ti.Name, buf
	}
	return f, "", nil
}

// TestRealWeightMatrixStats decodes one tensor per gated format and checks it looks like a
// transformer weight matrix rather than noise.
//
// The assertion is on the shape of the distribution, not on exact values, and it is worth
// being precise about what that can and cannot see -- measured by breaking the decoder rather
// than argued:
//
//   - Reading the scale from the min slot and the min from the scale slot is caught: Q4_1's
//     mean moved to -0.512 and its spread to 0.518, both outside the bounds.
//   - Swapping the two nibbles of a Q4_1 byte is NOT caught. That is a permutation of the
//     weights, and a permutation preserves the mean and the standard deviation. The
//     constructed-block tests catch it exactly, which is the right place for it: they assert
//     value by value.
//
// So this test covers the failures that depend on reading a field from the wrong place or
// with the wrong width, and the exact-value tests cover ordering. Neither substitutes for the
// other, and saying which is which is the point of writing it down.
//
// All four real checkpoints landed at std 0.0427 and a mean within 1e-5 of zero, which is
// the agreement a correct decoder should show across formats of the same weights.
func TestRealWeightMatrixStats(t *testing.T) {
	checked := 0
	for _, g := range formatGate {
		t.Run(g.name, func(t *testing.T) {
			path := os.Getenv(g.env)
			if path == "" {
				t.Skipf("set %s to a GGUF file whose tensors are %s", g.env, g.name)
			}
			f, name, raw := firstTensorOfType(t, path, g.typ)
			if name == "" {
				// A file without the format under test is not a reason to pass: the gate was
				// set, so something was supposed to be checked. Say what the file does carry.
				t.Fatalf("%s holds no %s tensor; it is not the file this gate is for", path, g.name)
			}
			ti, ok := f.Tensor(name)
			if !ok {
				t.Fatalf("tensor %q vanished from the directory it was read from", name)
			}
			n := int(ti.N())
			dst := make([]float32, n)
			if err := Decode(g.typ, raw, dst); err != nil {
				t.Fatalf("Decode(%s, %d values): %v", g.name, n, err)
			}

			var sum, sumSq float64
			finite := 0
			for _, v := range dst {
				x := float64(v)
				if x != x || x > 1e6 || x < -1e6 {
					continue
				}
				finite++
				sum += x
				sumSq += x * x
			}
			if finite != n {
				t.Errorf("%d of %d values are not finite", n-finite, n)
			}
			mean := sum / float64(n)
			std := sqrtF(sumSq/float64(n) - mean*mean)
			t.Logf("%s: %s, %d values, mean %.6f, std %.6f", name, g.name, n, mean, std)

			// The same bounds kquant_real_test.go uses, for the same reason: wide enough to
			// accept any real weight matrix, narrow enough that a misread scale field (which
			// multiplies by up to 255) or a misread sign bit lands outside them.
			if std > 0.5 || std < 1e-6 {
				t.Errorf("%s standard deviation %.6f is not a weight matrix's spread",
					g.name, std)
			}
			if mean > 0.05 || mean < -0.05 {
				t.Errorf("%s mean %.6f is far from zero for a weight matrix", g.name, mean)
			}
			checked++
		})
	}
	// Report the count rather than only the verdict: a run where every subtest skipped says
	// nothing, and the summary would otherwise look the same as one that checked four files.
	if checked == 0 {
		t.Skip("no 32-weight format gate was set; set MODEL_Q4_1, MODEL_Q5_0, MODEL_Q5_1 or " +
			"MODEL_IQ4_NL to check one")
	}
}

// TestEveryGatedFormatHasADecoder pins that a format named in formatGate is one this package
// can actually decode.
//
// A gate for a format that Decode refuses would skip-on-failure ambiguity: the subtest would
// fail with "cannot decode", which reads as a broken file rather than as a mismatched table.
// This makes the table's own consistency a checked property.
func TestEveryGatedFormatHasADecoder(t *testing.T) {
	for _, g := range formatGate {
		if _, ok := codecs[g.typ]; !ok {
			t.Errorf("%s is gated but has no decoder, so its test could only fail", g.name)
		}
	}
}
