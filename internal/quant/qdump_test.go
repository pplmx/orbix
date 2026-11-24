package quant

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
)

// TestDumpDequant writes decoded blocks for scripts/quant_xcheck.py to compare.
//
// The Go tests pin the arithmetic against hand-computed values, and FuzzDecode covers the
// length checks. Neither catches a misreading of the block layout that the test itself repeats:
// the Q4_0 comment says an interleaving mistake yields "a plausible-looking but completely
// scrambled matrix", and a scrambled matrix passes any test written from the same reading.
//
// The script decodes the same bytes from ggml's layout independently, so the two sides are
// written from different sources.
//
// Skipped unless ORBIX_QUANTDUMP names an output file.
func TestDumpDequant(t *testing.T) {
	out := os.Getenv("ORBIX_QUANTDUMP")
	if out == "" {
		t.Skip("set ORBIX_QUANTDUMP to write a comparison corpus")
	}

	// Formats with a published block layout and no super-block indirection.
	formats := []struct {
		name  string
		typ   gguf.QuantType
		block int
		size  int
	}{
		{"Q4_0", gguf.Q4_0, 32, 18},
		{"Q4_1", gguf.Q4_1, 32, 20},
		{"Q5_0", gguf.Q5_0, 32, 22},
		{"Q5_1", gguf.Q5_1, 32, 24},
		{"Q8_0", gguf.Q8_0, 32, 34},
		{"BF16", gguf.BF16, 1, 2},
		{"F16", gguf.F16, 1, 2},
	}

	type block struct {
		// Src is the encoded bytes, so the script decodes the same input rather than
		// regenerating it and risking an off-by-one in the generator instead of the decoder.
		Src []byte `json:"src"`
		// Bits rather than floats: JSON has no way to spell -0.0 that every parser reads back
		// with its sign, and Python's json module turns it into the integer 0. A decoder that
		// dropped the sign would look correct.
		Bits []uint32 `json:"bits"`
	}
	got := make(map[string]block, len(formats))
	for _, f := range formats {
		// Two blocks of bytes with every pattern the layout can carry: a scale, then
		// nibbles or bytes that exercise both ends of the code range.
		src := make([]byte, f.size*2)
		for i := range src {
			src[i] = byte(i*37 + 11)
		}
		dst := make([]float32, f.block*2)
		if err := Decode(f.typ, src, dst); err != nil {
			t.Fatalf("%s: %v", f.name, err)
		}
		bits := make([]uint32, len(dst))
		for i, v := range dst {
			bits[i] = math.Float32bits(v)
		}
		got[f.name] = block{Src: src, Bits: bits}
	}

	b, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(out, b, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
}
