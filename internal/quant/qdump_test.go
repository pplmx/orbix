package quant

import (
	"encoding/json"
	"math"
	"math/rand"
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
		{"F32", gguf.F32, 1, 4},
		// The K-quants put eight or sixteen sub-blocks inside one super-block, each with its
		// own 6-bit scale, and locate the pieces in an order that differs between formats --
		// Q6_K's super-scale is the last field, not the first. One block each is enough: the
		// arithmetic repeats per sub-block.
		{"Q4_K", gguf.Q4_K, 256, 144},
		{"Q6_K", gguf.Q6_K, 256, 210},
		{"IQ4_NL", gguf.IQ4_NL, 32, 18},
		{"Q2_K", gguf.Q2_K, 256, 84},
		{"Q3_K", gguf.Q3_K, 256, 110},
		{"Q5_K", gguf.Q5_K, 256, 176},
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
		// Two corpora, because each finds a different kind of mistake.
		//
		// The arithmetic run is byte i = i*37+11 over two blocks: a deterministic pattern that
		// gives both blocks different scale fields and walks the code range, so a layout read that
		// is off by one byte shows up immediately.
		//
		// The random run is eight blocks from a fixed seed. It exists because the arithmetic run is
		// too regular to reach every combination -- measured, it produced 64 values for the
		// 32-weight formats and 512 for the 256-weight ones, and running the same comparison over
		// 11648 random values agreed with ggml too. A seed rather than a time source keeps the
		// corpus reproducible: the point is coverage, not novelty.
		srcs := [][]byte{make([]byte, f.size*2)}
		for i := range srcs[0] {
			srcs[0][i] = byte(i*37 + 11)
		}
		// Eight blocks, whatever a block is: for the one-value formats that is eight values, which
		// is why the block count rather than a byte count sets the size.
		rng := rand.New(rand.NewSource(0x9e3779b9))
		random := make([]byte, f.size*8)
		rng.Read(random)
		// The scale fields stay at a sane magnitude so the comparison is about the layout rather
		// than about an infinity times zero. Their offsets differ per format and are listed in the
		// table above only for the formats this file decodes through offsets; the rest are found
		// by simply leaving the bytes as drawn, since every decoder here tolerates an odd f16.
		srcs = append(srcs, random)

		var src []byte
		var bits []uint32
		for _, s := range srcs {
			dst := make([]float32, len(s)/f.size*f.block)
			if err := Decode(f.typ, s, dst); err != nil {
				t.Fatalf("%s: %v", f.name, err)
			}
			src = append(src, s...)
			for _, v := range dst {
				bits = append(bits, math.Float32bits(v))
			}
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
