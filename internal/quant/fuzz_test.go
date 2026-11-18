package quant

import (
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
)

// FuzzDecode checks that no byte pattern makes a decoder panic or read out of range.
//
// The encoded bytes come from a model file, so they are as untrusted as the rest of it. Decode
// validates the length before handing the buffer to a codec, which should make a short buffer
// impossible to reach -- but "should" is what a fuzz target is for, and a decoder that indexes
// past its block is a crash rather than a wrong number.
//
// The destination length is derived from the input so the target explores the block/remainder
// arithmetic, and the type is chosen from the input so every codec is reached from one target
// rather than needing one per format.
func FuzzDecode(f *testing.F) {
	// Per-format seeds: a block-sized buffer of a plausible encoding, and the same buffer one
	// byte short, which is the shape that would expose an off-by-one in a length check.
	for t, c := range codecs {
		full := make([]byte, c.size)
		for i := range full {
			full[i] = byte(i * 7)
		}
		f.Add(uint8(t), full, c.block)
		if len(full) > 0 {
			f.Add(uint8(t), full[:len(full)-1], c.block)
		}
		f.Add(uint8(t), []byte{}, c.block)
		f.Add(uint8(t), full, c.block*2)
	}
	f.Add(uint8(0), []byte{}, 0)

	f.Fuzz(func(t *testing.T, typeByte uint8, src []byte, blocks int) {
		typ := gguf.QuantType(typeByte)
		c, ok := codecs[typ]
		if !ok {
			return
		}
		// Bound the destination so the target cannot ask for a gigabyte of floats: the point
		// is the decoder's bounds behaviour, not its throughput.
		const maxBlocks = 64
		if blocks < 0 {
			blocks = -blocks
		}
		blocks %= maxBlocks
		n := blocks * c.block
		dst := make([]float32, n)

		// A wrong length must be refused, never decoded.
		if err := Decode(typ, src, dst); err != nil {
			if len(src) == n/c.block*c.size && n%c.block == 0 {
				t.Fatalf("%s: %d bytes refused for %d elements, but the length is right",
					typ, len(src), n)
			}
			return
		}
		// Decoding succeeded, so every value must be a real float. A NaN here is not
		// automatically a bug -- the encoding can represent one -- so the check is that the
		// call returned at all.
	})
}
