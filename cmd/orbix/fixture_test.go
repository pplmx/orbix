package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// writeTestModel writes a minimal but complete GGUF file to a temp dir and
// returns its path.
//
// The CLI tests need a real file on disk because Open takes a path. Writing
// one here rather than shipping a binary fixture keeps the test data visible
// and reviewable in the test source itself.
func writeTestModel(t *testing.T) string {
	t.Helper()

	var b bytes.Buffer
	u32 := func(v uint32) {
		var tmp [4]byte
		binary.LittleEndian.PutUint32(tmp[:], v)
		b.Write(tmp[:])
	}
	u64 := func(v uint64) {
		var tmp [8]byte
		binary.LittleEndian.PutUint64(tmp[:], v)
		b.Write(tmp[:])
	}
	str := func(s string) {
		u64(uint64(len(s)))
		b.WriteString(s)
	}

	b.WriteString("GGUF")
	u32(3) // version
	u64(2) // tensor count
	u64(4) // metadata count

	str("general.architecture")
	u32(8) // string
	str("llama")

	str("llama.block_count")
	u32(4) // uint32
	u32(1)

	str("llama.attention.layer_norm_rms_epsilon")
	u32(6) // float32
	u32(math.Float32bits(1e-5))

	str("tokenizer.ggml.tokens")
	u32(9) // array
	u32(8) // of strings
	u64(2) // two elements
	str("<unk>")
	str("a")

	// Tensor 0: token_embd.weight, dims [4 2], F32, offset 0.
	str("token_embd.weight")
	u32(2)
	u64(4)
	u64(2)
	u32(0) // F32
	u64(0)

	// Tensor 1: blk.0.attn_q.weight, dims [32], Q4_0, offset 32.
	str("blk.0.attn_q.weight")
	u32(1)
	u64(32)
	u32(2) // Q4_0
	u64(32)

	// Pad to the 32-byte alignment, then write 32 bytes of F32 data and one
	// 18-byte Q4_0 block.
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	data := make([]byte, 32+18)
	for i := 0; i < 8; i++ {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(float32(i+1)))
	}
	b.Write(data)

	path := filepath.Join(t.TempDir(), "test.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}
