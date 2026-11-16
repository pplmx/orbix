package gguf

import (
	"bytes"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// sampleModel is a small but complete GGUF file: one F32 tensor, one Q4_0
// tensor, and enough metadata to look like a tiny Llama checkpoint.
func sampleModel() []byte {
	return buildSample(nil)
}

// buildSample renders the sample model, optionally overriding metadata keys.
func buildSample(overrides []kv) []byte {
	tensors := []TensorInfo{
		{Name: "token_embd.weight", Dims: []uint64{4, 2}, Type: F32, Offset: 0},
		// Q4_0 has 32 weights per 18-byte block, so 32 weights occupy 18 bytes.
		{Name: "blk.0.attn_q.weight", Dims: []uint64{32}, Type: Q4_0, Offset: 32},
	}
	data := make([]byte, 32+18)
	// token_embd: 8 float32 values, 1, 2, ..., 8.
	for i := 0; i < 8; i++ {
		f := math.Float32bits(float32(i + 1))
		data[i*4] = byte(f)
		data[i*4+1] = byte(f >> 8)
		data[i*4+2] = byte(f >> 16)
		data[i*4+3] = byte(f >> 24)
	}
	kvs := []kv{
		{"general.architecture", strVal("llama")},
		{"general.name", strVal("sample")},
		{"general.alignment", u32Val(32)},
		{"llama.block_count", u32Val(1)},
		{"llama.attention.head_count", u32Val(2)},
		{"llama.attention.layer_norm_rms_epsilon", f32Val(1e-5)},
		{"tokenizer.ggml.model", strVal("llama")},
		{"tokenizer.ggml.tokens", strArray("<unk>", "a", "b", "<s>")},
		{"tokenizer.ggml.token_type", i32Array(2, 1, 1, 3)},
		{"tokenizer.ggml.bos_token_id", u32Val(3)},
		{"orbix.test.flag", boolVal(true)},
		{"orbix.test.big", u64Val(1 << 40)},
	}
	return buildFile(append(kvs, overrides...), tensors, 32, data)
}

func TestReadHeaderAndDirectory(t *testing.T) {
	f, err := Read(bytes.NewReader(sampleModel()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if f.Version != Version {
		t.Errorf("Version = %d, want %d", f.Version, Version)
	}
	if len(f.Metadata) != 12 {
		t.Errorf("metadata entries = %d, want 12", len(f.Metadata))
	}
	if len(f.Tensors) != 2 {
		t.Fatalf("tensors = %d, want 2", len(f.Tensors))
	}
	if got := f.Tensors[0].Name; got != "token_embd.weight" {
		t.Errorf("tensor[0].Name = %q", got)
	}
	if got, want := f.Tensors[0].N(), int64(8); got != want {
		t.Errorf("tensor[0].N() = %d, want %d", got, want)
	}
	if got, want := f.Tensors[1].Name, "blk.0.attn_q.weight"; got != want {
		t.Errorf("tensor[1].Name = %q, want %q", got, want)
	}
	if f.Alignment != 32 {
		t.Errorf("Alignment = %d, want 32", f.Alignment)
	}
	// The data section must be aligned relative to the start of the file.
	if f.TensorDataLoc%32 != 0 {
		t.Errorf("TensorDataLoc = %d, not 32-byte aligned", f.TensorDataLoc)
	}
}

func TestMetadataAccessors(t *testing.T) {
	f, err := Read(bytes.NewReader(sampleModel()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}

	if got, ok := f.String("general.architecture"); !ok || got != "llama" {
		t.Errorf("String(general.architecture) = %q, %v", got, ok)
	}
	if _, ok := f.String("llama.block_count"); ok {
		t.Error("String(llama.block_count) reported ok for a uint32 value")
	}
	if got, ok := f.Uint32("llama.block_count"); !ok || got != 1 {
		t.Errorf("Uint32(llama.block_count) = %d, %v", got, ok)
	}
	if got, ok := f.Uint32("llama.attention.head_count"); !ok || got != 2 {
		t.Errorf("Uint32(head_count) = %d, %v", got, ok)
	}
	if got, ok := f.Uint64("orbix.test.big"); !ok || got != 1<<40 {
		t.Errorf("Uint64(orbix.test.big) = %d, %v", got, ok)
	}
	if got, ok := f.Float64("llama.attention.layer_norm_rms_epsilon"); !ok || math.Abs(got-1e-5) > 1e-9 {
		t.Errorf("Float64(eps) = %v, %v", got, ok)
	}
	// Integer values are readable as floats; real checkpoints rely on writers
	// storing some hyperparameters as integers.
	if got, ok := f.Float64("llama.block_count"); !ok || got != 1 {
		t.Errorf("Float64(block_count) = %v, %v", got, ok)
	}
	if got, ok := f.Strings("tokenizer.ggml.tokens"); !ok || len(got) != 4 || got[3] != "<s>" {
		t.Errorf("Strings(tokens) = %q, %v", got, ok)
	}

	// Missing keys and type mismatches must report false rather than zero.
	if _, ok := f.Uint32("llama.missing"); ok {
		t.Error("Uint32 on a missing key reported ok")
	}
	if _, ok := f.Strings("llama.block_count"); ok {
		t.Error("Strings on a scalar reported ok")
	}
	if _, ok := f.String("orbix.test.flag"); ok {
		t.Error("String on a bool reported ok")
	}
}

func TestTokenTypesKeepSignedness(t *testing.T) {
	// tokenizer.ggml.token_type holds int32 values. Reading them as unsigned
	// would corrupt any file that carries a negative value.
	f, err := Read(bytes.NewReader(sampleModel()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	v, ok := f.Lookup("tokenizer.ggml.token_type")
	if !ok {
		t.Fatal("token_type missing")
	}
	if v.Type != TypeArray || v.ElemType != TypeInt32 {
		t.Fatalf("token_type = %s of %s", v.Type, v.ElemType)
	}
	if len(v.Array) != 4 {
		t.Fatalf("token_type length = %d, want 4", len(v.Array))
	}
	if v.Array[3].Int != 3 {
		t.Errorf("token_type[3].Int = %d, want 3", v.Array[3].Int)
	}
}

func TestTensorBytes(t *testing.T) {
	f, err := Read(bytes.NewReader(sampleModel()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	tests := []struct {
		name string
		want int64
	}{
		{"token_embd.weight", 32},   // 8 F32
		{"blk.0.attn_q.weight", 18}, // 32 Q4_0 = one block
	}
	for _, tc := range tests {
		ti, ok := f.Tensor(tc.name)
		if !ok {
			t.Fatalf("Tensor(%q) not found", tc.name)
		}
		got, ok := ti.Bytes()
		if !ok {
			t.Fatalf("Bytes(%q) reported unsupported type", tc.name)
		}
		if got != tc.want {
			t.Errorf("Bytes(%q) = %d, want %d", tc.name, got, tc.want)
		}
	}
	if _, ok := f.Tensor("nope"); ok {
		t.Error("Tensor reported a missing tensor as found")
	}
}

func TestTensorData(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sample.gguf")
	if err := os.WriteFile(path, sampleModel(), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	rc, err := f.TensorData(path, "token_embd.weight")
	if err != nil {
		t.Fatalf("TensorData: %v", err)
	}
	raw, err := io.ReadAll(rc)
	rc.Close()
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(raw) != 32 {
		t.Fatalf("len = %d, want 32", len(raw))
	}
	// The sample stores 1..8 as little-endian float32.
	want := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	for i, w := range want {
		bits := uint32(raw[i*4]) | uint32(raw[i*4+1])<<8 |
			uint32(raw[i*4+2])<<16 | uint32(raw[i*4+3])<<24
		if got := math.Float32frombits(bits); got != w {
			t.Errorf("value[%d] = %v, want %v", i, got, w)
		}
	}

	if _, err := f.TensorData(path, "nope"); err == nil {
		t.Error("TensorData on a missing tensor succeeded")
	}
}

func TestTensorDataRejectsOutOfRange(t *testing.T) {
	// A file whose directory claims more data than the file holds must not
	// produce a reader that silently returns short data.
	data := buildFile(nil, []TensorInfo{
		{Name: "big", Dims: []uint64{1024}, Type: F32, Offset: 0},
	}, 32, make([]byte, 64))
	path := filepath.Join(t.TempDir(), "short.gguf")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if _, err := f.TensorData(path, "big"); err == nil {
		t.Fatal("TensorData on an out-of-range tensor succeeded")
	}
}

func TestReadRejectsBadInput(t *testing.T) {
	good := sampleModel()

	// Read stops after the tensor directory, so a file truncated anywhere
	// inside the header, metadata, or directory must fail, while a file
	// truncated inside the alignment padding may still parse. Find the exact
	// boundary: the first length that parses must be the directory end, and
	// the only thing it can be missing is padding (fewer than alignment
	// bytes).
	parsed, err := Read(bytes.NewReader(good))
	if err != nil {
		t.Fatalf("Read(sample): %v", err)
	}
	firstOK := -1
	for n := 0; n < len(good); n++ {
		if _, err := Read(bytes.NewReader(good[:n])); err == nil {
			firstOK = n
			break
		}
	}
	if firstOK < 0 {
		t.Fatal("every truncation failed, including one that keeps the directory")
	}
	if missing := parsed.TensorDataLoc - int64(firstOK); missing < 0 || missing >= parsed.Alignment {
		t.Errorf("first parsable prefix = %d bytes, TensorDataLoc = %d, so it is missing %d bytes; "+
			"only alignment padding (fewer than %d) may be missing",
			firstOK, parsed.TensorDataLoc, missing, parsed.Alignment)
	}
	for n := firstOK; n <= len(good); n++ {
		if _, err := Read(bytes.NewReader(good[:n])); err != nil {
			t.Errorf("Read(%d bytes) failed after the directory was complete: %v", n, err)
			break
		}
	}

	badMagic := append([]byte("NOPE"), good[4:]...)
	if _, err := Read(bytes.NewReader(badMagic)); err == nil || !strings.Contains(err.Error(), "magic") {
		t.Errorf("Read(bad magic) = %v, want a magic error", err)
	}

	badVersion := append([]byte(nil), good...)
	badVersion[4] = 2
	if _, err := Read(bytes.NewReader(badVersion)); err == nil || !strings.Contains(err.Error(), "version") {
		t.Errorf("Read(version 2) = %v, want a version error", err)
	}
}

func TestOpenMissingFile(t *testing.T) {
	if _, err := Open(filepath.Join(t.TempDir(), "absent.gguf")); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("Open(missing) = %v, want a not-exist error", err)
	}
}

func TestReadRejectsInvalidAlignment(t *testing.T) {
	// alignment 0 would divide by zero when padding the data section.
	data := buildFile([]kv{{"general.alignment", u32Val(0)}}, nil, 0, nil)
	if _, err := Read(bytes.NewReader(data)); err == nil {
		t.Fatal("Read(alignment 0) succeeded, want error")
	}
}

func TestDefaultAlignment(t *testing.T) {
	// Files without general.alignment use the specification's default of 32.
	data := buildFile([]kv{{"general.architecture", strVal("llama")}}, nil, 32, nil)
	f, err := Read(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if f.Alignment != 32 {
		t.Errorf("Alignment = %d, want 32", f.Alignment)
	}
}

func TestArch(t *testing.T) {
	f, err := Read(bytes.NewReader(sampleModel()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if got, err := f.Arch(); err != nil || got != "llama" {
		t.Errorf("Arch() = %q, %v", got, err)
	}

	noArch, err := Read(bytes.NewReader(buildFile(nil, nil, 32, nil)))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if _, err := noArch.Arch(); err == nil {
		t.Error("Arch() on a file without general.architecture succeeded")
	}
}

func TestKeysSorted(t *testing.T) {
	f, err := Read(bytes.NewReader(sampleModel()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	keys := f.Keys()
	if len(keys) != len(f.Metadata) {
		t.Fatalf("Keys() = %d entries, want %d", len(keys), len(f.Metadata))
	}
	for i := 1; i < len(keys); i++ {
		if keys[i-1] >= keys[i] {
			t.Errorf("Keys() not sorted at %d: %q then %q", i, keys[i-1], keys[i])
		}
	}
}

func TestBaseName(t *testing.T) {
	tests := map[string]string{
		"/models/qwen2.5-7b-q4_0.gguf": "qwen2.5-7b-q4_0",
		"model.gguf":                   "model",
		"/a/b/c":                       "c",
	}
	for in, want := range tests {
		if got := BaseName(in); got != want {
			t.Errorf("BaseName(%q) = %q, want %q", in, got, want)
		}
	}
}
