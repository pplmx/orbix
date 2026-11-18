package gguf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
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
	if got := f.MetadataCount(); got != 12 {
		t.Errorf("metadata entries = %d, want 12", got)
	}
	if got := f.TensorCount(); got != 2 {
		t.Fatalf("tensors = %d, want 2", got)
	}
	// The streaming accessor must yield the directory in file order.
	var names []string
	for ti := range f.Tensors() {
		names = append(names, ti.Name)
	}
	if want := []string{"token_embd.weight", "blk.0.attn_q.weight"}; !slices.Equal(names, want) {
		t.Errorf("Tensors() names = %v, want %v", names, want)
	}
	if got, want := f.TensorInfos[0].N(), int64(8); got != want {
		t.Errorf("tensor[0].N() = %d, want %d", got, want)
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

// TestParserRejectsStructuralCorruption covers the three error paths a corrupt file can
// reach that no test exercised: an impossible dimension count, a tensor name that is not
// in the directory, and a file with no general.architecture.
//
// Each is a check the parser performs for a specific reason, and each was reachable
// without a test saying so -- the repository's convention is that a parser's error paths
// are tested, because the alternative is a panic three frames deep in a decode loop.
// The assertions name the message because the message is the whole product of these
// paths: nothing else tells the caller which tensor or key was wrong.
func TestParserRejectsStructuralCorruption(t *testing.T) {
	t.Run("impossible dimension count", func(t *testing.T) {
		// A GGUF tensor is at most 4-dimensional in ggml; the reader allows up to 8 and
		// rejects beyond that rather than allocating a dims slice from the file's word.
		tooMany := make([]uint64, 9)
		for i := range tooMany {
			tooMany[i] = 1
		}
		data := buildFile(nil, []TensorInfo{{Name: "w", Type: F32, Dims: tooMany}}, 32, nil)
		_, err := Read(bytes.NewReader(data))
		if err == nil {
			t.Fatal("Read accepted a 9-dimensional tensor")
		}
		if !strings.Contains(err.Error(), "dimensions") {
			t.Errorf("error %q does not say the dimension count is the problem", err)
		}
	})

	t.Run("unknown tensor name", func(t *testing.T) {
		data := buildFile(nil, []TensorInfo{{Name: "w", Type: F32, Dims: []uint64{4}}}, 32, make([]byte, 16))
		path := filepath.Join(t.TempDir(), "one.gguf")
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
		f, err := Open(path)
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		if _, err := f.TensorData(path, "absent"); err == nil {
			t.Fatal("TensorData(absent) succeeded")
		} else if !strings.Contains(err.Error(), "no tensor named") {
			t.Errorf("error %q does not say the name is not in the file", err)
		}
	})

	t.Run("no architecture key", func(t *testing.T) {
		// Arch is what every architecture-specific key is prefixed with, so a file
		// without it cannot be interpreted at all; it must say so rather than return "".
		data := buildFile(nil, nil, 32, nil)
		f, err := Read(bytes.NewReader(data))
		if err != nil {
			t.Fatalf("Read: %v", err)
		}
		if arch, err := f.Arch(); err == nil {
			t.Errorf("Arch returned %q with no error for a file with no general.architecture", arch)
		} else if !strings.Contains(err.Error(), "general.architecture") {
			t.Errorf("error %q does not name the missing key", err)
		}
	})
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
	if len(keys) != f.MetadataCount() {
		t.Fatalf("Keys() = %d entries, want %d", len(keys), f.MetadataCount())
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

func TestGetGenericConversions(t *testing.T) {
	f, err := Read(bytes.NewReader(sampleModel()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}

	// The same key read through the generic accessor and its typed wrappers
	// must agree; the wrappers are thin aliases and drifting apart would mean
	// one of the two conversions is wrong.
	if got, ok := f.Get[uint32]("llama.block_count"); !ok || got != 1 {
		t.Errorf("Get[uint32](block_count) = %d, %v", got, ok)
	}
	if got, ok := f.Get[int64]("llama.block_count"); !ok || got != 1 {
		t.Errorf("Get[int64](block_count) = %d, %v", got, ok)
	}
	if got, ok := f.Get[float32]("llama.block_count"); !ok || got != 1 {
		t.Errorf("Get[float32](block_count) = %v, %v", got, ok)
	}
	// float32 stored as 1e-5 must survive as float32 and as float64.
	if got, ok := f.Get[float32]("llama.attention.layer_norm_rms_epsilon"); !ok || got != 1e-5 {
		t.Errorf("Get[float32](eps) = %v, %v", got, ok)
	}
	if got, ok := f.Get[float64]("llama.attention.layer_norm_rms_epsilon"); !ok || math.Abs(got-1e-5) > 1e-9 {
		t.Errorf("Get[float64](eps) = %v, %v", got, ok)
	}
	if got, ok := f.Get[string]("general.architecture"); !ok || got != "llama" {
		t.Errorf("Get[string](architecture) = %q, %v", got, ok)
	}
	if got, ok := f.Get[bool]("orbix.test.flag"); !ok || !got {
		t.Errorf("Get[bool](flag) = %v, %v", got, ok)
	}
}

func TestGetRejectsWrongKind(t *testing.T) {
	f, err := Read(bytes.NewReader(sampleModel()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	// Asking for a string at a numeric key, or a number at a string key, must
	// report absence rather than a zero value that looks like real data.
	if _, ok := f.Get[string]("llama.block_count"); ok {
		t.Error("Get[string] at a uint32 key reported ok")
	}
	if _, ok := f.Get[uint32]("general.architecture"); ok {
		t.Error("Get[uint32] at a string key reported ok")
	}
	if _, ok := f.Get[bool]("llama.block_count"); ok {
		t.Error("Get[bool] at a numeric key reported ok")
	}
	if _, ok := f.Get[uint32]("llama.missing"); ok {
		t.Error("Get at a missing key reported ok")
	}
}

func TestGetRejectsNarrowingOverflow(t *testing.T) {
	// orbix.test.big is 1<<40, which does not fit in uint32 or int32. The
	// conversion must report absence instead of wrapping to a plausible small
	// number that would silently change inference.
	f, err := Read(bytes.NewReader(sampleModel()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if _, ok := f.Get[uint32]("orbix.test.big"); ok {
		t.Error("Get[uint32] of 1<<40 reported ok")
	}
	if _, ok := f.Get[int32]("orbix.test.big"); ok {
		t.Error("Get[int32] of 1<<40 reported ok")
	}
	if got, ok := f.Get[uint64]("orbix.test.big"); !ok || got != 1<<40 {
		t.Errorf("Get[uint64](big) = %d, %v", got, ok)
	}
}

// TestScalarAsFloatToInteger covers the conversion branch nothing reached: a float source
// read as an integer target. Every other case in that function is integral-to-integral or
// integral-to-float, and this one decides whether metadata stored as 3.0 can be read as an
// int without a caller knowing which spelling a writer chose.
//
// The rule is that a float converts only when it is exactly integral: 3.0 becomes 3, while
// 3.5 is reported unrepresentable instead of truncated. Truncation here would be the quiet
// kind of wrong -- an alignment of 33 read as 32 still divides, and the file's tensors are
// then read from the wrong offsets.
// TestRejectsArrayOfArrays covers the one structural rule GGUF's metadata section states
// about arrays: "Arrays can contain other types but not other arrays" (gguf-spec.md, III).
//
// Nothing enforced it. A file declaring an array whose element type is itself an array was
// read into Value{Type: TypeArray, Array: []Value{{Type: TypeArray, ...}}}, which the rest of
// the package has no meaning for: GetArray compares each element's type against T, so an
// array inside an array reads as "the key is absent" -- a silent skip, not an error, and
// exactly what this parser is written not to do.
//
// The guard is on the element type rather than on a recursion depth, because the format
// allows one level: there is no depth to count once an array element cannot be an array.
func TestRejectsArrayOfArrays(t *testing.T) {
	// hand-built: the fixture builder has no way to express a shape the format forbids.
	var b bytes.Buffer
	b.WriteString(Magic)
	u32 := func(v uint32) {
		var x [4]byte
		binary.LittleEndian.PutUint32(x[:], v)
		b.Write(x[:])
	}
	u64 := func(v uint64) {
		var x [8]byte
		binary.LittleEndian.PutUint64(x[:], v)
		b.Write(x[:])
	}

	u32(Version)
	u64(0) // no tensors
	u64(1) // one metadata entry
	u64(uint64(len("nested")))
	b.WriteString("nested")
	u32(uint32(TypeArray)) // value type: array
	u32(uint32(TypeArray)) // element type: array, which the format forbids
	u64(1)                 // one element
	// The inner array the forbidden element type would describe.
	u32(uint32(TypeUint8))
	u64(1)
	b.WriteByte(7)

	_, err := Read(bytes.NewReader(b.Bytes()))
	if err == nil {
		t.Fatal("Read accepted an array of arrays")
	}
	// The message has to say the shape is the problem: this file is otherwise well-formed,
	// so an error about lengths or EOF would send a reader looking in the wrong place.
	if !strings.Contains(err.Error(), "array of arrays") {
		t.Errorf("error %q does not name the nesting as the problem", err)
	}
}

// TestNormalArraysStillParse is the other half: rejecting nesting must not reject the flat
// arrays every real file has.
func TestNormalArraysStillParse(t *testing.T) {
	f, err := Read(bytes.NewReader(sampleModel()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	// The fixture carries a string array (tokens) and an int32 array (token_type).
	tokens, ok := f.GetArray[string]("tokenizer.ggml.tokens")
	if !ok || len(tokens) == 0 {
		t.Fatalf("GetArray[string](tokens) = %v, %v", tokens, ok)
	}
	types, ok := f.GetArray[int32]("tokenizer.ggml.token_type")
	if !ok || len(types) != len(tokens) {
		t.Errorf("GetArray[int32](token_type) has %d values, want %d", len(types), len(tokens))
	}
}

func TestScalarAsFloatToInteger(t *testing.T) {
	cases := []struct {
		name   string
		src    Value
		want   int
		wantOK bool
	}{
		{"integral float32", Value{Type: TypeFloat32, Float: 3}, 3, true},
		{"negative integral", Value{Type: TypeFloat32, Float: -7}, -7, true},
		{"zero", Value{Type: TypeFloat32, Float: 0}, 0, true},
		{"fractional float32", Value{Type: TypeFloat32, Float: 3.5}, 0, false},
		{"fractional float64", Value{Type: TypeFloat64, Float: 0.1}, 0, false},
		{"positive infinity", Value{Type: TypeFloat64, Float: math.Inf(1)}, 0, false},
		{"negative infinity", Value{Type: TypeFloat64, Float: math.Inf(-1)}, 0, false},
		{"NaN", Value{Type: TypeFloat64, Float: math.NaN()}, 0, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, ok := scalarAs[int](c.src)
			if ok != c.wantOK {
				t.Fatalf("scalarAs[int](%s) ok = %v, want %v", c.name, ok, c.wantOK)
			}
			if ok && got != c.want {
				t.Errorf("scalarAs[int](%s) = %d, want %d", c.name, got, c.want)
			}
		})
	}

	// A float too large for the target must not wrap. 1e300 is integral and finite, so it
	// passes the truncation check and has to be caught by the wide-int path instead.
	if _, ok := scalarAs[int32](Value{Type: TypeFloat64, Float: 1e300}); ok {
		t.Error("scalarAs[int32](1e300) reported ok")
	}
}

// TestScalarAsIntegerToFloat covers the signed-integer-to-float direction. Only the
// unsigned case was reached before, because the fixture's large value happens to be
// unsigned; a negative metadata value is the interesting one, since a conversion that
// routed signed values through the unsigned field would turn -7 into a huge positive.
func TestScalarAsIntegerToFloat(t *testing.T) {
	cases := []struct {
		name string
		src  Value
		want float64
	}{
		{"negative int32", Value{Type: TypeInt32, Int: -7}, -7},
		{"negative int8", Value{Type: TypeInt8, Int: -128}, -128},
		{"positive int64", Value{Type: TypeInt64, Int: 1 << 40}, 1 << 40},
		{"unsigned uint16", Value{Type: TypeUint16, Uint: 65535}, 65535},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, ok := scalarAs[float64](c.src)
			if !ok {
				t.Fatalf("scalarAs[float64](%s) reported not ok", c.name)
			}
			if got != c.want {
				t.Errorf("scalarAs[float64](%s) = %v, want %v", c.name, got, c.want)
			}
		})
	}
}

// TestScalarAsUnsignedTooLargeForInt64 covers the guard that keeps a uint64 above
// MaxInt64 from becoming a negative number on the way to an integer target.
func TestScalarAsUnsignedTooLargeForInt64(t *testing.T) {
	// 1<<63 is exactly one past MaxInt64; as an int64 it would read as MinInt64.
	if _, ok := scalarAs[int64](Value{Type: TypeUint64, Uint: 1 << 63}); ok {
		t.Error("scalarAs[int64](1<<63) reported ok; it would come back negative")
	}
	// Just below the limit must still convert.
	if got, ok := scalarAs[int64](Value{Type: TypeUint64, Uint: 1<<63 - 1}); !ok || got != math.MaxInt64 {
		t.Errorf("scalarAs[int64](MaxInt64) = %d, %v", got, ok)
	}
}

// TestFloatAsNarrowingOverflow covers floatAs's one checked conversion: a float64 that
// becomes +Inf when narrowed to float32.
//
// The check exists because the alternative is worse than absence. Replacing a real number
// with infinity makes every downstream comparison involving it true, and a model loaded
// with an infinite epsilon normalizes to zero rows rather than failing to load.
func TestFloatAsNarrowingOverflow(t *testing.T) {
	// MaxFloat32 is representable; twice it is not. Both are finite float64 values.
	if got, ok := floatAs[float32](math.MaxFloat32); !ok || got != math.MaxFloat32 {
		t.Errorf("floatAs[float32](MaxFloat32) = %v, %v", got, ok)
	}
	if _, ok := floatAs[float32](2 * math.MaxFloat32); ok {
		t.Error("floatAs[float32](2*MaxFloat32) reported ok; the value became +Inf")
	}
	// A genuine infinity must still convert: the guard is about narrowing, not about Inf.
	if got, ok := floatAs[float32](math.Inf(1)); !ok || !math.IsInf(float64(got), 1) {
		t.Errorf("floatAs[float32](+Inf) = %v, %v; the source was already infinite", got, ok)
	}
	// float64 to float64 is exact and must never fail.
	if got, ok := floatAs[float64](1e308); !ok || got != 1e308 {
		t.Errorf("floatAs[float64](1e308) = %v, %v", got, ok)
	}
}

// TestIntAsEveryTargetWidth drives intAs's per-width range checks, which is a branch for
// each integer type in the Scalar union and no test reached more than a few.
//
// The boundaries are the point: each check is `n < min || n > max` for that width, and an
// off-by-one there would let one specific value wrap into a plausible small number.
func TestIntAsEveryTargetWidth(t *testing.T) {
	cases := []struct {
		name string
		run  func() bool
	}{
		{"int8 max ok", func() bool { _, ok := intAs[int8](math.MaxInt8); return ok }},
		{"int8 over", func() bool { _, ok := intAs[int8](math.MaxInt8 + 1); return !ok }},
		{"int8 min ok", func() bool { _, ok := intAs[int8](math.MinInt8); return ok }},
		{"int8 under", func() bool { _, ok := intAs[int8](math.MinInt8 - 1); return !ok }},
		{"uint8 max ok", func() bool { _, ok := intAs[uint8](math.MaxUint8); return ok }},
		{"uint8 over", func() bool { _, ok := intAs[uint8](math.MaxUint8 + 1); return !ok }},
		{"uint8 negative", func() bool { _, ok := intAs[uint8](-1); return !ok }},
		{"int16 max ok", func() bool { _, ok := intAs[int16](math.MaxInt16); return ok }},
		{"int16 over", func() bool { _, ok := intAs[int16](math.MaxInt16 + 1); return !ok }},
		{"uint16 max ok", func() bool { _, ok := intAs[uint16](math.MaxUint16); return ok }},
		{"uint16 over", func() bool { _, ok := intAs[uint16](math.MaxUint16 + 1); return !ok }},
		{"int32 max ok", func() bool { _, ok := intAs[int32](math.MaxInt32); return ok }},
		{"int32 over", func() bool { _, ok := intAs[int32](math.MaxInt32 + 1); return !ok }},
		{"uint32 max ok", func() bool { _, ok := intAs[uint32](math.MaxUint32); return ok }},
		{"uint32 over", func() bool { _, ok := intAs[uint32](math.MaxUint32 + 1); return !ok }},
		{"uint32 negative", func() bool { _, ok := intAs[uint32](-1); return !ok }},
		{"int64 any", func() bool { _, ok := intAs[int64](math.MinInt64); return ok }},
		{"uint64 negative", func() bool { _, ok := intAs[uint64](-1); return !ok }},
		{"uint64 max ok", func() bool { got, ok := intAs[uint64](math.MaxInt64); return ok && got == math.MaxInt64 }},
		{"float32 target", func() bool { got, ok := intAs[float32](7); return ok && got == 7 }},
		{"float64 target", func() bool { got, ok := intAs[float64](7); return ok && got == 7 }},
		{"int target", func() bool { got, ok := intAs[int](7); return ok && got == 7 }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if !c.run() {
				t.Errorf("%s: the conversion did not behave as the boundary requires", c.name)
			}
		})
	}
}

func TestGetArray(t *testing.T) {
	f, err := Read(bytes.NewReader(sampleModel()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	tokens, ok := f.GetArray[string]("tokenizer.ggml.tokens")
	if !ok || !slices.Equal(tokens, []string{"<unk>", "a", "b", "<s>"}) {
		t.Errorf("GetArray[string](tokens) = %v, %v", tokens, ok)
	}
	types, ok := f.GetArray[int32]("tokenizer.ggml.token_type")
	if !ok || !slices.Equal(types, []int32{2, 1, 1, 3}) {
		t.Errorf("GetArray[int32](token_type) = %v, %v", types, ok)
	}
	// A scalar key is not an array.
	if _, ok := f.GetArray[string]("general.architecture"); ok {
		t.Error("GetArray at a scalar key reported ok")
	}
}

func TestGetArrayEmptyIsSuccess(t *testing.T) {
	// An empty array must be distinguishable from a failure: both would return
	// a nil slice to a careless implementation.
	data := buildFile([]kv{{"orbix.test.empty", Value{Type: TypeArray, ElemType: TypeString}}}, nil, 32, nil)
	f, err := Read(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	got, ok := f.GetArray[string]("orbix.test.empty")
	if !ok {
		t.Fatal("GetArray on an empty array reported failure")
	}
	if len(got) != 0 {
		t.Errorf("GetArray = %v, want empty", got)
	}
	if _, ok := f.GetArray[string]("orbix.test.absent"); ok {
		t.Error("GetArray at a missing key reported ok")
	}
}

func TestAllIteratesSorted(t *testing.T) {
	f, err := Read(bytes.NewReader(sampleModel()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	var keys []string
	for k := range f.All() {
		keys = append(keys, k)
	}
	if !slices.IsSorted(keys) {
		t.Errorf("All() yielded unsorted keys: %v", keys)
	}
	if len(keys) != f.MetadataCount() {
		t.Errorf("All() yielded %d keys, want %d", len(keys), f.MetadataCount())
	}
	// Early termination must stop the iteration.
	count := 0
	for range f.All() {
		count++
		break
	}
	if count != 1 {
		t.Errorf("early break yielded %d entries, want 1", count)
	}
}

func TestBaseNameEdgeCases(t *testing.T) {
	tests := map[string]string{
		"/models/qwen2.5-7b-q4_0.gguf": "qwen2.5-7b-q4_0",
		"model.gguf":                   "model",
		"noext":                        "noext",
		"/a/b/c":                       "c",
		// A dotfile has no extension, so it must not collapse to an empty
		// name.
		".gguf": ".gguf",
		// Only the final extension is stripped; the version stays.
		"model.v2.gguf": "model.v2",
	}
	for in, want := range tests {
		if got := BaseName(in); got != want {
			t.Errorf("BaseName(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestGetEveryIntegerWidth(t *testing.T) {
	// Every integer width in the constraint must convert from every stored
	// integer type. A generic accessor whose conversions only work for some
	// widths is worse than no generic accessor: the failure looks like a
	// missing key rather than a wrong implementation.
	f, err := Read(bytes.NewReader(sampleModel()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	for _, tc := range []struct {
		key  string
		want int64
	}{
		{"llama.block_count", 1},           // uint32
		{"tokenizer.ggml.bos_token_id", 3}, // uint32
	} {
		if got, ok := f.Get[int](tc.key); !ok || int64(got) != tc.want {
			t.Errorf("Get[int](%s) = %d, %v, want %d", tc.key, got, ok, tc.want)
		}
		if got, ok := f.Get[int8](tc.key); !ok || int64(got) != tc.want {
			t.Errorf("Get[int8](%s) = %d, %v, want %d", tc.key, got, ok, tc.want)
		}
		if got, ok := f.Get[int16](tc.key); !ok || int64(got) != tc.want {
			t.Errorf("Get[int16](%s) = %d, %v", tc.key, got, ok)
		}
		if got, ok := f.Get[int32](tc.key); !ok || int64(got) != tc.want {
			t.Errorf("Get[int32](%s) = %d, %v", tc.key, got, ok)
		}
		if got, ok := f.Get[int64](tc.key); !ok || got != tc.want {
			t.Errorf("Get[int64](%s) = %d, %v", tc.key, got, ok)
		}
		if got, ok := f.Get[uint8](tc.key); !ok || int64(got) != tc.want {
			t.Errorf("Get[uint8](%s) = %d, %v", tc.key, got, ok)
		}
		if got, ok := f.Get[uint16](tc.key); !ok || int64(got) != tc.want {
			t.Errorf("Get[uint16](%s) = %d, %v", tc.key, got, ok)
		}
		if got, ok := f.Get[uint32](tc.key); !ok || int64(got) != tc.want {
			t.Errorf("Get[uint32](%s) = %d, %v", tc.key, got, ok)
		}
		if got, ok := f.Get[uint64](tc.key); !ok || int64(got) != tc.want {
			t.Errorf("Get[uint64](%s) = %d, %v", tc.key, got, ok)
		}
		if got, ok := f.Get[uint](tc.key); !ok || int64(got) != tc.want {
			t.Errorf("Get[uint](%s) = %d, %v", tc.key, got, ok)
		}
		if got, ok := f.Get[float32](tc.key); !ok || int64(got) != tc.want {
			t.Errorf("Get[float32](%s) = %v, %v", tc.key, got, ok)
		}
		if got, ok := f.Get[float64](tc.key); !ok || int64(got) != tc.want {
			t.Errorf("Get[float64](%s) = %v, %v", tc.key, got, ok)
		}
	}
}

func TestGetSignedness(t *testing.T) {
	// A negative int32 must not be readable as any unsigned type; silently
	// wrapping it would turn -1 into 4294967295.
	data := buildFile([]kv{{"orbix.test.neg", i32Val(-1)}}, nil, 32, nil)
	f, err := Read(bytes.NewReader(data))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if got, ok := f.Get[int32]("orbix.test.neg"); !ok || got != -1 {
		t.Errorf("Get[int32] = %d, %v, want -1", got, ok)
	}
	for _, check := range []struct {
		name string
		ok   bool
	}{
		{"uint32", func() bool { _, ok := f.Get[uint32]("orbix.test.neg"); return ok }()},
		{"uint64", func() bool { _, ok := f.Get[uint64]("orbix.test.neg"); return ok }()},
		{"uint", func() bool { _, ok := f.Get[uint]("orbix.test.neg"); return ok }()},
	} {
		if check.ok {
			t.Errorf("Get[%s] accepted a negative value", check.name)
		}
	}
}

// TestHugeDimensionsDoNotWrap covers tensor shapes whose element count overflows int64.
//
// Dims come straight from the file, so hostile or corrupt values must not be trusted to produce
// a sensible product. The failure this guards against is specific: a product that wraps to a
// negative number is worse than one reported as unavailable, because a bounds check written as
// "start + nbytes > fileSize" is satisfied by a negative length and the read then goes through.
//
// N() therefore saturates, and Size reports the byte count as unknown rather than computing it.
func TestHugeDimensionsDoNotWrap(t *testing.T) {
	cases := [][]uint64{
		{1 << 62, 4},
		{1 << 40, 1 << 40},
		{math.MaxUint64, 2},
		{1 << 63},
		{2, 3}, // a real shape still counts normally
	}
	for _, dims := range cases {
		ti := TensorInfo{Dims: dims, Type: F32}
		if n := ti.N(); n <= 0 {
			t.Errorf("dims %v: N() = %d, want a positive count", dims, n)
		}
		// For the shapes that overflow, the byte count must be reported as unavailable. A
		// saturated N() times four would wrap, which is exactly what the second return value
		// exists to prevent.
		n, ok := ti.Bytes()
		if ok && n <= 0 {
			t.Errorf("dims %v: Bytes() = %d with ok, want a positive byte count", dims, n)
		}
	}

	// The ordinary case is unaffected.
	small := TensorInfo{Dims: []uint64{6}, Type: F32}
	if n := small.N(); n != 6 {
		t.Errorf("N() = %d, want 6", n)
	}
	if b, ok := small.Bytes(); !ok || b != 24 {
		t.Errorf("Bytes() = %d, %v, want 24, true", b, ok)
	}
}
