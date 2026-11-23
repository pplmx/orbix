package llama

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestGenerateReferenceInputs writes the fixtures and this engine's logits for each spec, so a
// numpy implementation can reproduce them and the results can be written into
// testdata/reference/logits.json.
//
// It is skipped unless ORBIX_REFERENCE_DIR is set, so an ordinary run never writes files. See
// docs/models.md for what the numpy side has to do with them.
func TestGenerateReferenceInputs(t *testing.T) {
	dir := os.Getenv("ORBIX_REFERENCE_DIR")
	if dir == "" {
		t.Skip("set ORBIX_REFERENCE_DIR=<dir> to write the fixtures for the numpy reference")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("create %s: %v", dir, err)
	}

	out := make(map[string]referenceResult, len(referenceSpecs))
	for _, spec := range referenceSpecs {
		spec := spec
		t.Run(spec.name, func(t *testing.T) {
			raw, err := spec.model().Bytes()
			if err != nil {
				t.Fatalf("Bytes: %v", err)
			}
			ggufPath := filepath.Join(dir, spec.name+".gguf")
			if err := os.WriteFile(ggufPath, raw, 0o644); err != nil {
				t.Fatalf("write fixture: %v", err)
			}

			m, err := New(ggufPath)
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			cache, err := NewKVCache(m.Config(), 128)
			if err != nil {
				t.Fatalf("NewKVCache: %v", err)
			}
			ids := spec.prompt
			var logits []float32
			for pos, id := range ids {
				logits, err = m.Forward(cache, int(id), pos)
				if err != nil {
					t.Fatalf("Forward at %d: %v", pos, err)
				}
			}

			// Both forms: a flat binary a numpy script can read without a JSON parser, and the
			// JSON this package's test consumes once the numpy side has filled it in.
			writeReferenceBin(t, filepath.Join(dir, spec.name+".bin"), ids, logits)
			r := referenceResult{Ids: ids, Logits: make([]float64, len(logits))}
			for i := range logits {
				r.Logits[i] = float64(logits[i])
			}
			out[spec.name] = r
			t.Logf("%s: %d prompt tokens, %d logits", spec.name, len(ids), len(logits))
		})
	}

	b, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "engine.json"), append(b, '\n'), 0o644); err != nil {
		t.Fatalf("write engine.json: %v", err)
	}
}

func writeReferenceBin(t *testing.T, path string, ids []int32, logits []float32) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create %s: %v", path, err)
	}
	defer func() { _ = f.Close() }()
	putInt32s(f, []int32{int32(len(ids))})
	putInt32s(f, ids)
	putInt32s(f, []int32{int32(len(logits))})
	for _, x := range logits {
		if err := binary.Write(f, binary.LittleEndian, math.Float32bits(x)); err != nil {
			t.Fatalf("write: %v", err)
		}
	}
}

func putInt32s(f *os.File, v []int32) {
	for _, x := range v {
		if err := binary.Write(f, binary.LittleEndian, x); err != nil {
			panic(err)
		}
	}
}
