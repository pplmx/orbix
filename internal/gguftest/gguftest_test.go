package gguftest

import (
	"strings"
	"testing"

	"os"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/llama"
	"github.com/pplmx/orbix/internal/tokenize"
)

// The fixture is only useful if the packages it feeds accept it, so these tests exercise
// the same entry points a checkpoint goes through: the reader, the tokenizer, and a
// forward pass. A fixture that parses but cannot run would move 74 tests from "skipped"
// to "failing", which is worse than leaving them skipped.

// readFile reads a fixture, for the determinism test.
func readFile(path string) ([]byte, error) { return os.ReadFile(path) }

func TestFixtureParses(t *testing.T) {
	f, err := gguf.Open(Path(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	arch, err := f.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch != "llama" {
		t.Errorf("Arch = %q, want llama", arch)
	}
	cfg, err := llama.LoadConfig(f)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.Layers != 2 || cfg.Hidden != 32 || cfg.Heads != 2 {
		t.Errorf("config = layers %d hidden %d heads %d, want 2/32/2",
			cfg.Layers, cfg.Hidden, cfg.Heads)
	}
	// The vocabulary includes the 256 byte-fallback pieces, because a SentencePiece model
	// always carries them and the embedding rows have to match the declared count.
	if want := len(DefaultVocab) + 256; cfg.VocabSize != want {
		t.Errorf("VocabSize = %d, want %d (default pieces plus 256 byte fallbacks)",
			cfg.VocabSize, want)
	}
	// The special ids must point into the vocabulary, or generation has no valid stop.
	if cfg.EOSID < 0 || cfg.EOSID >= cfg.VocabSize {
		t.Errorf("EOSID = %d, outside the vocabulary", cfg.EOSID)
	}
	if cfg.BOSID < 0 || cfg.BOSID >= cfg.VocabSize {
		t.Errorf("BOSID = %d, outside the vocabulary", cfg.BOSID)
	}
}

func TestFixtureTokenizes(t *testing.T) {
	f, err := gguf.Open(Path(t))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(f)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	// A round trip is the property the vocabulary exists for: the fixture's pieces have
	// to be enough that ordinary text encodes and decodes unchanged, or a test that
	// checks prompt construction cannot use it.
	for _, text := range []string{"once upon a time", "the time", "abc"} {
		ids := tok.Encode(text, false)
		if len(ids) == 0 {
			t.Errorf("Encode(%q) returned nothing", text)
			continue
		}
		if got := tok.Decode(ids); got != text {
			t.Errorf("round trip of %q = %q (ids %v)", text, got, ids)
		}
	}
}

func TestFixtureRuns(t *testing.T) {
	path := Path(t)
	m, err := llama.New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	cache, err := llama.NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}
	f, _ := gguf.Open(path)
	tok, _ := tokenize.New(f)
	logits, err := m.Prefill(cache, tok.Encode("once upon a time", true), 0)
	if err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	if len(logits) != cfg.VocabSize {
		t.Fatalf("%d logits, want %d", len(logits), cfg.VocabSize)
	}
	// Finite logits are the assertion that matters: a fixture whose residual stream
	// overflows would make every test that uses it compare infinities.
	for i, v := range logits {
		if v != v || v > 1e6 || v < -1e6 {
			t.Fatalf("logit[%d] = %v, not a usable value", i, v)
		}
	}
}

// TestFixtureIsDeterministic checks that two builds produce the same bytes, which is what
// makes a failure reproducible and lets a test compare two runs of the same fixture.
func TestFixtureIsDeterministic(t *testing.T) {
	a := Path(t)
	b := Path(t)
	rawA, err := readFile(a)
	if err != nil {
		t.Fatal(err)
	}
	rawB, err := readFile(b)
	if err != nil {
		t.Fatal(err)
	}
	if len(rawA) != len(rawB) {
		t.Fatalf("sizes differ: %d vs %d", len(rawA), len(rawB))
	}
	for i := range rawA {
		if rawA[i] != rawB[i] {
			t.Fatalf("fixtures differ at byte %d", i)
		}
	}
}

// TestModelShapeOptions covers the variations the engine and llama tests need: a wider
// model, a mixture of experts, and a chat template.
func TestModelShapeOptions(t *testing.T) {
	t.Run("wider", func(t *testing.T) {
		path := Model{Hidden: 64, Heads: 4, KVHeads: 2, FFN: 128}.Write(t, t.TempDir())
		m, err := llama.New(path)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if cfg := m.Config(); cfg.Hidden != 64 || cfg.KVHeads != 2 {
			t.Errorf("config = hidden %d kvHeads %d", cfg.Hidden, cfg.KVHeads)
		}
	})
	t.Run("mixture of experts", func(t *testing.T) {
		path := Model{Experts: 4, ExpertUsed: 2}.Write(t, t.TempDir())
		m, err := llama.New(path)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		cfg := m.Config()
		if cfg.Experts != 4 || cfg.ExpertUsed != 2 {
			t.Errorf("config = experts %d used %d", cfg.Experts, cfg.ExpertUsed)
		}
		cache, _ := llama.NewKVCache(cfg, cfg.ContextLen)
		if _, err := m.Forward(cache, 4, 0); err != nil {
			t.Fatalf("Forward: %v", err)
		}
	})
	t.Run("chat template", func(t *testing.T) {
		const tmpl = `{% for m in messages %}{{ m.role }}: {{ m.content }}` + "\n" + `{% endfor %}`
		path := Model{ChatTemplate: tmpl}.Write(t, t.TempDir())
		f, err := gguf.Open(path)
		if err != nil {
			t.Fatalf("Open: %v", err)
		}
		got, ok := f.Get[string]("tokenizer.chat_template")
		if !ok || got != tmpl {
			t.Errorf("chat template = %q, present %v", got, ok)
		}
	})
	t.Run("context length", func(t *testing.T) {
		path := Model{ContextLen: 128}.Write(t, t.TempDir())
		f, _ := gguf.Open(path)
		cfg, err := llama.LoadConfig(f)
		if err != nil {
			t.Fatalf("LoadConfig: %v", err)
		}
		if cfg.ContextLen != 128 {
			t.Errorf("ContextLen = %d, want 128", cfg.ContextLen)
		}
	})
}

// TestModelRejectsBadShapes keeps a misconfigured fixture from producing a file that
// fails later with a slice-bounds panic somewhere unrelated.
func TestModelRejectsBadShapes(t *testing.T) {
	for name, m := range map[string]Model{
		"hidden not divisible by heads": {Hidden: 30, Heads: 4},
		"heads not divisible by kv":     {Hidden: 32, Heads: 4, KVHeads: 3},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := m.Bytes(); err == nil {
				t.Errorf("%s was accepted; Bytes should reject a shape the loader cannot run", name)
			}
		})
	}
}

// TestByteLevelVocabularySpelling checks the fixture spells a byte-level vocabulary the way
// that vocabulary expects.
//
// A byte-level model's pieces carry an escaped space (U+0120, "Ġ") where SentencePiece's
// carry a marker (U+2581, "▁"). The fixture emitted the marker either way, so a vocabulary
// built for a byte-level tokenizer contained pieces that tokenizer can never look up: every
// word fell back to one token per byte. Nothing failed -- the ids were valid, and the
// round trip still worked -- which is why this went unnoticed until a downstream test tried
// to compare two pre-tokenizers and got identical answers.
func TestByteLevelVocabularySpelling(t *testing.T) {
	bpe := Default()
	bpe.TokenizerModel = "gpt2"
	path := bpe.Write(t, t.TempDir())
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tokens, ok := f.GetArray[string]("tokenizer.ggml.tokens")
	if !ok {
		t.Fatal("no token list in the fixture")
	}
	sawEscaped := false
	for _, piece := range tokens {
		if strings.Contains(piece, "▁") {
			t.Errorf("byte-level vocabulary contains the SentencePiece marker in %q", piece)
		}
		if strings.Contains(piece, "Ġ") {
			sawEscaped = true
		}
	}
	if !sawEscaped {
		t.Error("byte-level vocabulary has no piece with an escaped space, so no word piece " +
			"can ever match")
	}

	// The SentencePiece spelling is unchanged.
	spm := Default()
	path = spm.Write(t, t.TempDir())
	f, err = gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tokens, _ = f.GetArray[string]("tokenizer.ggml.tokens")
	sawMarker := false
	for _, piece := range tokens {
		if strings.Contains(piece, "Ġ") {
			t.Errorf("SentencePiece vocabulary contains the byte-level escape in %q", piece)
		}
		if strings.Contains(piece, "▁") {
			sawMarker = true
		}
	}
	if !sawMarker {
		t.Error("SentencePiece vocabulary has no space marker")
	}
}
