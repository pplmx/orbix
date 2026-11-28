package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
	"github.com/pplmx/orbix/internal/llama"
)

// ggufOpens counts successful gguf.Open calls made while fn runs.
//
// The counter in internal/gguf is process-wide, so this reads a delta rather than an absolute:
// other tests in this package may have opened files already, and an absolute count would make
// this test's result depend on which tests ran before it.
func ggufOpens(t *testing.T, fn func()) int64 {
	t.Helper()
	before := gguf.OpenCount()
	fn()
	return gguf.OpenCount() - before
}

// TestOpenModelParsesTheFileOnce is the property the shared assembly exists for.
//
// Before cmd/orbix had one entry point, `serve` called gguf.Open twice -- once to check
// --block-size against the context, once for the tokenizer -- and engine.New reached through
// llama.NewWithOptions into quant.OpenLoader for a third parse of the same directory. Each parse
// is real work: measured on the 553 MiB Llama 3.2 file, one Open takes 1.47 s.
//
// openModel has to do exactly one. The assertion is on the count rather than on timing because a
// count is deterministic and a duration on this host is not, and because the count is the thing
// that was wrong.
func TestOpenModelParsesTheFileOnce(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 128}.Write(t, t.TempDir())

	var m *modelFiles
	n := ggufOpens(t, func() {
		var err error
		m, err = openModel(path)
		if err != nil {
			t.Fatalf("openModel: %v", err)
		}
	})
	if n != 1 {
		t.Errorf("openModel parsed the file %d times, want 1", n)
	}
	// The parts it hands back have to come from that one parse, so a caller that reads the
	// config and the tokenizer is reading the same file rather than two views of it.
	if m.cfg.ContextLen != 128 {
		t.Errorf("config context = %d, want the file's 128", m.cfg.ContextLen)
	}
	if m.tok == nil || m.renderer == nil {
		t.Error("openModel returned without a tokenizer or renderer")
	}
}

// TestCommandsParseTheMetadataOnce covers the three commands that were rewired onto openModel.
//
// The count is two, not one, and the difference is structural rather than an oversight:
//
//   - openModel parses the metadata directory once, for the config, the tokenizer and the
//     template;
//   - engine.New -> llama.LoadWeights -> quant.OpenLoader parses it again and opens a second
//     handle, because reading tensor bytes needs a reader positioned at the data section while
//     the metadata stays available.
//
// What the refactor removed is the *extra* metadata parse a command did to answer one question
// it could have answered from the config it was going to load anyway.
//
//   - serve called gguf.Open once to test --block-size against the context and again for the
//     tokenizer, plus the weight loader's -- so three parses, now two.
//   - bench opened once and checked --block-size from that same *File, so it was already at two
//     and stays there; the change for bench is that the check now lives in planCache with the
//     same condition serve uses, not that it parses less.
//   - chat opened once and called llama.LoadConfig on it for the same check, so it was also at
//     two. Its change is that the hardcoded 16 and the chat-local cfgForBlocks are gone.
//
// So the number to hold is "no command parses the metadata more than once for its own use",
// which is 2 with the weight loader counted. The earlier version of this test asserted 1 for all
// three, which was wrong about engine.New -- caught by running it rather than by reading it.
func TestCommandsParseTheMetadataOnce(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 128}.Write(t, t.TempDir())

	const want = 2 // metadata for openModel, plus the weight loader's own

	// serve, with a prefix cache and a block wider than the context: that combination makes
	// serve open the model, derive the config and refuse -- after the parse, before the listen.
	// The wider block is what stops it, so the invocation cannot start a server and the test
	// cannot block.
	//
	// A malformed --addr would be the wrong vehicle: that check runs BEFORE the model is opened
	// (deliberately, so a typo costs no load), so the model would never be reached and the count
	// would be 0 for reasons that say nothing about the assembly.
	t.Run("serve", func(t *testing.T) {
		n := ggufOpens(t, func() {
			err := cmdServe([]string{"--model", path, "--addr", "127.0.0.1:0",
				"--prefix-cache", "--blocks", "8", "--block-size", "1000"})
			if err == nil {
				t.Fatal("serve accepted a block wider than the context, so it ran to completion")
			}
		})
		// One: the metadata parse. serve refuses before engine.New, so the weight loader never
		// runs and the reference count is 1 here rather than the 2 bench and chat show.
		if n != 1 {
			t.Errorf("serve parsed the model %d times before refusing, want 1", n)
		}
	})

	t.Run("bench", func(t *testing.T) {
		n := ggufOpens(t, func() {
			err := cmdBench([]string{"--model", path, "--max-tokens", "1",
				"--prefix-cache", "--blocks", "8", "--block-size", "16"})
			if err != nil {
				t.Fatalf("cmdBench: %v", err)
			}
		})
		if n != want {
			t.Errorf("bench parsed the model %d times, want %d", n, want)
		}
	})

	t.Run("chat", func(t *testing.T) {
		n := ggufOpens(t, func() {
			var out bytes.Buffer
			err := chatSession(strings.NewReader("/exit\n"), &out, chatOptions{
				model:     path,
				maxTokens: 2,
				sampling:  samplingFlags{topK: 40},
			})
			if err != nil {
				t.Fatalf("chatSession: %v", err)
			}
		})
		if n != want {
			t.Errorf("chat parsed the model %d times, want %d", n, want)
		}
	})
}

// TestPlanCacheSkipsTheBlockGuardWhenThereIsNoPool pins the condition serve and bench disagreed on.
//
// The guard's purpose is narrow: a block wider than the context never fills, so the prefix index
// never publishes it. That only matters when blocks exist. serve used to require both --blocks
// and --block-size before checking, bench checked on --block-size alone, so a command line with
// --prefix-cache --block-size 1000 and no --blocks was accepted by bench's guard path and
// rejected by serve's -- two answers to one question.
func TestPlanCacheSkipsTheBlockGuardWhenThereIsNoPool(t *testing.T) {
	cfg := llama.Config{ContextLen: 64}

	t.Run("no pool accepts a wide block", func(t *testing.T) {
		blockSize, blocks := 1000, 0
		plan, err := planCache("serve", cfg, cacheFlags{blockSize: &blockSize, blocks: &blocks},
			false, true)
		if err != nil {
			t.Fatalf("a block size with no pool was refused: %v", err)
		}
		// The default still applies, so the plan is usable if a caller later asks for a pool.
		if plan.blocks != 0 {
			t.Errorf("blocks = %d, want 0: the caller asked for no pool", plan.blocks)
		}
	})

	t.Run("a pool reaches the guard", func(t *testing.T) {
		blockSize, blocks := 1000, 8
		_, err := planCache("serve", cfg, cacheFlags{blockSize: &blockSize, blocks: &blocks},
			false, true)
		if err == nil {
			t.Fatal("a block wider than the context was accepted with a pool")
		}
		// Both numbers and the flag, so the refusal is actionable.
		for _, want := range []string{"1000", "64", "64 or less", "--prefix-cache"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error %q does not mention %q", err, want)
			}
		}
	})

	t.Run("reuse off does not refuse", func(t *testing.T) {
		// Without --prefix-cache a wide block is just a pool with one block in practice, which
		// serves text correctly. Refusing it would reject a working configuration.
		blockSize, blocks := 1000, 8
		if _, err := planCache("serve", cfg,
			cacheFlags{blockSize: &blockSize, blocks: &blocks}, false, false); err != nil {
			t.Errorf("reuse off refused a wide block: %v", err)
		}
	})
}

// TestRunStillOpensTheModelSeparately records a known gap rather than hiding it.
//
// `run` was not moved onto openModel: it generates through llama.Model directly instead of
// through the engine, so it wants the weights and a contiguous cache rather than a scheduler,
// and giving it the engine would change what it measures. It therefore still opens the file with
// gguf.Open and again inside llama.NewWithOptions.
//
// This test exists so the number is a decision with a recorded value instead of an oversight: if
// someone later moves run onto the shared path, this fails and they update it deliberately.
func TestRunStillOpensTheModelSeparately(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 128}.Write(t, t.TempDir())

	n := ggufOpens(t, func() {
		err := cmdRun([]string{"--model", path, "--max-tokens", "1", "Once upon a time"})
		if err != nil {
			t.Fatalf("cmdRun: %v", err)
		}
	})
	// Two: the metadata parse and the weight loader's own. Documented in cmd/orbix/run.go as a
	// deliberate split -- the weight loader closes the handle it opens, so sharing the *File
	// would need the loader to accept one rather than a path.
	if n != 2 {
		t.Errorf("run parsed the model %d times, want the documented 2", n)
	}
}

// TestOpenModelRefusesAMissingFile keeps the one-parse claim from hiding a swallowing error.
func TestOpenModelRefusesAMissingFile(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "nope.gguf")
	before := gguf.OpenCount()
	if _, err := openModel(missing); err == nil {
		t.Fatal("openModel accepted a path that does not exist")
	}
	// A failed Open is not counted, so the counter means "parses", not "attempts".
	if got := gguf.OpenCount() - before; got != 0 {
		t.Errorf("a failed open was counted %d times", got)
	}
}
