package engine

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
	"github.com/pplmx/orbix/internal/llama"
	"github.com/pplmx/orbix/internal/tokenize"
)

// These tests check that two sequences interleaved on one model produce the same tokens
// they would alone. That failure mode is a shared scratch buffer, and a model has one
// whatever its weights are -- a constructed fixture has exactly as many buffers as a
// checkpoint does, so it exposes the same pollution. They run on a fixture, which is what
// lets them run everywhere.

func loadModel(t *testing.T) (*Engine, *tokenize.Tokenizer) {
	t.Helper()
	e, tok, _ := loadModelPath(t, Options{})
	return e, tok
}

// loadModelPath also returns the fixture's path, for a test that needs to build a second
// engine over the same weights.
func loadModelPath(t *testing.T, opts Options) (*Engine, *tokenize.Tokenizer, string) {
	t.Helper()
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	e, tok := loadModelAt(t, path, opts)
	return e, tok, path
}

// loadModelAt is loadModel over a fixture and options a test chooses.
func loadModelAt(t *testing.T, path string, opts Options) (*Engine, *tokenize.Tokenizer) {
	t.Helper()
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	e, err := New(path, opts)
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	return e, tok
}

// collect drains a sequence's channel.
func collect(t *testing.T, ch <-chan Event) ([]int32, string) {
	t.Helper()
	var out []int32
	reason := ""
	for ev := range ch {
		if ev.Err != nil {
			t.Fatalf("sequence failed: %v", ev.Err)
		}
		if ev.HasToken {
			out = append(out, ev.Token)
		}
		if ev.Done {
			reason = ev.FinishReason
		}
	}
	return out, reason
}

func TestSubmitRejectsBadRequests(t *testing.T) {
	e, tok := loadModel(t)
	cfg := e.Model().Config()

	if _, err := e.Submit(Request{}); err == nil {
		t.Error("Submit accepted an empty prompt")
	}
	long := make([]int32, cfg.ContextLen+1)
	if _, err := e.Submit(Request{Prompt: long, MaxTokens: 1}); err == nil {
		t.Error("Submit accepted a prompt longer than the context")
	}
	if _, err := e.Submit(Request{Prompt: tok.Encode("x", true), MaxTokens: -1}); err == nil {
		t.Error("Submit accepted a negative token budget")
	}
}

func TestMaxTokensRespected(t *testing.T) {
	e, tok := loadModel(t)
	ch, err := e.Submit(Request{Prompt: tok.Encode("Once upon a time", true), MaxTokens: 5})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	for e.Pump() > 0 {
	}
	out, reason := collect(t, ch)
	if len(out) != 5 {
		t.Errorf("generated %d tokens, want 5", len(out))
	}
	if reason != "length" {
		t.Errorf("finish reason = %q, want length", reason)
	}
}

func TestInterleavingMatchesSequentialResults(t *testing.T) {
	// The core correctness property: a sequence's tokens must not depend on what
	// else is in flight. Running two requests together and alone must give the
	// same output for each.
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	gf, _ := gguf.Open(path)
	tok, _ := tokenize.New(gf)

	prompts := []string{"Once upon a time", "The little girl"}
	tokens := make([][]int32, len(prompts))
	for i, p := range prompts {
		tokens[i] = tok.Encode(p, true)
	}

	run := func(parallel bool) [][]int32 {
		e, err := New(path, Options{})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		results := make([][]int32, len(prompts))
		if parallel {
			chans := make([]<-chan Event, len(prompts))
			for i := range prompts {
				ch, err := e.Submit(Request{Prompt: tokens[i], MaxTokens: 10})
				if err != nil {
					t.Fatalf("Submit: %v", err)
				}
				chans[i] = ch
			}
			for e.Pump() > 0 {
			}
			for i, ch := range chans {
				results[i], _ = collect(t, ch)
			}
			return results
		}
		for i := range prompts {
			ch, err := e.Submit(Request{Prompt: tokens[i], MaxTokens: 10})
			if err != nil {
				t.Fatalf("Submit: %v", err)
			}
			for e.Pump() > 0 {
			}
			results[i], _ = collect(t, ch)
		}
		return results
	}

	alone := run(false)
	together := run(true)
	for i := range prompts {
		if len(alone[i]) != len(together[i]) {
			t.Fatalf("seq %d: %d tokens alone, %d together", i, len(alone[i]), len(together[i]))
		}
		for j := range alone[i] {
			if alone[i][j] != together[i][j] {
				t.Errorf("seq %d token %d: %d alone, %d together (%q vs %q)",
					i, j, alone[i][j], together[i][j],
					tok.Decode(alone[i]), tok.Decode(together[i]))
				break
			}
		}
	}
}

func TestStopToken(t *testing.T) {
	// One fixture, two engines: the first learns the greedy continuation, the second
	// checks that stopping on it ends the request. Sharing the path is what makes the
	// second engine's tokens comparable to the first's.
	e, tok, path := loadModelPath(t, Options{})
	prompt := tok.Encode("Once upon a time", true)
	// First learn what the greedy continuation is, then stop on its second token.
	ch, _ := e.Submit(Request{Prompt: prompt, MaxTokens: 3})
	for e.Pump() > 0 {
	}
	full, _ := collect(t, ch)
	if len(full) < 2 {
		t.Skip("model produced too few tokens to test stopping")
	}
	stopAt := full[1]

	e2, _ := loadModelAt(t, path, Options{})
	ch2, err := e2.Submit(Request{Prompt: prompt, MaxTokens: 10, Stop: []int32{stopAt}})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	for e2.Pump() > 0 {
	}
	out, reason := collect(t, ch2)
	if reason != "stop" {
		t.Errorf("finish reason = %q, want stop", reason)
	}
	// The stop token itself is not emitted.
	for _, id := range out {
		if id == stopAt {
			t.Errorf("output contains the stop token %d", stopAt)
		}
	}
}

func TestPumpReportsActiveCount(t *testing.T) {
	e, tok := loadModel(t)
	h, err := e.Submit(Request{Prompt: tok.Encode("Once", true), MaxTokens: 100})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	// The generator must run to completion somewhere; drain it in a goroutine
	// so Pump never blocks writing to an unread channel.
	go func() {
		for range h {
		}
	}()
	rounds := 0
	for e.Pump() > 0 {
		rounds++
		if rounds > 500 {
			t.Fatal("Pump never drained the sequence")
		}
	}
	if e.Active() != 0 {
		t.Errorf("Active = %d after draining, want 0", e.Active())
	}
}

func TestGreedySamplerDefault(t *testing.T) {
	// A nil sampler must behave like Greedy, so callers do not have to know the
	// default.
	e, tok, path := loadModelPath(t, Options{})
	ch, err := e.Submit(Request{Prompt: tok.Encode("Once upon a time", true), MaxTokens: 4})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	for e.Pump() > 0 {
	}
	got, _ := collect(t, ch)

	// A second engine over the same weights, to hold the explicit sampler against the
	// implicit one.
	e2, _ := loadModelAt(t, path, Options{})
	ch2, err := e2.Submit(Request{
		Prompt:    tok.Encode("Once upon a time", true),
		MaxTokens: 4,
		Sampler:   llama.Greedy{},
	})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	for e2.Pump() > 0 {
	}
	want, _ := collect(t, ch2)

	if len(got) != len(want) {
		t.Fatalf("nil sampler produced %d tokens, greedy produced %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("token %d: nil sampler gave %d, greedy gave %d", i, got[i], want[i])
		}
	}
}

// TestEngineMatchesGenerate compares the engine's output with the single-stream
// llama.Generate path over a prompt long enough to exercise batched prefill.
// They share the model but not the loop, so agreement means the engine's
// batched prefill and per-round decode reproduce the straightforward path.
func TestEngineMatchesGenerate(t *testing.T) {
	// The comparison is against llama.Generate over the same weights, so the path has to
	// travel with the engine.
	e, tok, path := loadModelPath(t, Options{})
	cfg := e.Model().Config()
	prompt := tok.Encode("Once upon a time there was a little girl named Lily who lived near the river.", cfg.BOSID >= 0)
	if len(prompt) < 8 {
		t.Skipf("prompt tokenized to only %d tokens", len(prompt))
	}
	const maxTokens = 8

	ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: maxTokens})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	for e.Pump() > 0 {
	}
	got, _ := collect(t, ch)

	m, err := llama.New(path)
	if err != nil {
		t.Fatalf("llama.New: %v", err)
	}
	cache, err := llama.NewKVCache(cfg, cfg.ContextLen)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}
	out, err := m.Generate(cache, prompt, llama.Greedy{}, llama.GenerateOptions{MaxTokens: maxTokens})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	want := out[len(prompt):]

	if len(got) != len(want) {
		t.Fatalf("engine produced %d tokens, Generate produced %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token %d: engine %d, Generate %d (%q vs %q)",
				i, got[i], want[i], tok.Decode(got), tok.Decode(want))
		}
	}
}

// TestWorkersOptionReachesTheModel checks the plumbing: the engine's Workers
// option must configure the model's kernels, or a caller tuning concurrency
// would silently get the default instead.
func TestWorkersOptionReachesTheModel(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	e, err := New(path, Options{Workers: 1})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// A serial engine and a parallel one must still generate the same tokens;
	// the option changes the schedule, not the numbers.
	gf, _ := gguf.Open(path)
	tok, _ := tokenize.New(gf)
	prompt := tok.Encode("Once upon a time", true)

	run := func(e *Engine) []int32 {
		ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: 5})
		if err != nil {
			t.Fatalf("Submit: %v", err)
		}
		for e.Pump() > 0 {
		}
		out, _ := collect(t, ch)
		return out
	}
	serial := run(e)

	pe, err := New(path, Options{Workers: 8})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	parallel := run(pe)
	if len(serial) != len(parallel) {
		t.Fatalf("serial produced %d tokens, parallel %d", len(serial), len(parallel))
	}
	for i := range serial {
		if serial[i] != parallel[i] {
			t.Errorf("token %d: serial %d, parallel %d", i, serial[i], parallel[i])
		}
	}
}

// TestMaxActiveQueuesExcessRequests checks the cap actually applies. Before the
// batched decode existed MaxActive was stored but never consulted, so a caller
// could set it to 2 and still have all eight requests advance in one round.
func TestMaxActiveQueuesExcessRequests(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	gf, _ := gguf.Open(path)
	tok, _ := tokenize.New(gf)
	prompt := tok.Encode("Once upon a time", true)

	e, err := New(path, Options{MaxActive: 2, Workers: 1})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	const requests = 5
	chans := make([]<-chan Event, requests)
	for i := range chans {
		ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: 3})
		if err != nil {
			t.Fatalf("Submit: %v", err)
		}
		chans[i] = ch
	}
	if got := e.Active(); got != requests {
		t.Fatalf("Active = %d, want %d", got, requests)
	}

	// Drain concurrently: five sequences x three tokens exceeds the channel
	// buffer, so Pump would block if nobody is reading.
	results := make([][]int32, requests)
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i, ch := range chans {
			for ev := range ch {
				if ev.Err != nil {
					t.Errorf("sequence %d: %v", i, ev.Err)
				}
				if ev.HasToken {
					results[i] = append(results[i], ev.Token)
				}
			}
		}
	}()
	for e.Pump() > 0 {
	}
	<-done
	for i := range results {
		if len(results[i]) != 3 {
			t.Errorf("sequence %d produced %d tokens, want 3", i, len(results[i]))
		}
	}
}

// TestMaxActiveZeroDefaultsToEight pins the default, which is otherwise only
// visible by reading the constructor.
func TestMaxActiveZeroDefaultsToEight(t *testing.T) {
	e, _ := loadModel(t)
	if got := e.MaxActive(); got != 8 {
		t.Errorf("default MaxActive = %d, want 8", got)
	}
}

// TestPagedEngineMatchesContiguous is the property that matters for switching a
// deployment to paged caches: the tokens must not change. It runs the same
// requests through both layouts and requires identical output.
func TestPagedEngineMatchesContiguous(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	gf, _ := gguf.Open(path)
	tok, _ := tokenize.New(gf)
	prompts := []string{"Once upon a time", "The little girl was"}
	const maxTokens = 6

	run := func(opts Options) [][]int32 {
		e, err := New(path, opts)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		chans := make([]<-chan Event, len(prompts))
		for i, p := range prompts {
			ch, err := e.Submit(Request{Prompt: tok.Encode(p, true), MaxTokens: maxTokens})
			if err != nil {
				t.Fatalf("Submit: %v", err)
			}
			chans[i] = ch
		}
		for e.Pump() > 0 {
		}
		out := make([][]int32, len(prompts))
		for i, ch := range chans {
			out[i], _ = collect(t, ch)
		}
		return out
	}

	want := run(Options{})
	// A pool with room to spare, and a block size that does not divide the prompt
	// lengths, so the last block of every sequence is partly filled.
	got := run(Options{BlockSize: 5, Blocks: 256})

	for i := range want {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("sequence %d: paged produced %d tokens, contiguous %d", i, len(got[i]), len(want[i]))
		}
		for j := range want[i] {
			if got[i][j] != want[i][j] {
				t.Fatalf("sequence %d token %d: paged %d, contiguous %d", i, j, got[i][j], want[i][j])
			}
		}
	}
}

// TestPagedEngineReservesLessMemory is the reason paging exists. A queued request
// must not reserve a whole context: the pool is a fixed budget, and the sequences
// draw from it as they generate.
func TestPagedEngineReservesLessMemory(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	gf, _ := gguf.Open(path)
	tok, _ := tokenize.New(gf)

	// A pool sized for about one context. It has to cover the sequences that are
	// *active* (the engine advances at most MaxActive per round), not every queued
	// one: that is the whole point of sharing a budget rather than reserving per
	// request.
	probe, err := New(path, Options{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	mcfg := probe.Model().Config()
	const blockSize = 16
	const maxActive = 4
	// 12 positions per sequence (a 4-token prompt plus 3 generated, rounded up)
	// across maxActive sequences, with slack for the block the next token needs.
	// A block id covers every layer at once, so the pool counts position slots.
	perSeq := (len(tok.Encode("Once upon a time", true)) + 3 + blockSize - 1) / blockSize
	blocks := maxActive * (perSeq + 1)
	e, err := New(path, Options{BlockSize: blockSize, Blocks: blocks, MaxActive: maxActive})
	if err != nil {
		t.Fatalf("New with pool: %v", err)
	}
	if e.Pool() == nil {
		t.Fatal("Pool() is nil for a paged engine")
	}
	poolBytes := e.Pool().Bytes()

	prompt := tok.Encode("Once upon a time", true)
	const queued = 32
	chans := make([]<-chan Event, queued)
	for i := range chans {
		ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: 3})
		if err != nil {
			t.Fatalf("Submit: %v", err)
		}
		chans[i] = ch
	}
	// Every request is admitted and runs to completion out of one context-sized
	// pool, which is impossible with whole-context reservations: 32 sequences
	// would need 32 contexts.
	for e.Pump() > 0 {
	}
	for i, ch := range chans {
		out, reason := collect(t, ch)
		if len(out) != 3 {
			t.Errorf("sequence %d produced %d tokens, want 3", i, len(out))
		}
		if reason != "length" {
			t.Errorf("sequence %d finish reason = %q, want length", i, reason)
		}
	}

	// The pool is the whole cache budget, so its size is the number the contiguous
	// layout would have paid per sequence.
	perSequence := int64(mcfg.ContextLen) * int64(mcfg.KVHeads) * int64(mcfg.HeadDim) * 2 * 4 * int64(mcfg.Layers)
	if poolBytes > perSequence {
		t.Errorf("pool is %d bytes, larger than one whole-context cache (%d)", poolBytes, perSequence)
	}
	t.Logf("pool %.2f MiB served %d sequences that would each have reserved %.2f MiB",
		float64(poolBytes)/(1<<20), queued, float64(perSequence)/(1<<20))
}

// TestBlockPoolDefaultIsClamped checks the engine keeps working when a caller asks
// for paging but leaves the block size at its zero value.
func TestBlockPoolDefaultIsClamped(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	e, err := New(path, Options{Blocks: 8})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := e.Pool().BlockSize(); got <= 0 {
		t.Errorf("BlockSize = %d, want a positive default", got)
	}
}

// TestCancelStopsASequence checks that a cancelled request stops generating and
// releases its resources. Without this, a client that disconnects would keep the
// model busy generating into a channel nobody reads.
func TestCancelStopsASequence(t *testing.T) {
	e, tok := loadModel(t)

	prompt := tok.Encode("Once upon a time", true)
	cancel := make(chan struct{})
	ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: 1000, Cancel: cancel})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	var generated int
	var reason string
	// Run a few rounds, then cancel.
	for range 3 {
		if e.Pump() == 0 {
			break
		}
	}
	close(cancel)

	deadline := 200
	for e.Pump() > 0 && deadline > 0 {
		deadline--
	}
	if e.Active() != 0 {
		t.Fatalf("cancelled sequence is still active (%d)", e.Active())
	}
	for ev := range ch {
		if ev.Err != nil {
			t.Fatalf("cancelled sequence reported an error: %v", ev.Err)
		}
		if ev.HasToken {
			generated++
		}
		if ev.Done {
			reason = ev.FinishReason
		}
	}
	if reason != "cancel" {
		t.Errorf("finish reason = %q, want cancel", reason)
	}
	// Cancel has to arrive quickly: the sequence must not have run out its budget.
	if generated >= 1000 {
		t.Errorf("generated %d tokens despite cancellation", generated)
	}
}

// TestCancelBeforeStartNeverGenerates checks the boundary: a request cancelled
// before any round must not produce a token.
func TestCancelBeforeStartNeverGenerates(t *testing.T) {
	e, tok := loadModel(t)
	cancel := make(chan struct{})
	close(cancel)
	ch, err := e.Submit(Request{Prompt: tok.Encode("Once", true), MaxTokens: 10, Cancel: cancel})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	for e.Pump() > 0 {
	}
	out, reason := collect(t, ch)
	if len(out) != 0 {
		t.Errorf("cancelled request produced %d tokens", len(out))
	}
	if reason != "cancel" {
		t.Errorf("finish reason = %q, want cancel", reason)
	}
}

// TestNilCancelNeverFires pins that the common case is unaffected: no cancel
// channel means no cancellation, and the request runs its whole budget.
func TestNilCancelNeverFires(t *testing.T) {
	e, tok := loadModel(t)
	ch, err := e.Submit(Request{Prompt: tok.Encode("Once", true), MaxTokens: 4})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	for e.Pump() > 0 {
	}
	out, reason := collect(t, ch)
	if len(out) != 4 {
		t.Errorf("produced %d tokens, want 4", len(out))
	}
	if reason != "length" {
		t.Errorf("finish reason = %q, want length", reason)
	}
}

// TestPrefixCacheMatchesNoCache is the property that makes prefix reuse safe to
// turn on: a request's tokens must not change. It runs the same prompts through an
// engine with prefix reuse and one without, several times over, so the second round
// is served from a warm index.
func TestPrefixCacheMatchesNoCache(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	gf, _ := gguf.Open(path)
	tok, _ := tokenize.New(gf)
	prompts := []string{
		"Once upon a time",
		"Once upon a time, there was a little girl",
		"Once upon a time, there was a little boy",
		"The little girl",
	}
	const maxTokens = 5

	run := func(opts Options) [][]int32 {
		e, err := New(path, opts)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		out := make([][]int32, len(prompts))
		// Two rounds: the first fills the index, the second is served from it.
		for round := range 2 {
			chans := make([]<-chan Event, len(prompts))
			for i, p := range prompts {
				ch, err := e.Submit(Request{Prompt: tok.Encode(p, true), MaxTokens: maxTokens})
				if err != nil {
					t.Fatalf("Submit: %v", err)
				}
				chans[i] = ch
			}
			for e.Pump() > 0 {
			}
			for i, ch := range chans {
				tokens, _ := collect(t, ch)
				if round == 0 {
					out[i] = tokens
					continue
				}
				if len(tokens) != len(out[i]) {
					t.Fatalf("round 2 prompt %d produced %d tokens, round 1 produced %d", i, len(tokens), len(out[i]))
				}
				for j := range tokens {
					if tokens[j] != out[i][j] {
						t.Errorf("prompt %d token %d: round 2 %d, round 1 %d", i, j, tokens[j], out[i][j])
					}
				}
			}
		}
		return out
	}

	plain := run(Options{BlockSize: 4, Blocks: 256})
	cached := run(Options{BlockSize: 4, Blocks: 256, PrefixCache: true})
	for i := range plain {
		if len(cached[i]) != len(plain[i]) {
			t.Fatalf("prompt %d: prefix cache produced %d tokens, plain produced %d", i, len(cached[i]), len(plain[i]))
		}
		for j := range plain[i] {
			if cached[i][j] != plain[i][j] {
				t.Errorf("prompt %d token %d: prefix cache %d, plain %d", i, j, cached[i][j], plain[i][j])
			}
		}
	}
}

// TestPrefixCacheReusesAcrossRequests checks reuse actually happens at the engine
// level, by observing that the index filled up.
func TestPrefixCacheReusesAcrossRequests(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	gf, _ := gguf.Open(path)
	tok, _ := tokenize.New(gf)
	e, err := New(path, Options{BlockSize: 4, Blocks: 256, PrefixCache: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if e.Pool().PrefixIndex() == nil {
		t.Fatal("PrefixCache option did not enable the index")
	}
	prompt := tok.Encode("Once upon a time there was a little girl named Lily", true)
	for range 2 {
		ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: 3})
		if err != nil {
			t.Fatalf("Submit: %v", err)
		}
		for e.Pump() > 0 {
		}
		if _, reason := collect(t, ch); reason != "length" {
			t.Errorf("finish reason = %q, want length", reason)
		}
	}
	if got := e.Pool().PrefixIndex().Len(); got == 0 {
		t.Error("index remembered no blocks across two identical requests")
	}
}

// TestPrefixCacheOffByDefault pins that the optimization is opt-in: a plain paged
// engine must not allocate an index or retain blocks it cannot use.
func TestPrefixCacheOffByDefault(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	e, err := New(path, Options{BlockSize: 4, Blocks: 32})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if idx := e.Pool().PrefixIndex(); idx != nil {
		t.Errorf("PrefixIndex() = %+v without PrefixCache", idx)
	}
}

// TestRenderChatPrompt pins the shared template: the REPL and the HTTP server both go
// through it, so its exact shape is what keeps the same conversation producing the
// same tokens however it was asked.
func TestRenderChatPrompt(t *testing.T) {
	got := RenderChatPrompt([]Turn{
		{Role: "system", Text: "be brief"},
		{Role: "user", Text: "  hi  "},
		{Role: "assistant", Text: "hello"},
		{Role: "", Text: "and?"},
	})
	want := "System: be brief\nUser: hi\nAssistant: hello\nUser: and?\nAssistant:"
	if got != want {
		t.Errorf("RenderChatPrompt =\n%q\nwant\n%q", got, want)
	}
	// An empty conversation still leaves the assistant's turn open, which is what a
	// prompt-only request needs.
	if got, want := RenderChatPrompt(nil), "Assistant:"; got != want {
		t.Errorf("RenderChatPrompt(nil) = %q, want %q", got, want)
	}
}

// TestUnreadEventsDoNotStallPump checks the engine survives a caller that stops reading.
//
// The event sends used to block. A test that drains the channel never notices, but the HTTP
// handlers return as soon as the response cannot be written -- a disconnected client, or an
// error path that reports and stops -- and nothing drains the rest. Pump is a loop shared by
// every request in the server, so one stalled sequence stops all of them: the whole process
// keeps accepting connections and generating nothing.
//
// The buffer absorbs 64 events, so this asks for more than that and never reads.
func TestUnreadEventsDoNotStallPump(t *testing.T) {
	e, tok, _ := loadModelPath(t, Options{MaxActive: 1})
	if _, err := e.Submit(Request{
		Prompt:    tok.Encode("Once upon a time", true),
		MaxTokens: 200,
	}); err != nil {
		t.Fatalf("Submit: %v", err)
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		for e.Pump() > 0 {
		}
	}()
	select {
	case <-done:
	case <-time.After(20 * time.Second):
		t.Fatal("Pump blocked on a channel nobody reads")
	}
	// The abandoned sequence must also be retired, or Active never reaches zero and the
	// server's pump loop never idles.
	if n := e.Active(); n != 0 {
		t.Errorf("Active() = %d after the pump drained, want 0", n)
	}
}

// TestReadingConsumerReceivesEveryToken is the other half: abandoning is for callers that
// have gone, and a caller that is merely slower than the pump must still get everything.
func TestReadingConsumerReceivesEveryToken(t *testing.T) {
	e, tok, _ := loadModelPath(t, Options{MaxActive: 1})
	const want = 100
	ch, err := e.Submit(Request{
		Prompt:    tok.Encode("Once upon a time", true),
		MaxTokens: want,
	})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}

	var got int
	finished := make(chan struct{})
	go func() {
		defer close(finished)
		for ev := range ch {
			if ev.HasToken {
				got++
			}
		}
	}()
	for e.Active() > 0 {
		e.Pump()
	}
	<-finished
	if got != want {
		t.Errorf("read %d tokens, want %d", got, want)
	}
}

// The block pool is reference counted, and a missed release shows up as a pool that
// shrinks over time rather than as a failing request: the first few cycles work, then
// requests start failing with "block pool exhausted" on a server with plenty of memory.
// A single lifecycle is therefore not enough to catch it, which is why these run many.

// TestPoolIsBalancedAcrossManyLifecycles checks that finishing requests returns every block
// they took.
func TestPoolIsBalancedAcrossManyLifecycles(t *testing.T) {
	e, tok, _ := loadModelPath(t, Options{MaxActive: 2, Blocks: 32, BlockSize: 8})
	pool := e.Pool()
	if pool == nil {
		t.Fatal("the paged cache was not configured, so this would test nothing")
	}
	start := pool.Free()
	for i := range 20 {
		ch, err := e.Submit(Request{Prompt: tok.Encode("Once upon a time", true), MaxTokens: 12})
		if err != nil {
			t.Fatalf("cycle %d Submit: %v", i, err)
		}
		// Drain and pump concurrently: events only exist because Pump makes them, so
		// reading first would wait forever.
		drained := make(chan struct{})
		go func() {
			defer close(drained)
			for range ch {
			}
		}()
		for e.Pump() > 0 {
		}
		<-drained
		if f := pool.Free(); f != start {
			t.Fatalf("cycle %d left %d of %d blocks free, want %d", i, f, pool.Blocks(), start)
		}
	}
}

// TestAbandonedSequenceReturnsItsBlocks checks the path a disconnected client takes. It is
// the one most likely to leak, because the sequence ends without its caller ever seeing a
// normal completion.
func TestAbandonedSequenceReturnsItsBlocks(t *testing.T) {
	e, tok, _ := loadModelPath(t, Options{MaxActive: 1, Blocks: 16, BlockSize: 4})
	pool := e.Pool()
	if pool == nil {
		t.Fatal("the paged cache was not configured")
	}
	start := pool.Free()
	// Never read the channel, so the sequence is abandoned mid-generation.
	if _, err := e.Submit(Request{
		Prompt:    tok.Encode("Once upon a time", true),
		MaxTokens: 200,
	}); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	for e.Pump() > 0 {
	}
	if f := pool.Free(); f != start {
		t.Errorf("an abandoned sequence left %d of %d blocks free, want %d",
			f, pool.Blocks(), start)
	}
	if n := e.Active(); n != 0 {
		t.Errorf("Active() = %d, want 0", n)
	}
}

// TestPoolExhaustionStillCompletes checks that a pool too small for the requests degrades
// into per-request errors rather than into a stuck engine. The engine reports exhaustion
// through the sequence's event stream, so the pump still reaches zero.
func TestPoolExhaustionStillCompletes(t *testing.T) {
	e, tok, _ := loadModelPath(t, Options{MaxActive: 4, Blocks: 2, BlockSize: 4})
	pool := e.Pool()
	if pool == nil {
		t.Fatal("the paged cache was not configured")
	}
	var chans []<-chan Event
	for range 4 {
		ch, err := e.Submit(Request{Prompt: tok.Encode("Once upon a time", true), MaxTokens: 20})
		if err != nil {
			continue
		}
		chans = append(chans, ch)
	}
	go func() {
		for _, ch := range chans {
			for range ch {
			}
		}
	}()
	done := make(chan struct{})
	go func() {
		defer close(done)
		for e.Pump() > 0 {
		}
	}()
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatalf("the pump stalled with %d sequences active", e.Active())
	}
	if f := pool.Free(); f != pool.Blocks() {
		t.Errorf("pool not restored after exhaustion: %d of %d free", f, pool.Blocks())
	}
}

// TestEmitDistinguishesSlowFromGone covers the matrix that the non-blocking send has to get
// right. Blocking unconditionally wedges the whole server on a caller that left; dropping
// unconditionally loses tokens for a caller that is merely behind, because a pump running
// flat out fills the 64-event buffer before a reading goroutine is scheduled even once.
//
// The cancel channel is what separates the two, so all four combinations are checked.
func TestEmitDistinguishesSlowFromGone(t *testing.T) {
	t.Run("reading, no cancel: gets every token", func(t *testing.T) {
		e, tok, _ := loadModelPath(t, Options{MaxActive: 1})
		const want = 100
		ch, err := e.Submit(Request{Prompt: tok.Encode("Once upon a time", true), MaxTokens: want})
		if err != nil {
			t.Fatalf("Submit: %v", err)
		}
		var got int
		finished := make(chan struct{})
		go func() {
			defer close(finished)
			for ev := range ch {
				if ev.HasToken {
					got++
				}
			}
		}()
		for e.Active() > 0 {
			e.Pump()
		}
		<-finished
		if got != want {
			t.Errorf("read %d tokens, want %d", got, want)
		}
	})

	t.Run("unread, no cancel: abandoned without stalling", func(t *testing.T) {
		e, tok, _ := loadModelPath(t, Options{MaxActive: 1})
		if _, err := e.Submit(Request{
			Prompt: tok.Encode("Once upon a time", true), MaxTokens: 200,
		}); err != nil {
			t.Fatalf("Submit: %v", err)
		}
		done := make(chan struct{})
		go func() {
			defer close(done)
			for e.Pump() > 0 {
			}
		}()
		select {
		case <-done:
		case <-time.After(20 * time.Second):
			t.Fatal("Pump blocked on a channel nobody reads and nobody cancelled")
		}
		if n := e.Active(); n != 0 {
			t.Errorf("Active() = %d, want 0", n)
		}
	})

	t.Run("unread, cancelled: cancel unblocks the send", func(t *testing.T) {
		e, tok, _ := loadModelPath(t, Options{MaxActive: 1})
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		if _, err := e.Submit(Request{
			Prompt:    tok.Encode("Once upon a time", true),
			MaxTokens: 500,
			Cancel:    ctx.Done(),
		}); err != nil {
			t.Fatalf("Submit: %v", err)
		}
		done := make(chan struct{})
		go func() {
			defer close(done)
			for e.Pump() > 0 {
			}
		}()
		// Let the pump fill the buffer first, so the cancel is what releases it.
		time.Sleep(50 * time.Millisecond)
		cancel()
		select {
		case <-done:
		case <-time.After(20 * time.Second):
			t.Fatal("cancel did not unblock a full event buffer")
		}
	})
}

// TestQueueFullIsRefused checks that a backlog deeper than the engine will hold is refused
// rather than accepted.
//
// A contiguous cache reserves its whole context at Submit, so an accepted request costs
// about 1.7MB for a small model before a round has looked at it. There was no limit: 400
// concurrent requests drove a running server from 146MB to 847MB of resident memory, and a
// client that simply kept sending could exhaust the process. With the bound in place the
// same 400 peak at 224MB.
func TestQueueFullIsRefused(t *testing.T) {
	const maxActive, maxQueued = 1, 2
	e, tok, _ := loadModelPath(t, Options{MaxActive: maxActive, MaxQueued: maxQueued})
	prompt := tok.Encode("Once upon a time", true)

	var accepted, refused int
	for range 10 {
		ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: 20})
		if err != nil {
			if !errors.Is(err, ErrQueueFull) {
				t.Fatalf("unexpected error: %v", err)
			}
			refused++
			continue
		}
		accepted++
		_ = ch
	}
	if refused == 0 {
		t.Error("every request was accepted, so the limit did nothing")
	}
	// One active plus the queue depth; anything beyond that would be waiting for a slot
	// that no round is going to reach.
	if want := maxActive + maxQueued; accepted != want {
		t.Errorf("accepted %d requests, want %d (MaxActive + MaxQueued)", accepted, want)
	}
}

// TestQueueFullHoldsUnderConcurrentSubmit covers the limit from several goroutines at once,
// which is the case that broke it.
//
// Submit used to read len(active) under the lock, release it, build the cache, and re-acquire
// to insert. Every goroutine arriving during that gap saw the same count, so all of them
// passed the check: measured with MaxActive 1 and MaxQueued 2, twelve concurrent submits were
// all accepted and twelve sequences ended up active. The sequential test above cannot see it,
// because with one caller no gap ever contains a second request.
//
// It is a TOCTOU rather than a data race -- each read of len(active) is guarded, and the bug is
// that the value outlives the guard -- so -race alone would not have found it either; the
// assertion is on the count.
// TestFullBufferWithLiveCancelDoesNotStallPump covers the caller that keeps the connection
// open and reads nothing.
//
// emit fell back to an unbounded send whenever the request had a cancel channel, on the
// reasoning that a cancelled request is a definite goodbye. It is not: a cancel channel says
// the caller *can* announce that it left, not that it will. A socket whose peer stops reading
// is precisely that shape -- TCP holds the connection, r.Context() stays live, and the HTTP
// handler blocks writing into a zero window -- so the sequence's buffer never drains and the
// cancel never fires. Pump then waits for a slot that cannot arrive, and because Pump is
// shared, every other request stops advancing with it.
//
// Measured before the fix, with a 1000-position fixture and MaxTokens 900 against an undrained
// channel and an un-fired cancel: Pump was still inside the round after five seconds, and the
// test had to be killed. Now it returns after abandonGrace.
//
// The assertion is a deadline rather than an exact duration, so a slower machine does not fail
// it while a regression to unbounded waiting still does.
// TestTerminalEventReportsGeneratedTotal pins what Generated means and where it is absent.
//
// The field exists so a consumer that stops reading early -- the streaming path breaks out at
// a stop string -- can still report how many tokens the model produced. Two things follow, and
// both are easy to get wrong: for a fully-read sequence it must equal the number of token
// events delivered, and for an abandoned one there is no terminal event at all to carry it.
//
// The second is not a defect to fix, it is what abandonment means -- emit refuses everything
// once the caller is judged gone -- but a consumer that assumed otherwise would wait for an
// event that never comes.
// TestFinishReasonStringsRoundTrip binds the two mappings that have to agree.
//
// FinishReason.String produces the label /metrics uses, and reasonIndex accepts the string
// finish is called with. They are written as two switches rather than one table, so a reason
// added to the enum without a matching case in reasonIndex would be counted in the total and
// silently missing from the split -- the sum test in internal/server would catch it, but only
// for traffic that actually reaches that reason, which for "error" is rare.
//
// Round-tripping every enum value is the property that makes the pair safe: string -> index ->
// string must return the input, and index -> string -> index must too.
func TestFinishReasonStringsRoundTrip(t *testing.T) {
	all := []FinishReason{
		FinishStop, FinishLength, FinishCancel, FinishAbandoned, FinishError,
	}
	if len(all) != int(FinishReasonCount) {
		t.Fatalf("FinishReasonCount is %d but this test lists %d reasons; add the new one here",
			FinishReasonCount, len(all))
	}

	seen := map[string]bool{}
	for _, r := range all {
		name := r.String()
		if name == "unknown" {
			t.Errorf("FinishReason(%d).String() = %q; every value below FinishReasonCount needs a name", r, name)
			continue
		}
		if seen[name] {
			t.Errorf("two reasons share the name %q", name)
		}
		seen[name] = true

		back, ok := reasonIndex(name)
		if !ok {
			t.Errorf("reasonIndex(%q) does not accept the name FinishReason(%d).String() produces", name, r)
			continue
		}
		if back != r {
			t.Errorf("%q maps back to FinishReason(%d), want (%d)", name, back, r)
		}
	}

	// And the other direction: a string that is not one of the names must not be accepted,
	// since finish would then count it nowhere.
	for _, bad := range []string{"", "stop ", "Stop", "content_filter", "unknown"} {
		if _, ok := reasonIndex(bad); ok {
			t.Errorf("reasonIndex(%q) accepted a name the enum does not define", bad)
		}
	}
}

// TestFinishReasonsMatchTheEventStream keeps the enum and the events in step.
//
// It restates the five strings rather than reading them out of engine.go, which is worth being
// honest about: the map below is a second copy, so a rename in finish does not fail this test
// on its own. What it does catch is the enum growing a reason the events never produce, or
// losing one they do -- the case where /metrics would report a label that no traffic reaches,
// or silently under-count. Pairing it with the round-trip test, which fails if the two switches
// disagree, covers the rename case indirectly: a renamed string in reasonIndex would stop
// matching String(), and that is the test that fires.
func TestFinishReasonsMatchTheEventStream(t *testing.T) {
	// Copied from the FinishReason values in engine.go; see the note above on what that costs.
	want := map[string]bool{"stop": true, "length": true, "cancel": true, "abandoned": true, "error": true}
	got := map[string]bool{}
	for r := FinishStop; r < FinishReasonCount; r++ {
		got[r.String()] = true
	}
	for name := range want {
		if !got[name] {
			t.Errorf("the event stream uses %q but the enum has no such reason", name)
		}
	}
	for name := range got {
		if !want[name] {
			t.Errorf("the enum defines %q but no event carries it", name)
		}
	}
}

func TestTerminalEventReportsGeneratedTotal(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 200}.Write(t, t.TempDir())

	t.Run("equals the delivered tokens", func(t *testing.T) {
		for _, budget := range []int{1, 5, 40} {
			e, tok := loadEngineAt(t, path, Options{MaxActive: 1})
			prompt := tok.Encode("Once upon a time", true)
			ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: budget})
			if err != nil {
				t.Fatalf("Submit: %v", err)
			}
			go func() {
				for e.Pump() > 0 {
				}
			}()
			delivered, generated := 0, -1
			for ev := range ch {
				if ev.HasToken {
					delivered++
				}
				if ev.Done {
					generated = ev.Generated
				}
			}
			if generated < 0 {
				t.Fatalf("budget %d: no terminal event carried a total", budget)
			}
			if delivered != generated {
				t.Errorf("budget %d: delivered %d tokens but Generated = %d",
					budget, delivered, generated)
			}
			if generated != budget {
				t.Errorf("budget %d: Generated = %d, want the whole budget", budget, generated)
			}
		}
	})

	t.Run("an abandoned sequence sends no terminal event", func(t *testing.T) {
		// Big context so the budget alone would keep it running; the test relies on the
		// abandon, not on running out of positions.
		long := gguftest.Model{NoEOS: true, ContextLen: 1000}.Write(t, t.TempDir())
		e, tok := loadEngineAt(t, long, Options{MaxActive: 1})
		prompt := tok.Encode("Once upon a time", true)

		ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: 900})
		if err != nil {
			t.Fatalf("Submit: %v", err)
		}
		go func() {
			for e.Pump() > 0 {
			}
		}()
		// Read a few, then stop long enough for the buffer to fill and emit's grace to retire
		// the sequence. The sleep is the load-bearing part: ranging over ch immediately after
		// would keep draining the buffer, emit would succeed again, and no abandon would
		// happen -- which is exactly how my first version of this subtest read 897 tokens and
		// saw a Done.
		for range 3 {
			<-ch
		}
		time.Sleep(3 * time.Second)
		sawDone, received := false, 0
		for ev := range ch {
			if ev.HasToken {
				received++
			}
			if ev.Done {
				sawDone = true
			}
		}
		if sawDone {
			t.Errorf("an abandoned sequence delivered a terminal event; received %d tokens", received)
		}
		if got := e.Stats().Active; got != 0 {
			t.Errorf("Active = %d after the sequence was abandoned", got)
		}
	})
}

func TestFullBufferWithLiveCancelDoesNotStallPump(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 1000}.Write(t, t.TempDir())
	e, tok := loadEngineAt(t, path, Options{MaxActive: 1, Workers: 1})
	prompt := tok.Encode("Once upon a time", true)

	// Never closed: the caller is slow, not gone.
	cancel := make(chan struct{})
	ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: 900, Cancel: cancel})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	_ = ch // never drained, so the 64-slot buffer fills

	start := time.Now()
	done := make(chan struct{})
	go func() {
		defer close(done)
		for e.Pump() > 0 {
		}
	}()

	// abandonGrace is 2s; allow a wide margin so this fails only on unbounded waiting.
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("Pump never returned: a full buffer with a live cancel channel stalled the engine")
	}
	elapsed := time.Since(start)
	if got := e.Stats().Active; got != 0 {
		t.Errorf("engine still reports %d active sequences after the round finished", got)
	}

	// The run has to have gone through the deadline, not merely finished quickly. A fixture
	// that generated fewer than 64 tokens would never fill the buffer, so this test would pass
	// on broken code -- exactly how my first two attempts at reproducing the stall failed to
	// reproduce anything. Requiring at least one abandonGrace makes the fixture's size part of
	// the assertion instead of an undocumented hope.
	if elapsed < abandonGrace {
		t.Fatalf("Pump finished in %v, before abandonGrace (%v): the buffer never filled, "+
			"so this run did not exercise the deadline and proves nothing", elapsed, abandonGrace)
	}
	// It also has to retire the sequence *because it was abandoned*, not because the budget
	// ran out. This is what the first version of this assertion missed: deleting the
	// reapAbandoned call still passed, because sample keeps counting tokens (out grows before
	// emit refuses them) and the request then ends on its own MaxTokens limit. With a budget
	// of 900 that happened 902 rounds later in the same 2.004s the reaped run took, so no
	// elapsed-time bound can tell them apart.
	//
	// The distinguishing fact is the token count: a reaped sequence stops where it was
	// abandoned, while an unreaped one runs to its whole budget. Measured without the reap,
	// that is 900 tokens against roughly 64 with it -- the buffer's own size plus the round
	// that filled it.
	acct := e.Stats()
	if acct.RequestsFinished != 1 {
		t.Errorf("finished = %d after abandoning one request, want 1", acct.RequestsFinished)
	}
	const budget = 900
	if acct.TokensGenerated >= budget/2 {
		t.Errorf("TokensGenerated = %d of a %d-token budget: the sequence ran on after being "+
			"abandoned instead of being reaped, so its slot was held for the whole run",
			acct.TokensGenerated, budget)
	}

	// And the engine must still serve after abandoning: an abandoned sequence that is reaped
	// but left holding its active slot turns a deadlock into a leak, which looks identical
	// from the outside -- requests stop completing.
	e2ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: 3})
	if err != nil {
		t.Fatalf("Submit after abandon: %v", err)
	}
	// Pump first, then read: the engine produces events only while Pump runs, so ranging
	// before pumping blocks forever -- which is what my first version of this did.
	for e.Pump() > 0 {
	}
	for ev := range e2ch {
		if ev.Err != nil {
			t.Fatalf("follow-up request failed: %v", ev.Err)
		}
	}
	st := e.Stats()
	if st.Active != 0 {
		t.Errorf("Active = %d after the follow-up finished", st.Active)
	}
	if st.RequestsSubmitted != st.RequestsFinished {
		t.Errorf("submitted %d but finished %d: the abandoned sequence was never accounted for",
			st.RequestsSubmitted, st.RequestsFinished)
	}
}

func TestQueueFullHoldsUnderConcurrentSubmit(t *testing.T) {
	const maxActive, maxQueued = 1, 2
	e, tok, _ := loadModelPath(t, Options{MaxActive: maxActive, MaxQueued: maxQueued})
	prompt := tok.Encode("Once upon a time", true)

	var wg sync.WaitGroup
	var mu sync.Mutex
	accepted, refused := 0, 0
	for range 12 {
		wg.Go(func() {
			ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: 20})
			if err != nil {
				if !errors.Is(err, ErrQueueFull) {
					return
				}
				mu.Lock()
				refused++
				mu.Unlock()
				return
			}
			mu.Lock()
			accepted++
			mu.Unlock()
			go func() {
				for range ch {
				}
			}()
		})
	}
	wg.Wait()

	if want := maxActive + maxQueued; accepted != want {
		t.Errorf("accepted %d concurrent requests, want %d (MaxActive + MaxQueued); "+
			"the limit admitted more than it should", accepted, want)
	}
	if refused == 0 {
		t.Error("every concurrent request was accepted, so the limit did nothing")
	}
	if got := e.Stats().Active; got != accepted {
		t.Errorf("engine reports %d active after %d accepted", got, accepted)
	}
}

// TestQueueFullMessageNamesTheCapacityOnce checks the text a client actually reads.
//
// ErrQueueFull is wrapped with %w, so the sentinel's own "engine: too many requests
// waiting" is appended to whatever the wrapper writes. Writing the detail first produced
//
//	engine: 8 requests already waiting for 8 slots: engine: too many requests waiting
//
// in the 503 body -- the package prefix twice, and the same capacity described two
// different ways. Nothing asserted the text: the server test checks the status code and
// the Retry-After header, and this package checks errors.Is, so a doubled prefix could
// ship and did.
//
// The assertion is deliberately on the prefix count rather than on exact wording, so
// rewording the message is not a test break, while repeating the prefix still is.
func TestQueueFullMessageNamesTheCapacityOnce(t *testing.T) {
	const maxActive, maxQueued = 1, 2
	e, tok, _ := loadModelPath(t, Options{MaxActive: maxActive, MaxQueued: maxQueued})
	prompt := tok.Encode("Once upon a time", true)
	for range maxActive + maxQueued {
		if _, err := e.Submit(Request{Prompt: prompt, MaxTokens: 20}); err != nil {
			t.Fatalf("filling the engine: %v", err)
		}
	}
	_, err := e.Submit(Request{Prompt: prompt, MaxTokens: 20})
	if !errors.Is(err, ErrQueueFull) {
		t.Fatalf("Submit = %v, want ErrQueueFull", err)
	}
	msg := err.Error()
	if n := strings.Count(msg, "engine:"); n != 1 {
		t.Errorf("message names the package %d times, want 1 (a client reads this):\n%s", n, msg)
	}
	// The detail still has to describe the capacity the caller ran into, or a message
	// without a doubled prefix has been bought by saying nothing.
	if !strings.Contains(msg, "queued") || !strings.Contains(msg, "slots") {
		t.Errorf("message does not describe the queue: %q", msg)
	}
}

// TestQueueFullFreesUpAgain checks the limit is not permanent: once the backlog drains,
// new requests are accepted.
func TestQueueFullFreesUpAgain(t *testing.T) {
	e, tok, _ := loadModelPath(t, Options{MaxActive: 1, MaxQueued: 1})
	prompt := tok.Encode("Once upon a time", true)
	ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: 2})
	if err != nil {
		t.Fatalf("first Submit: %v", err)
	}
	if _, err := e.Submit(Request{Prompt: prompt, MaxTokens: 2}); err != nil {
		t.Fatalf("second Submit should fit the queue: %v", err)
	}
	if _, err := e.Submit(Request{Prompt: prompt, MaxTokens: 2}); !errors.Is(err, ErrQueueFull) {
		t.Fatalf("third Submit: %v, want ErrQueueFull", err)
	}
	go func() {
		for range ch {
		}
	}()
	for e.Pump() > 0 {
	}
	// The slot is free again once everything has finished.
	if _, err := e.Submit(Request{Prompt: prompt, MaxTokens: 2}); err != nil {
		t.Errorf("Submit after the queue drained: %v", err)
	}
}

// TestPrefixCacheSharedWhileConcurrent is the case the sequential reuse test misses: two
// requests in flight at once, over the same prompt, with the cache on.
//
// Sequential requests are the easy case -- the first finishes, its blocks are released into the
// index, and the second finds them. Concurrent ones share those blocks while both are still
// reading them, which is the state that a block-count or refcount bug corrupts: the two sequences
// are advanced in the same rounds, and a block handed to one must not be recycled under the other.
//
// The assertion is the same property the sequential test relies on and which makes the whole
// optimization safe: turning reuse on must not change the output. Two identical greedy requests
// that share a prefix have to agree with each other and with the same run under no cache.
func TestPrefixCacheSharedWhileConcurrent(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	gf, _ := gguf.Open(path)
	tok, _ := tokenize.New(gf)
	prompt := tok.Encode("Once upon a time there was a little girl named Lily who lived", true)

	// run submits n identical requests before pumping any of them, so all n are resident and
	// advancing together rather than one at a time.
	run := func(opts Options, n int) [][]int32 {
		t.Helper()
		e, err := New(path, opts)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		chans := make([]<-chan Event, n)
		for i := range n {
			ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: 4})
			if err != nil {
				t.Fatalf("Submit %d: %v", i, err)
			}
			chans[i] = ch
		}
		// Drain until every sequence has finished; Pump returns how many advanced.
		for e.Pump() > 0 {
		}
		out := make([][]int32, n)
		for i, ch := range chans {
			out[i], _ = collect(t, ch)
		}
		return out
	}

	const n = 3
	plain := run(Options{BlockSize: 4, Blocks: 256, MaxActive: 8}, n)
	cached := run(Options{BlockSize: 4, Blocks: 256, MaxActive: 8, PrefixCache: true}, n)

	for i := range n {
		// Each cached sequence must produce exactly what the uncached run did.
		if len(cached[i]) != len(plain[i]) {
			t.Fatalf("request %d: cached produced %d tokens, uncached %d", i, len(cached[i]), len(plain[i]))
		}
		for j := range cached[i] {
			if cached[i][j] != plain[i][j] {
				t.Fatalf("request %d token %d: cached %d, uncached %d", i, j, cached[i][j], plain[i][j])
			}
		}
		// And the concurrent requests agree with each other, which is what shows they did not
		// read each other's KV state.
		if len(cached[i]) != len(cached[0]) {
			t.Fatalf("request %d produced %d tokens, request 0 produced %d", i, len(cached[i]), len(cached[0]))
		}
		for j := range cached[i] {
			if cached[i][j] != cached[0][j] {
				t.Fatalf("request %d token %d: %d, request 0: %d", i, j, cached[i][j], cached[0][j])
			}
		}
	}
}

// TestFractionFull pins the arithmetic behind orbix_kv_pool_used_fraction.
//
// The gauge is the one reading that says a paged deployment is close to refusing requests,
// and it is invisible until that happens. Its formula is a single division, so a mistake
// reverses or rescales it silently: a scrape showing 0.3 where the pool is 70% used reads as
// plenty of headroom.
//
// Every case is recomputed here rather than taken from the function. The zero-block case is
// not hypothetical -- a contiguous engine reports no pool at all -- and returning NaN there
// would make the gauge parse as a string rather than a number.
func TestFractionFull(t *testing.T) {
	cases := []struct {
		name         string
		blocks, free int
		want         float64
	}{
		{"no pool", 0, 0, 0},
		{"no pool with a stray free count", 0, 5, 0},
		{"empty pool", 10, 10, 0},
		{"half used", 10, 5, 0.5},
		{"seven of ten", 10, 3, 0.7},
		{"one block of three used", 3, 2, 1.0 / 3},
		{"two blocks of three used", 3, 1, 2.0 / 3},
		{"full", 10, 0, 1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := Stats{PoolBlocks: c.blocks, PoolFree: c.free}.FractionFull()
			if got != c.want {
				t.Errorf("FractionFull(blocks=%d, free=%d) = %v, want %v",
					c.blocks, c.free, got, c.want)
			}
		})
	}
}
