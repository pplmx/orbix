package engine

import (
	"os"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/llama"
	"github.com/pplmx/orbix/internal/tokenize"
)

// These tests need a real model, because the property under test is that two
// sequences interleaved on one model produce the same tokens they would alone.
// A synthetic model cannot show that: the failure mode is the shared scratch
// buffer, which only exists with real weights.
//
// Set MODEL to a GGUF file to run them.

func loadModel(t *testing.T) (*Engine, *tokenize.Tokenizer) {
	t.Helper()
	path := os.Getenv("MODEL")
	if path == "" {
		t.Skip("set MODEL to a GGUF file to run engine tests")
	}
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	e, err := New(path, Options{})
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
	path := os.Getenv("MODEL")
	if path == "" {
		t.Skip("set MODEL to a GGUF file to run engine tests")
	}
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
	e, tok := loadModel(t)
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

	e2, _ := New(os.Getenv("MODEL"), Options{})
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
	e, tok := loadModel(t)
	ch, err := e.Submit(Request{Prompt: tok.Encode("Once upon a time", true), MaxTokens: 4})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	for e.Pump() > 0 {
	}
	got, _ := collect(t, ch)

	e2, _ := New(os.Getenv("MODEL"), Options{})
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
	e, tok := loadModel(t)
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

	m, err := llama.New(os.Getenv("MODEL"))
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
	path := os.Getenv("MODEL")
	if path == "" {
		t.Skip("set MODEL to a GGUF file to run engine tests")
	}
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
	path := os.Getenv("MODEL")
	if path == "" {
		t.Skip("set MODEL to a GGUF file to run engine tests")
	}
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
	path := os.Getenv("MODEL")
	if path == "" {
		t.Skip("set MODEL to a GGUF file to run engine tests")
	}
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
	path := os.Getenv("MODEL")
	if path == "" {
		t.Skip("set MODEL to a GGUF file to run engine tests")
	}
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
	path := os.Getenv("MODEL")
	if path == "" {
		t.Skip("set MODEL to a GGUF file to run engine tests")
	}
	e, err := New(path, Options{Blocks: 8})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := e.Pool().BlockSize(); got <= 0 {
		t.Errorf("BlockSize = %d, want a positive default", got)
	}
}
