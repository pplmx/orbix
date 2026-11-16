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
