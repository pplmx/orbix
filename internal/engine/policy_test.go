package engine

import (
	"fmt"
	"os"
	"testing"

	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/tokenize"
)

// Policies change which requests advance, never what a request produces: these tests
// use fake sequences wherever the property is about selection order, so they run
// without a model and stay fast. The end-to-end tests at the bottom check the property
// a scheduler is actually judged on -- that a short request is not stuck behind a long
// one -- against a real model.

// fakeSeq builds a sequence with just the fields the policies read.
func fakeSeq(id, maxTokens, emitted, lastServed int) *sequence {
	s := &sequence{id: id, lastServed: lastServed}
	s.req.MaxTokens = maxTokens
	s.out = make([]int32, emitted)
	return s
}

func ids(seqs []*sequence) []int {
	out := make([]int, len(seqs))
	for i, s := range seqs {
		out[i] = s.id
	}
	return out
}

func TestAdmitFIFOIsSubmissionOrder(t *testing.T) {
	seqs := []*sequence{fakeSeq(0, 5, 0, -1), fakeSeq(1, 5, 0, -1), fakeSeq(2, 5, 0, -1)}
	got := admit(seqs, 2, PriorityFIFO)
	if want := []int{0, 1}; fmt.Sprint(ids(got)) != fmt.Sprint(want) {
		t.Errorf("FIFO admitted %v, want %v", ids(got), want)
	}
}

func TestAdmitShortestPicksLeastRemaining(t *testing.T) {
	// Sequence 1 has 1 token left, sequence 2 has 4, sequence 0 has 5. Shortest must
	// take 1 first even though it was not submitted first.
	seqs := []*sequence{fakeSeq(0, 5, 0, -1), fakeSeq(1, 4, 3, -1), fakeSeq(2, 6, 2, -1)}
	got := admit(seqs, 2, PriorityShortest)
	if want := []int{1, 2}; fmt.Sprint(ids(got)) != fmt.Sprint(want) {
		t.Errorf("shortest admitted %v, want %v", ids(got), want)
	}
}

func TestAdmitShortestTiesByID(t *testing.T) {
	// Equal remaining work must fall back to submission order, or two runs of the same
	// workload could select different sequences.
	seqs := []*sequence{fakeSeq(2, 4, 0, -1), fakeSeq(0, 4, 0, -1), fakeSeq(1, 4, 0, -1)}
	got := admit(seqs, 2, PriorityShortest)
	if want := []int{0, 1}; fmt.Sprint(ids(got)) != fmt.Sprint(want) {
		t.Errorf("shortest with ties admitted %v, want %v", ids(got), want)
	}
}

func TestAdmitFairPicksLeastRecentlyServed(t *testing.T) {
	// Sequence 2 ran longest ago, so it goes first even though it was submitted last.
	seqs := []*sequence{fakeSeq(0, 5, 0, 10), fakeSeq(1, 5, 0, 9), fakeSeq(2, 5, 0, 2)}
	got := admit(seqs, 2, PriorityFair)
	if want := []int{2, 1}; fmt.Sprint(ids(got)) != fmt.Sprint(want) {
		t.Errorf("fair admitted %v, want %v", ids(got), want)
	}
}

func TestAdmitFairNeverServedGoesFirst(t *testing.T) {
	// A sequence that has never run (lastServed -1) outranks every one that has, which
	// is what stops a new request from waiting behind the whole active set.
	seqs := []*sequence{fakeSeq(0, 5, 0, 1), fakeSeq(1, 5, 0, 2), fakeSeq(2, 5, 0, -1)}
	got := admit(seqs, 1, PriorityFair)
	if want := []int{2}; fmt.Sprint(ids(got)) != fmt.Sprint(want) {
		t.Errorf("fair admitted %v, want %v", ids(got), want)
	}
}

func TestAdmitReturnsEverythingWhenItFits(t *testing.T) {
	// No policy applies when the whole queue fits, and the input must come back
	// unchanged rather than reordered: reordering without a reason would make trace
	// output harder to follow.
	seqs := []*sequence{fakeSeq(2, 5, 0, 5), fakeSeq(0, 1, 0, 1), fakeSeq(1, 9, 0, 3)}
	for _, policy := range []Priority{PriorityFIFO, PriorityShortest, PriorityFair} {
		got := admit(seqs, 5, policy)
		if want := []int{2, 0, 1}; fmt.Sprint(ids(got)) != fmt.Sprint(want) {
			t.Errorf("%v admitted %v for an unbounded round, want the input order %v", policy, ids(got), want)
		}
	}
}

func TestAdmitEdgeCases(t *testing.T) {
	seqs := []*sequence{fakeSeq(0, 5, 0, -1), fakeSeq(1, 5, 0, -1)}
	if got := admit(seqs, 0, PriorityFIFO); got != nil {
		t.Errorf("limit 0 admitted %v, want nil", ids(got))
	}
	if got := admit(nil, 3, PriorityFIFO); got != nil {
		t.Errorf("empty queue admitted %v, want nil", ids(got))
	}
	// A limit larger than the queue must not grow the result.
	if got := admit(seqs, 10, PriorityFair); len(got) != 2 {
		t.Errorf("limit 10 admitted %d sequences, want 2", len(got))
	}
}

func TestPriorityNames(t *testing.T) {
	for _, tc := range []struct {
		name string
		want Priority
	}{
		{"fifo", PriorityFIFO},
		{"", PriorityFIFO},
		{"shortest", PriorityShortest},
		{"fair", PriorityFair},
	} {
		got, err := ParsePriority(tc.name)
		if err != nil {
			t.Errorf("ParsePriority(%q) = %v", tc.name, err)
			continue
		}
		if got != tc.want {
			t.Errorf("ParsePriority(%q) = %v, want %v", tc.name, got, tc.want)
		}
	}
	if _, err := ParsePriority("random"); err == nil {
		t.Error("ParsePriority accepted an unknown policy")
	}
	// The names have to round-trip, since the CLI prints them.
	for _, p := range []Priority{PriorityFIFO, PriorityShortest, PriorityFair} {
		back, err := ParsePriority(p.String())
		if err != nil || back != p {
			t.Errorf("%v round-trips to %v (%v)", p, back, err)
		}
	}
}

// --- end-to-end: the property a scheduler is judged on ---

// loadEngine returns an engine and tokenizer for the model in MODEL.
func loadEngine(t *testing.T, opts Options) (*Engine, *tokenize.Tokenizer) {
	t.Helper()
	path := os.Getenv("MODEL")
	if path == "" {
		t.Skip("set MODEL to a GGUF file")
	}
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
		t.Fatalf("New: %v", err)
	}
	return e, tok
}

func TestSchedulingPolicyKeepsShortRequestResponsive(t *testing.T) {
	// The failure this policy exists to prevent: with MaxActive=1, a three-token
	// request submitted behind a sixty-token one under FIFO waits for the long one to
	// finish (measured: first token on round 62 of 64). PriorityShortest and
	// PriorityFair both have to cut that down.
	const (
		longTokens  = 60
		shortTokens = 3
	)
	bfs := func(t *testing.T, policy Priority) int {
		e, tok := loadEngine(t, Options{MaxActive: 1, Workers: 2, Policy: policy})
		prompt := tok.Encode("Once upon a time", true)

		longCh, err := e.Submit(Request{Prompt: prompt, MaxTokens: longTokens})
		if err != nil {
			t.Fatalf("Submit: %v", err)
		}
		shortCh, err := e.Submit(Request{Prompt: prompt, MaxTokens: shortTokens})
		if err != nil {
			t.Fatalf("Submit: %v", err)
		}
		// Drain both in the background: the channels are buffered but a 60-token
		// sequence would fill the buffer and block Pump.
		go func() {
			for range longCh {
			}
		}()
		var firstTokenRound int
		done := make(chan struct{})
		go func() {
			defer close(done)
			for ev := range shortCh {
				if ev.HasToken && firstTokenRound == 0 {
					firstTokenRound = e.Round()
				}
			}
		}()
		for e.Pump() > 0 {
		}
		<-done
		return firstTokenRound
	}

	t.Run("fifo", func(t *testing.T) {
		round := bfs(t, PriorityFIFO)
		t.Logf("FIFO: the short request's first token arrived on round %d", round)
		if round < longTokens/2 {
			t.Errorf("FIFO served the short request on round %d; the baseline was 62, so the queue changed", round)
		}
	})
	t.Run("shortest", func(t *testing.T) {
		round := bfs(t, PriorityShortest)
		t.Logf("shortest: the short request's first token arrived on round %d", round)
		if round > 10 {
			t.Errorf("shortest served the short request on round %d, want under 10", round)
		}
	})
	t.Run("fair", func(t *testing.T) {
		round := bfs(t, PriorityFair)
		t.Logf("fair: the short request's first token arrived on round %d", round)
		if round > 10 {
			t.Errorf("fair served the short request on round %d, want under 10", round)
		}
	})
}

// TestPolicyDoesNotChangeOutput is the safety property: a scheduling policy changes
// when a request runs, never what it produces. Greedy decoding is deterministic, so two
// policies must return the same tokens for the same prompt.
func TestPolicyDoesNotChangeOutput(t *testing.T) {
	run := func(policy Priority) [][]int32 {
		e, tok := loadEngine(t, Options{MaxActive: 2, Workers: 2, Policy: policy})
		prompt := tok.Encode("Once upon a time", true)
		const n = 4
		chans := make([]<-chan Event, n)
		for i := range chans {
			ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: 6})
			if err != nil {
				t.Fatalf("Submit: %v", err)
			}
			chans[i] = ch
		}
		for e.Pump() > 0 {
		}
		out := make([][]int32, n)
		for i, ch := range chans {
			for ev := range ch {
				if ev.Err != nil {
					t.Fatalf("sequence %d: %v", i, ev.Err)
				}
				if ev.HasToken {
					out[i] = append(out[i], ev.Token)
				}
			}
		}
		return out
	}

	fifo := run(PriorityFIFO)
	for _, policy := range []Priority{PriorityShortest, PriorityFair} {
		got := run(policy)
		for i := range fifo {
			if len(got[i]) != len(fifo[i]) {
				t.Fatalf("%v: request %d produced %d tokens, fifo produced %d", policy, i, len(got[i]), len(fifo[i]))
			}
			for j := range fifo[i] {
				if got[i][j] != fifo[i][j] {
					t.Errorf("%v: request %d token %d = %d, fifo = %d", policy, i, j, got[i][j], fifo[i][j])
				}
			}
		}
	}
}
