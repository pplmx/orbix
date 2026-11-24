package engine

import (
	"slices"
	"testing"

	"github.com/pplmx/orbix/internal/gguftest"
)

// TestDriveDrainsBeyondTheEventBuffer is the regression Drive exists for.
//
// Each sequence's channel holds 64 events. A caller that drained between rounds -- or that
// pumped first and read afterwards -- stalls once a sequence produces more than that: the
// engine waits two seconds for a reader, retires the sequence as abandoned, and the run stops
// at 64 tokens with no finish reason at all.
//
// Measured in cmd/orbix/bench before it drained concurrently, with --max-tokens 65, 100 and 200
// against an idle fixture: every run generated exactly 64 tokens and printed no "finish:" line.
// This test asks for well past the buffer so the same defect fails here rather than only in a
// benchmark nobody runs.
func TestDriveDrainsBeyondTheEventBuffer(t *testing.T) {
	e, tok, _ := loadModelPath(t, Options{MaxActive: 2})
	const budget = 200 // three times the 64-event buffer

	ch, err := e.Submit(Request{
		Prompt:    tok.Encode("Once upon a time", true),
		MaxTokens: budget,
	})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}

	var generated int
	var reason string
	Drive(e, []<-chan Event{ch}, func(ch <-chan Event) {
		for ev := range ch {
			if ev.HasToken {
				generated++
			}
			if ev.Done {
				reason = ev.FinishReason
			}
		}
	})

	if generated != budget {
		t.Errorf("Drive collected %d tokens, want the full budget %d: a short count means the "+
			"sequence was abandoned at the event buffer", generated, budget)
	}
	// Without a finish reason the caller cannot tell a complete run from a truncated one, which
	// is the other half of the same failure.
	if reason != "length" {
		t.Errorf("finish reason = %q, want length: the budget is what stopped it", reason)
	}
	if active := e.Active(); active != 0 {
		t.Errorf("%d sequences still active after Drive returned", active)
	}
}

// TestDriveConsumesEverySequence checks that Drive gives each channel its own consumer.
//
// The four hand-written copies this replaced all had to start one goroutine per sequence. One
// consumer shared across channels would work only while the sequences happened to interleave,
// and would deadlock as soon as two were active at once -- which the multi-sequence case here
// makes immediate rather than occasional.
func TestDriveConsumesEverySequence(t *testing.T) {
	e, tok, _ := loadModelPath(t, Options{MaxActive: 4})
	prompt := tok.Encode("Once upon a time", true)

	const n = 4
	chans := make([]<-chan Event, 0, n)
	for range n {
		ch, err := e.Submit(Request{Prompt: prompt, MaxTokens: 8})
		if err != nil {
			t.Fatalf("Submit: %v", err)
		}
		chans = append(chans, ch)
	}

	// One entry per sequence, written only by that sequence's own consumer. A shared counter
	// without a lock would be a data race the detector reports; this shape avoids needing one.
	counts := make([]int, n)
	index := make(map[<-chan Event]int, n)
	for i, ch := range chans {
		index[ch] = i
	}
	Drive(e, chans, func(ch <-chan Event) {
		i := index[ch]
		for ev := range ch {
			if ev.HasToken {
				counts[i]++
			}
		}
	})

	if slices.Contains(counts, 0) {
		t.Errorf("token counts per sequence = %v; a zero means a channel had no consumer of "+
			"its own", counts)
	}
	if active := e.Active(); active != 0 {
		t.Errorf("%d sequences still active after Drive returned", active)
	}
}

// TestDriveReturnsAfterPumpReachesZero pins the ordering the join depends on.
//
// Pump has to return zero before the consumers are joined: a consumer blocks until its channel
// closes, and the channel closes when the sequence finishes, which only happens inside Pump. A
// version that joined first would hang -- so this test passing at all is the assertion, and it
// runs under the package's own timeout rather than an explicit deadline.
func TestDriveReturnsAfterPumpReachesZero(t *testing.T) {
	e, tok, _ := loadModelPath(t, Options{MaxActive: 1})
	ch, err := e.Submit(Request{
		Prompt:    tok.Encode("Once upon a time", true),
		MaxTokens: 4,
	})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	roundsBefore := e.Round()
	Drive(e, []<-chan Event{ch}, func(ch <-chan Event) {
		for range ch {
		}
	})
	// The round counter is a scheduling clock: if Drive returned without pumping, it would not
	// have moved, and the assertion says which half is missing.
	if e.Round() <= roundsBefore {
		t.Errorf("round count did not advance (%d -> %d), so Drive never pumped",
			roundsBefore, e.Round())
	}
}

// TestDriveIsUsableOnAnEmptyList keeps the boundary from panicking.
//
// A caller with nothing submitted is not an error -- a scenario whose requests were all refused
// before submission reaches here -- and the loop must simply do nothing rather than wait for a
// zero it never computes.
func TestDriveIsUsableOnAnEmptyList(t *testing.T) {
	e, _, _ := loadModelPath(t, Options{MaxActive: 1})
	called := false
	Drive(e, nil, func(ch <-chan Event) { called = true })
	if called {
		t.Error("Drive called the consumer for a list it was not given")
	}
}

// TestDriveKeepsAFixtureModelUsable is a small end-to-end guard on the helper's inputs: the
// fixture is written and loaded the same way the other engine tests do it, so a change to the
// test helper shows up here too.
func TestDriveKeepsAFixtureModelUsable(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	e, tok := loadModelAt(t, path, Options{MaxActive: 1})
	ch, err := e.Submit(Request{Prompt: tok.Encode("hi", true), MaxTokens: 2})
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	var got int
	Drive(e, []<-chan Event{ch}, func(ch <-chan Event) {
		for ev := range ch {
			if ev.HasToken {
				got++
			}
		}
	})
	if got != 2 {
		t.Errorf("collected %d tokens, want 2", got)
	}
}
