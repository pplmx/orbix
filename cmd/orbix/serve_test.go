package main

import (
	"fmt"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/llama"
)

// TestContiguousMemoryNote pins the KV footprint arithmetic.
//
// The number is the only thing that tells an operator what the offline default costs: a
// contiguous cache reserves a whole context per accepted request, so a backlog of eight
// active plus sixty-four queued reserves far more than the model file. A factor-of-two
// error here would read as plausible and be wrong, which is exactly the class of mistake
// the repository convention says to recompute independently in the test.
func TestContiguousMemoryNote(t *testing.T) {
	// The shape of the stories15M fixture: six layers, six KV heads of forty-eight
	// dimensions, a 128-position window.
	cfg := llama.Config{Layers: 6, KVHeads: 6, HeadDim: 48, ContextLen: 128}

	// Recomputed here rather than taken from the function: positions times the per-position
	// KV width, for both K and V, in every layer, at four bytes each.
	const (
		// One position holds a K and a V vector of KVHeads*HeadDim floats in every layer.
		perPos = 2 * 6 * 48 * 4 * 6
		// A request reserves its whole context.
		perReq   = 128 * perPos
		wantMiB  = float64(perReq*(8+64)) / (1 << 20)
		wantPerM = float64(perReq) / (1 << 20)
	)
	got := contiguousMemoryNote(cfg, 8, 64)
	t.Logf("note: %s", got)

	if !strings.Contains(got, "contiguous") {
		t.Errorf("note does not name the cache layout: %q", got)
	}
	// The per-request figure, to one decimal, and the total for the configured slots.
	if want := itoa1(wantPerM); !strings.Contains(got, want) {
		t.Errorf("note %q has no per-request figure %q (want %.2f MiB)", got, want, wantPerM)
	}
	if want := itoa1(wantMiB); !strings.Contains(got, want) {
		t.Errorf("note %q has no total %q (want %.2f MiB)", got, want, wantMiB)
	}

	// Doubling the slots doubles the total, which is the property the line exists to convey.
	doubled := contiguousMemoryNote(cfg, 8, 136)
	if want := itoa1(2 * wantMiB); !strings.Contains(doubled, want) {
		t.Errorf("doubling the queue gave %q, want a total of %q", doubled, want)
	}
}

// itoa1 formats like the function under test, so the assertion compares the same rendering.
func itoa1(v float64) string {
	return fmt.Sprintf("%.1f MB", v)
}

// TestReservationWarning covers the line that tells an operator a contiguous reservation is
// too large to leave alone.
//
// The figure it accompanies is virtual, not resident: the arenas are one make per layer and
// Linux commits a page only when it is written, so a server reserving hundreds of gigabytes
// measures a few gigabytes of RSS and looks healthy right up until long requests fault in the
// pages. Nothing else at startup distinguishes a planner that reserves 1.7MB per request from
// one that reserves 8GB.
func TestReservationWarning(t *testing.T) {
	// The shape of the Llama 3.2 checkpoint: 16 layers, 8 KV heads of 64 dimensions, a
	// 128k window.
	wide := llama.Config{Layers: 16, KVHeads: 8, HeadDim: 64, ContextLen: 131072}
	if got := reservationWarning(wide, 8, 64); got == "" {
		t.Error("no warning for a 128k-context model, whose reservation is hundreds of gigabytes")
	} else {
		// The advice is the part that has to be there: a warning without a way out is noise.
		if !strings.Contains(got, "--blocks") {
			t.Errorf("warning does not name the flags that fix it: %q", got)
		}
		if !strings.Contains(got, "virtual") {
			t.Errorf("warning does not explain why resident memory will look smaller: %q", got)
		}
	}

	// The small fixture: 1.7MB per request, 121.5MB for the default slots. Well under the
	// threshold, so the line would be noise.
	small := llama.Config{Layers: 6, KVHeads: 6, HeadDim: 48, ContextLen: 128}
	if got := reservationWarning(small, 8, 64); got != "" {
		t.Errorf("warning for a 121.5MB reservation: %q", got)
	}

	// The threshold is on the total, not on one request: the wide model with a single slot
	// and no queue still crosses it, and a narrow one with a huge queue does too.
	// 131072 * 8 * 64 * 2 * 4 * 16 = 8GiB for one request alone.
	if got := reservationWarning(wide, 1, 0); got == "" {
		t.Error("no warning for a single 8GiB request")
	}
}

// TestPoolReservationWarning covers the counterpart of the contiguous warning, for the mode
// where the reservation comes from a block count rather than a context length.
//
// A block pool allocates every block up front, so --blocks 20000 on a 128k-window model asks
// the kernel for 19.5GB whether or not a request ever touches those blocks. Measured on a
// running server at that setting: VmSize 32GB against VmRSS 7.4GB after eight requests, so
// the resident figure is not the reservation and an operator watching RSS would not see it.
func TestPoolReservationWarning(t *testing.T) {
	// Below the threshold: the small fixture's 2048 x 16 pool is 432MiB, which is a normal
	// thing to configure and should not print a warning.
	if got := poolReservationWarning(432 << 20); got != "" {
		t.Errorf("warning for a 432MiB pool: %q", got)
	}
	// Exactly at the threshold stays quiet: the condition is "more than a gibibyte".
	if got := poolReservationWarning(1 << 30); got != "" {
		t.Errorf("warning for exactly 1GiB: %q", got)
	}

	got := poolReservationWarning(20 << 30)
	if got == "" {
		t.Fatal("no warning for a 20GiB pool")
	}
	// The advice and the reason both have to be there: a warning that does not say what to
	// change, or why the resident number will disagree, is not actionable.
	if !strings.Contains(got, "--blocks") {
		t.Errorf("warning does not name the flag that fixes it: %q", got)
	}
	if !strings.Contains(got, "20.0 GB") {
		t.Errorf("warning does not state the reservation: %q", got)
	}
	if !strings.Contains(got, "virtual") && !strings.Contains(got, "commits pages") {
		t.Errorf("warning does not explain why resident memory will be lower: %q", got)
	}
}

// TestPrefixCacheNeedsABlockPool checks that the combination which cannot work is refused at
// startup rather than ignored.
//
// Prefix reuse lives in the paged cache: attaching a previous request's KV blocks needs a
// block index, and a contiguous arena has none. Passing --prefix-cache without --blocks used
// to start a contiguous server that silently did no reuse, and the startup log said "kv
// cache: contiguous" with no mention of the flag, so the operator who passed it had nothing
// to notice it by.
//
// The check runs before the model is opened, so this needs no checkpoint: the flag
// combination is rejected on its own terms.
func TestPrefixCacheNeedsABlockPool(t *testing.T) {
	err := cmdServe([]string{"--model", "unused.gguf", "--prefix-cache"})
	if err == nil {
		t.Fatal("cmdServe accepted --prefix-cache without --blocks")
	}
	msg := err.Error()
	if !strings.Contains(msg, "prefix-cache") {
		t.Errorf("error %q does not name the flag that was passed", msg)
	}
	if !strings.Contains(msg, "blocks") {
		t.Errorf("error %q does not say what the flag needs", msg)
	}
}
