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
