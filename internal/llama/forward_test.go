package llama

import (
	"testing"

	"github.com/pplmx/orbix/internal/gguftest"
)

// TestForwardAllocatesPerToken pins what per-token decoding actually allocates.
//
// The field comment on Model's scratch buffers claimed "per-token allocation at zero after the
// first call". Measured: 23 per token on the two-layer fixture, and every one of them from a
// kernel rather than from the scratch slices the comment was about. `Runner.splitRows` takes a
// closure, and the escape analysis sends it to the heap on each call -- at every size, including
// the size where `split` takes its serial path, so it is the closure and not the fan-out.
//
// A count is asserted rather than a ceiling because the number is a property of the fixture and
// the kernels, and it is stable: five measurements over the same fixture all returned 23.0. The
// bound is exact so that a change in the kernels shows up here, where the reason is written down,
// instead of leaving the comment wrong again.
//
// Timing is deliberately not asserted. The machine running this had a load average above 13 on
// 128 cores, and per AGENTS.md a parallel-code timing on a loaded machine measures whether
// goroutines got scheduled, not the code.
func TestForwardAllocatesPerToken(t *testing.T) {
	m, err := New(gguftest.Model{NoEOS: true, ContextLen: 256}.Write(t, t.TempDir()))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	cfg := m.Config()
	cache, err := NewKVCache(cfg, 256)
	if err != nil {
		t.Fatalf("NewKVCache: %v", err)
	}
	// The first call is what builds the scratch buffers, which is the exception the comment
	// always carved out.
	if _, err := m.Forward(cache, 1, 0); err != nil {
		t.Fatalf("warm-up Forward: %v", err)
	}

	pos := 1
	var errs int
	got := testing.AllocsPerRun(60, func() {
		if _, err := m.Forward(cache, 1, pos); err != nil {
			errs++
		}
		pos++
	})
	if errs > 0 {
		t.Fatalf("%d of the measured Forward calls returned an error; a call on the error path "+
			"allocates a fmt.Errorf and would make this figure meaningless", errs)
	}
	// The exact figure for a two-layer model: 6 kernels a layer x 2 + the output projection.
	const want = 23.0
	if got != want {
		t.Errorf("Forward allocates %.1f per token, want %.1f.\n"+
			"If this went down because the kernels stopped allocating, update the field comment "+
			"on Model and this constant together.\n"+
			"If it went up, something on the decode path started allocating.", got, want)
	}
}
