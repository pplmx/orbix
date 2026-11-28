// Command genfixture writes a synthetic GGUF checkpoint large enough to be worth interrupting.
//
// The in-repo fixtures used by the test suite are two layers of a few dozen positions, which
// finish in single-digit milliseconds -- the engine never has time to be slow, so a client that
// hangs up mid-generation has nothing to hang up on. `just stress` needs the opposite: a request
// that keeps the engine busy long enough for a queued client to leave. This writes one.
//
//	go run ./scripts/genfixture -out /tmp/stress.gguf
//
// The weights are random, so the text it generates is meaningless. That is fine: the script under
// stress measures whether a disconnect is handled, not what the model said.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/pplmx/orbix/internal/gguftest"
)

func main() {
	out := flag.String("out", "stress.gguf", "path to write")
	layers := flag.Int("layers", 4, "transformer layers")
	hidden := flag.Int("hidden", 256, "hidden dimension")
	ctx := flag.Int("context", 4096, "context length in positions")
	flag.Parse()

	if *layers <= 0 || *hidden <= 0 || *ctx <= 0 {
		fmt.Fprintf(os.Stderr, "genfixture: layers, hidden and context must all be positive (got %d, %d, %d)\n",
			*layers, *hidden, *ctx)
		os.Exit(1)
	}

	m := gguftest.Model{
		NoEOS:      true,
		Layers:     *layers,
		Hidden:     *hidden,
		ContextLen: *ctx,
	}
	raw, err := m.Bytes()
	if err != nil {
		fmt.Fprintf(os.Stderr, "genfixture: %v\n", err)
		os.Exit(1)
	}
	if err := os.WriteFile(*out, raw, 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "genfixture: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("genfixture: wrote %s (%d bytes, %d layers, hidden %d, context %d)\n",
		*out, len(raw), *layers, *hidden, *ctx)
}
