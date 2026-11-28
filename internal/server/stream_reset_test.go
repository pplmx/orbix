package server

import (
	"bufio"
	"context"
	"fmt"
	"net"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/pplmx/orbix/internal/engine"
	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
	"github.com/pplmx/orbix/internal/tokenize"
)

// TestTCPResetStopsGeneration covers what the comment in streamResponse claims and nothing tested:
// that a client which resets the connection mid-stream stops the engine.
//
// The claim is easy to state and hard to test, because it depends on the request context rather
// than on any error the handler sees: a flush error is deliberately not checked, and the reasoning
// is that the context notices first. What has been tested until now is the other half -- a
// ResponseWriter whose Write fails, with a hand-built channel standing in for the engine -- which
// says the handler returns and says nothing about whether the model stops.
//
// This runs the real thing over a real socket, because the property is about TCP: SetLinger(0)
// makes Close send RST rather than FIN, which is what distinguishes "the client is gone" from "the
// client has stopped writing but is still reading". An earlier attempt over http.Post measured
// nothing because a plain close left generation running to its full budget.
//
// It used to be gated on MODEL, with the fixture described as too fast for a reset to land
// mid-stream. That was wrong. Measured on the fixture, the reset lands after two SSE chunks with 38
// of a 40-token budget still to generate, and the run stops 1 or 2 tokens later; removing the
// cancellation wiring -- Cancel: nil instead of r.Context().Done() -- makes eight runs out of eight
// fail, at 31 to 35 tokens past the reset. The gate only kept the property from being checked at
// all on a machine without a checkpoint.
//
// The budget has to fit the fixture's window (64 positions, 20-token prompt) or the run is cut
// short by the context rather than by the reset, which passes for the wrong reason. At 200 the
// budget leaves the window only 44 positions, so the injected run stops at 35 to 41 tokens, and the
// bound below is 200/4 = 50 -- a figure the engine cannot reach while the window holds, which makes
// that test pass whatever the cancellation does. 40 keeps the bound (10) far below what the window
// allows.
func TestTCPResetStopsGeneration(t *testing.T) {
	path := os.Getenv("MODEL")
	if path == "" {
		path = gguftest.Path(t)
	}
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	eng, err := engine.New(path, engine.Options{MaxActive: 1, BlockSize: 16, Blocks: 512})
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	cfg := eng.Model().Config()
	srv, err := New(eng, tok, Config{ModelName: cfg.Name, ContextLen: cfg.ContextLen})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	srv.StartPump(ctx)

	ts := httptest.NewUnstartedServer(srv.Handler())
	ts.Start()
	defer ts.Close()
	addr := ts.Listener.Addr().String()

	conn, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	const budget = 40
	body := fmt.Sprintf(`{"model":%q,"prompt":"The quick brown fox","max_tokens":%d,"stream":true}`,
		cfg.Name, budget)
	req := fmt.Sprintf("POST /v1/completions HTTP/1.1\r\nHost: %s\r\n"+
		"Content-Type: application/json\r\nContent-Length: %d\r\n\r\n%s", addr, len(body), body)
	if _, err := conn.Write([]byte(req)); err != nil {
		t.Fatalf("write request: %v", err)
	}

	// Read until the stream is genuinely running. A 404 or a refused request would satisfy a
	// single Read, so waiting for the SSE framing is what makes the reset land mid-stream.
	_ = conn.SetReadDeadline(time.Now().Add(90 * time.Second))
	br := bufio.NewReader(conn)
	chunks := 0
	for chunks < 2 {
		line, err := br.ReadString('\n')
		if err != nil {
			t.Fatalf("reading the stream after %d chunks: %v", chunks, err)
		}
		if strings.HasPrefix(line, "data: ") {
			chunks++
		}
	}

	before := eng.Stats().TokensGenerated
	// RST rather than FIN: SetLinger(0) makes Close send a reset, which is what makes the server
	// notice. A plain close is indistinguishable from a client that is still reading.
	if tc, ok := conn.(*net.TCPConn); ok {
		if err := tc.SetLinger(0); err != nil {
			t.Fatalf("SetLinger: %v", err)
		}
	}
	if err := conn.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	deadline := time.Now().Add(90 * time.Second)
	for time.Now().Before(deadline) && eng.Stats().Active != 0 {
		time.Sleep(100 * time.Millisecond)
	}
	s := eng.Stats()
	ran := int64(s.TokensGenerated) - int64(before)
	t.Logf("%d chunks read, then %d of a %d-token budget generated after the reset", chunks, ran, budget)

	if s.Active != 0 {
		t.Errorf("the engine still holds a request whose client sent RST")
	}
	// The budget is what the engine would run to if the reset went unnoticed, so the bound is a
	// small fraction of it rather than an absolute token count -- the exact number depends on how
	// many rounds were already in flight when the reset arrived.
	if ran > budget/4 {
		t.Errorf("%d tokens of a %d-token budget were generated after the client reset the"+
			" connection; the request context did not cancel the sequence", ran, budget)
	}
}
