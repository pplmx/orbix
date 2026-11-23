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
// This runs the real thing over a real socket, because the property is about TCP: a graceful FIN
// leaves the connection usable and generation runs to its full budget, so the test sets SO_LINGER
// to zero to send RST instead. That difference is the reason an earlier attempt over http.Post
// measured nothing.
//
// Gated on MODEL because it needs a checkpoint; the fixture is too fast for a reset to land in the
// middle of a stream.
func TestTCPResetStopsGeneration(t *testing.T) {
	path := os.Getenv("MODEL")
	if path == "" {
		t.Skip("set MODEL to a checkpoint; a fixture generates too fast for a reset to land mid-stream")
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
	const budget = 200
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
