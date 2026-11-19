package server

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/gguftest"
)

// The server produced no records at all before this: a failure was written to the HTTP
// response and forgotten, which is fine for a test and wrong for a server, where "what
// happened to that request" has no answer once the client is gone. These tests pin the
// replacement.

// loggingServer returns a server that writes JSON records into buf.
func loggingServer(t *testing.T) (*Server, *bytes.Buffer) {
	t.Helper()
	var buf bytes.Buffer
	srv, _ := newTestServerConfig(t, gguftest.Default(), Config{
		Logger: slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})),
	})
	return srv, &buf
}

// firstRecord decodes the first JSON record written to the buffer.
func firstRecord(t *testing.T, buf *bytes.Buffer) map[string]any {
	t.Helper()
	line, _, _ := strings.Cut(buf.String(), "\n")
	var rec map[string]any
	if err := json.Unmarshal([]byte(line), &rec); err != nil {
		t.Fatalf("log line is not JSON: %v\n%s", err, buf.String())
	}
	return rec
}

// TestLoggerRecordsOnePerRequest checks the fields a reader needs to answer "what
// happened to that request" after the client has gone.
func TestLoggerRecordsOnePerRequest(t *testing.T) {
	srv, buf := loggingServer(t)

	post(t, srv.Handler(), "/v1/completions", `{"prompt": "Once", "max_tokens": 2}`)

	rec := firstRecord(t, buf)
	for _, key := range []string{"method", "path", "status", "duration", "bytes"} {
		if _, ok := rec[key]; !ok {
			t.Errorf("record is missing %q: %s", key, buf.String())
		}
	}
	if rec["path"] != "/v1/completions" {
		t.Errorf("path = %v", rec["path"])
	}
	if status, _ := rec["status"].(float64); status != http.StatusOK {
		t.Errorf("status = %v, want 200", rec["status"])
	}
}

// TestLoggerBadRequestIsNotAWarning checks the level choice: a 4xx is the client's
// mistake, and logging those as warnings teaches a reader to ignore the level.
func TestLoggerBadRequestIsNotAWarning(t *testing.T) {
	srv, buf := loggingServer(t)

	post(t, srv.Handler(), "/v1/completions", `{"prompt": ""}`)

	rec := firstRecord(t, buf)
	if status, _ := rec["status"].(float64); status != http.StatusBadRequest {
		t.Errorf("status = %v, want 400", rec["status"])
	}
	if rec["level"] != "INFO" {
		t.Errorf("level = %v, want INFO for a client error", rec["level"])
	}
}

// TestNilLoggerIsSilent pins the default: a server with no logger still answers, and its
// accessor never returns nil, so a caller does not have to guard it.
func TestNilLoggerIsSilent(t *testing.T) {
	srv, _ := newTestServer(t)
	if srv.Logger() == nil {
		t.Fatal("Logger() returned nil; an unconfigured server must still answer")
	}
	resp := post(t, srv.Handler(), "/v1/completions", `{"prompt": "Once", "max_tokens": 2}`)
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d with no logger configured", resp.StatusCode)
	}
}

// TestRecordingWriterPreservesFlush is what keeps streaming working. The SSE handlers
// flush after each token; a wrapper that swallowed Flush would hold every token until the
// response ended, turning a working stream into a hung one with no error anywhere -- so
// this is checked directly rather than only through the end-to-end path.
func TestRecordingWriterPreservesFlush(t *testing.T) {
	rec := httptest.NewRecorder()
	var w http.ResponseWriter = &statusRecorder{ResponseWriter: rec, status: http.StatusOK}
	f, ok := w.(http.Flusher)
	if !ok {
		t.Fatal("the recording writer is not an http.Flusher; streaming would stall")
	}
	f.Flush()
	if !rec.Flushed {
		t.Error("Flush did not reach the underlying writer")
	}
}

// TestStatusRecorderDefaultsTo200 covers a handler that writes without calling
// WriteHeader: the record must not report status 0.
func TestStatusRecorderDefaultsTo200(t *testing.T) {
	rec := httptest.NewRecorder()
	sr := &statusRecorder{ResponseWriter: rec, status: http.StatusOK}
	if _, err := sr.Write([]byte("hi")); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if sr.status != http.StatusOK {
		t.Errorf("status = %d, want 200", sr.status)
	}
	if sr.written != 2 {
		t.Errorf("written = %d, want 2", sr.written)
	}
}

// TestStreamingProducesARecord is the end-to-end version: a configured logger must not
// break the streaming path, and must record it.
func TestStreamingProducesARecord(t *testing.T) {
	srv, buf := loggingServer(t)

	post(t, srv.Handler(), "/v1/completions",
		`{"prompt": "Once", "max_tokens": 2, "stream": true}`)

	if buf.Len() == 0 {
		t.Fatal("a streaming request produced no log record")
	}
	if !strings.Contains(buf.String(), "/v1/completions") {
		t.Errorf("record does not name the path: %s", buf.String())
	}
}

// TestLoggerReturnsTheConfiguredOne covers the non-nil arm of Logger.
//
// The nil arm is tested because a server with no logger must not panic;
// the arm that returns the configured logger was not, and it is the one a caller who asks
// for the server's logger actually gets. Worth pinning since Logger is exported: its
// contract is "never nil, and mine when I supplied one".
func TestLoggerReturnsTheConfiguredOne(t *testing.T) {
	var buf bytes.Buffer
	want := slog.New(slog.NewJSONHandler(&buf, nil))
	srv, _ := newTestServerConfig(t, gguftest.Default(), Config{Logger: want})
	if got := srv.Logger(); got != want {
		t.Errorf("Logger() = %p, want the configured %p", got, want)
	}
	// And it is the same logger the request wrapper writes through, which is what makes
	// returning it useful rather than cosmetic.
	srv.Handler().ServeHTTP(httptest.NewRecorder(),
		httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if buf.Len() == 0 {
		t.Error("no record was written through the logger Logger returned")
	}
}

// TestServerErrorIsLoggedAsWarning covers the 5xx level split.
//
// Only a 5xx is a server-side failure worth a warning; a 4xx is the client's mistake, and
// logging those as warnings trains a reader to ignore the level. Both arms had the 4xx half
// covered.
func TestServerErrorIsLoggedAsWarning(t *testing.T) {
	var buf bytes.Buffer
	h := slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})
	srv, _ := newTestServerConfig(t, gguftest.Default(), Config{Logger: slog.New(h)})

	// A route that exists with the wrong method: handleUnimplemented answers 405, which is
	// below 500, so drive a real 5xx instead through the handler that writes one.
	srv.Handler().ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(
		http.MethodPost, "/v1/completions", strings.NewReader(`{"prompt":`)))
	var rec map[string]any
	if err := json.Unmarshal(bytes.TrimSpace(buf.Bytes()), &rec); err != nil {
		t.Fatalf("log record is not JSON: %v (%q)", err, buf.String())
	}
	// A malformed body is a 400, so this asserts the *other* arm: it must be Info.
	if lvl, _ := rec["level"].(string); lvl != "INFO" {
		t.Errorf("level = %v, want INFO for a 400", rec["level"])
	}

	buf.Reset()
	// Drive the 5xx arm directly through the wrapper: a handler that answers 500 is the
	// smallest way to reach it, and it does not depend on a particular engine state.
	srv.logRequest(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	})).ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/x", nil))
	if err := json.Unmarshal(bytes.TrimSpace(buf.Bytes()), &rec); err != nil {
		t.Fatalf("log record is not JSON: %v (%q)", err, buf.String())
	}
	if lvl, _ := rec["level"].(string); lvl != "WARN" {
		t.Errorf("level = %v, want WARN for a 5xx", rec["level"])
	}
	if msg, _ := rec["msg"].(string); msg != "request failed" {
		t.Errorf("msg = %v, want request failed", rec["msg"])
	}
}
