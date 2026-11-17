package server

import (
	"log/slog"
	"net/http"
	"time"
)

// The server had no logging at all: errors were written to the HTTP response and
// nowhere else, so a request that failed left no record on the machine that ran it. That
// is fine for a test and wrong for a server, where the question "what happened to that
// request" has no answer after the client has gone.
//
// The logger is the standard library's log/slog rather than a logging library. What a
// server needs here is leveled, structured records and a handler the caller picks --
// text for a terminal, JSON for a collector -- and that is what slog is. A library would
// add a dependency to gain the same three call sites.

// Logger returns the server's logger, which is never nil: an unconfigured server
// discards its records rather than panicking on a nil interface, so a caller that does
// not care about logs does not have to supply one.
func (s *Server) Logger() *slog.Logger {
	if s.log == nil {
		return slog.New(slog.DiscardHandler)
	}
	return s.log
}

// logRequest wraps a handler with one record per request.
//
// It records the method, path, status, and duration, and logs at Warn for a 5xx and Info
// otherwise.
//
// This is deliberately not a general-purpose middleware stack. One record per request
// with the fields that answer "what happened" is the whole requirement, and a chain of
// wrappers would be more machinery than that.
func (s *Server) logRequest(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// A discarding logger still costs an interface call and a handler check per
		// request, so the wrapper short-circuits when nothing will be written.
		if s.log == nil {
			next.ServeHTTP(w, r)
			return
		}
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r)

		attrs := []slog.Attr{
			slog.String("method", r.Method),
			slog.String("path", r.URL.Path),
			slog.Int("status", rec.status),
			slog.Duration("duration", time.Since(start)),
			// The byte count distinguishes "answered" from "answered with nothing",
			// which a status code alone cannot.
			slog.Int("bytes", rec.written),
		}
		// A 4xx is the client's mistake rather than the server's, so only a 5xx is a
		// warning: logging bad requests as errors trains a reader to ignore the level.
		level := slog.LevelInfo
		msg := "request"
		if rec.status >= 500 {
			level, msg = slog.LevelWarn, "request failed"
		}
		s.log.LogAttrs(r.Context(), level, msg, attrs...)
	})
}

// statusRecorder captures the status code and byte count a handler produced, which the
// http.ResponseWriter does not expose after the fact.
type statusRecorder struct {
	http.ResponseWriter
	status  int
	written int
}

// WriteHeader records the status before passing it on.
func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

// Write records the byte count, defaulting the status to 200 for a handler that never
// called WriteHeader.
func (r *statusRecorder) Write(b []byte) (int, error) {
	n, err := r.ResponseWriter.Write(b)
	r.written += n
	return n, err
}

// Flush forwards to the underlying writer when it supports flushing.
//
// The SSE handlers need this: without it, wrapping the writer would break streaming, and
// the wrapper would have turned a working feature into a hung one.
func (r *statusRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}
