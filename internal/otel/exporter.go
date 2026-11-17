package otel

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

// Exporter posts spans to an OTLP collector.
//
// It batches and sends from one goroutine rather than per span, because a chat request
// produces a handful of spans and one HTTP request each would cost more than the work
// being traced. Record never blocks: a full queue drops spans and counts them, since
// tracing must not become the reason a request is slow.
type Exporter struct {
	endpoint string
	client   *http.Client
	tracer   *Tracer

	mu      sync.Mutex
	queue   []Span
	dropped int

	// maxBatch and maxQueue bound the memory a stalled collector can cause.
	maxBatch int
	maxQueue int

	stop    chan struct{}
	stopped chan struct{}
	once    sync.Once
}

// ExporterOptions configures an Exporter.
type ExporterOptions struct {
	// Endpoint is the collector's OTLP/HTTP address, without the /v1/traces path --
	// for example http://localhost:4318. A value that already ends in /v1/traces is
	// accepted too, because both spellings are common in collector documentation.
	Endpoint string
	// Interval is how often the queue is flushed. Zero means one second.
	Interval time.Duration
	// MaxBatch is the most spans sent in one request. Zero means 512.
	MaxBatch int
	// MaxQueue is the most spans held before new ones are dropped. Zero means 2048.
	MaxQueue int
	// Client is the HTTP client, or nil for a default one with a short timeout.
	Client *http.Client
	// Resource is attached to every exported span, typically service.name.
	Resource []Attribute
}

// tracesPath is the OTLP/HTTP path for trace data.
const tracesPath = "/v1/traces"

// NewExporter starts an exporter and its background sender.
//
// The caller owns its lifetime and must Close it, which flushes what is queued. An
// exporter that is never closed loses its last interval of spans, which is usually the
// interesting part of a short run.
func NewExporter(opts ExporterOptions) (*Exporter, error) {
	if opts.Endpoint == "" {
		return nil, errors.New("otel: exporter needs an endpoint")
	}
	interval := opts.Interval
	if interval <= 0 {
		interval = time.Second
	}
	maxBatch := opts.MaxBatch
	if maxBatch <= 0 {
		maxBatch = 512
	}
	maxQueue := opts.MaxQueue
	if maxQueue <= 0 {
		maxQueue = 2048
	}
	client := opts.Client
	if client == nil {
		// A collector that hangs must not pin a goroutine and hold spans forever.
		client = &http.Client{Timeout: 5 * time.Second}
	}

	e := &Exporter{
		endpoint: normalizeEndpoint(opts.Endpoint),
		client:   client,
		tracer:   NewTracer(opts.Resource...),
		maxBatch: maxBatch,
		maxQueue: maxQueue,
		stop:     make(chan struct{}),
		stopped:  make(chan struct{}),
	}
	go e.run(interval)
	return e, nil
}

// normalizeEndpoint strips a trailing /v1/traces so both spellings work.
func normalizeEndpoint(s string) string {
	for len(s) > len(tracesPath) && s[len(s)-len(tracesPath):] == tracesPath {
		s = s[:len(s)-len(tracesPath)]
	}
	return trimTrailingSlash(s)
}

// trimTrailingSlash removes trailing slashes.
func trimTrailingSlash(s string) string {
	for len(s) > 0 && s[len(s)-1] == '/' {
		s = s[:len(s)-1]
	}
	return s
}

// Tracer returns the tracer that stamps spans for this exporter's resource.
func (e *Exporter) Tracer() *Tracer { return e.tracer }

// Record queues a span for export.
//
// A span that is unfinished or invalid is dropped silently: it is not the caller's job to
// check, and half a span is not worth sending. A full queue drops the newest span rather
// than blocking or evicting an older one, so a burst cannot lose the start of a trace.
func (e *Exporter) Record(s Span) {
	if !s.TraceID.IsValid() || !s.SpanID.IsValid() || s.End.IsZero() {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(e.queue) >= e.maxQueue {
		e.dropped++
		return
	}
	e.queue = append(e.queue, s)
}

// Dropped reports how many spans were discarded, whether for a full queue or a failed
// send. A non-zero value means the collector is not keeping up.
func (e *Exporter) Dropped() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.dropped + e.tracer.Dropped()
}

// run flushes on a ticker until Close is called.
func (e *Exporter) run(interval time.Duration) {
	defer close(e.stopped)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			_ = e.flush(context.Background())
		case <-e.stop:
			return
		}
	}
}

// flush sends everything queued.
//
// On a send error the spans are dropped rather than re-queued: a collector that is down
// would otherwise turn the exporter into an unbounded retry buffer, and a trace is worth
// less than the request it describes. The count is kept so the loss is visible.
func (e *Exporter) flush(ctx context.Context) error {
	e.mu.Lock()
	if len(e.queue) == 0 {
		e.mu.Unlock()
		return nil
	}
	batch := e.queue
	e.queue = nil
	e.mu.Unlock()

	var firstErr error
	for len(batch) > 0 {
		n := min(e.maxBatch, len(batch))
		chunk := batch[:n]
		batch = batch[n:]
		if err := e.send(ctx, chunk); err != nil {
			if firstErr == nil {
				firstErr = err
			}
			e.mu.Lock()
			e.dropped += n
			e.mu.Unlock()
		}
	}
	return firstErr
}

// send posts one batch.
func (e *Exporter) send(ctx context.Context, spans []Span) error {
	body, err := Encode(e.tracer.Resource(), spans)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.endpoint+tracesPath, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("otel: build export request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := e.client.Do(req)
	if err != nil {
		return fmt.Errorf("otel: export: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// Read a bounded amount of the body: a collector's error message is the useful
		// part and is short, and an unbounded read from an unknown server is not.
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("otel: export: %s: %s", resp.Status, bytes.TrimSpace(msg))
	}
	return nil
}

// Flush sends everything queued now. It is what a test or a one-shot command wants
// instead of waiting for the interval.
func (e *Exporter) Flush(ctx context.Context) error { return e.flush(ctx) }

// Close flushes and stops the sender. It is safe to call more than once.
func (e *Exporter) Close() error {
	var err error
	e.once.Do(func() {
		close(e.stop)
		<-e.stopped
		err = e.flush(context.Background())
	})
	return err
}
