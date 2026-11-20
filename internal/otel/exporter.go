package otel

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
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

	// closeBudget bounds how long Close may spend flushing. Without it the flush ran on
	// context.Background() and a stalled collector held the caller for one client timeout per
	// batch: measured with a collector that accepts and never answers, Close took 603ms for
	// three batches at a 200ms timeout, and the default 5s timeout would multiply the same way.
	// For a server that is shutdown latency -- measured end to end with SIGTERM, a serving
	// process went from 30ms to 5044ms -- and a trace is worth less than an orderly exit.
	closeBudget time.Duration

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
	// CloseBudget is how long Close may spend delivering what is queued. Zero means two
	// seconds. Spans still queued when it expires are counted as dropped.
	CloseBudget time.Duration
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
	closeBudget := opts.CloseBudget
	if closeBudget <= 0 {
		closeBudget = 2 * time.Second
	}
	client := opts.Client
	if client == nil {
		// A collector that hangs must not pin a goroutine and hold spans forever.
		client = &http.Client{Timeout: 5 * time.Second}
	}

	e := &Exporter{
		endpoint:    normalizeEndpoint(opts.Endpoint),
		client:      client,
		tracer:      NewTracer(opts.Resource...),
		maxBatch:    maxBatch,
		maxQueue:    maxQueue,
		closeBudget: closeBudget,
		stop:        make(chan struct{}),
		stopped:     make(chan struct{}),
	}
	go e.run(interval)
	return e, nil
}

// normalizeEndpoint strips a trailing /v1/traces so both spellings work.
//
// The slash comes off first. Stripping the path first leaves "http://host:4318/v1/traces/" with
// its suffix unmatched, and the trailing-slash trim that follows then produces
// "http://host:4318/v1/traces", to which the caller appends the path again -- a request to
// /v1/traces/v1/traces.
func normalizeEndpoint(s string) string {
	s = trimTrailingSlash(s)
	if len(s) >= len(tracesPath) && s[len(s)-len(tracesPath):] == tracesPath {
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

// Dropped reports how many spans were discarded: for a full queue, for a failed send, or
// because the collector acknowledged the request while rejecting spans inside it. A non-zero
// value means a trace is incomplete, whether or not the export itself succeeded -- a collector
// over quota answers 200 and keeps nothing.
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
			// Bounded for the same reason Close is: this goroutine's flush is what
			// Close waits for through <-e.stopped, so an unbounded send here would hold
			// the shutdown even though Close's own flush has a budget. Measured before
			// this: with a hung collector the ticker was mid-flush when SIGTERM arrived,
			// and the process took 4522ms even though Close's budget was 2s.
			ctx, cancel := context.WithTimeout(context.Background(), e.closeBudget)
			_ = e.flush(ctx)
			cancel()
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
	// A 2xx does not mean every span was kept. OTLP's response carries partialSuccess for
	// exactly this: a collector that accepted the request and dropped some or all of the
	// spans inside it, usually for quota or for an attribute it rejects. Without reading it
	// those spans vanish with Dropped() at zero, which is the opposite of what the counter
	// is for -- measured against a collector answering
	// {"partialSuccess":{"rejectedSpans":"3","errorMessage":"quota exceeded"}}, six requests
	// left orbix_trace_spans_dropped_total at 0 while no span reached a backend.
	//
	// Read bounded, and failure to parse is not an export failure: the spans were sent, and
	// an unreadable acknowledgement is the collector's problem rather than a reason to
	// report the send as failed.
	ack, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	if rejected := partialSuccessRejected(ack); rejected > 0 {
		e.mu.Lock()
		e.dropped += rejected
		e.mu.Unlock()
	}
	return nil
}

// partialSuccessRejected reads the rejected span count from an OTLP/HTTP response.
//
// proto3 omits zero-valued fields, so a clean export has no partialSuccess at all and a
// present one with no rejectedSpans is indistinguishable from an absent one -- both mean
// nothing was dropped. rejectedSpans is an int64 in the proto, which proto3's JSON mapping
// writes as a string, so the string form is what a collector sends; a number is accepted too
// because the mapping permits it and a collector in the wild may use it.
func partialSuccessRejected(body []byte) int {
	if len(body) == 0 {
		return 0
	}
	var resp struct {
		PartialSuccess struct {
			RejectedSpans json.RawMessage `json:"rejectedSpans"`
		} `json:"partialSuccess"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return 0
	}
	raw := bytes.TrimSpace(resp.PartialSuccess.RejectedSpans)
	if len(raw) == 0 {
		return 0
	}
	var n int64
	if err := json.Unmarshal(raw, &n); err == nil {
		return int(n)
	}
	// The string form proto3 uses for 64-bit integers.
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return 0
	}
	if v, err := strconv.ParseInt(s, 10, 64); err == nil {
		return int(v)
	}
	return 0
}

// Flush sends everything queued now. It is what a test or a one-shot command wants
// instead of waiting for the interval.
func (e *Exporter) Flush(ctx context.Context) error { return e.flush(ctx) }

// Close flushes and stops the sender. It is safe to call more than once.
//
// Only the first call can report an error. A later call returns nil because the flush already
// happened and its result was that call's business: measured with a collector answering 500,
// the first Close returned "otel: export: 500 Internal Server Error" and the second returned
// nil. A caller that defers Close and also checks the error from an explicit call will see the
// deferred one fail and the checked one succeed, so the error has to be read from the first
// call -- the alternative, caching the result for every caller afterwards, would let a
// shutdown path that closes twice report the same failure twice.
//
// The flush is bounded by CloseBudget rather than running on context.Background(). A
// collector that accepts a connection and never answers otherwise holds the caller for one
// client timeout per batch, and Close is deferred in a server's shutdown path: measured end
// to end, SIGTERM to process exit went from 30ms to 5044ms with a hung collector, which is
// longer than the shutdown grace the server promises for in-flight responses. A trace is
// worth less than an orderly exit, so spans still queued when the budget expires are
// counted as dropped and the call returns.
func (e *Exporter) Close() error {
	var err error
	e.once.Do(func() {
		close(e.stop)
		<-e.stopped
		ctx, cancel := context.WithTimeout(context.Background(), e.closeBudget)
		defer cancel()
		err = e.flush(ctx)
	})
	return err
}
