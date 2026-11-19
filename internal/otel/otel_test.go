package otel

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// testSpan builds a finished span with explicit ids so a test can check the encoding.
func testSpan(t *testing.T, name string) Span {
	t.Helper()
	var tid TraceID
	var sid SpanID
	for i := range tid {
		tid[i] = byte(i + 1)
	}
	for i := range sid {
		sid[i] = byte(0x80 + i)
	}
	start := time.Unix(1700000000, 0)
	return Span{
		TraceID: tid, SpanID: sid, Name: name, Kind: 1,
		Start: start, End: start.Add(1500 * time.Millisecond),
		Status: 1,
	}
}

// unmarshalRequest decodes an exported body into the wire types, which is how these
// tests check field names: a wrong json tag produces a zero field here rather than a
// silent reordering.
func unmarshalRequest(t *testing.T, body []byte) exportRequest {
	t.Helper()
	var req exportRequest
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("unmarshal: %v\nbody: %s", err, body)
	}
	return req
}

// TestEncodeIdsAreHex is the one place OTLP/JSON departs from proto3's standard mapping.
//
// proto3 would put a bytes field in base64, and this exporter did exactly that until the
// specification was read: "The traceId and spanId byte arrays are represented as
// case-insensitive hex-encoded strings; they are not base64-encoded as is defined in the
// standard Protobuf JSON Mapping."
func TestEncodeIdsAreHex(t *testing.T) {
	span := testSpan(t, "chat")
	body, err := Encode(nil, []Span{span})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	req := unmarshalRequest(t, body)
	spans := req.ResourceSpans[0].ScopeSpans[0].Spans
	if len(spans) != 1 {
		t.Fatalf("%d spans, want 1", len(spans))
	}

	if want := span.TraceID.Hex(); spans[0].TraceID != want {
		t.Errorf("traceId = %q, want hex %q", spans[0].TraceID, want)
	}
	if want := span.SpanID.Hex(); spans[0].SpanID != want {
		t.Errorf("spanId = %q, want hex %q", spans[0].SpanID, want)
	}

	// Hex has a fixed length per id width, and a collector rejects anything else: 32
	// characters for a 16-byte trace id, 16 for an 8-byte span id.
	if got := len(spans[0].TraceID); got != 32 {
		t.Errorf("traceId is %d characters, want 32", got)
	}
	if got := len(spans[0].SpanID); got != 16 {
		t.Errorf("spanId is %d characters, want 16", got)
	}
	// A base64 id would decode here, which is the mistake being guarded against.
	if _, err := base64.StdEncoding.DecodeString(spans[0].TraceID); err == nil &&
		len(spans[0].TraceID) != 32 {
		t.Errorf("traceId %q looks base64-encoded", spans[0].TraceID)
	}
}

// TestEncodeTimestampsAreNanos checks that timestamps are strings of nanoseconds, not
// numbers: a JSON number cannot hold a nanosecond timestamp at float64 precision.
func TestEncodeTimestampsAreNanos(t *testing.T) {
	span := testSpan(t, "chat")
	body, err := Encode(nil, []Span{span})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	req := unmarshalRequest(t, body)
	got := req.ResourceSpans[0].ScopeSpans[0].Spans[0]
	if want := "1700000000000000000"; got.StartTimeUnixNano != want {
		t.Errorf("startTimeUnixNano = %q, want %q", got.StartTimeUnixNano, want)
	}
	if want := "1700000001500000000"; got.EndTimeUnixNano != want {
		t.Errorf("endTimeUnixNano = %q, want %q", got.EndTimeUnixNano, want)
	}
	// A numeric timestamp would round-trip as a float, so check the raw JSON too.
	if !strings.Contains(string(body), `"startTimeUnixNano":"1700000000000000000"`) {
		t.Error("startTimeUnixNano is not JSON-quoted; a number loses nanosecond precision")
	}
}

// TestEncodeFieldNames pins the proto3 JSON names. An exporter that emits "trace_id"
// instead of "traceId" sends traces a collector silently ignores.
func TestEncodeFieldNames(t *testing.T) {
	body, err := Encode(nil, []Span{testSpan(t, "chat")})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	for _, want := range []string{
		`"resourceSpans"`, `"scopeSpans"`, `"spans"`,
		`"traceId"`, `"spanId"`, `"name"`, `"kind"`,
		`"startTimeUnixNano"`, `"endTimeUnixNano"`,
	} {
		if !strings.Contains(string(body), want) {
			t.Errorf("body is missing %s: %s", want, body)
		}
	}
	// snake_case would decode to zero fields and drop the trace.
	for _, bad := range []string{`"trace_id"`, `"span_id"`, `"start_time_unix_nano"`, `"resource_spans"`} {
		if strings.Contains(string(body), bad) {
			t.Errorf("body contains proto field name %s, not its JSON mapping", bad)
		}
	}
}

// TestEncodeAttributes checks the typed value union: an integer must use intValue, not a
// quoted string, or a backend cannot sum it.
func TestEncodeAttributes(t *testing.T) {
	span := testSpan(t, "chat")
	span.Attributes = []Attribute{
		StringAttr("gen_ai.request.model", "qwen2"),
		IntAttr("gen_ai.usage.output_tokens", 42),
	}
	body, err := Encode(nil, []Span{span})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	var probe struct {
		ResourceSpans []struct {
			ScopeSpans []struct {
				Spans []struct {
					Attributes []struct {
						Key   string `json:"key"`
						Value struct {
							StringValue *string `json:"stringValue"`
							IntValue    *string `json:"intValue"`
						} `json:"value"`
					} `json:"attributes"`
				} `json:"spans"`
			} `json:"scopeSpans"`
		} `json:"resourceSpans"`
	}
	if err := json.Unmarshal(body, &probe); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	attrs := probe.ResourceSpans[0].ScopeSpans[0].Spans[0].Attributes
	if len(attrs) != 2 {
		t.Fatalf("%d attributes, want 2", len(attrs))
	}
	if attrs[0].Value.StringValue == nil || *attrs[0].Value.StringValue != "qwen2" {
		t.Errorf("string attribute = %+v", attrs[0].Value)
	}
	if attrs[1].Value.IntValue == nil || *attrs[1].Value.IntValue != "42" {
		t.Errorf("int attribute = %+v, want intValue 42", attrs[1].Value)
	}
}

// TestEncodeSkipsInvalidSpans protects the batch: a collector rejects a request whose
// trace id is invalid, so one malformed span would discard every good one with it.
func TestEncodeSkipsInvalidSpans(t *testing.T) {
	good := testSpan(t, "good")
	bad := testSpan(t, "bad")
	bad.TraceID = TraceID{}
	unfinished := testSpan(t, "unfinished")
	unfinished.End = time.Time{}
	// A span whose end precedes its start is the other half of the guard, and it is not the
	// same condition: a zero End is Before every Start, but a finite End can be moved back
	// without becoming zero. TestEncodeSkipsInvalidSpans only covered the zero case, so
	// dropping `s.End.Before(s.Start)` passed the otel and engine suites and let a span with
	// a negative duration reach the collector -- which renders as a bar that ends before it
	// begins, or is dropped by the backend's own validation, depending on the viewer.
	backdated := testSpan(t, "backdated")
	backdated.End = backdated.End.Add(-time.Hour)

	body, err := Encode(nil, []Span{good, bad, unfinished, backdated})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	spans := unmarshalRequest(t, body).ResourceSpans[0].ScopeSpans[0].Spans
	if len(spans) != 1 {
		t.Fatalf("%d spans, want only the valid one", len(spans))
	}
	if spans[0].Name != "good" {
		t.Errorf("kept %q, want good", spans[0].Name)
	}
}

// TestEncodeResource checks that resource attributes land on the resource and not on the
// span, where a backend would not treat them as identity.
func TestEncodeResource(t *testing.T) {
	body, err := Encode([]Attribute{StringAttr("service.name", "orbix")}, []Span{testSpan(t, "chat")})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if !strings.Contains(string(body), `"service.name"`) {
		t.Errorf("resource attribute missing: %s", body)
	}
	res := unmarshalRequest(t, body).ResourceSpans[0].Resource
	if res == nil || len(res.Attributes) != 1 {
		t.Fatalf("resource = %+v", res)
	}
}

// TestExporterPostsToTracesPath checks the request the collector actually receives: the
// path, the content type, and that the body decodes.
func TestExporterPostsToTracesPath(t *testing.T) {
	type received struct {
		path        string
		contentType string
		body        []byte
	}
	got := make(chan received, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		got <- received{path: r.URL.Path, contentType: r.Header.Get("Content-Type"), body: body}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	exp, err := NewExporter(ExporterOptions{
		Endpoint: srv.URL,
		Resource: []Attribute{StringAttr("service.name", "orbix")},
		Interval: time.Hour, // flush manually
	})
	if err != nil {
		t.Fatalf("NewExporter: %v", err)
	}
	defer func() { _ = exp.Close() }()

	exp.Record(testSpan(t, "chat"))
	if err := exp.Flush(context.Background()); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	select {
	case r := <-got:
		if r.path != "/v1/traces" {
			t.Errorf("path = %q, want /v1/traces", r.path)
		}
		if !strings.HasPrefix(r.contentType, "application/json") {
			t.Errorf("Content-Type = %q", r.contentType)
		}
		spans := unmarshalRequest(t, r.body).ResourceSpans[0].ScopeSpans[0].Spans
		if len(spans) != 1 || spans[0].Name != "chat" {
			t.Errorf("spans = %+v", spans)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("collector received no request")
	}
}

// TestExporterReportsCollectorRefusal covers the failure a real deployment sees most: a
// collector that answers, but answers with an error.
//
// The exporter was tested against a server that accepted everything, so this branch -- read
// a bounded amount of the body, include it in the error -- was at 0%. It is the branch that
// decides whether an operator learns *why* the collector refused: "400 Bad Request" alone
// sends them to the exporter's logs again, while the collector's own message usually names
// the problem.
func TestExporterReportsCollectorRefusal(t *testing.T) {
	// A body long enough to prove the 512-byte limit bites, with the useful part first.
	const collectorMsg = `{"error":"unknown field ResourceSpans"}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, collectorMsg+strings.Repeat("x", 2000))
	}))
	defer srv.Close()

	exp, err := NewExporter(ExporterOptions{Endpoint: srv.URL, Interval: time.Hour})
	if err != nil {
		t.Fatalf("NewExporter: %v", err)
	}
	defer func() { _ = exp.Close() }()

	exp.Record(testSpan(t, "chat"))
	err = exp.Flush(context.Background())
	if err == nil {
		t.Fatal("Flush succeeded against a collector that returned 400")
	}
	msg := err.Error()
	if !strings.Contains(msg, "400") {
		t.Errorf("error %q does not carry the status code", msg)
	}
	if !strings.Contains(msg, "unknown field ResourceSpans") {
		t.Errorf("error %q does not carry the collector's message", msg)
	}
	// Bounded read: the 2000 filler bytes must not all appear.
	if len(msg) > 700 {
		t.Errorf("error is %d bytes; the body read is supposed to be bounded at 512", len(msg))
	}
}

// TestExporterReportsUnreachableCollector covers the other half: nothing is listening.
//
// This is the case that must not crash and must not hang. The default client has a 5s
// timeout so a hung collector cannot pin the sender goroutine, and a connection refusal
// reaches the error return rather than a panic.
func TestExporterReportsUnreachableCollector(t *testing.T) {
	// Grab a port, then close it: connecting to it is refused rather than hanging.
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	url := srv.URL
	srv.Close()

	exp, err := NewExporter(ExporterOptions{Endpoint: url, Interval: time.Hour})
	if err != nil {
		t.Fatalf("NewExporter: %v", err)
	}
	defer func() { _ = exp.Close() }()

	exp.Record(testSpan(t, "chat"))
	err = exp.Flush(context.Background())
	if err == nil {
		t.Fatal("Flush succeeded with no collector listening")
	}
	if !strings.Contains(err.Error(), "export") {
		t.Errorf("error %q does not say the export failed", err)
	}
}

// TestExporterToleratesEndpointWithPath accepts both spellings of the endpoint, which
// appear interchangeably in collector documentation.
func TestExporterToleratesEndpointWithPath(t *testing.T) {
	// The trailing-slash spelling of the full path is the one that used to break: the path
	// was stripped before the slash, so the suffix did not match, the trim left the path in
	// place, and the caller appended it a second time.
	for _, in := range []string{
		"http://host:4318",
		"http://host:4318/",
		"http://host:4318/v1/traces",
		"http://host:4318/v1/traces/",
		"http://host:4318//",
		// A collector mounted under a prefix keeps it.
		"http://host:4318/otlp/",
	} {
		want := "http://host:4318"
		if strings.Contains(in, "/otlp") {
			want = "http://host:4318/otlp"
		}
		if got := normalizeEndpoint(in); got != want {
			t.Errorf("normalizeEndpoint(%q) = %q, want %q", in, got, want)
		}
		// The property that actually matters: appending the path once yields exactly one
		// copy of it.
		if got := normalizeEndpoint(in) + tracesPath; strings.Count(got, tracesPath) != 1 {
			t.Errorf("%q normalizes to %q, which has the path twice", in, got)
		}
	}
}

// TestExporterDropsRatherThanBlocks is the availability property: tracing must never make
// a request slow, so a full queue loses spans and counts them.
// TestExporterRecordAccountsUnderConcurrency pins the queue bound from several goroutines at
// once, which the sequential drop test cannot see.
//
// Record checks the length and appends inside one critical section, so the count it compares
// against is the count its append extends. The engine's Submit had the same intent but read
// the length in one lock and inserted in another, and admitted twelve requests against a cap
// of three the moment submissions were concurrent -- a defect no sequential test could reach.
// Recording it here keeps the correct shape from being refactored into the wrong one for the
// same reasons.
//
// Every span must land in exactly one place: queued, or counted as dropped. A race would show
// up as a span that is neither, or as a queue longer than its limit.
//
// What this test does not do is catch a split critical section on its own, which is worth
// stating so nobody trusts it further than it goes. Injecting the engine's old shape here --
// read the length, unlock, re-lock to append -- passed this test: with no work between the two
// locks the window is nanoseconds wide, and the incorrect version admitted 8 spans instead of
// overrunning. Adding a microsecond of work between them made the same injection overrun to 11
// against a limit of 8, which is what confirms the injected shape was wrong and the test was
// simply not sensitive enough.
//
// The honest summary: this pins that the accounting adds up and that the bound is respected in
// normal operation, not that the critical section is single.
func TestExporterRecordAccountsUnderConcurrency(t *testing.T) {
	const maxQueue = 64
	const goroutines, each = 16, 40

	exp, err := NewExporter(ExporterOptions{
		Endpoint: "http://127.0.0.1:1", Interval: time.Hour, MaxQueue: maxQueue,
	})
	if err != nil {
		t.Fatalf("NewExporter: %v", err)
	}
	defer func() { _ = exp.Close() }()

	var wg sync.WaitGroup
	for range goroutines {
		wg.Go(func() {
			for range each {
				exp.Record(testSpan(t, "op"))
			}
		})
	}
	wg.Wait()

	exp.mu.Lock()
	queued, dropped := len(exp.queue), exp.dropped
	exp.mu.Unlock()

	if queued > maxQueue {
		t.Errorf("queue holds %d spans, above its limit of %d", queued, maxQueue)
	}
	if want := goroutines * each; queued+dropped != want {
		t.Errorf("queued %d + dropped %d = %d, want every one of %d accounted for",
			queued, dropped, queued+dropped, want)
	}
	if dropped == 0 {
		t.Error("nothing was dropped, so the limit was never reached and this proves nothing")
	}
}

func TestExporterDropsRatherThanBlocks(t *testing.T) {
	// A collector that never answers, so nothing drains. The client is given a short
	// timeout so Close does not wait for the default one: the point of this test is
	// that Record does not block, not how long a hung request takes to give up.
	block := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-block
	}))

	exp, err := NewExporter(ExporterOptions{
		Endpoint: srv.URL, Interval: time.Hour, MaxQueue: 2, MaxBatch: 1,
		Client: &http.Client{Timeout: 50 * time.Millisecond},
	})
	if err != nil {
		t.Fatalf("NewExporter: %v", err)
	}
	defer func() {
		_ = exp.Close()
		close(block)
		srv.Close()
	}()

	// Recording more than the queue holds must return promptly.
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range 10 {
			exp.Record(testSpan(t, "chat"))
		}
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Record blocked on a full queue")
	}
	if exp.Dropped() == 0 {
		t.Error("Dropped = 0 after overflowing a queue of 2")
	}
}

// TestExporterCloseIsIdempotent makes the documented lifetime safe to follow literally.
func TestExporterCloseIsIdempotent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()
	exp, err := NewExporter(ExporterOptions{Endpoint: srv.URL, Interval: time.Hour})
	if err != nil {
		t.Fatalf("NewExporter: %v", err)
	}
	if err := exp.Close(); err != nil {
		t.Errorf("first Close: %v", err)
	}
	if err := exp.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
}

// TestExporterRequiresEndpoint keeps a misconfiguration from silently discarding traces.
func TestExporterRequiresEndpoint(t *testing.T) {
	if _, err := NewExporter(ExporterOptions{}); err == nil {
		t.Fatal("NewExporter with no endpoint succeeded")
	}
}

// TestExporterBatchSplitting checks that a queue larger than MaxBatch goes out in several
// requests rather than one oversized body.
func TestExporterBatchSplitting(t *testing.T) {
	var mu sync.Mutex
	var requests int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requests++
		mu.Unlock()
	}))
	defer srv.Close()

	exp, err := NewExporter(ExporterOptions{
		Endpoint: srv.URL, Interval: time.Hour, MaxBatch: 2,
	})
	if err != nil {
		t.Fatalf("NewExporter: %v", err)
	}
	defer func() { _ = exp.Close() }()
	for range 5 {
		exp.Record(testSpan(t, "chat"))
	}
	if err := exp.Flush(context.Background()); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if requests != 3 {
		t.Errorf("%d requests for 5 spans at MaxBatch 2, want 3", requests)
	}
}

// TestTracerProducesValidSpans checks that generated ids are usable and distinct.
func TestTracerProducesValidSpans(t *testing.T) {
	tr := NewTracer()
	root, ok := tr.StartTrace("request", 2)
	if !ok {
		t.Fatal("StartTrace failed")
	}
	if !root.TraceID.IsValid() || !root.SpanID.IsValid() {
		t.Fatal("StartTrace produced an invalid id")
	}
	child, ok := tr.StartChild(root, "prefill", 1)
	if !ok {
		t.Fatal("StartChild failed")
	}
	if child.TraceID != root.TraceID {
		t.Error("child has a different trace id")
	}
	if child.ParentID != root.SpanID {
		t.Error("child does not point at its parent")
	}
	if child.SpanID == root.SpanID {
		t.Error("child reused the parent's span id")
	}
}

// TestSpanFinishSetsStatus checks that Finish records both the outcome and its message,
// which is what makes a failed request visible in a trace.
func TestSpanFinishSetsStatus(t *testing.T) {
	tr := NewTracer()
	s, ok := tr.StartTrace("request", 2)
	if !ok {
		t.Fatal("StartTrace failed")
	}
	done := s.Finish(nil)
	if done.Status != 1 {
		t.Errorf("status = %d after a nil error, want 1", done.Status)
	}
	failed := s.Finish(errors.New("boom"))
	if failed.Status != 2 {
		t.Errorf("status = %d after an error, want 2", failed.Status)
	}
	if failed.StatusMessage != "boom" {
		t.Errorf("status message = %q, want boom", failed.StatusMessage)
	}
	if failed.Duration() < 0 {
		t.Errorf("negative duration %v", failed.Duration())
	}
}

// TestCloseFlushesQueuedSpans checks that a span recorded just before shutdown still reaches
// the collector.
//
// The interval is set to an hour so nothing is sent on a tick: the only thing that can
// deliver the span is Close. That is the path a server takes on SIGTERM, where the last
// request's span is recorded during shutdown, and losing it would leave a trace that ends
// one request early with nothing to say why.
//
// TestExporterCloseIsIdempotent covers calling Close twice; this covers Close doing its job
// once. They were separate gaps: the earlier test would pass an implementation that dropped
// the queue on the floor.
func TestCloseFlushesQueuedSpans(t *testing.T) {
	var (
		mu       sync.Mutex
		requests int
		spans    int
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			ResourceSpans []struct {
				ScopeSpans []struct {
					Spans []json.RawMessage `json:"spans"`
				} `json:"scopeSpans"`
			} `json:"resourceSpans"`
		}
		_ = json.NewDecoder(r.Body).Decode(&payload)
		n := 0
		for _, rs := range payload.ResourceSpans {
			for _, ss := range rs.ScopeSpans {
				n += len(ss.Spans)
			}
		}
		mu.Lock()
		requests++
		spans += n
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	exp, err := NewExporter(ExporterOptions{
		Endpoint: srv.URL,
		// An hour, so the ticker cannot be what sends this.
		Interval: time.Hour,
	})
	if err != nil {
		t.Fatalf("NewExporter: %v", err)
	}
	tr := exp.Tracer()
	span, ok := tr.StartTrace("completion", 2)
	if !ok {
		t.Fatal("StartTrace failed")
	}
	exp.Record(span.Finish(nil))

	if err := exp.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if requests == 0 {
		t.Fatal("Close returned without posting anything: a span recorded before shutdown was lost")
	}
	if spans != 1 {
		t.Errorf("posted %d spans, want 1", spans)
	}
}

// TestExporterRecordsFromManyGoroutines covers the concurrency the exporter exists for.
//
// A span is recorded from whatever goroutine finished the work, so the queue is written by as many
// goroutines as the server has requests in flight while the background sender drains it. The
// existing tests record from one goroutine, so the locking was only ever exercised serially.
//
// Two properties matter and they are different: nothing may be lost (every span recorded under the
// queue limit has to arrive), and the split between sent and dropped must add up. A lock that
// dropped a span silently would still produce a valid-looking export, and the count is what says
// otherwise.
func TestExporterRecordsFromManyGoroutines(t *testing.T) {
	const (
		goroutines = 8
		perG       = 25
		total      = goroutines * perG
	)
	var got atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err == nil {
			if res, ok := payload["resourceSpans"].([]any); ok {
				for _, rs := range res {
					m, _ := rs.(map[string]any)
					if ss, ok := m["scopeSpans"].([]any); ok {
						for _, s := range ss {
							sm, _ := s.(map[string]any)
							spans, _ := sm["spans"].([]any)
							got.Add(int64(len(spans)))
						}
					}
				}
			}
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	// A queue large enough to hold everything, so the drop path is not what is under test here:
	// this test is about nothing being lost to a race on the queue itself.
	exp, err := NewExporter(ExporterOptions{
		Endpoint: srv.URL, Interval: 20 * time.Millisecond,
		MaxQueue: total * 2, MaxBatch: 7,
	})
	if err != nil {
		t.Fatalf("NewExporter: %v", err)
	}

	tr := exp.Tracer()
	var wg sync.WaitGroup
	for g := range goroutines {
		wg.Go(func() {
			for i := range perG {
				span, ok := tr.StartTrace(fmt.Sprintf("g%d-i%d", g, i), 1)
				if !ok {
					t.Errorf("StartTrace failed")
					return
				}
				exp.Record(span.Finish(nil))
			}
		})
	}
	wg.Wait()

	// Close flushes what is still queued, so after it returns every recorded span has either
	// been posted or counted as dropped.
	if err := exp.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if dropped := exp.Dropped(); dropped != 0 {
		t.Errorf("dropped %d spans with a queue of %d", dropped, total*2)
	}
	if n := got.Load(); n != total {
		t.Errorf("collector received %d spans, want %d", n, total)
	}
}

// TestEncodeStringArrayAttribute covers the arrayValue branch of the value union.
//
// gen_ai.response.finish_reasons is typed string[] by the convention, so it has to go out as
// an array rather than a scalar under the singular name -- a backend queries the documented
// shape, and a scalar is invisible to it.
func TestEncodeStringArrayAttribute(t *testing.T) {
	span := testSpan(t, "chat")
	span.Attributes = append(span.Attributes,
		StringsAttr("gen_ai.response.finish_reasons", "stop", "length"))
	body, err := Encode(nil, []Span{span})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	req := unmarshalRequest(t, body)
	attrs := req.ResourceSpans[0].ScopeSpans[0].Spans[0].Attributes

	var found *keyValue
	for i := range attrs {
		if attrs[i].Key == "gen_ai.response.finish_reasons" {
			found = &attrs[i]
		}
	}
	if found == nil {
		t.Fatalf("finish_reasons attribute is missing: %+v", attrs)
	}
	if found.Value.ArrayValue == nil {
		t.Fatalf("finish_reasons is not an array: %+v", found.Value)
	}
	vals := found.Value.ArrayValue.Values
	if len(vals) != 2 {
		t.Fatalf("%d values, want 2", len(vals))
	}
	for i, want := range []string{"stop", "length"} {
		if vals[i].StringValue == nil || *vals[i].StringValue != want {
			t.Errorf("values[%d] = %+v, want %q", i, vals[i], want)
		}
	}
	// An array attribute must not also carry a scalar, or a reader that checks for one
	// gets a value the convention says is not there.
	if found.Value.StringValue != nil {
		t.Error("array attribute also has stringValue set")
	}
}

// TestEncodeBoolAttribute covers the boolValue branch.
//
// gen_ai.request.stream is a boolean in the convention, and a backend reads it as one; a quoted
// "true" in stringValue is a different attribute value type.
func TestEncodeBoolAttribute(t *testing.T) {
	span := testSpan(t, "chat")
	span.Attributes = append(span.Attributes, BoolAttr("gen_ai.request.stream", true))
	body, err := Encode(nil, []Span{span})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	req := unmarshalRequest(t, body)
	attrs := req.ResourceSpans[0].ScopeSpans[0].Spans[0].Attributes

	var found *keyValue
	for i := range attrs {
		if attrs[i].Key == "gen_ai.request.stream" {
			found = &attrs[i]
		}
	}
	if found == nil {
		t.Fatalf("gen_ai.request.stream is missing: %+v", attrs)
	}
	if found.Value.BoolValue == nil {
		t.Fatalf("stream is not a boolean: %+v", found.Value)
	}
	if !*found.Value.BoolValue {
		t.Error("stream is false, want true")
	}
	// A boolean must not also arrive as a string, which is how a backend would read it as
	// a non-empty value regardless of true or false.
	if found.Value.StringValue != nil {
		t.Error("boolean attribute also has stringValue set")
	}
}

// TestEncodeDoesNotEscapeHTML pins the encoder setting that keeps attributes readable.
//
// A prompt is ordinary text that can contain <, > and &, and Go's encoder escapes those to
// \u003c and friends by default. The bytes mean the same thing to a collector, so nothing
// breaks either way -- which is why no test noticed: turning SetEscapeHTML back on passed the
// otel and engine suites. What changes is the size of every trace that carries prompt text,
// and whether the attribute can be read in a collector's UI without decoding.
func TestEncodeDoesNotEscapeHTML(t *testing.T) {
	sp := testSpan(t, "chat")
	sp.Attributes = []Attribute{StringAttr("gen_ai.prompt", "x < y & z > w")}
	body, err := Encode(nil, []Span{sp})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if !strings.Contains(string(body), "x < y & z > w") {
		t.Errorf("attribute text was escaped: %s", body)
	}
	for _, esc := range []string{`\u003c`, `\u0026`, `\u003e`} {
		if strings.Contains(string(body), esc) {
			t.Errorf("body contains %s: the encoder escaped HTML characters", esc)
		}
	}
	// The escaping must not have been turned off to the point of producing invalid JSON.
	unmarshalRequest(t, body)
}

// TestExporterTreatsNon2xxAsFailure pins both ends of the success range.
//
// send accepts 200..299 and reports anything else. The upper half of that guard is what these
// cases pin: the only refusal test in the suite used 400, so covering 3xx and 5xx says the
// comparison is against a range rather than against one value.
//
// The lower comparison, `resp.StatusCode < 200`, cannot be proved here and is not claimed to
// be. net/http treats a 1xx WriteHeader as an interim response, so an httptest handler cannot
// hand the client a bare 199 to answer; simplifying the guard to `>= 300` still passes this
// test and every other. The comparison stays in the code because 200..299 is the range that
// is actually meant, and one comparison would be wrong the first time an interim status
// reached an HTTP/2 client -- but a test that passes for the wrong reason is worse than an
// honest note.
func TestExporterTreatsNon2xxAsFailure(t *testing.T) {
	// 1xx is deliberately absent: net/http treats a 1xx WriteHeader as an interim response
	// rather than the final one, so an httptest server cannot hand the client a bare 199.
	// The lower comparison is still worth having in the code -- the range is 200..299, and
	// writing it as one comparison would be wrong even where no test can reach it -- but it
	// is not something this suite can prove, and pretending otherwise would be a test that
	// passes for the wrong reason.
	for _, code := range []int{300, 418, 500} {
		t.Run(fmt.Sprint(code), func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(code)
			}))
			defer srv.Close()

			exp, err := NewExporter(ExporterOptions{Endpoint: srv.URL, Interval: time.Hour})
			if err != nil {
				t.Fatalf("NewExporter: %v", err)
			}
			defer func() { _ = exp.Close() }()

			exp.Record(testSpan(t, "chat"))
			if err := exp.Flush(context.Background()); err == nil {
				t.Errorf("Flush reported success for status %d", code)
			}
			if got := exp.Dropped(); got != 1 {
				t.Errorf("dropped %d spans after a %d, want 1", got, code)
			}
		})
	}

	// The accepting side, so a guard that rejects everything cannot pass either.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()
	exp, err := NewExporter(ExporterOptions{Endpoint: srv.URL, Interval: time.Hour})
	if err != nil {
		t.Fatalf("NewExporter: %v", err)
	}
	defer func() { _ = exp.Close() }()
	exp.Record(testSpan(t, "chat"))
	if err := exp.Flush(context.Background()); err != nil {
		t.Errorf("Flush failed against a 204: %v", err)
	}
}

// TestExporterFlushesOnItsInterval covers the ticker path, which is how a running server exports.
//
// Every other exporter test sets Interval to an hour and calls Flush itself, so the branch that
// sends without being asked never ran. That is the path production uses: orbix serve creates an
// exporter and never flushes it, so spans reach the collector only through this ticker.
//
// A short interval is used rather than waiting on the default second, and the assertion is that a
// request arrives without any call to Flush or Close.
func TestExporterFlushesOnItsInterval(t *testing.T) {
	got := make(chan struct{}, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.WriteHeader(http.StatusOK)
		select {
		case got <- struct{}{}:
		default:
		}
	}))
	defer srv.Close()

	exp, err := NewExporter(ExporterOptions{
		Endpoint: srv.URL,
		Interval: 20 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("NewExporter: %v", err)
	}
	// Close waits for run to stop, so a leak here would fail the test rather than the process.
	defer func() { _ = exp.Close() }()

	exp.Record(testSpan(t, "ticked"))

	select {
	case <-got:
		// The ticker sent it with no help from the test.
	case <-time.After(5 * time.Second):
		t.Fatal("no export arrived on the interval; the ticker branch is not running")
	}
}

// TestExporterDropsWhenTheCollectorIsDown covers the flush error path under the ticker.
//
// run ignores the error from flush, which is deliberate -- there is nowhere to report it from a
// goroutine -- but the spans must still be dropped rather than re-queued, or a collector that is
// down turns the exporter into an unbounded buffer. Nothing asserted the queue drains when a send
// fails, so a change to re-queue would only show up as growing memory in production.
func TestExporterDropsWhenTheCollectorIsDown(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	exp, err := NewExporter(ExporterOptions{Endpoint: srv.URL, Interval: 20 * time.Millisecond})
	if err != nil {
		t.Fatalf("NewExporter: %v", err)
	}
	defer func() { _ = exp.Close() }()

	exp.Record(testSpan(t, "lost"))
	// Flush reports the failure, so the caller knows; the queue must be empty afterwards.
	if err := exp.Flush(context.Background()); err == nil {
		t.Fatal("Flush succeeded against a collector returning 500")
	}
	if n := exp.Dropped(); n == 0 {
		t.Error("the failed send was not counted as dropped")
	}
	// A second flush has nothing left to send, which is what proves the spans were dropped rather
	// than kept for another attempt.
	if err := exp.Flush(context.Background()); err != nil {
		t.Errorf("the second flush still had work: %v", err)
	}
}

// TestMaxBatchSplitsTheQueue covers the option that bounds one request's size.
//
// A stalled collector must not be given arbitrarily much at once, so a flush sends the queue in
// chunks of MaxBatch. Nothing asserted it: changing the default from 512 to 1 passed every test, and
// so did making the interval an hour, because the tests that need a flush call it themselves.
//
// The count of requests is what distinguishes the two: six spans with MaxBatch 2 must arrive as three
// posts of two, not one of six.
func TestMaxBatchSplitsTheQueue(t *testing.T) {
	var mu sync.Mutex
	var sizes []int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		// Count the spans in the export request rather than trusting the client's own accounting.
		spans := unmarshalRequest(t, body).ResourceSpans[0].ScopeSpans[0].Spans
		mu.Lock()
		sizes = append(sizes, len(spans))
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	exp, err := NewExporter(ExporterOptions{
		Endpoint: srv.URL,
		Interval: time.Hour, // flushed by hand
		MaxBatch: 2,
	})
	if err != nil {
		t.Fatalf("NewExporter: %v", err)
	}
	defer func() { _ = exp.Close() }()

	for i := range 6 {
		exp.Record(testSpan(t, fmt.Sprintf("span-%d", i)))
	}
	if err := exp.Flush(context.Background()); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(sizes) != 3 {
		t.Fatalf("the flush sent %d requests for 6 spans at MaxBatch 2, want 3: %v", len(sizes), sizes)
	}
	for i, n := range sizes {
		if n != 2 {
			t.Errorf("request %d carried %d spans, want 2", i, n)
		}
	}
}

// TestMaxBatchDefaultIsUsed covers the branch that picks 512 when the option is unset.
//
// The explicit-value case above cannot see it: with MaxBatch 2 in the options, the default is never
// consulted, so changing 512 to 1 passed. This records 512 spans -- one more than the default -- and
// expects the default rather than a request per span.
func TestMaxBatchDefaultIsUsed(t *testing.T) {
	if testing.Short() {
		t.Skip("512 spans through a real server")
	}
	var mu sync.Mutex
	var posts int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		mu.Lock()
		posts++
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	exp, err := NewExporter(ExporterOptions{
		Endpoint: srv.URL,
		Interval: time.Hour,
		// MaxBatch left unset on purpose.
	})
	if err != nil {
		t.Fatalf("NewExporter: %v", err)
	}
	defer func() { _ = exp.Close() }()

	spans := make([]Span, 0, 512)
	for i := range 512 {
		spans = append(spans, testSpan(t, fmt.Sprintf("s-%d", i)))
	}
	for _, s := range spans {
		exp.Record(s)
	}
	if err := exp.Flush(context.Background()); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	// 512 spans fit exactly one default batch; a smaller default would need more than one post.
	if posts != 1 {
		t.Errorf("512 spans took %d requests with the default batch, want 1", posts)
	}
}

// TestRecordDropsInvalidSpans covers the guard before a span reaches the queue.
//
// Encode has the same rule and is tested, but it runs at flush time -- by then the span has already
// occupied queue space, and a caller recording thousands of unfinished spans would push the queue to
// its limit with work that can never be sent. Record's own check is what keeps them out.
//
// The three conditions are separate: an invalid trace id, an invalid span id, and a zero end time. The
// assertion is on what reaches the collector, which is the only thing a caller can observe.
func TestRecordDropsInvalidSpans(t *testing.T) {
	got := make(chan int, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		spans := unmarshalRequest(t, body).ResourceSpans[0].ScopeSpans[0].Spans
		select {
		case got <- len(spans):
		default:
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	exp, err := NewExporter(ExporterOptions{Endpoint: srv.URL, Interval: time.Hour})
	if err != nil {
		t.Fatalf("NewExporter: %v", err)
	}
	defer func() { _ = exp.Close() }()

	exp.Record(testSpan(t, "good"))

	noTrace := testSpan(t, "no-trace")
	noTrace.TraceID = TraceID{}
	exp.Record(noTrace)

	noSpan := testSpan(t, "no-span")
	noSpan.SpanID = SpanID{}
	exp.Record(noSpan)

	unfinished := testSpan(t, "unfinished")
	unfinished.End = time.Time{}
	exp.Record(unfinished)

	// The queue's length, not only what arrives: Encode applies the same rule at flush time, so the
	// end-to-end view cannot tell whether Record rejected these spans or merely queued them for the
	// encoder to drop. Reading the queue is what separates the two.
	exp.mu.Lock()
	queued := len(exp.queue)
	exp.mu.Unlock()
	if queued != 1 {
		t.Errorf("%d spans queued, want only the valid one -- the guard belongs in Record, not only "+
			"in Encode", queued)
	}

	if err := exp.Flush(context.Background()); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	select {
	case n := <-got:
		if n != 1 {
			t.Errorf("%d spans reached the collector, want only the valid one", n)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("no export arrived")
	}
}
