package otel

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
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

// TestEncodeIdsAreBase64 is the detail most hand-written exporters get wrong: OTLP/JSON
// follows proto3's mapping, where a bytes field is base64, not hex.
func TestEncodeIdsAreBase64(t *testing.T) {
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

	wantTrace := base64.StdEncoding.EncodeToString(span.TraceID[:])
	if spans[0].TraceID != wantTrace {
		t.Errorf("traceId = %q, want base64 %q (hex would be %s)",
			spans[0].TraceID, wantTrace, span.TraceID.Hex())
	}
	wantSpan := base64.StdEncoding.EncodeToString(span.SpanID[:])
	if spans[0].SpanID != wantSpan {
		t.Errorf("spanId = %q, want base64 %q", spans[0].SpanID, wantSpan)
	}

	// The decoded id must be the right length: a collector rejects anything that is not
	// 16 bytes for a trace and 8 for a span.
	raw, err := base64.StdEncoding.DecodeString(spans[0].TraceID)
	if err != nil {
		t.Fatalf("traceId is not base64: %v", err)
	}
	if len(raw) != 16 {
		t.Errorf("trace id decodes to %d bytes, want 16", len(raw))
	}
	rawSpan, _ := base64.StdEncoding.DecodeString(spans[0].SpanID)
	if len(rawSpan) != 8 {
		t.Errorf("span id decodes to %d bytes, want 8", len(rawSpan))
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

	body, err := Encode(nil, []Span{good, bad, unfinished})
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

// TestExporterToleratesEndpointWithPath accepts both spellings of the endpoint, which
// appear interchangeably in collector documentation.
func TestExporterToleratesEndpointWithPath(t *testing.T) {
	for _, in := range []string{"http://host:4318", "http://host:4318/", "http://host:4318/v1/traces"} {
		if got := normalizeEndpoint(in); got != "http://host:4318" {
			t.Errorf("normalizeEndpoint(%q) = %q, want http://host:4318", in, got)
		}
	}
}

// TestExporterDropsRatherThanBlocks is the availability property: tracing must never make
// a request slow, so a full queue loses spans and counts them.
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
