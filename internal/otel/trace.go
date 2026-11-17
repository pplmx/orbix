package otel

import (
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"sync"
	"time"
)

// TraceID is a 16-byte trace identifier shared by every span in one trace.
//
// The zero value is invalid, which is why NewTraceID exists rather than letting a
// literal stand: a collector rejects an all-zero id, so a default-constructed span would
// be dropped with no error anywhere the caller can see.
type TraceID [16]byte

// SpanID is an 8-byte span identifier, unique within its trace.
type SpanID [8]byte

// IsValid reports whether the id is non-zero.
func (t TraceID) IsValid() bool { return t != TraceID{} }

// IsValid reports whether the id is non-zero.
func (s SpanID) IsValid() bool { return s != SpanID{} }

// String renders the id as base64, the encoding OTLP/JSON uses for a bytes field.
func (t TraceID) String() string { return base64.StdEncoding.EncodeToString(t[:]) }

// String renders the id as base64.
func (s SpanID) String() string { return base64.StdEncoding.EncodeToString(s[:]) }

// Hex renders the id as hex, for logs. It is not the wire encoding.
func (t TraceID) Hex() string { return fmt.Sprintf("%x", t[:]) }

// Hex renders the id as hex, for logs.
func (s SpanID) Hex() string { return fmt.Sprintf("%x", s[:]) }

// NewTraceID returns a random trace id.
func NewTraceID() (TraceID, error) {
	var id TraceID
	for id == (TraceID{}) {
		if _, err := rand.Read(id[:]); err != nil {
			return TraceID{}, fmt.Errorf("otel: generate trace id: %w", err)
		}
	}
	return id, nil
}

// NewSpanID returns a random span id.
func NewSpanID() (SpanID, error) {
	var id SpanID
	for id == (SpanID{}) {
		if _, err := rand.Read(id[:]); err != nil {
			return SpanID{}, fmt.Errorf("otel: generate span id: %w", err)
		}
	}
	return id, nil
}

// Attribute is one key/value pair attached to a span.
//
// The value set is small on purpose: OTLP distinguishes typed values, and a collector
// renders each differently. Numbers are integers because every attribute this package
// emits -- token counts, batch sizes, rounds -- is one, and sending them as strings would
// make them unusable for arithmetic in a backend.
type Attribute struct {
	Key   string
	Value string
	// Int reports whether Value holds an integer, which OTLP marks with a different
	// field name than a string.
	Int bool
}

// StringAttr returns a string attribute.
func StringAttr(key, value string) Attribute { return Attribute{Key: key, Value: value} }

// IntAttr returns an integer attribute.
func IntAttr(key string, value int64) Attribute {
	return Attribute{Key: key, Value: fmt.Sprintf("%d", value), Int: true}
}

// Span is one timed operation.
//
// It is a plain value rather than a mutable object with a start/end pair, because a span
// is written exactly once at the end of the operation: sharing a live object across
// goroutines would need a lock for no benefit, and the timestamps are the only thing a
// caller has to supply.
type Span struct {
	TraceID TraceID
	SpanID  SpanID
	// ParentID is the enclosing span, or the zero value for a root span.
	ParentID SpanID
	Name     string
	// Kind is OTLP's span kind: 1 internal, 2 server, 3 client. Zero means unspecified.
	Kind       int
	Start, End time.Time
	Attributes []Attribute
	// Status is OTLP's status code: 1 ok, 2 error. Zero means unset.
	Status int
	// StatusMessage explains an error status.
	StatusMessage string
}

// Duration is how long the span ran.
func (s Span) Duration() time.Duration { return s.End.Sub(s.Start) }

// Tracer creates spans for one process.
//
// It generates ids without an error return because a generator failure is not something
// a request path can act on: tracing is observability, and a span with no id is dropped
// rather than propagated as an error. Seen() reports how many spans were dropped that
// way so a broken entropy source is still visible.
type Tracer struct {
	mu       sync.Mutex
	dropped  int
	resource []Attribute
}

// NewTracer returns a tracer that tags every span with the given resource attributes,
// typically service.name.
func NewTracer(resource ...Attribute) *Tracer {
	return &Tracer{resource: resource}
}

// Resource returns the attributes every span from this tracer carries.
func (t *Tracer) Resource() []Attribute { return t.resource }

// Dropped reports how many spans could not be created because id generation failed.
func (t *Tracer) Dropped() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.dropped
}

// StartTrace begins a new trace and returns its root span.
//
// The span's End is the zero time; Finish sets it. A caller that never finishes a span
// leaves it with a zero end time, which Export treats as "still running" and skips
// rather than emitting a span ending at the epoch.
func (t *Tracer) StartTrace(name string, kind int, attrs ...Attribute) (Span, bool) {
	traceID, err := NewTraceID()
	if err != nil {
		t.noteDropped()
		return Span{}, false
	}
	spanID, err := NewSpanID()
	if err != nil {
		t.noteDropped()
		return Span{}, false
	}
	return Span{
		TraceID: traceID, SpanID: spanID, Name: name, Kind: kind,
		Start: time.Now(), Attributes: attrs,
	}, true
}

// StartChild begins a span inside an existing trace.
func (t *Tracer) StartChild(parent Span, name string, kind int, attrs ...Attribute) (Span, bool) {
	spanID, err := NewSpanID()
	if err != nil {
		t.noteDropped()
		return Span{}, false
	}
	return Span{
		TraceID: parent.TraceID, SpanID: spanID, ParentID: parent.SpanID,
		Name: name, Kind: kind, Start: time.Now(), Attributes: attrs,
	}, true
}

// noteDropped counts a span that could not be created.
func (t *Tracer) noteDropped() {
	t.mu.Lock()
	t.dropped++
	t.mu.Unlock()
}

// Finish sets the span's end time and status. It returns the span so a caller can write
//
//	span = span.Finish(err)
func (s Span) Finish(err error) Span {
	s.End = time.Now()
	if err != nil {
		s.Status = 2
		s.StatusMessage = err.Error()
	} else {
		s.Status = 1
	}
	return s
}

// unixNanos formats a timestamp the way OTLP/JSON requires: a string of nanoseconds,
// because a JSON number cannot hold one exactly at float64 precision.
func unixNanos(t time.Time) string {
	return fmt.Sprintf("%d", t.UnixNano())
}
