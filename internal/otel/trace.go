package otel

import (
	"crypto/rand"
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

// String renders the id as lowercase hex.
//
// OTLP's JSON encoding deviates from proto3's standard mapping here: the spec says traceId and
// spanId are "case-insensitive hex-encoded strings; they are not base64-encoded as is defined in
// the standard Protobuf JSON Mapping". This is the wire form.
func (t TraceID) String() string { return t.Hex() }

// String renders the id as lowercase hex; see TraceID.String.
func (s SpanID) String() string { return s.Hex() }

// Hex renders the id as lowercase hex, which is both the wire encoding and what logs show.
func (t TraceID) Hex() string { return fmt.Sprintf("%x", t[:]) }

// Hex renders the id as lowercase hex.
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
	// Bool reports whether Value holds a boolean, which OTLP marks with its own field
	// rather than a quoted "true".
	Bool bool
	// Strings, when non-nil, makes this a string[] attribute and Value is ignored. Some
	// convention attributes are arrays rather than scalars, and a backend queries them by
	// that shape.
	Strings []string
}

// StringAttr returns a string attribute.
func StringAttr(key, value string) Attribute { return Attribute{Key: key, Value: value} }

// BoolAttr returns a boolean attribute.
func BoolAttr(key string, value bool) Attribute {
	return Attribute{Key: key, Value: fmt.Sprintf("%t", value), Bool: true}
}

// StringsAttr returns a string[] attribute.
func StringsAttr(key string, values ...string) Attribute {
	return Attribute{Key: key, Strings: values}
}

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
//
// A zero parent's TraceID is refused rather than inherited. NewTraceID loops until its id is
// non-zero because OTLP reads an all-zero trace id as "invalid", and a child that copies the
// zero would carry that invalidity past every check the root went through: the collector drops
// the span or files it under a trace that does not exist, and nothing reports it. Measured
// before this guard: StartChild(Span{}, ...) returned ok with TraceID 00000000000000000000000000000000.
//
// ParentID is left alone when the parent has no SpanID -- a root span is the normal parent and
// it has one, but a caller assembling a span by hand may not.
func (t *Tracer) StartChild(parent Span, name string, kind int, attrs ...Attribute) (Span, bool) {
	if parent.TraceID == (TraceID{}) {
		t.noteDropped()
		return Span{}, false
	}
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
//
// The message is cleared on the success path rather than left alone: Finish takes a value
// receiver, so calling it twice returns two independent spans, and the second call must
// describe its own outcome. Measured before this, Finish(err) followed by Finish(nil) produced
// status 1 with StatusMessage still holding the earlier error, which encodes as an OK status
// carrying a message -- a combination OTLP reserves for an error, so a backend filtering on the
// message would count a successful span as failed.
func (s Span) Finish(err error) Span {
	s.End = time.Now()
	if err != nil {
		s.Status = 2
		s.StatusMessage = err.Error()
	} else {
		s.Status = 1
		s.StatusMessage = ""
	}
	return s
}

// unixNanos formats a timestamp the way OTLP/JSON requires: a string of nanoseconds,
// because a JSON number cannot hold one exactly at float64 precision.
func unixNanos(t time.Time) string {
	return fmt.Sprintf("%d", t.UnixNano())
}
