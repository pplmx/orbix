package otel

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// TestEncodeOmitsAbsentParentAndZeroStatus covers the two fields the encoder leaves out.
//
// parentSpanId is optional in the OTLP schema and its absence is meaningful: a span with no parent
// is a root. An invalid id rendered anyway becomes sixteen hex zeroes, which a collector reads as a
// real parent -- a root span quietly attached to a node that does not exist.
//
// status is omitted when it is zero, which the schema defines as unset. Emitting it as 0 writes the
// same value a collector would assume, but the field's presence is what distinguishes "not set"
// from "set to unset in the enum", and it costs a field on every span.
//
// Both were uncovered: the encoder test asserted the fields that are present and never that these
// two are absent.
func TestEncodeOmitsAbsentParentAndZeroStatus(t *testing.T) {
	traceID, err := NewTraceID()
	if err != nil {
		t.Fatalf("NewTraceID: %v", err)
	}
	spanID, err := NewSpanID()
	if err != nil {
		t.Fatalf("NewSpanID: %v", err)
	}
	spans := []Span{{
		TraceID: traceID,
		SpanID:  spanID,
		Name:    "root",
		Kind:    2,
		Start:   time.Unix(0, 0),
		End:     time.Unix(1, 0),
		// No ParentID, and Status left at its zero value.
	}}

	raw, err := Encode(nil, spans)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("the output is not JSON: %v\n%s", err, raw)
	}
	// Find the span object wherever it sits in the envelope.
	var span map[string]any
	var walk func(v any)
	walk = func(v any) {
		switch x := v.(type) {
		case map[string]any:
			if _, ok := x["spanId"]; ok {
				span = x
				return
			}
			for _, e := range x {
				walk(e)
			}
		case []any:
			for _, e := range x {
				walk(e)
			}
		}
	}
	walk(got)
	if span == nil {
		t.Fatalf("no span in the output:\n%s", raw)
	}

	if _, ok := span["parentSpanId"]; ok {
		t.Errorf("a span with no parent carried parentSpanId = %v; its absence is what makes the "+
			"span a root", span["parentSpanId"])
	}
	if _, ok := span["status"]; ok {
		t.Errorf("a span with the zero status carried a status field: %v", span["status"])
	}

	// The same encoder must include both when they are set, so the omissions above are about the
	// values rather than about the fields never being written.
	parent, err := NewSpanID()
	if err != nil {
		t.Fatalf("NewSpanID: %v", err)
	}
	spans[0].ParentID = parent
	spans[0].Status = 2
	spans[0].StatusMessage = "boom"
	raw, err = Encode(nil, spans)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	for _, want := range []string{`"parentSpanId":"` + parent.String() + `"`, `"status":{`, `"boom"`} {
		if !strings.Contains(string(raw), want) {
			t.Errorf("the set case is missing %s:\n%s", want, raw)
		}
	}
}
