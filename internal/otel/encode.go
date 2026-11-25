package otel

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// The wire types below mirror opentelemetry-proto's ExportTraceServiceRequest as proto3
// JSON maps it. They are unexported and live next to the encoder because they exist only
// to be serialised: a caller works with Span, and the shape of the message is this
// package's business.
//
// Field names follow proto3's JSON mapping exactly -- lowerCamelCase of the protobuf
// field name -- because a collector parses them by those names. Renaming one here would
// make the whole trace invisible with no error on either side.

// exportRequest is the top-level message posted to /v1/traces.
type exportRequest struct {
	ResourceSpans []resourceSpans `json:"resourceSpans"`
}

// resourceSpans groups spans produced by one resource.
type resourceSpans struct {
	Resource   *resource      `json:"resource,omitempty"`
	ScopeSpans []scopeSpansIn `json:"scopeSpans"`
}

// resource describes the producer.
type resource struct {
	Attributes []keyValue `json:"attributes,omitempty"`
}

// scopeSpansIn groups spans produced by one instrumentation scope.
type scopeSpansIn struct {
	Scope *instrumentationScope `json:"scope,omitempty"`
	Spans []spanIn              `json:"spans"`
}

// instrumentationScope names the library that produced the spans.
type instrumentationScope struct {
	Name    string `json:"name"`
	Version string `json:"version,omitempty"`
}

// spanIn is one span on the wire.
type spanIn struct {
	// TraceID and SpanID are lowercase hex, which OTLP's JSON encoding requires and
	// which explicitly overrides proto3's base64 rule for bytes fields.
	TraceID           string     `json:"traceId"`
	SpanID            string     `json:"spanId"`
	ParentSpanID      string     `json:"parentSpanId,omitempty"`
	Name              string     `json:"name"`
	Kind              int        `json:"kind,omitempty"`
	StartTimeUnixNano string     `json:"startTimeUnixNano"`
	EndTimeUnixNano   string     `json:"endTimeUnixNano"`
	Attributes        []keyValue `json:"attributes,omitempty"`
	Status            *status    `json:"status,omitempty"`
}

// keyValue is one attribute.
type keyValue struct {
	Key   string   `json:"key"`
	Value anyValue `json:"value"`
}

// anyValue is OTLP's typed value union: exactly one field is set.
type anyValue struct {
	StringValue *string     `json:"stringValue,omitempty"`
	IntValue    *string     `json:"intValue,omitempty"`
	BoolValue   *bool       `json:"boolValue,omitempty"`
	ArrayValue  *arrayValue `json:"arrayValue,omitempty"`
}

// arrayValue wraps a repeated field. OTLP uses it for attributes typed as string[],
// such as gen_ai.response.finish_reasons.
type arrayValue struct {
	Values []anyValue `json:"values"`
}

// status is a span's outcome.
type status struct {
	Code    int    `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
}

// Scope identifies the library that produced the spans.
var Scope = instrumentationScope{Name: "github.com/pplmx/orbix"}

// Encode renders spans as an ExportTraceServiceRequest in OTLP/JSON.
//
// Spans are grouped under one resource and one scope, which is what a single-process
// exporter produces. A span with an invalid id or a zero end time is skipped rather than
// emitted: a collector rejects an invalid id and would discard the whole request, so one
// bad span would take every good one with it.
func Encode(resourceAttrs []Attribute, spans []Span) ([]byte, error) {
	encoded := make([]spanIn, 0, len(spans))
	for _, s := range spans {
		if !s.TraceID.IsValid() || !s.SpanID.IsValid() {
			continue
		}
		if s.End.IsZero() || s.End.Before(s.Start) {
			continue
		}
		out := spanIn{
			TraceID:           s.TraceID.String(),
			SpanID:            s.SpanID.String(),
			Name:              s.Name,
			Kind:              s.Kind,
			StartTimeUnixNano: unixNanos(s.Start),
			EndTimeUnixNano:   unixNanos(s.End),
			Attributes:        encodeAttrs(s.Attributes),
		}
		if s.ParentID.IsValid() {
			out.ParentSpanID = s.ParentID.String()
		}
		if s.Status != 0 {
			out.Status = &status{Code: s.Status, Message: s.StatusMessage}
		}
		encoded = append(encoded, out)
	}

	scope := Scope
	req := exportRequest{ResourceSpans: []resourceSpans{{
		ScopeSpans: []scopeSpansIn{{Scope: &scope, Spans: encoded}},
	}}}
	if len(resourceAttrs) > 0 {
		req.ResourceSpans[0].Resource = &resource{Attributes: encodeAttrs(resourceAttrs)}
	}

	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	// Trace attribute values can contain HTML-shaped text (a prompt), and the default
	// encoder escapes <, >, and & into \u forms, which inflates the payload and makes it
	// unreadable in a collector's UI without changing its meaning.
	enc.SetEscapeHTML(false)
	if err := enc.Encode(req); err != nil {
		return nil, fmt.Errorf("otel: encode trace request: %w", err)
	}
	return buf.Bytes(), nil
}

// encodeAttrs converts attributes to their wire form, dropping any whose key is empty.
//
// An empty key carries no name for a backend to store the value under, so such an attribute
// cannot be represented and is dropped rather than sent with "key": "". This said "dropping
// keys with no value", which describes a different filter -- a value is always present here,
// since Attribute holds one by construction.
func encodeAttrs(attrs []Attribute) []keyValue {
	if len(attrs) == 0 {
		return nil
	}
	out := make([]keyValue, 0, len(attrs))
	for _, a := range attrs {
		if a.Key == "" {
			continue
		}
		if a.Int {
			v := a.Value
			out = append(out, keyValue{Key: a.Key, Value: anyValue{IntValue: &v}})
			continue
		}
		if a.Bool {
			// A bool gets its own field, not a string: a backend that reads
			// gen_ai.request.stream as a boolean cannot use "true" quoted.
			v := a.Value == "true"
			out = append(out, keyValue{Key: a.Key, Value: anyValue{BoolValue: &v}})
			continue
		}
		if a.Strings != nil {
			vals := make([]anyValue, len(a.Strings))
			for i := range a.Strings {
				s := a.Strings[i]
				vals[i] = anyValue{StringValue: &s}
			}
			out = append(out, keyValue{Key: a.Key, Value: anyValue{ArrayValue: &arrayValue{Values: vals}}})
			continue
		}
		v := a.Value
		out = append(out, keyValue{Key: a.Key, Value: anyValue{StringValue: &v}})
	}
	return out
}
