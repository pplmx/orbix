// Package otel exports traces in OpenTelemetry's wire format, without the SDK.
//
// Orbix has no module dependencies, and the tracing story is built to keep it that
// way. The official OpenTelemetry SDK is a large tree of packages; what it produces is
// a protobuf message posted to an HTTP endpoint, and the JSON encoding of that message
// is a handful of structs. Writing those structs is the same choice this project makes
// everywhere else -- a GGUF parser and a Prometheus exposition are in the tree for the
// same reason -- and it keeps a reader able to follow a span from creation to wire
// format inside one language.
//
// The protocol details that are easy to get wrong, and that this package pins:
//
//   - Trace and span ids are base64 in OTLP/HTTP+JSON, not hex. Most hand-written
//     exporters in the wild emit hex and are silently rejected or mis-joined by a
//     collector; the specification is explicit that JSON follows proto3's mapping, where
//     a bytes field is base64.
//   - Timestamps are strings holding nanoseconds since the epoch, because a JSON number
//     cannot represent them exactly.
//   - The body is an ExportTraceServiceRequest: resourceSpans, then scopeSpans, then
//     spans. A span without a valid 16-byte trace id or 8-byte span id is invalid.
//   - The endpoint path is /v1/traces and the request must set Content-Type.
//
// Export is best-effort by design. Tracing is observability, not correctness: a
// collector that is down must not fail a generation request, so an export error is
// returned for the caller to log and never propagates into the serving path.
package otel
