// Package otel exports traces in OpenTelemetry's wire format, without the SDK.
//
// The official OpenTelemetry SDK is a large tree of packages, and what it ultimately
// produces is a protobuf message posted to an HTTP endpoint. The JSON encoding of that
// message is a handful of structs, and this package writes them for the same reason the
// GGUF parser and the Prometheus exposition are written here: a reader can follow a span
// from creation to wire format inside one language.
//
// Dependencies are allowed in this project -- it is a judgement per dependency, not a
// rule. This one was not taken because the surface it replaces is small enough to read,
// not because importing anything is forbidden.
//
// The protocol details that are easy to get wrong, and that this package pins:
//
//   - Trace and span ids are lowercase hex in OTLP/HTTP+JSON, not base64. This is the
//     one place OTLP departs from proto3's standard JSON mapping, which would put a bytes
//     field in base64: the specification says the ids "are represented as case-insensitive
//     hex-encoded strings; they are not base64-encoded". An exporter that emits base64 is
//     the one a collector rejects or mis-joins.
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
