package server

import (
	"bytes"
	"fmt"
	"net/http"
	"strings"

	"github.com/pplmx/orbix/internal/engine"
)

// metricsNamespace prefixes every exported name, so a scrape cannot collide with an
// unrelated exporter in the same process.
const metricsNamespace = "orbix"

// handleMetrics serves the engine's counters in Prometheus's text exposition format.
//
// The format is written by hand rather than pulled in as a client library, for the same
// reason the GGUF parser is: it is four lines of syntax, and a dependency that
// serialises ten lines would make the exposition opaque in a repository whose point is
// to show how each layer works. The format is stable and documented, and the tests below
// parse it back so a mistake in the hand-written encoder fails a test rather than a
// dashboard.
//
// Prometheus is the de-facto convention here; the same numbers are what any other
// collector would want, so a different system only needs a different encoder, not
// different counters.
func (s *Server) handleMetrics(w http.ResponseWriter, _ *http.Request) {
	st := s.eng.Stats()

	var b metricsBuilder
	b.help("requests_submitted_total", "Requests accepted into the engine, in cumulative total.",
		"counter", st.RequestsSubmitted)
	b.help("requests_finished_total", "Requests that reached their end, whatever the reason.",
		"counter", st.RequestsFinished)
	// The split matters more than the total for the same reason the pool gauges do: a server
	// whose clients disconnect mid-generation looks healthy in every other number. Those
	// sequences are counted as finished, they contribute a normal-looking latency, and the
	// only sign is that most of them ended in "cancel" or "abandoned" rather than "stop".
	// The reasons come from the engine rather than a list restated here: a new one would
	// otherwise keep this file compiling while its label never appeared in a scrape.
	all := engine.FinishReasons()
	reasons := make([]labeledValue, 0, len(all))
	for _, r := range all {
		reasons = append(reasons, labeledValue{
			label: r.String(),
			value: float64(st.FinishedByReason[r]),
		})
	}
	b.labeledSeries("requests_finished_by_reason_total",
		"Requests that reached their end, split by why.",
		"counter", "reason", reasons)
	b.help("tokens_generated_total", "Tokens delivered to callers, in cumulative total.",
		"counter", st.TokensGenerated)
	b.help("rounds_total", "Scheduling rounds that advanced at least one sequence.",
		"counter", st.Rounds)
	b.help("prefix_positions_reused_total",
		"Prompt positions served from a shared KV prefix instead of a forward pass, in cumulative total.",
		"counter", st.PrefixReused)

	b.gauge("requests_active", "Sequences being advanced right now.",
		"gauge", float64(st.Active))
	b.gauge("requests_queued", "Accepted requests waiting for a slot.",
		"gauge", float64(st.Queued))
	b.gauge("round_max_active", "Sequences a round can advance.",
		"gauge", float64(st.MaxActive))

	// Refusals are a server-side decision, so they are not in engine.Stats: Submit returns
	// ErrQueueFull and the pool error arrives on an event, and neither reaches the engine's
	// counters -- by design, since a request that never entered the queue must not inflate
	// the admission rate. That left them invisible: measured with --max-active 1 --max-queued
	// 1 and 24 requests in bursts of 8, 16 were refused with 503 while
	// orbix_requests_submitted_total still read 8 and the queue gauge sat at 0, so a
	// deployment shedding most of its load looked healthy.
	//
	// Two families rather than one labelled series, because the causes need different
	// reactions: a full queue wants more --max-queued or more replicas, while an exhausted
	// pool wants more --blocks or shorter prompts. A single "refused" total would say
	// capacity is short without saying which.
	b.help("requests_refused_queue_full_total",
		"Requests refused with 503 because the backlog was full, in cumulative total.",
		"counter", s.refusedQueueFull.Load())
	b.help("requests_refused_pool_total",
		"Requests refused with 503 because the KV pool had no room, in cumulative total.",
		"counter", s.refusedPool.Load())

	// The pool block counts are only meaningful for a paged engine. Exporting zeros for
	// a contiguous one would make "the pool is empty" indistinguishable from "there is no
	// pool", so they are omitted entirely instead.
	if st.PoolBlocks > 0 {
		b.gauge("kv_pool_blocks", "Blocks in the shared KV pool.",
			"gauge", float64(st.PoolBlocks))
		b.gauge("kv_pool_free", "Unused blocks in the shared KV pool.",
			"gauge", float64(st.PoolFree))
		b.gauge("kv_pool_block_size", "Positions held by one KV block.",
			"gauge", float64(st.PoolBlockSize))
		b.gauge("kv_pool_used_fraction", "Share of KV pool blocks in use.",
			"gauge", st.FractionFull())
	}

	// Tracing is exported only when it is on, for the same reason the pool counts are:
	// a zero would read as "no spans were dropped" when the truth is "nothing is being
	// traced". A non-zero value here is the signal that a collector is down or slow, and
	// it is otherwise invisible -- requests keep succeeding and the traces just have gaps.
	if s.cfg.Trace != nil {
		b.help("trace_spans_dropped_total",
			"Spans discarded because the export queue was full or a send failed, in cumulative total.",
			"counter", uint64(s.cfg.Trace.Dropped()))
	}

	// The model name is the one label here. It is a constant for the process lifetime,
	// so it costs nothing and it tells a scraper which checkpoint the numbers describe --
	// the difference between throughput on a 15M and a 7B model is the whole reading.
	b.labeled("model_info", "Checkpoint the engine is serving.", "gauge", 1,
		"model", s.cfg.ModelName)

	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write(b.buf.Bytes()); err != nil {
		// The response is already committed; there is no status left to change, and a
		// client that went away mid-scrape has nothing to receive.
		return
	}
}

// metricsBuilder accumulates one exposition response.
type metricsBuilder struct {
	buf bytes.Buffer
}

// help writes a metric's HELP and TYPE lines followed by its value.
//
// Prometheus requires HELP and TYPE to precede the samples for a name, and it treats a
// repeated HELP for the same name as an error, so emitting them together here is what
// keeps a caller from getting that ordering wrong.
func (b *metricsBuilder) help(name, doc, typ string, value uint64) {
	fmt.Fprintf(&b.buf, "# HELP %s_%s %s\n", metricsNamespace, name, doc)
	fmt.Fprintf(&b.buf, "# TYPE %s_%s %s\n", metricsNamespace, name, typ)
	fmt.Fprintf(&b.buf, "%s_%s %d\n", metricsNamespace, name, value)
}

// gauge is help for a floating-point value.
func (b *metricsBuilder) gauge(name, doc, typ string, value float64) {
	fmt.Fprintf(&b.buf, "# HELP %s_%s %s\n", metricsNamespace, name, doc)
	fmt.Fprintf(&b.buf, "# TYPE %s_%s %s\n", metricsNamespace, name, typ)
	fmt.Fprintf(&b.buf, "%s_%s %s\n", metricsNamespace, name, formatFloat(value))
}

// labeled writes a metric with one label pair and a floating-point value.
func (b *metricsBuilder) labeled(name, doc, typ string, value float64, label, labelValue string) {
	fmt.Fprintf(&b.buf, "# HELP %s_%s %s\n", metricsNamespace, name, doc)
	fmt.Fprintf(&b.buf, "# TYPE %s_%s %s\n", metricsNamespace, name, typ)
	fmt.Fprintf(&b.buf, "%s_%s{%s=\"%s\"} %s\n", metricsNamespace, name, label,
		escapeLabelValue(labelValue), formatFloat(value))
}

// labeledSeries writes one metric name with several label values, sharing one HELP and one
// TYPE line.
//
// Prometheus treats a repeated HELP for the same name as an error, so the per-sample helper
// above cannot express a metric with more than one label value: N values would emit N HELP
// lines. This takes the whole series at once instead, which is the shape a counter split by
// reason actually has.
func (b *metricsBuilder) labeledSeries(name, doc, typ, label string, values []labeledValue) {
	fmt.Fprintf(&b.buf, "# HELP %s_%s %s\n", metricsNamespace, name, doc)
	fmt.Fprintf(&b.buf, "# TYPE %s_%s %s\n", metricsNamespace, name, typ)
	for _, v := range values {
		fmt.Fprintf(&b.buf, "%s_%s{%s=\"%s\"} %s\n", metricsNamespace, name, label,
			escapeLabelValue(v.label), formatFloat(v.value))
	}
}

// labeledValue is one label value and its number.
type labeledValue struct {
	label string
	value float64
}

// escapeLabelValue escapes a label value for the text exposition format.
//
// The format requires exactly three characters to be escaped: backslash, double quote, and line
// feed. Go's %q is close but not the same rule -- it also escapes tabs and control characters as
// \t and \xNN, which Prometheus reads back literally. A model name containing a tab would come
// out of a scrape as the two characters backslash-t rather than a tab.
func escapeLabelValue(v string) string {
	if !strings.ContainsAny(v, "\\\"\n") {
		return v
	}
	var b strings.Builder
	b.Grow(len(v) + 8)
	for _, r := range v {
		switch r {
		case '\\':
			b.WriteString(`\\`)
		case '"':
			b.WriteString(`\"`)
		case '\n':
			b.WriteString(`\n`)
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// formatFloat renders a value the way the exposition format expects.
//
// Integers are written without a decimal point because Prometheus prints them that way
// and a reader comparing output by eye should see the same thing from Orbix.
func formatFloat(v float64) string {
	if v == float64(int64(v)) {
		return fmt.Sprintf("%d", int64(v))
	}
	return fmt.Sprintf("%g", v)
}
