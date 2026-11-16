package server

import (
	"bytes"
	"fmt"
	"net/http"
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
	b.help("tokens_generated_total", "Tokens delivered to callers, in cumulative total.",
		"counter", st.TokensGenerated)
	b.help("rounds_total", "Scheduling rounds that advanced at least one sequence.",
		"counter", st.Rounds)

	b.gauge("requests_active", "Sequences being advanced right now.",
		"gauge", float64(st.Active))
	b.gauge("requests_queued", "Accepted requests waiting for a slot.",
		"gauge", float64(st.Queued))
	b.gauge("round_max_active", "Sequences a round can advance.",
		"gauge", float64(st.MaxActive))

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
	fmt.Fprintf(&b.buf, "%s_%s{%s=%q} %s\n", metricsNamespace, name, label, labelValue, formatFloat(value))
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
