package server

import (
	"context"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/pplmx/orbix/internal/engine"
	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
	"github.com/pplmx/orbix/internal/otel"
	"github.com/pplmx/orbix/internal/tokenize"
)

// get issues a GET and returns the raw response.
func get(t *testing.T, h http.Handler, path string) *http.Response {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, path, nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	return w.Result()
}

// readBody reads a response's whole body.
func readBody(t *testing.T, resp *http.Response) string {
	t.Helper()
	defer func() { _ = resp.Body.Close() }()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	return string(raw)
}

// TestMetricsUnderConcurrentRequests scrapes /metrics while requests are running.
//
// Every existing metrics test scrapes between requests, so nothing exercised the one thing
// the endpoint exists for: an operator polls it while the pump is advancing sequences. The
// two disagree about locking -- Stats takes the engine mutex once for its fields and the pool
// takes its own for Free -- and a scrape is allowed to see a slightly stale combination, but
// it must not race. That is what -race decides, and the test only has to get both sides
// running at once to ask the question.
//
// The body is also checked to be complete on every scrape: a handler that raced would most
// likely show up as a truncated or half-written exposition rather than as a failed request.
// TestMetricsSplitFinishedByReason covers the counter that makes a client walking away
// visible.
//
// Every other number a server exports looks healthy while clients disconnect mid-generation:
// the request is counted as finished, it contributes an ordinary latency, and nothing in the
// queue depth or the pool occupancy says otherwise. Measured before this counter existed, an
// aborted stream and a completed one produced access-log lines that differed only in byte
// count.
//
// The assertion is that the parts sum to requests_finished_total. That is the property that
// makes the split trustworthy: a reason missed in finish, or counted twice, breaks the sum
// rather than showing up as a slightly wrong bucket nobody checks.
func TestMetricsSplitFinishedByReason(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()

	// One request that runs to its budget, and one whose client leaves straight away.
	post(t, h, "/v1/completions", `{"prompt": "hi", "max_tokens": 2}`)
	cancelCtx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest(http.MethodPost, "/v1/completions",
		strings.NewReader(`{"prompt": "hi", "max_tokens": 50, "stream": true}`))
	req = req.WithContext(cancelCtx)
	cancel() // the client is gone before the handler starts
	h.ServeHTTP(httptest.NewRecorder(), req)

	body := readBody(t, get(t, h, "/metrics"))

	var byReason = map[string]int{}
	for _, line := range strings.Split(body, "\n") {
		rest, ok := strings.CutPrefix(line, "orbix_requests_finished_by_reason_total{reason=\"")
		if !ok {
			continue
		}
		label, value, ok := strings.Cut(rest, "\"} ")
		if !ok {
			t.Fatalf("malformed sample %q", line)
		}
		var n int
		if _, err := fmt.Sscanf(value, "%d", &n); err != nil {
			t.Fatalf("sample %q is not an integer: %v", line, err)
		}
		byReason[label] = n
	}

	// Every reason the engine defines has to be present, so a dashboard does not see a label
	// appear and disappear as traffic changes.
	for _, want := range []string{"stop", "length", "cancel", "abandoned", "error"} {
		if _, ok := byReason[want]; !ok {
			t.Errorf("no sample for reason %q; present: %v", want, byReason)
		}
	}

	parts := 0
	for _, n := range byReason {
		parts += n
	}
	total := finishedTotal(t, body)
	if parts != total {
		t.Errorf("reasons sum to %d but requests_finished_total is %d (parts: %v)", parts, total, byReason)
	}
	if total == 0 {
		t.Fatal("nothing finished, so this proves nothing")
	}
	if byReason["cancel"] == 0 {
		t.Errorf("a cancelled request did not show up as reason=cancel: %v", byReason)
	}
}

// TestMetricsExportsEveryFinishReason ties the exposition to the engine's own list.
//
// The handler used to name the five reasons itself. That copy kept compiling when the engine
// added one, and the failure would have been a label no scrape ever showed -- a dashboard
// missing a reason rather than a build error. It now iterates engine.FinishReasons, and this
// test compares the labels in the body against that list rather than against a literal, so the
// two cannot disagree.
func TestMetricsExportsEveryFinishReason(t *testing.T) {
	srv, _ := newTestServer(t)
	body := readBody(t, get(t, srv.Handler(), "/metrics"))

	var have []string
	for _, line := range strings.Split(body, "\n") {
		rest, ok := strings.CutPrefix(line, "orbix_requests_finished_by_reason_total{reason=\"")
		if !ok {
			continue
		}
		label, _, ok := strings.Cut(rest, "\"}")
		if !ok {
			t.Fatalf("malformed sample %q", line)
		}
		have = append(have, label)
	}

	want := engine.FinishReasons()
	if len(have) != len(want) {
		t.Fatalf("exported %d reasons %v, engine defines %d %v", len(have), have, len(want), want)
	}
	// Order is not part of the contract, so compare as sets.
	seen := map[string]bool{}
	for _, h := range have {
		seen[h] = true
	}
	for _, r := range want {
		if !seen[r.String()] {
			t.Errorf("engine defines reason %q but /metrics does not export it", r.String())
		}
	}
}

// finishedTotal reads orbix_requests_finished_total out of an exposition body.
func finishedTotal(t *testing.T, body string) int {
	t.Helper()
	for _, line := range strings.Split(body, "\n") {
		rest, ok := strings.CutPrefix(line, "orbix_requests_finished_total ")
		if !ok {
			continue
		}
		var n int
		if _, err := fmt.Sscanf(rest, "%d", &n); err != nil {
			t.Fatalf("requests_finished_total %q is not an integer: %v", line, err)
		}
		return n
	}
	t.Fatal("requests_finished_total is missing")
	return 0
}

func TestMetricsUnderConcurrentRequests(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()

	var wg sync.WaitGroup
	// Scrapers. Failures are collected rather than reported from the goroutine: t.Errorf
	// is safe there but t.Fatalf is not, and a bare return would drop the failure entirely.
	var mu sync.Mutex
	var problems []string
	note := func(format string, args ...any) {
		mu.Lock()
		defer mu.Unlock()
		problems = append(problems, fmt.Sprintf(format, args...))
	}
	for range 4 {
		wg.Go(func() {
			for range 25 {
				resp := get(t, h, "/metrics")
				body := readBody(t, resp)
				if resp.StatusCode != http.StatusOK {
					note("scrape returned %d", resp.StatusCode)
					return
				}
				// The model label is written last, so its presence means the whole
				// exposition reached the buffer.
				if !strings.Contains(body, "model_info{model=") {
					note("scrape body is missing the final metric; len=%d", len(body))
					return
				}
			}
		})
	}
	// Requests, so the pump has something to advance while the scrapes read.
	for range 4 {
		wg.Go(func() {
			for range 3 {
				resp := post(t, h, "/v1/completions", `{"prompt": "Once upon a time", "max_tokens": 4}`)
				_ = readBody(t, resp)
			}
		})
	}
	wg.Wait()
	for _, p := range problems {
		t.Error(p)
	}

	// And the counters the scrapers read must add up afterwards: a race that corrupted a
	// counter would leave a request accepted but never finished.
	st := srv.eng.Stats()
	if st.RequestsSubmitted != st.RequestsFinished {
		t.Errorf("submitted %d but finished %d with no requests outstanding",
			st.RequestsSubmitted, st.RequestsFinished)
	}
}

// scrape fetches /metrics and returns its body.
func scrape(t *testing.T, h http.Handler) string {
	t.Helper()
	resp := get(t, h, "/metrics")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /metrics = %d, want 200", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/plain") {
		t.Errorf("Content-Type = %q, want text/plain", ct)
	}
	return readBody(t, resp)
}

// parseMetrics reads a Prometheus exposition body into name->value, verifying the format
// as it goes.
//
// Parsing it back is the point: the encoder is hand-written, so the only way to know it
// emits valid text is to consume it the way a scraper would. A grep for one metric would
// pass even if a HELP line were in the wrong place.
func parseMetrics(t *testing.T, body string) map[string]string {
	t.Helper()
	values := map[string]string{}
	helped := map[string]bool{}
	typed := map[string]bool{}

	for i, line := range strings.Split(strings.TrimRight(body, "\n"), "\n") {
		lineno := i + 1
		switch {
		case line == "":
			t.Errorf("line %d: blank line inside an exposition body", lineno)
		case strings.HasPrefix(line, "# HELP "):
			rest, ok := strings.CutPrefix(line, "# HELP ")
			if !ok {
				t.Fatalf("line %d: %q", lineno, line)
			}
			name, doc, ok := strings.Cut(rest, " ")
			if !ok || doc == "" {
				t.Errorf("line %d: HELP %q has no doc string", lineno, name)
			}
			if helped[name] {
				t.Errorf("line %d: duplicate HELP for %s", lineno, name)
			}
			helped[name] = true
		case strings.HasPrefix(line, "# TYPE "):
			rest, _ := strings.CutPrefix(line, "# TYPE ")
			name, typ, ok := strings.Cut(rest, " ")
			if !ok {
				t.Fatalf("line %d: TYPE has no type: %q", lineno, line)
			}
			switch typ {
			case "counter", "gauge":
			default:
				t.Errorf("line %d: %s has type %q, want counter or gauge", lineno, name, typ)
			}
			typed[name] = true
			if !helped[name] {
				t.Errorf("line %d: TYPE for %s came before its HELP", lineno, name)
			}
		case strings.HasPrefix(line, "#"):
			t.Errorf("line %d: unexpected comment %q", lineno, line)
		default:
			sample, value, ok := strings.Cut(line, " ")
			if !ok {
				t.Fatalf("line %d: sample has no value: %q", lineno, line)
			}
			name, _, hasLabel := strings.Cut(sample, "{")
			if !typed[name] {
				t.Errorf("line %d: sample %s has no preceding TYPE", lineno, name)
			}
			if hasLabel {
				// Only one label shape is emitted; check it is quoted.
				if !strings.Contains(sample, `="`) {
					t.Errorf("line %d: label value is not quoted: %q", lineno, sample)
				}
			}
			if _, err := strconv.ParseFloat(value, 64); err != nil {
				t.Errorf("line %d: %s value %q is not a number", lineno, name, value)
			}
			if _, dup := values[sample]; dup {
				t.Errorf("line %d: duplicate sample %s", lineno, sample)
			}
			values[sample] = value
		}
	}
	if len(values) == 0 {
		t.Fatal("no samples in the exposition body")
	}
	return values
}

// num reads one sample as a float.
func num(t *testing.T, values map[string]string, name string) float64 {
	t.Helper()
	raw, ok := values[name]
	if !ok {
		t.Fatalf("metric %s is missing; got %v", name, values)
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		t.Fatalf("metric %s = %q: %v", name, raw, err)
	}
	return v
}

func TestMetricsExpositionIsParseable(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	srv := serverWithPolicy(t, path, engine.PriorityFIFO)
	values := parseMetrics(t, scrape(t, srv.Handler()))

	// The counters must all be present before anything ran, and zero.
	for _, name := range []string{
		"orbix_requests_submitted_total",
		"orbix_requests_finished_total",
		"orbix_tokens_generated_total",
		"orbix_rounds_total",
	} {
		if got := num(t, values, name); got != 0 {
			t.Errorf("%s = %v on a fresh server, want 0", name, got)
		}
	}
	// The gauges likewise, except MaxActive which echoes the configured limit.
	if got := num(t, values, "orbix_requests_active"); got != 0 {
		t.Errorf("orbix_requests_active = %v, want 0", got)
	}
	if got := num(t, values, "orbix_round_max_active"); got != 1 {
		t.Errorf("orbix_round_max_active = %v, want 1 (serverWithPolicy uses MaxActive 1)", got)
	}
}

func TestMetricsCountARequest(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	srv := serverWithPolicy(t, path, engine.PriorityFIFO)

	// One request whose token count is known, so the counter can be checked exactly
	// rather than against an inequality that would pass for any positive number.
	const wantTokens = 5
	resp := post(t, srv.Handler(), "/v1/completions",
		fmt.Sprintf(`{"prompt": "Once upon a time", "max_tokens": %d}`, wantTokens))
	decodeBody[completionResponse](t, resp)

	values := parseMetrics(t, scrape(t, srv.Handler()))
	if got := num(t, values, "orbix_requests_submitted_total"); got != 1 {
		t.Errorf("requests_submitted_total = %v, want 1", got)
	}
	if got := num(t, values, "orbix_requests_finished_total"); got != 1 {
		t.Errorf("requests_finished_total = %v, want 1", got)
	}
	if got := num(t, values, "orbix_requests_active"); got != 0 {
		t.Errorf("requests_active = %v after the request finished, want 0", got)
	}
	// The request may stop early on EOS, so the budget is an upper bound.
	if got := num(t, values, "orbix_tokens_generated_total"); got <= 0 || got > wantTokens {
		t.Errorf("tokens_generated_total = %v, want in (0, %d]", got, wantTokens)
	}
	if got := num(t, values, "orbix_rounds_total"); got <= 0 {
		t.Errorf("rounds_total = %v, want positive", got)
	}
}

func TestMetricsOmitsPoolMetricsWithoutAPool(t *testing.T) {
	// A contiguous engine has no pool. Exporting zeros would make "no pool" and "empty
	// pool" the same reading, and an alert on free blocks would fire forever.
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	srv := serverWithPolicy(t, path, engine.PriorityFIFO)
	values := parseMetrics(t, scrape(t, srv.Handler()))
	for _, name := range []string{
		"orbix_kv_pool_blocks",
		"orbix_kv_pool_free",
		"orbix_kv_pool_used_fraction",
	} {
		if _, ok := values[name]; ok {
			t.Errorf("%s is exported by a contiguous engine; it should be omitted", name)
		}
	}
}

func TestMetricsModelLabel(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	srv := serverWithPolicy(t, path, engine.PriorityFIFO)
	values := parseMetrics(t, scrape(t, srv.Handler()))

	var found bool
	for sample := range values {
		if strings.HasPrefix(sample, "orbix_model_info{") {
			found = true
			if !strings.Contains(sample, `model="`) {
				t.Errorf("model_info sample has no model label: %q", sample)
			}
		}
	}
	if !found {
		t.Error("orbix_model_info is missing; a scrape cannot tell which checkpoint the numbers describe")
	}
}

func TestMetricsRejectsNonGet(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	srv := serverWithPolicy(t, path, engine.PriorityFIFO)
	// The route is registered for GET only, so another method must not reach the handler.
	resp := post(t, srv.Handler(), "/metrics", "{}")
	if resp.StatusCode == http.StatusOK {
		t.Error("POST /metrics returned 200; the route should be GET-only")
	}
}

// TestFormatFloat pins the value formatting: Prometheus prints whole numbers without a
// decimal point, and matching that keeps the output readable next to other exporters.
//
// The large integers are the part that discriminates. Every case used to be small enough that
// %g and %d agree, so replacing the whole function with `fmt.Sprintf("%g", v)` passed this
// test and the rest of the server suite. The two only diverge where %g switches to exponent
// notation, at a million -- and orbix_tokens_generated_total passes that within minutes of
// real traffic, so a scrape would start reporting "1e+06" where every other exporter in the
// process reports a plain integer.
func TestFormatFloat(t *testing.T) {
	tests := []struct {
		in   float64
		want string
	}{
		{0, "0"},
		{1, "1"},
		{1672, "1672"},
		{0.5, "0.5"},
		{0.125, "0.125"},
		// The exponent threshold: %g writes 1e+06 here, Prometheus writes 1000000.
		{999999, "999999"},
		{1e6, "1000000"},
		{1e9, "1000000000"},
		{1e12, "1000000000000"},
		// Still exact at the top of int64's exact range for float64.
		{1e18, "1000000000000000000"},
		// Past it, int64(v) overflows and the equality check fails, so %g is the honest
		// answer rather than a wrapped negative. Nothing counts that high, but the branch
		// exists and returning a negative total would be worse than a rounded one.
		{1e21, "1e+21"},
		// Negative counters are not meaningful either, and the same path handles them.
		{-5, "-5"},
	}
	for _, tc := range tests {
		if got := formatFloat(tc.in); got != tc.want {
			t.Errorf("formatFloat(%v) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// serverWithPagedEngine builds a server whose engine uses the shared block pool, so the
// pool metrics have something to report.
func serverWithPagedEngine(t *testing.T, path string) *Server {
	t.Helper()
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	eng, err := engine.New(path, engine.Options{MaxActive: 1, BlockSize: 16, Blocks: 64})
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	cfg := eng.Model().Config()
	srv, err := New(eng, tok, Config{ModelName: cfg.Name, ContextLen: cfg.ContextLen})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	srv.StartPump(ctx)
	return srv
}

// TestMetricsExportsPoolMetrics is the counterpart to the contiguous case: with a pool,
// the block counts must appear and must describe the pool that was configured.
func TestMetricsExportsPoolMetrics(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	srv := serverWithPagedEngine(t, path)
	values := parseMetrics(t, scrape(t, srv.Handler()))

	if got := num(t, values, "orbix_kv_pool_blocks"); got != 64 {
		t.Errorf("kv_pool_blocks = %v, want 64", got)
	}
	if got := num(t, values, "orbix_kv_pool_free"); got != 64 {
		t.Errorf("kv_pool_free = %v before any request, want 64", got)
	}
	if got := num(t, values, "orbix_kv_pool_block_size"); got != 16 {
		t.Errorf("kv_pool_block_size = %v, want 16", got)
	}
	if got := num(t, values, "orbix_kv_pool_used_fraction"); got != 0 {
		t.Errorf("kv_pool_used_fraction = %v before any request, want 0", got)
	}

	// After a request, its blocks must be back: a pool that leaks is the failure this
	// metric exists to surface, and it is invisible in latency until the pool is empty.
	resp := post(t, srv.Handler(), "/v1/completions",
		`{"prompt": "Once upon a time", "max_tokens": 4}`)
	decodeBody[completionResponse](t, resp)

	values = parseMetrics(t, scrape(t, srv.Handler()))
	if got := num(t, values, "orbix_kv_pool_free"); got != 64 {
		t.Errorf("kv_pool_free = %v after one request finished, want 64; blocks leaked", got)
	}
}

// TestMetricsExportPrefixReuse covers the counter a deployment running with a prefix cache
// needs: it is the only reading that says the cache is doing anything, since a reused prompt
// costs no forward pass and therefore moves no other number.
//
// The value is checked exactly. A block is published only when its last position is written,
// so with 8-position blocks a prompt of three whole blocks is cached by the first request and
// each identical later request reuses all of it -- a number that a broken counter (counting
// requests, or blocks, or nothing) cannot produce by accident.
func TestMetricsExportPrefixReuse(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	eng, err := engine.New(path, engine.Options{
		MaxActive: 1, BlockSize: 8, Blocks: 64, PrefixCache: true,
	})
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	cfg := eng.Model().Config()
	srv, err := New(eng, tok, Config{ModelName: cfg.Name, ContextLen: cfg.ContextLen})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	srv.StartPump(ctx)

	// A prose prompt long enough to fill several 8-position blocks, so there is something
	// for the index to publish.
	text := strings.Repeat("Once upon a time there was a little girl. ", 4)
	body := `{"prompt": ` + quoteJSON(text) + `, "max_tokens": 2}`

	if got := num(t, parseMetrics(t, scrape(t, srv.Handler())),
		"orbix_prefix_positions_reused_total"); got != 0 {
		t.Errorf("before any request: prefix_positions_reused_total = %v, want 0", got)
	}

	// The first request fills the cache; the next three reuse it.
	for range 4 {
		decodeBody[completionResponse](t, post(t, srv.Handler(), "/v1/completions", body))
	}
	got := num(t, parseMetrics(t, scrape(t, srv.Handler())),
		"orbix_prefix_positions_reused_total")
	// Three reuses of a whole-block-aligned prefix: positive and a multiple of the block
	// size, without pinning a block count the encoder could legitimately change.
	if got <= 0 {
		t.Fatalf("prefix_positions_reused_total = %v after four identical requests, want positive", got)
	}
	if rem := math.Mod(got, 8); rem != 0 {
		t.Errorf("prefix_positions_reused_total = %v, not a multiple of the 8-position block", got)
	}
}

// TestMetricsExportDroppedSpans checks the reading that tells an operator tracing is losing
// data.
//
// A dead or slow collector is otherwise invisible: requests keep succeeding, latency does not
// move (the export runs off the request path), and the only symptom is gaps in the traces
// that nobody is looking at because they are not arriving. The queue is bounded, so the
// spans are dropped rather than piling up -- and until this counter was exported, the
// exporter's own Dropped() had no caller outside its tests.
//
// Tracing is off by default, so the metric is absent for a server without it, in the same
// way the pool counts are absent for a contiguous engine: a zero would be indistinguishable
// from "nothing is being traced".
func TestMetricsExportDroppedSpans(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	eng, err := engine.New(path, engine.Options{MaxActive: 1})
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	cfg := eng.Model().Config()

	// A collector that cannot be reached, and a queue of one, so the second span is
	// dropped without waiting for a tick.
	exp, err := otel.NewExporter(otel.ExporterOptions{
		Endpoint: "http://127.0.0.1:1/v1/traces",
		Interval: time.Hour,
		MaxQueue: 1,
	})
	if err != nil {
		t.Fatalf("NewExporter: %v", err)
	}
	t.Cleanup(func() { _ = exp.Close() })

	srv, err := New(eng, tok, Config{
		ModelName: cfg.Name, ContextLen: cfg.ContextLen, Trace: exp,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	srv.StartPump(ctx)

	body := `{"prompt": "Once upon a time", "max_tokens": 2}`
	// Three requests: the first queues its span, the rest have nowhere to go.
	for range 3 {
		decodeBody[completionResponse](t, post(t, srv.Handler(), "/v1/completions", body))
	}

	values := parseMetrics(t, scrape(t, srv.Handler()))
	if got := num(t, values, "orbix_trace_spans_dropped_total"); got <= 0 {
		t.Errorf("trace_spans_dropped_total = %v with a dead collector and a queue of one, want positive", got)
	}
}

// TestMetricsOmitDroppedSpansWithoutTracing is the counterpart: with tracing off there is no
// exporter, and a zero would read as a healthy collector rather than as tracing being off.
func TestMetricsOmitDroppedSpansWithoutTracing(t *testing.T) {
	srv, _ := newTestServer(t)
	values := parseMetrics(t, scrape(t, srv.Handler()))
	if _, ok := values["orbix_trace_spans_dropped_total"]; ok {
		t.Error("trace_spans_dropped_total is exported although tracing is off")
	}
}

// TestPrefixReuseOnChatCompletions checks the same reuse on the chat endpoint.
//
// The test above drives /v1/completions, where the prompt is the caller's text. A chat
// request's prompt comes out of the model's template, so reuse there depends on the template
// producing the same prefix for the same conversation -- which it does, and which is what
// makes a multi-turn session cheap. The two endpoints share the metric but not the path that
// produces the prompt, so one passing does not mean the other does.
func TestPrefixReuseOnChatCompletions(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512, Pre: "qwen2"}.Write(t, t.TempDir())
	eng, err := engine.New(path, engine.Options{
		MaxActive: 1, BlockSize: 8, Blocks: 64, PrefixCache: true,
	})
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	cfg := eng.Model().Config()
	srv, err := New(eng, tok, Config{ModelName: cfg.Name, ContextLen: cfg.ContextLen})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	srv.StartPump(ctx)

	// A conversation long enough that the rendered prompt spans several blocks.
	long := strings.Repeat("Tell me about the history of the city. ", 4)
	body := `{"messages": [{"role": "user", "content": ` + quoteJSON(long) + `}], "max_tokens": 2}`

	if got := num(t, parseMetrics(t, scrape(t, srv.Handler())),
		"orbix_prefix_positions_reused_total"); got != 0 {
		t.Errorf("before any request: prefix_positions_reused_total = %v, want 0", got)
	}
	for range 3 {
		decodeBody[chatResponse](t, post(t, srv.Handler(), "/v1/chat/completions", body))
	}
	got := num(t, parseMetrics(t, scrape(t, srv.Handler())),
		"orbix_prefix_positions_reused_total")
	if got <= 0 {
		t.Fatalf("prefix_positions_reused_total = %v after three identical chat requests, "+
			"want positive", got)
	}
}

// TestEscapeLabelValue pins the three characters the text exposition format requires to be
// escaped, and the ones it does not.
//
// Go's %q was used here first. It agrees on backslash, quote, and line feed but also escapes
// tabs and control characters, which Prometheus reads back literally -- so a label value
// containing a tab would arrive as the two characters backslash and t.
func TestEscapeLabelValue(t *testing.T) {
	cases := []struct{ in, want, why string }{
		{"plain", "plain", "nothing to escape"},
		{"qwen2.5-7b", "qwen2.5-7b", "dots and dashes are legal as-is"},
		{`back\slash`, `back\\slash`, "backslash"},
		{`has"quote`, `has\"quote`, "double quote"},
		{"new\nline", `new\nline`, "line feed"},
		// The three below are where %q and the exposition format disagree. A tab, a carriage
		// return, and a non-ASCII byte are all allowed verbatim by the format.
		{"tab\there", "tab\there", "a tab stays a tab"},
		{"cr\rhere", "cr\rhere", "a carriage return stays literal"},
		{"中文模型", "中文模型", "UTF-8 passes through"},
	}
	for _, c := range cases {
		if got := escapeLabelValue(c.in); got != c.want {
			t.Errorf("%s: escapeLabelValue(%q) = %q, want %q", c.why, c.in, got, c.want)
		}
	}
}
