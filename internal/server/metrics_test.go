package server

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/pplmx/orbix/internal/engine"
	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
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
	go srv.Run(ctx)
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
