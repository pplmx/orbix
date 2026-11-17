package server

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/pplmx/orbix/internal/engine"
	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
	"github.com/pplmx/orbix/internal/tokenize"
)

// These tests need a real model: the properties under test are the HTTP and SSE
// contract, but the tokens that flow through them have to be real to see a
// streaming or template bug. Set MODEL to a GGUF file to run them.

func newTestServer(t *testing.T) (*Server, *tokenize.Tokenizer) {
	t.Helper()
	return newTestServerModel(t, gguftest.Default())
}

// newTestServerModel is newTestServer over a fixture the test describes, for tests that
// need a property the default does not have -- generation that runs to its budget, say.
func newTestServerModel(t *testing.T, model gguftest.Model) (*Server, *tokenize.Tokenizer) {
	t.Helper()
	return newTestServerConfig(t, model, Config{})
}

// newTestServerConfig is newTestServerModel with extra Config fields, for tests that need
// a logger or another knob the default does not set.
func newTestServerConfig(t *testing.T, model gguftest.Model, extra Config) (*Server, *tokenize.Tokenizer) {
	t.Helper()
	path := model.Write(t, t.TempDir())
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	eng, err := engine.New(path, engine.Options{MaxActive: 4})
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	cfg := eng.Model().Config()
	// Defaults first, then the caller's fields, so a test that sets one knob does not
	// have to restate the model name and context length.
	conf := Config{ModelName: cfg.Name, ContextLen: cfg.ContextLen}
	if extra.ModelName != "" {
		conf.ModelName = extra.ModelName
	}
	if extra.MaxTokens != 0 {
		conf.MaxTokens = extra.MaxTokens
	}
	if extra.Temperature != 0 {
		conf.Temperature = extra.Temperature
	}
	if extra.TopK != 0 {
		conf.TopK = extra.TopK
	}
	if extra.ChatRenderer != nil {
		conf.ChatRenderer = extra.ChatRenderer
	}
	if extra.Logger != nil {
		conf.Logger = extra.Logger
	}
	if extra.Trace != nil {
		conf.Trace = extra.Trace
	}
	srv, err := New(eng, tok, conf)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	srv.StartPump(ctx)
	return srv, tok
}

// Response shapes, named so the tests read as assertions rather than as type
// declarations.
type chatResponse struct {
	Object  string `json:"object"`
	Choices []struct {
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage usage `json:"usage"`
}

type completionResponse struct {
	Object  string `json:"object"`
	Choices []struct {
		Text         string `json:"text"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage usage `json:"usage"`
}

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type errorResponse struct {
	Error struct {
		Message string `json:"message"`
	} `json:"error"`
}

type modelsResponse struct {
	Object string `json:"object"`
	Data   []struct {
		ID string `json:"id"`
	} `json:"data"`
}

// post sends a JSON body to path and returns the response.
func post(t *testing.T, h http.Handler, path, body string) *http.Response {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	return w.Result()
}

func decodeBody[T any](t *testing.T, resp *http.Response) T {
	t.Helper()
	defer func() { _ = resp.Body.Close() }()
	var out T
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	return out
}

func TestChatCompletion(t *testing.T) {
	srv, _ := newTestServer(t)
	resp := post(t, srv.Handler(), "/v1/chat/completions", `{
		"model": "orbix",
		"messages": [{"role": "user", "content": "Once upon a time"}],
		"max_tokens": 8
	}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	body := decodeBody[chatResponse](t, resp)

	if body.Object != "chat.completion" {
		t.Errorf("object = %q, want chat.completion", body.Object)
	}
	if len(body.Choices) != 1 {
		t.Fatalf("choices = %d, want 1", len(body.Choices))
	}
	if body.Choices[0].Message.Role != "assistant" {
		t.Errorf("role = %q, want assistant", body.Choices[0].Message.Role)
	}
	if body.Choices[0].Message.Content == "" {
		t.Error("content is empty")
	}
	if got := body.Usage.CompletionTokens; got == 0 || got > 8 {
		t.Errorf("completion_tokens = %d, want 1..8", got)
	}
	if body.Usage.TotalTokens != body.Usage.PromptTokens+body.Usage.CompletionTokens {
		t.Errorf("total_tokens = %d, want %d", body.Usage.TotalTokens,
			body.Usage.PromptTokens+body.Usage.CompletionTokens)
	}
}

func TestCompletionMatchesRun(t *testing.T) {
	// The server's non-streaming text must be the same whatever the client does,
	// so two identical requests have to agree token for token. This is also what
	// makes greedy output a useful regression check.
	srv, _ := newTestServer(t)
	const req = `{"prompt": "Once upon a time", "max_tokens": 12}`
	var first, second string
	for i := range 2 {
		resp := post(t, srv.Handler(), "/v1/completions", req)
		body := decodeBody[completionResponse](t, resp)
		if len(body.Choices) != 1 {
			t.Fatalf("choices = %d, want 1", len(body.Choices))
		}
		if i == 0 {
			first = body.Choices[0].Text
		} else {
			second = body.Choices[0].Text
		}
	}
	if first != second {
		t.Errorf("two identical greedy requests differ:\n%q\n%q", first, second)
	}
}

func TestStreamingReassemblesToSameText(t *testing.T) {
	// The contract that makes streaming safe: concatenating the deltas must equal
	// the non-streaming text for the same request. This is where a per-token Decode
	// would fail, because it strips the space before every word.
	//
	// A fixture whose vocabulary is mostly single characters can generate byte-token
	// runs that never form a valid character, and the two paths legitimately differ on
	// those: Decode joins every pending byte before deciding, while a stream must decide
	// when the run ends. A model that emits valid text exercises the contract without
	// that ambiguity, so the fixture here has a byte-fallback-free vocabulary -- every
	// token it can produce decodes to a character.
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, NoBytes: true})
	const body = `{"prompt": "Once upon a time", "max_tokens": 12}`

	resp := post(t, srv.Handler(), "/v1/completions", body)
	plain := decodeBody[completionResponse](t, resp)
	want := plain.Choices[0].Text

	streamReq := `{"prompt": "Once upon a time", "max_tokens": 12, "stream": true}`
	sresp := post(t, srv.Handler(), "/v1/completions", streamReq)
	if got := sresp.Header.Get("Content-Type"); !strings.HasPrefix(got, "text/event-stream") {
		t.Fatalf("Content-Type = %q, want text/event-stream", got)
	}
	got := collectStream(t, sresp.Body)
	if got != want {
		t.Errorf("streamed %q, non-streaming %q", got, want)
	}
}

// collectStream reads an SSE response and concatenates its text deltas.
func collectStream(t *testing.T, body io.ReadCloser) string {
	t.Helper()
	defer func() { _ = body.Close() }()
	var out strings.Builder
	sc := bufio.NewScanner(body)
	sawDone := false
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			sawDone = true
			break
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
				Text string `json:"text"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("chunk %q: %v", payload, err)
		}
		for _, c := range chunk.Choices {
			out.WriteString(c.Delta.Content)
			out.WriteString(c.Text)
		}
	}
	if !sawDone {
		t.Error("stream ended without [DONE]")
	}
	return out.String()
}

func TestStreamingEndsWithFinishReason(t *testing.T) {
	srv, _ := newTestServer(t)
	resp := post(t, srv.Handler(), "/v1/chat/completions", `{
		"messages": [{"role": "user", "content": "Once"}],
		"max_tokens": 3, "stream": true
	}`)
	defer func() { _ = resp.Body.Close() }()
	sc := bufio.NewScanner(resp.Body)
	var last string
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "data: ") && strings.HasPrefix(line, "data: {") {
			last = line
		}
	}
	if last == "" {
		t.Fatal("no data chunks")
	}
	var chunk struct {
		Choices []struct {
			FinishReason *string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(strings.TrimPrefix(last, "data: ")), &chunk); err != nil {
		// The last chunk before [DONE] may be the empty-delta one.
		t.Logf("last chunk not parseable as a finish chunk: %v", err)
		return
	}
	var reason string
	if len(chunk.Choices) > 0 && chunk.Choices[0].FinishReason != nil {
		reason = *chunk.Choices[0].FinishReason
	}
	if reason != "length" && reason != "stop" {
		t.Errorf("finish_reason = %q, want length or stop", reason)
	}
}

func TestRejectsBadRequests(t *testing.T) {
	srv, _ := newTestServer(t)
	tests := []struct {
		name string
		path string
		body string
	}{
		{"no messages", "/v1/chat/completions", `{}`},
		{"empty messages", "/v1/chat/completions", `{"messages": []}`},
		{"no prompt", "/v1/completions", `{}`},
		{"blank prompt", "/v1/completions", `{"prompt": "   "}`},
		{"bad json", "/v1/completions", `{"prompt":`},
		{"negative temperature", "/v1/completions", `{"prompt": "x", "temperature": -1}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			resp := post(t, srv.Handler(), tc.path, tc.body)
			if resp.StatusCode != http.StatusBadRequest {
				t.Errorf("status = %d, want 400", resp.StatusCode)
			}
			got := decodeBody[errorResponse](t, resp)
			if got.Error.Message == "" {
				t.Error("error message is empty")
			}
		})
	}
}

func TestPromptLongerThanContextRejected(t *testing.T) {
	srv, tok := newTestServer(t)
	cfg := srv.eng.Model().Config()
	// Encode far more text than the window holds.
	long := strings.Repeat("Once upon a time there was a little girl. ", cfg.ContextLen)
	ids := tok.Encode(long, false)
	if len(ids) <= cfg.ContextLen {
		t.Skipf("encoded only %d tokens for context %d", len(ids), cfg.ContextLen)
	}
	resp := post(t, srv.Handler(), "/v1/completions",
		`{"prompt": `+quoteJSON(long)+`, "max_tokens": 4}`)
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
	_ = resp.Body.Close()
}

// quoteJSON marshals a string, for building a request body around long input.
func quoteJSON(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func TestMaxTokensClampedToContextRoom(t *testing.T) {
	// Asking for more than the window can hold must be clamped, not fail: a client
	// with a 512-token default against a 128-token model should get what fits.
	srv, _ := newTestServer(t)
	resp := post(t, srv.Handler(), "/v1/completions", `{"prompt": "Once upon a time", "max_tokens": 100000}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	body := decodeBody[completionResponse](t, resp)
	cfg := srv.eng.Model().Config()
	if body.Usage.CompletionTokens > cfg.ContextLen {
		t.Errorf("generated %d tokens, more than the context %d", body.Usage.CompletionTokens, cfg.ContextLen)
	}
}

func TestModelsAndHealth(t *testing.T) {
	srv, _ := newTestServer(t)

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("/v1/models status = %d", w.Code)
	}
	models := decodeBody[modelsResponse](t, w.Result())
	if models.Object != "list" || len(models.Data) != 1 {
		t.Errorf("models = %+v", models)
	}
	if models.Data[0].ID != srv.cfg.ModelName {
		t.Errorf("model id = %q, want %q", models.Data[0].ID, srv.cfg.ModelName)
	}

	req = httptest.NewRequest(http.MethodGet, "/healthz", nil)
	w = httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("/healthz status = %d", w.Code)
	}
}

func TestStopStringStops(t *testing.T) {
	srv, tok := newTestServer(t)
	// Learn the greedy continuation, then stop on a token it contains.
	resp := post(t, srv.Handler(), "/v1/completions", `{"prompt": "Once upon a time", "max_tokens": 8}`)
	plain := decodeBody[completionResponse](t, resp)
	full := plain.Choices[0].Text
	if len(full) < 4 {
		t.Skipf("continuation %q is too short to pick a stop from", full)
	}
	// Pick a single-token stop string from the middle of the continuation.
	var stopText string
	for _, tokText := range strings.SplitAfter(full, " ") {
		if tokText == "" {
			continue
		}
		if ids := tok.Encode(tokText, false); len(ids) == 1 {
			stopText = tokText
			break
		}
	}
	if stopText == "" {
		t.Skip("could not find a single-token stop string in the continuation")
	}

	body := `{"prompt": "Once upon a time", "max_tokens": 30, "stop": ` + quoteJSON(stopText) + `}`
	resp2 := post(t, srv.Handler(), "/v1/completions", body)
	stopped := decodeBody[completionResponse](t, resp2)
	if stopped.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want stop", stopped.Choices[0].FinishReason)
	}
	if strings.Contains(stopped.Choices[0].Text, stopText) {
		t.Errorf("output %q contains the stop string %q", stopped.Choices[0].Text, stopText)
	}
}

// waitFor polls cond until it holds or the timeout expires, then fails with msg.
func waitFor(t *testing.T, timeout time.Duration, cond func() bool, msg string) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatal(msg)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// TestClientDisconnectStopsGeneration checks the cancellation path: a client that
// goes away must not leave the model generating into a channel nobody reads.
func TestClientDisconnectStopsGeneration(t *testing.T) {
	// Generation must outlast the cancel, so the fixture needs both a context long
	// enough that 1000 tokens is not cut short by the position limit and no
	// end-of-sequence token to stop on.
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 2048})
	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest(http.MethodPost, "/v1/completions",
		strings.NewReader(`{"prompt": "Once upon a time", "max_tokens": 1000}`)).WithContext(ctx)
	w := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		defer close(done)
		srv.Handler().ServeHTTP(w, req)
	}()
	// Wait until the request is actually in the engine rather than sleeping a
	// guessed interval: the submission is what makes cancel meaningful, and a
	// fixed sleep would turn a slow machine into a flaky failure.
	waitFor(t, 5*time.Second, func() bool { return srv.eng.Active() > 0 },
		"request never reached the engine")
	cancel()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("handler did not return after the client cancelled")
	}
	// The engine must not still be working on the cancelled request.
	waitFor(t, 5*time.Second, func() bool { return srv.eng.Active() == 0 },
		"engine still has an active sequence after the client left")
}

func TestConcurrentRequestsAllComplete(t *testing.T) {
	srv, _ := newTestServer(t)
	const n = 6
	var wg sync.WaitGroup
	texts := make([]string, n)
	for i := range n {
		wg.Go(func() {
			resp := post(t, srv.Handler(), "/v1/completions", `{"prompt": "Once upon a time", "max_tokens": 6}`)
			body := decodeBody[completionResponse](t, resp)
			if len(body.Choices) != 1 {
				return
			}
			texts[i] = body.Choices[0].Text
		})
	}
	wg.Wait()
	for i, text := range texts {
		if text == "" {
			t.Errorf("request %d produced no text", i)
		}
	}
	// Greedy requests with the same prompt must agree, which is what proves the
	// sequences did not read each other's state.
	for i := 1; i < n; i++ {
		if texts[i] != texts[0] {
			t.Errorf("request %d produced %q, request 0 produced %q", i, texts[i], texts[0])
		}
	}
}

func TestRenderChatTemplate(t *testing.T) {
	srv, _ := newTestServer(t)
	got := srv.renderChat([]chatMessage{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hi"},
	})
	want := "System: You are helpful.\nUser: Hi\nAssistant:"
	if got != want {
		t.Errorf("renderChat = %q, want %q", got, want)
	}
	// An empty role defaults to user rather than producing a bare ": ".
	got = srv.renderChat([]chatMessage{{Content: "Hi"}})
	if !strings.HasPrefix(got, "User: Hi") {
		t.Errorf("renderChat with no role = %q", got)
	}
}

func TestStopFieldAcceptsBothSpellings(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{`"x"`, "x"},
		{`["a","b"]`, "a,b"},
		{`null`, ""},
	} {
		var f stopField
		if err := json.Unmarshal([]byte(tc.in), &f); err != nil {
			t.Fatalf("%s: %v", tc.in, err)
		}
		if got := strings.Join(f, ","); got != tc.want {
			t.Errorf("%s decoded to %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestPolicyAffectsShortRequestLatency checks that an admission policy reaches the HTTP
// layer, by measuring whether a short request queued behind a long one gets served before
// the long one finishes.
//
// The assertion is on ordering rather than on elapsed time. A time ratio compared two
// quantities that both move with machine load, so it failed whenever something else was
// running -- the policy rows of `orbix bench --suite` measure the same property more
// precisely, and this test exists to prove the wiring, not the magnitude.
func TestPolicyAffectsShortRequestLatency(t *testing.T) {
	// The long request has to still be running when the short one is submitted, so the
	// fixture needs a context that holds a long budget and no end-of-sequence token to
	// stop on. A constructed model is fast enough that forty tokens finish inside one
	// poll interval, which would make this test measure scheduling luck.
	path := gguftest.Model{NoEOS: true, ContextLen: 4096}.Write(t, t.TempDir())

	type outcome struct {
		// shortWait is how long the short request waited for its response. Under FIFO it
		// queues behind the long request's whole budget; under the other policies it is
		// admitted almost immediately.
		shortWait time.Duration
	}
	measure := func(t *testing.T, policy engine.Priority) outcome {
		t.Helper()
		srv := serverWithPolicy(t, path, policy)

		var wg sync.WaitGroup
		wg.Go(func() {
			resp := post(t, srv.Handler(), "/v1/completions",
				`{"prompt": "Once upon a time", "max_tokens": 2000}`)
			decodeBody[completionResponse](t, resp)
		})
		// Wait until the long request occupies the only slot, so the short one is
		// queued behind it rather than racing it.
		waitFor(t, 5*time.Second, func() bool {
			st := srv.eng.Stats()
			return st.Active > 0 && st.RequestsFinished == 0
		}, "the long request never started")

		var shortWait time.Duration
		wg.Go(func() {
			start := time.Now()
			resp := post(t, srv.Handler(), "/v1/completions",
				`{"prompt": "Hi", "max_tokens": 2}`)
			decodeBody[completionResponse](t, resp)
			shortWait = time.Since(start)
		})
		wg.Wait()
		return outcome{shortWait: shortWait}
	}

	// The assertion is on how long the short request waited, not on which response
	// arrived first. A short request that waits its turn still finishes before a long
	// one that started earlier and has thousands of tokens to go -- comparing arrival
	// order would call that a policy failure when it is the policy working.
	fifo := measure(t, engine.PriorityFIFO)
	t.Logf("FIFO: the short request waited %v", fifo.shortWait)
	for _, policy := range []engine.Priority{engine.PriorityShortest, engine.PriorityFair} {
		got := measure(t, policy)
		t.Logf("%v: the short request waited %v", policy, got.shortWait)
		// An admitted-immediately request waits for two tokens, which on this fixture is
		// milliseconds; a queued one waits for the long request's whole budget.
		if got.shortWait >= fifo.shortWait/2 {
			t.Errorf("%v made the short request wait %v against FIFO's %v; the policy did "+
				"not reach the HTTP layer", policy, got.shortWait, fifo.shortWait)
		}
	}
}

// serverWithPolicy builds a server whose engine runs exactly one request at a time under
// the given policy, so the test can observe queueing rather than batching.
func serverWithPolicy(t *testing.T, path string, policy engine.Priority) *Server {
	t.Helper()
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	eng, err := engine.New(path, engine.Options{MaxActive: 1, Policy: policy})
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

// TestPriorityFieldReachesTheEngine checks the HTTP surface: `priority` is Orbix's own
// extension to the OpenAI schema, so nothing else would catch it being dropped between the
// JSON body and engine.Request.
//
// The observation is indirect by necessity -- the response carries no scheduling
// information -- so the test compares completion order between a low-priority request
// holding the only slot and a high-priority one queued behind it.
func TestPriorityFieldReachesTheEngine(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 4096}.Write(t, t.TempDir())
	srv := serverWithPolicy(t, path, engine.PriorityFIFO)

	var mu sync.Mutex
	finished := map[string]time.Time{}
	record := func(name string, at time.Time) {
		mu.Lock()
		finished[name] = at
		mu.Unlock()
	}

	var wg sync.WaitGroup
	wg.Go(func() {
		resp := post(t, srv.Handler(), "/v1/completions",
			`{"prompt": "Once upon a time", "max_tokens": 2000, "priority": -10}`)
		decodeBody[completionResponse](t, resp)
		record("low", time.Now())
	})
	waitFor(t, 5*time.Second, func() bool { return srv.eng.Active() > 0 },
		"the long request never started")

	// Submitted second, ranked far higher: it must finish first even though the low
	// priority request is holding the only slot and still has 40 tokens to go.
	wg.Go(func() {
		resp := post(t, srv.Handler(), "/v1/completions",
			`{"prompt": "Hi", "max_tokens": 2, "priority": 100}`)
		decodeBody[completionResponse](t, resp)
		record("high", time.Now())
	})
	wg.Wait()

	mu.Lock()
	defer mu.Unlock()
	high, low, ok := finished["high"], finished["low"], len(finished) == 2
	if !ok {
		t.Fatalf("both requests must finish; got %v", finished)
	}
	t.Logf("high finished %v before low", low.Sub(high).Round(time.Millisecond))
	if !high.Before(low) {
		t.Error("the priority-100 request finished after the priority -10 one; " +
			"the priority field did not reach the engine")
	}
}

// TestPriorityIsOptional pins the compatibility property: a client that never heard of
// `priority` must keep working, and an explicit 0 must mean the same as omitting it.
func TestPriorityIsOptional(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	srv := serverWithPolicy(t, path, engine.PriorityFIFO)
	for _, body := range []string{
		`{"prompt": "Hi", "max_tokens": 2}`,
		`{"prompt": "Hi", "max_tokens": 2, "priority": 0}`,
		`{"prompt": "Hi", "max_tokens": 2, "priority": -3}`,
		`{"prompt": "Hi", "max_tokens": 2, "priority": 7}`,
	} {
		resp := post(t, srv.Handler(), "/v1/completions", body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("%s: status = %d, want 200", body, resp.StatusCode)
		}
		got := decodeBody[completionResponse](t, resp)
		if len(got.Choices) != 1 {
			t.Fatalf("%s: %d choices", body, len(got.Choices))
		}
	}
}

// TestShutdownTracksThePumpGoroutine checks Shutdown actually waits for the pump it stops.
//
// The waitgroup was incremented after the deferred decrement, which leaves a window where
// Wait sees a zero counter and returns while the pump goroutine is not yet tracked: the
// server reports a clean shutdown while the goroutine is still starting up. The window is
// narrow, which is why this runs many iterations and why the race detector is the thing
// that sees it rather than an assertion.
//
// Each iteration is a subtest so it gets its own temp directory; sharing one across the
// loop made the detector report races in the test harness rather than in the server.
func TestShutdownTracksThePumpGoroutine(t *testing.T) {
	for range 20 {
		t.Run("", func(t *testing.T) {
			path := gguftest.Model{NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
			gf, err := gguf.Open(path)
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			tok, err := tokenize.New(gf)
			if err != nil {
				t.Fatalf("tokenize: %v", err)
			}
			eng, err := engine.New(path, engine.Options{MaxActive: 4})
			if err != nil {
				t.Fatalf("engine.New: %v", err)
			}
			cfg := eng.Model().Config()
			srv, err := New(eng, tok, Config{ModelName: cfg.Name, ContextLen: cfg.ContextLen})
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			// StartPump accounts for the goroutine before it exists, so Shutdown is
			// guaranteed to wait for it however soon it is called.
			srv.StartPump(context.Background())
			srv.Shutdown()
		})
	}
}

// TestShutdownIsIdempotent checks the lifecycle edges a server actually reaches: two
// shutdown paths running at once, and a shutdown before the pump was started.
//
// Closing the stop channel twice panics, and a process that installs both a signal handler
// and a deferred cleanup will close it twice on the ordinary path. A waitgroup with no Add
// is fine, so the second case needed no fix -- it is pinned so a future change to the
// accounting cannot quietly reintroduce the panic.
func TestShutdownIsIdempotent(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	eng, err := engine.New(path, engine.Options{MaxActive: 2})
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	cfg := eng.Model().Config()
	newSrv := func() *Server {
		srv, err := New(eng, tok, Config{ModelName: cfg.Name, ContextLen: cfg.ContextLen})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		return srv
	}

	t.Run("twice after StartPump", func(t *testing.T) {
		srv := newSrv()
		srv.StartPump(context.Background())
		srv.Shutdown()
		srv.Shutdown()
	})
	t.Run("before StartPump", func(t *testing.T) {
		newSrv().Shutdown()
	})
}

// TestResponseObjectAndIDFollowOpenAI checks the two envelope fields a client reads before
// it reads any text, across all four endpoint and streaming combinations.
//
// They are easy to get subtly wrong and impossible to notice by testing text: the body is
// right, the model answers correctly, and only a client that switches on "object" or logs
// the id sees the difference. OpenAI's own streaming example for a completion carries
// object="text_completion", the same as the non-streaming one -- the chat endpoint is the
// one that changes to "chat.completion.chunk" when it streams.
func TestResponseObjectAndIDFollowOpenAI(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()

	cases := []struct {
		name       string
		path       string
		body       string
		stream     bool
		wantObject string
		wantPrefix string
	}{
		{"completion", "/v1/completions",
			`{"prompt": "Once", "max_tokens": 2}`, false, "text_completion", "cmpl-"},
		{"completion streaming", "/v1/completions",
			`{"prompt": "Once", "max_tokens": 2, "stream": true}`, true, "text_completion", "cmpl-"},
		{"chat", "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}`, false,
			"chat.completion", "chatcmpl-"},
		{"chat streaming", "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "stream": true}`, true,
			"chat.completion.chunk", "chatcmpl-"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			resp := post(t, h, c.path, c.body)
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("status = %d, want 200", resp.StatusCode)
			}
			var envelope struct {
				ID     string `json:"id"`
				Object string `json:"object"`
			}
			if c.stream {
				// The first SSE frame carries the envelope.
				var first string
				sc := bufio.NewScanner(resp.Body)
				for sc.Scan() {
					if p := strings.TrimPrefix(sc.Text(), "data: "); p != sc.Text() && p != "[DONE]" {
						first = p
						break
					}
				}
				_ = resp.Body.Close()
				if first == "" {
					t.Fatal("no SSE frame")
				}
				if err := json.Unmarshal([]byte(first), &envelope); err != nil {
					t.Fatalf("frame %q: %v", first, err)
				}
			} else {
				if err := json.NewDecoder(resp.Body).Decode(&envelope); err != nil {
					t.Fatalf("decode: %v", err)
				}
				_ = resp.Body.Close()
			}
			if envelope.Object != c.wantObject {
				t.Errorf("object = %q, want %q", envelope.Object, c.wantObject)
			}
			if !strings.HasPrefix(envelope.ID, c.wantPrefix) {
				t.Errorf("id = %q, want the %q prefix", envelope.ID, c.wantPrefix)
			}
		})
	}
}

// TestPumpPicksUpWorkAfterIdle checks that the idle backoff does not delay a request.
//
// The pump sleeps longer and longer while the engine is empty, to keep an idle server from
// costing 2.4% of a core just to notice it has nothing to do. The risk is the first request
// after a quiet period: it waits for the next poll, where before the backoff the wait was
// at most a millisecond.
//
// The bound is the maximum backoff and little else. A slower pump is caught by the request
// not being served at all; the point of this test is that the sleep after an idle spell is
// a poll interval, not a stall.
func TestPumpPicksUpWorkAfterIdle(t *testing.T) {
	srv, _ := newTestServer(t)
	// Longer than the deepest backoff, so the pump is certainly in its slow state.
	time.Sleep(60 * time.Millisecond)

	start := time.Now()
	resp := post(t, srv.Handler(), "/v1/completions",
		`{"prompt": "Once upon a time", "max_tokens": 2}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	decodeBody[completionResponse](t, resp)
	// Wide enough for a loaded machine, narrow enough that a pump asleep for anything other
	// than a poll interval fails it.
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("a request after an idle period took %v", elapsed)
	}
}
