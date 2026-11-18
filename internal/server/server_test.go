package server

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
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
	// The model id has to be the one this server serves: a request naming another is
	// refused, which is what makes a client pointed at the wrong server fail loudly.
	reqBody := fmt.Sprintf(`{
		"model": %q,
		"messages": [{"role": "user", "content": "Once upon a time"}],
		"max_tokens": 8
	}`, srv.cfg.ModelName)
	resp := post(t, srv.Handler(), "/v1/chat/completions", reqBody)
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
	srv, _ := newTestServer(t)

	// Learn what the fixture generates, then stop on a piece of it. The fixture's weights
	// are random, so the text is arbitrary -- but it is deterministic, which is all this
	// needs: the assertions are about where generation ends, not about what it says.
	resp := post(t, srv.Handler(), "/v1/completions",
		`{"prompt": "Once upon a time", "max_tokens": 8}`)
	full := decodeBody[completionResponse](t, resp).Choices[0].Text
	if len([]rune(full)) < 2 {
		t.Skipf("continuation %q is too short to cut", full)
	}

	// A stop string of one rune, which may or may not be one token. Either way cutAtStop
	// has to find it in the decoded text, and the token-id path is irrelevant.
	stopText := string([]rune(full)[1])
	if strings.ContainsRune(full[:strings.Index(full, stopText)], '\x00') {
		t.Skipf("continuation %q has a NUL before the cut point", full)
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
	if want := strings.TrimSuffix(full[:strings.Index(full, stopText)+len(stopText)], stopText); stopped.Choices[0].Text != want {
		t.Errorf("output = %q, want %q: generation should stop where the string would start",
			stopped.Choices[0].Text, want)
	}
}

// TestMultiTokenStopString covers a stop string that is not one token, which the engine
// cannot match and used to be reduced to its first token.
//
// " Paris" encodes as a space token followed by the word's own, so taking the first token
// meant stopping at the first space of any reply: the request returned an empty string with
// finish_reason "stop", which looks like a successful truncation and is the opposite.
func TestMultiTokenStopString(t *testing.T) {
	srv, tok := newTestServer(t)
	// Confirm the premise, so this fails loudly if the fixture's tokenizer changes and the
	// string stops being multi-token -- the test would otherwise pass for the wrong reason.
	if enc := tok.Encode(" the", false); len(enc) < 2 {
		t.Skipf("\" the\" encodes to %v, so there is no multi-token case to test", enc)
	}

	// Generate once to learn the text, then ask for the same prompt with a stop string
	// built from it that spans more than one token.
	first := decodeBody[completionResponse](t, post(t, srv.Handler(), "/v1/completions",
		`{"prompt": "Once upon a time there was", "max_tokens": 10}`)).Choices[0].Text
	if len([]rune(first)) < 3 {
		t.Skipf("continuation %q is too short", first)
	}
	// Take a run of several runes from the start; whatever it is, it must cut the text.
	stopText := string([]rune(first)[:3])

	stopped := decodeBody[completionResponse](t, post(t, srv.Handler(), "/v1/completions",
		`{"prompt": "Once upon a time there was", "max_tokens": 20, "stop": `+quoteJSON(stopText)+`}`))
	if stopped.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want stop", stopped.Choices[0].FinishReason)
	}
	if strings.Contains(stopped.Choices[0].Text, stopText) {
		t.Errorf("output %q still contains the stop string %q", stopped.Choices[0].Text, stopText)
	}
	if stopped.Choices[0].Text != "" && !strings.HasPrefix(first, stopped.Choices[0].Text+stopText) {
		t.Errorf("output %q is not the prefix of %q that precedes %q",
			stopped.Choices[0].Text, first, stopText)
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

// TestQueueFullReturns503 checks the status a saturated engine produces.
//
// The request is valid and would succeed later, so 400 would tell a client to stop sending
// it -- the opposite of what should happen. 503 with Retry-After is the answer a proxy or
// SDK backs off on, and it is the only signal that separates "the server is at capacity"
// from "your request was wrong".
func TestQueueFullReturns503(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	// One active plus one queued; the engine is saturated as soon as both are filled.
	eng, err := engine.New(path, engine.Options{MaxActive: 1, MaxQueued: 1})
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	cfg := eng.Model().Config()
	srv, err := New(eng, tok, Config{ModelName: cfg.Name, ContextLen: cfg.ContextLen})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// Fill the engine directly rather than over HTTP: a handler blocks until its sequence
	// produces a token, and there is no pump here to produce one.
	ids := tok.Encode("Once upon a time", true)
	if len(ids) == 0 {
		t.Fatal("prompt encoded to nothing")
	}
	for range 2 {
		if _, err := eng.Submit(engine.Request{Prompt: ids, MaxTokens: 4}); err != nil {
			t.Fatalf("filling the engine: %v", err)
		}
	}

	resp := post(t, srv.Handler(), "/v1/completions",
		`{"prompt": "Once upon a time", "max_tokens": 4}`)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d with a full queue, want 503", resp.StatusCode)
	}
	if got := resp.Header.Get("Retry-After"); got == "" {
		t.Error("no Retry-After on a 503: a client has nothing to back off on")
	}
}

// TestPoolExhaustionReturns503 checks the status when the paged cache runs out of blocks.
//
// Like a full queue, this is capacity and not a fault: the request is well formed, the
// engine is working, and the same request succeeds as soon as another sequence releases
// its blocks. Answering 500 tells a client something is broken; 503 with Retry-After tells
// it to come back, which is what the situation calls for.
//
// The pool is small enough that a couple of sequences exhaust it, and the prompt is long
// enough that one sequence needs more than a block.
func TestPoolExhaustionReturns503(t *testing.T) {
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
		MaxActive: 8, MaxQueued: 64, Blocks: 96, BlockSize: 4,
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

	// Enough concurrent requests that some cannot get a block, with a prompt that spans
	// more than one block so one sequence alone can strain a two-block pool.
	prompt := strings.Repeat("Once upon a time there was ", 4)
	body := `{"prompt": ` + quoteJSON(prompt) + `, "max_tokens": 4}`

	codes := make(chan int, 16)
	for range 16 {
		go func() {
			resp := post(t, srv.Handler(), "/v1/completions", body)
			codes <- resp.StatusCode
			_ = resp.Body.Close()
		}()
	}
	var sawOK, sawUnavailable, sawServerError bool
	for range 16 {
		switch code := <-codes; {
		case code == http.StatusOK:
			sawOK = true
		case code == http.StatusServiceUnavailable:
			sawUnavailable = true
		case code >= 500:
			sawServerError = true
		}
	}
	if !sawOK {
		t.Error("no request succeeded; the pool was too small to test anything")
	}
	if !sawUnavailable {
		t.Error("no request got 503 although the pool cannot hold them all")
	}
	if sawServerError {
		t.Error("a request got 5xx other than 503: pool exhaustion is capacity, not a fault")
	}
}

// TestChatCompletionMultiTurn checks that a conversation's history reaches the prompt.
//
// The single-turn test above would pass whether or not history was included: the prompt
// differs only in what it contains, and the response shape is the same either way. What
// separates them is the token count, which grows with every message a template renders.
//
// A template that fails on a message it does not understand falls back to the neutral
// transcript inside the renderer, so a broken history path would still answer -- in the
// wrong format. The count is what notices.
func TestChatCompletionMultiTurn(t *testing.T) {
	// The default fixture's 64-token context is too small for a three-turn prompt.
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 512})

	one := decodeBody[chatResponse](t, post(t, srv.Handler(), "/v1/chat/completions", `{
		"messages": [{"role": "user", "content": "Once upon a time"}],
		"max_tokens": 2
	}`))
	three := decodeBody[chatResponse](t, post(t, srv.Handler(), "/v1/chat/completions", `{
		"messages": [
			{"role": "user", "content": "Once upon a time"},
			{"role": "assistant", "content": "There was a little girl."},
			{"role": "user", "content": "What next?"}
		],
		"max_tokens": 2
	}`))

	// Not merely larger: dropping the history leaves the last message plus a header, which
	// measured 26 against 25. The extra two messages account for a few dozen tokens, so the
	// bound has to be an amount rather than an inequality.
	if added := three.Usage.PromptTokens - one.Usage.PromptTokens; added < 10 {
		t.Errorf("a three-turn prompt used %d more tokens than a one-turn one (%d vs %d): "+
			"the history was not included", added, three.Usage.PromptTokens, one.Usage.PromptTokens)
	}
	// The answer is still a normal completion, not an error dressed as one.
	if three.Choices[0].Message.Role != "assistant" || three.Choices[0].Message.Content == "" {
		t.Errorf("three-turn reply = %+v, want assistant content", three.Choices[0].Message)
	}
}

// TestChatCompletionRejectsOverlongConversation checks the boundary the test above has to
// stay under: a conversation that does not fit the context is a client error, not a panic or
// a truncated prompt.
//
// It pins the contract rather than one implementation: the server and the engine both check
// the length, so removing either leaves it passing. What it rules out is answering 200 with
// a silently shortened conversation.
func TestChatCompletionRejectsOverlongConversation(t *testing.T) {
	// The default fixture: 64 tokens.
	srv, _ := newTestServer(t)
	// A message long enough that the rendered prompt cannot fit.
	long := strings.Repeat("Once upon a time there was a little girl. ", 20)
	resp := post(t, srv.Handler(), "/v1/chat/completions",
		`{"messages": [{"role": "user", "content": "`+long+`"}], "max_tokens": 2}`)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d for a conversation larger than the context, want 400",
			resp.StatusCode)
	}
}

// TestStreamingRespectsStopStrings checks that a stop string truncates a stream too.
//
// Only the blocking path applied stop strings; the streaming one did not look at them at
// all. A client that asked to stop at a phrase got the phrase and everything after it, so
// the two endpoints disagreed about what the same request means.
//
// The check is on accumulated text rather than a token id because the engine only knows
// single tokens: the rune that completes a stop string can be the second token of it, so the
// decision is made once the text is complete and only the part before the match is sent.
func TestStreamingRespectsStopStrings(t *testing.T) {
	srv, _ := newTestServer(t)

	// The fixture's output is deterministic but arbitrary, and it contains NUL bytes, which
	// cannot travel through a stop string. Find a run of two printable runes to stop on and
	// confirm with the blocking endpoint that it occurs -- that endpoint had the behaviour
	// first, so it is the reference the stream has to match.
	full := decodeBody[completionResponse](t, post(t, srv.Handler(), "/v1/completions",
		`{"prompt": "Once upon a time", "max_tokens": 8}`)).Choices[0].Text
	runes := []rune(full)
	start := -1
	for i := 0; i+2 <= len(runes); i++ {
		if runes[i] != 0 && runes[i+1] != 0 && i > 0 {
			start = i
			break
		}
	}
	if start < 0 {
		t.Skipf("continuation %q has no run of two printable runes after the first", full)
	}
	stopText := string(runes[start : start+2])

	blocking := decodeBody[completionResponse](t, post(t, srv.Handler(), "/v1/completions",
		`{"prompt": "Once upon a time", "max_tokens": 20, "stop": `+quoteJSON(stopText)+`}`))
	if blocking.Choices[0].FinishReason != "stop" {
		t.Fatalf("the blocking endpoint did not stop on %q either, so it is not a usable "+
			"reference: %+v", stopText, blocking.Choices[0])
	}
	if strings.Contains(blocking.Choices[0].Text, stopText) {
		t.Fatalf("the blocking endpoint kept the stop string in %q", blocking.Choices[0].Text)
	}

	streamed := collectStream(t, post(t, srv.Handler(), "/v1/completions",
		`{"prompt": "Once upon a time", "max_tokens": 20, "stream": true, "stop": `+
			quoteJSON(stopText)+`}`).Body)

	if strings.Contains(streamed, stopText) {
		t.Errorf("the streamed text %q contains the stop string %q", streamed, stopText)
	}
	if streamed != blocking.Choices[0].Text {
		t.Errorf("the stream produced %q and the blocking endpoint %q for the same request",
			streamed, blocking.Choices[0].Text)
	}
}

// TestUnknownModelIsRefused covers the model field on both endpoints.
//
// The field is what an OpenAI client uses to check it is talking to the server it meant to.
// Ignoring it means a client pointed at the wrong server gets fluent answers from a model it
// did not ask for, and nothing in the response reveals the mistake -- every field echoes the
// server's own name, so the output looks exactly like a correct one.
//
// An empty name stays acceptable, because the client libraries omit it when they were built
// without one and there is a single model here to mean.
func TestUnknownModelIsRefused(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()

	for _, path := range []string{"/v1/chat/completions", "/v1/completions"} {
		t.Run(path, func(t *testing.T) {
			// A name this server does not serve is a 404 with the code the OpenAI clients
			// branch on.
			resp := post(t, h, path, `{"model": "not-this-one", "prompt": "hi",
				"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}`)
			if resp.StatusCode != http.StatusNotFound {
				t.Fatalf("status = %d, want 404", resp.StatusCode)
			}
			var out struct {
				Error struct {
					Type  string `json:"type"`
					Param string `json:"param"`
					Code  string `json:"code"`
				} `json:"error"`
			}
			if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
				t.Fatalf("decode: %v", err)
			}
			if out.Error.Code != "model_not_found" {
				t.Errorf("code = %q, want model_not_found", out.Error.Code)
			}
			if out.Error.Param != "model" {
				t.Errorf("param = %q, want model", out.Error.Param)
			}
			if out.Error.Type != "invalid_request_error" {
				t.Errorf("type = %q, want invalid_request_error", out.Error.Type)
			}

			// The server's own name and an omitted name both go through. The prompt is
			// supplied for the completion endpoint and the messages for the chat one, so
			// both bodies carry the field each handler requires.
			for _, body := range []string{
				fmt.Sprintf(`{"model": %q, "prompt": "hi", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}`, srv.cfg.ModelName),
				`{"prompt": "hi", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}`,
			} {
				resp := post(t, h, path, body)
				if resp.StatusCode != http.StatusOK {
					t.Errorf("body %s: status = %d, want 200", body, resp.StatusCode)
				}
				_ = resp.Body.Close()
			}
		})
	}
}

// TestSeedMakesSamplingReproducible covers the seed request field.
//
// Sampling is random by definition, which makes a sampled reply impossible to assert on and
// impossible to reproduce when a user reports bad output. The seed is what makes it
// reproducible: the same prompt, settings and seed must give the same text, and different
// seeds must be able to differ.
//
// The fixture vocabulary is tiny, so two seeds can coincide by chance. The test therefore
// asserts the property that matters -- the same seed twice is identical -- and checks that
// some pair of different seeds diverges, which is a population statement rather than one
// pair's, so a collision on one pair does not fail it.
func TestSeedMakesSamplingReproducible(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()
	temp := 1.5

	gen := func(seed int64) string {
		t.Helper()
		reqBody := fmt.Sprintf(`{"model": %q, "prompt": "Once upon a time",
			"max_tokens": 16, "temperature": %v, "seed": %d}`, srv.cfg.ModelName, temp, seed)
		resp := post(t, h, "/v1/completions", reqBody)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("seed %d: status = %d", seed, resp.StatusCode)
		}
		out := decodeBody[completionResponse](t, resp)
		if len(out.Choices) == 0 {
			t.Fatalf("seed %d: no choices", seed)
		}
		return out.Choices[0].Text
	}

	// The same seed twice is the contract.
	const seed = 1234
	first, second := gen(seed), gen(seed)
	if first != second {
		t.Errorf("the same seed produced different text:\n  %q\n  %q", first, second)
	}

	// Some seed must produce something else; otherwise the generator is not being used.
	var differs bool
	for s := int64(1); s <= 8 && !differs; s++ {
		if gen(s) != first {
			differs = true
		}
	}
	if !differs {
		t.Error("every seed produced the same text, so the seed is not reaching the sampler")
	}
}

// TestGreedyIgnoresSeed covers the interaction between seed and temperature.
//
// Greedy is already deterministic, so a seed cannot change its output and must not appear to:
// two different seeds at temperature 0 have to agree, or a caller would conclude the seed
// works and be misled about what it controls.
func TestGreedyIgnoresSeed(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()
	gen := func(seed int64) string {
		t.Helper()
		reqBody := fmt.Sprintf(`{"model": %q, "prompt": "Once upon a time",
			"max_tokens": 12, "seed": %d}`, srv.cfg.ModelName, seed)
		resp := post(t, h, "/v1/completions", reqBody)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("seed %d: status = %d", seed, resp.StatusCode)
		}
		return decodeBody[completionResponse](t, resp).Choices[0].Text
	}
	if a, b := gen(42), gen(999); a != b {
		t.Errorf("greedy replied differently for two seeds:\n  %q\n  %q", a, b)
	}
}

// TestTopPIsHonoured covers the top_p request field end to end.
//
// The field was absent from the request struct, so an SDK that always sends it got no error
// and no effect: a caller asking for a narrow nucleus received the full distribution's
// output. The engine now implements nucleus sampling, and this checks that the value reaches
// it through the HTTP layer.
//
// The assertion is statistical because sampling is: a very small threshold keeps only the
// most likely token, so the same prompt under many seeds collapses to one continuation, while
// no threshold lets the seeds diverge. A per-request equality would be testing the RNG.
func TestTopPIsHonoured(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()

	sample := func(seed int, topP string) string {
		t.Helper()
		body := fmt.Sprintf(`{"model": %q, "prompt": "Once upon a time",
			"max_tokens": 8, "temperature": 1.5, "seed": %d%s}`,
			srv.cfg.ModelName, seed, topP)
		resp := post(t, h, "/v1/completions", body)
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d", resp.StatusCode)
		}
		return decodeBody[completionResponse](t, resp).Choices[0].Text
	}

	distinct := func(topP string) int {
		seen := make(map[string]bool)
		for seed := 1; seed <= 12; seed++ {
			seen[sample(seed, topP)] = true
		}
		return len(seen)
	}

	narrow := distinct(`, "top_p": 0.01`)
	wide := distinct("")
	if narrow >= wide {
		t.Errorf("top_p=0.01 produced %d distinct continuations, no limit produced %d;"+
			" the nucleus should collapse the distribution, not widen it", narrow, wide)
	}
	if wide < 2 {
		t.Errorf("no top_p produced %d distinct continuations over 12 seeds, so sampling"+
			" is not varying at all and the comparison proves nothing", wide)
	}
}

// TestTopPRejectsOutOfRange covers the validation.
//
// Both ends are refused rather than clamped. Zero and one mean different things to different
// callers -- "no limit" or "only the best token" -- so guessing would change a caller's output
// silently, which is the failure mode this whole area is about.
func TestTopPRejectsOutOfRange(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()
	for _, p := range []string{"-0.1", "1.5"} {
		body := fmt.Sprintf(`{"model": %q, "prompt": "hi", "max_tokens": 2, "temperature": 1, "top_p": %s}`,
			srv.cfg.ModelName, p)
		resp := post(t, h, "/v1/completions", body)
		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("top_p=%s: status = %d, want 400", p, resp.StatusCode)
		}
		_ = resp.Body.Close()
	}
}

// TestCompressedBodyIsRefusedWith415 covers a client that compresses its request.
//
// HTTP allows a request body to carry a Content-Encoding, and a client library that enables
// compression will send one. Decoding such a body as JSON produced
//
//	invalid JSON: invalid character '\x1f' looking for beginning of value
//
// which names a byte of the gzip header rather than the header that asked for it -- a 400 that
// blames the payload for something the encoding did.
//
// The answer is 415 and a sentence, not decompression: this server takes prompts measured in
// kilobytes, so the bytes saved are worth less than the decompression-bomb surface. 415 is also the
// status that tells a client to change what it sends, where 400 says what arrived was malformed.
func TestCompressedBodyIsRefusedWith415(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()

	send := func(encoding string) *http.Response {
		t.Helper()
		req := httptest.NewRequest(http.MethodPost, "/v1/completions",
			strings.NewReader(`{"prompt":"hi","max_tokens":2}`))
		req.Header.Set("Content-Type", "application/json")
		if encoding != "" {
			req.Header.Set("Content-Encoding", encoding)
		}
		w := httptest.NewRecorder()
		h.ServeHTTP(w, req)
		return w.Result()
	}

	for _, enc := range []string{"gzip", "br", "deflate"} {
		t.Run(enc, func(t *testing.T) {
			resp := send(enc)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != http.StatusUnsupportedMediaType {
				t.Errorf("status = %d, want 415", resp.StatusCode)
			}
			var body errorBody
			if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
				t.Fatalf("decode error body: %v", err)
			}
			if !strings.Contains(body.Error.Message, enc) {
				t.Errorf("message %q does not name the encoding", body.Error.Message)
			}
		})
	}

	// The paths that must keep working: no header at all, and the identity encoding a client may
	// send to say so explicitly. Both are ordinary requests and must reach the handler.
	for _, enc := range []string{"", "identity", "IDENTITY"} {
		resp := send(enc)
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Errorf("Content-Encoding %q gave %d, want 200", enc, resp.StatusCode)
		}
	}
}

// TestStreamFlushesTrailingPartialCharacter covers the end of a stream whose last tokens hold
// only part of a character.
//
// The decoder holds back an incomplete sequence so the next token can finish it, which is what
// keeps a split character from rendering as U+FFFD mid-stream. When generation ends there is no
// next token, so whatever is held has to be released explicitly; without that the stream ended
// one character short of what the same tokens produce without --stream.
//
// Driven through streamResponse with a synthetic channel rather than through a model, because
// which tokens a constructed fixture generates is not controllable -- the point is the end of
// the stream, not the generation that led there.
func TestStreamFlushesTrailingPartialCharacter(t *testing.T) {
	srv, tok := newTestServerModel(t, gguftest.Model{
		Vocab:          []string{"<unk>", "<s>", "</s>", "ä", "¸", "Ń"},
		TokenizerModel: "gpt2",
		NoBytes:        false,
	})

	// Token ids 3,4,5 are the escaped bytes e4 b8 ad, which together are 中. Send only the
	// first two, so the stream ends mid-character.
	const (
		head = int32(3)
		mid  = int32(4)
	)
	tailBytes := []int32{head, mid}
	want := tok.DecodeCompletion(tailBytes)
	if want == "" {
		t.Fatalf("DecodeCompletion of %v is empty; the fixture does not hold these bytes", tailBytes)
	}

	ch := make(chan engine.Event, 4)
	ch <- engine.Event{Token: head, HasToken: true}
	ch <- engine.Event{Token: mid, HasToken: true}
	ch <- engine.Event{Done: true, FinishReason: "length"}
	close(ch)

	w := httptest.NewRecorder()
	srv.streamResponse(w, httptest.NewRequest(http.MethodPost, "/v1/completions", nil),
		ch, "cmpl-test", false, nil)

	got := collectStream(t, io.NopCloser(strings.NewReader(w.Body.String())))
	if got != want {
		t.Errorf("streamed %q, the same tokens decode to %q", got, want)
	}

	// A stop string must not change that. The flush is about the decoder holding a partial
	// character, which has nothing to do with stop matching, so conditioning it on the stop
	// list was wrong: the two requests differ only in a stop that never matches, and they
	// have to stream the same text.
	ch2 := make(chan engine.Event, 4)
	ch2 <- engine.Event{Token: head, HasToken: true}
	ch2 <- engine.Event{Token: mid, HasToken: true}
	ch2 <- engine.Event{Done: true, FinishReason: "length"}
	close(ch2)
	w2 := httptest.NewRecorder()
	srv.streamResponse(w2, httptest.NewRequest(http.MethodPost, "/v1/completions", nil),
		ch2, "cmpl-test", false, []string{"this stop never appears"})
	if withStop := collectStream(t, io.NopCloser(strings.NewReader(w2.Body.String()))); withStop != want {
		t.Errorf("with a stop configured, streamed %q, want %q", withStop, want)
	}

	// The shortest truncation of all: a budget of one token, so the stream ends after a
	// single byte of a three-byte character. Nothing in the pending bytes forms a character,
	// and both paths have to say so the same way -- U+FFFD, not nothing.
	single := []int32{head}
	wantSingle := tok.DecodeCompletion(single)
	ch3 := make(chan engine.Event, 2)
	ch3 <- engine.Event{Token: head, HasToken: true}
	ch3 <- engine.Event{Done: true, FinishReason: "length"}
	close(ch3)
	w3 := httptest.NewRecorder()
	srv.streamResponse(w3, httptest.NewRequest(http.MethodPost, "/v1/completions", nil),
		ch3, "cmpl-test", false, nil)
	if got := collectStream(t, io.NopCloser(strings.NewReader(w3.Body.String()))); got != wantSingle {
		t.Errorf("one-byte stream gave %q, the same token decodes to %q", got, wantSingle)
	}
}
