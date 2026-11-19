package server

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
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

// TestOversizedBodyIs413 covers the bound on how much a request may send.
//
// Before it existed the decoder read whatever arrived: a 64 MiB body was read in full,
// tokenized, and rejected as context_length_exceeded after 82 seconds of CPU. No status
// code can be chosen until the body is parsed, so all that work happened before the request
// was known to be impossible -- and making the body bigger always cost more, with no point
// at which the server stopped reading.
//
// 413 rather than 400 is the part a client acts on: the JSON was fine, the size was not.
func TestOversizedBodyIs413(t *testing.T) {
	srv, _ := newTestServer(t)
	limit := srv.maxBodyBytes()

	// One byte past the limit, with a body that is valid JSON up to the truncation so the
	// failure cannot be blamed on syntax.
	body := `{"prompt":"` + strings.Repeat("x", int(limit)) + `","max_tokens":1}`
	resp := post(t, srv.Handler(), "/v1/completions", body)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want 413 for a body over %d bytes", resp.StatusCode, limit)
	}
	env := decodeBody[errorBody](t, resp)
	if !strings.Contains(env.Error.Message, "too large") {
		t.Errorf("message %q does not say the body is too large", env.Error.Message)
	}

	// The same limit must reject a chat request, since it shares the decoder.
	chatBody := `{"messages":[{"role":"user","content":"` + strings.Repeat("y", int(limit)) + `"}]}`
	resp2 := post(t, srv.Handler(), "/v1/chat/completions", chatBody)
	defer func() { _ = resp2.Body.Close() }()
	if resp2.StatusCode != http.StatusRequestEntityTooLarge {
		t.Errorf("chat status = %d, want 413", resp2.StatusCode)
	}
}

// TestNormalRequestFitsUnderTheBodyLimit is the other half: the bound must not reject
// requests the model can actually serve. A prompt filling the context is the largest
// legitimate body there is, and it has to pass the size check -- whether the engine can
// generate from it is a separate question.
func TestNormalRequestFitsUnderTheBodyLimit(t *testing.T) {
	srv, _ := newTestServer(t)
	limit := srv.maxBodyBytes()
	if limit < 1<<20 {
		t.Errorf("body limit is %d bytes; a small context must not produce a limit that "+
			"rejects ordinary conversation", limit)
	}
	// A body comfortably inside the limit must get past decodeJSON to a semantic error,
	// which proves the size check did not fire.
	body := `{"prompt":"` + strings.Repeat("a", 4096) + `","max_tokens":1}`
	resp := post(t, srv.Handler(), "/v1/completions", body)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode == http.StatusRequestEntityTooLarge {
		t.Errorf("a %d-byte body was rejected against a limit of %d", len(body), limit)
	}
}

// TestMaxBodyBytesScalesWithContext pins the derivation: the limit is the context size
// times a per-token byte allowance, with a floor so a tiny model still accepts normal
// requests.
func TestMaxBodyBytesScalesWithContext(t *testing.T) {
	cases := []struct {
		ctx  int
		want int64
		why  string
	}{
		{ctx: 64, want: 1 << 20, why: "a tiny context still gets the 1 MiB floor"},
		{ctx: 1000, want: 1 << 20, why: "48 KB is below the floor"},
		{ctx: 131072, want: 131072 * 48, why: "a 128k context allows 6 MiB of text"},
	}
	for _, c := range cases {
		srv := &Server{cfg: Config{ContextLen: c.ctx}}
		if got := srv.maxBodyBytes(); got != c.want {
			t.Errorf("maxBodyBytes(ctx=%d) = %d, want %d (%s)", c.ctx, got, c.want, c.why)
		}
	}
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

// TestStreamingResponseDeclaresItselfUnbuffered covers the header that keeps a reverse proxy
// from buffering the stream.
//
// nginx buffers proxied responses by default (`proxy_buffering on`), which holds an SSE
// stream until the response ends. The events still arrive, all at once, which for a token
// stream is the same as not streaming. nginx will not infer the intent from the media type --
// its maintainers declined to special-case text/event-stream, since a media type describes how
// a payload is formatted rather than how it is delivered -- so the application has to say so.
//
// Only the streaming responses may carry it. Sending it on everything would turn off
// buffering for the whole API, and proxy buffers exist to decouple a fast upstream from a slow
// client, which is worth keeping for the responses that are not streams.
func TestStreamingResponseDeclaresItselfUnbuffered(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, NoBytes: true})
	h := srv.Handler()

	for _, path := range []string{"/v1/completions", "/v1/chat/completions"} {
		t.Run("stream "+strings.ReplaceAll(strings.TrimPrefix(path, "/v1/"), "/", "_"), func(t *testing.T) {
			body := `{"prompt": "Once upon a time", "max_tokens": 2, "stream": true}`
			if strings.HasPrefix(path, "/v1/chat") {
				body = `{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2,
					"stream": true}`
			}
			resp := post(t, h, path, body)
			defer func() { _ = resp.Body.Close() }()
			if got := resp.Header.Get("X-Accel-Buffering"); got != "no" {
				t.Errorf("X-Accel-Buffering = %q on a stream, want \"no\"; "+
					"a proxy would hold every token until the response ends", got)
			}
			// The response must still be a stream: the header is a request to the proxy, not
			// a substitute for the content type.
			if got := resp.Header.Get("Content-Type"); !strings.HasPrefix(got, "text/event-stream") {
				t.Errorf("Content-Type = %q, want text/event-stream", got)
			}
		})
	}

	// The non-streaming paths keep the proxy's buffering, so they must not carry the header.
	for _, path := range []string{"/v1/completions", "/v1/chat/completions"} {
		t.Run("blocking "+strings.ReplaceAll(strings.TrimPrefix(path, "/v1/"), "/", "_"), func(t *testing.T) {
			body := `{"prompt": "Once upon a time", "max_tokens": 2}`
			if strings.HasPrefix(path, "/v1/chat") {
				body = `{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}`
			}
			resp := post(t, h, path, body)
			defer func() { _ = resp.Body.Close() }()
			if got := resp.Header.Get("X-Accel-Buffering"); got != "" {
				t.Errorf("X-Accel-Buffering = %q on a buffered response, want empty", got)
			}
		})
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

// TestModelRetrieveAndUnknownEndpoints covers the two paths an OpenAI SDK takes that this
// server used to answer with Go's plain-text 404.
//
// Both matter for the same reason: the client libraries decode the response body as JSON
// before they raise. A text/plain "404 page not found" is not a not-found error to them, it
// is a parse failure, so the failure a caller sees names the decoder rather than the
// endpoint. The SDKs' models.retrieve hits GET /v1/models/{id}, and a client may call
// endpoints this engine does not implement (embeddings, images, responses); both now answer
// in the API's own envelope.
// TestModelNameSurvivesJSONAndURL covers a model name that is not a plain identifier.
//
// The id comes from the checkpoint's general.name, which is free-form text: a converted file
// can carry quotes, a slash, or CJK. Two encodings have to hold up and neither was tested --
// the JSON body must escape it (json.Encoder does) and models.retrieve must decode an escaped
// path segment back to the same string (Go's pattern matching does, including %2F, which
// PathValue decodes without treating it as a separator).
//
// Measured on the seven names below: every one round-trips through /v1/models and retrieves
// with 200 once path-escaped. The one that would look like a failure -- a name containing "/"
// requested unescaped -- is not: a bare slash is a different path, and a client that wants
// that model escapes it, which is what the test does.
func TestModelNameSurvivesJSONAndURL(t *testing.T) {
	names := []string{
		"plain",
		`has"quote`,
		"has\nnewline",
		`back\slash`,
		"a/b",
		"a b",
		"中文模型🎉",
	}
	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			srv, _ := newTestServerConfig(t,
				gguftest.Model{Name: name, NoEOS: true, ContextLen: 64}, Config{})

			// The list endpoint must escape the name and come back as valid JSON.
			resp := get(t, srv.Handler(), "/v1/models")
			body := readBody(t, resp)
			var parsed struct {
				Data []struct {
					ID string `json:"id"`
				} `json:"data"`
			}
			if err := json.Unmarshal([]byte(body), &parsed); err != nil {
				t.Fatalf("/v1/models is not valid JSON: %v", err)
			}
			if len(parsed.Data) != 1 || parsed.Data[0].ID != name {
				t.Errorf("listed id = %q, want %q", parsed.Data[0].ID, name)
			}

			// And the retrieve endpoint must accept it once escaped.
			resp = get(t, srv.Handler(), "/v1/models/"+url.PathEscape(name))
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("retrieve %q = %d, want 200", url.PathEscape(name), resp.StatusCode)
			}
			got := decodeBody[map[string]any](t, resp)
			if got["id"] != name {
				t.Errorf("retrieved id = %v, want %q", got["id"], name)
			}
		})
	}
}

func TestModelRetrieveAndUnknownEndpoints(t *testing.T) {
	srv, _ := newTestServer(t)
	name := srv.cfg.ModelName

	// models.retrieve by the served name returns the same object the list contains.
	t.Run("retrieve a served model", func(t *testing.T) {
		resp := get(t, srv.Handler(), "/v1/models/"+name)
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want 200", resp.StatusCode)
		}
		got := decodeBody[map[string]any](t, resp)
		if got["id"] != name {
			t.Errorf("id = %v, want %q", got["id"], name)
		}
		if got["object"] != "model" {
			t.Errorf("object = %v, want model", got["object"])
		}
	})

	// An unknown id is the same model_not_found the completion endpoints use, so a client
	// that retrieves before sending gets the answer it would have got from the send.
	t.Run("retrieve an unknown model", func(t *testing.T) {
		resp := get(t, srv.Handler(), "/v1/models/does-not-exist")
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusNotFound {
			t.Fatalf("status = %d, want 404", resp.StatusCode)
		}
		body := decodeBody[errorBody](t, resp)
		if body.Error.Code != "model_not_found" {
			t.Errorf("code = %q, want model_not_found", body.Error.Code)
		}
		if body.Error.Param != "model" {
			t.Errorf("param = %q, want model", body.Error.Param)
		}
	})

	// An endpoint this server does not implement answers in the envelope, with a code that
	// says so rather than a generic not-found.
	t.Run("unimplemented endpoint", func(t *testing.T) {
		resp := post(t, srv.Handler(), "/v1/embeddings", `{}`)
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusNotFound {
			t.Fatalf("status = %d, want 404", resp.StatusCode)
		}
		if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
			t.Errorf("Content-Type = %q, want application/json (an SDK decodes this)", ct)
		}
		body := decodeBody[errorBody](t, resp)
		if body.Error.Code != "unknown_endpoint" {
			t.Errorf("code = %q, want unknown_endpoint", body.Error.Code)
		}
		if !strings.Contains(body.Error.Message, "/v1/embeddings") {
			t.Errorf("message %q does not name the path", body.Error.Message)
		}
	})

	// A known path with the wrong method is a 405 with Allow, not a 404. The fallback route is
	// method-less, so it takes the path from the mux and has to reproduce the mux's own answer:
	// OPTIONS is the CORS preflight, and README's account of why a cross-origin
	// application/json request never reaches the engine rests on the preflight being refused.
	t.Run("wrong method on a known route", func(t *testing.T) {
		cases := []struct{ method, path, allow string }{
			{http.MethodOptions, "/v1/chat/completions", http.MethodPost},
			{http.MethodGet, "/v1/completions", http.MethodPost},
			{http.MethodPost, "/v1/models", http.MethodGet},
			{http.MethodPost, "/v1/models/llama", http.MethodGet},
		}
		for _, c := range cases {
			req := httptest.NewRequest(c.method, c.path, nil)
			w := httptest.NewRecorder()
			srv.Handler().ServeHTTP(w, req)
			if w.Code != http.StatusMethodNotAllowed {
				t.Errorf("%s %s = %d, want 405", c.method, c.path, w.Code)
			}
			if got := w.Header().Get("Allow"); got != c.allow {
				t.Errorf("%s %s Allow = %q, want %q", c.method, c.path, got, c.allow)
			}
		}
	})

	// A path outside the API keeps the plain-text 404: it is not an API route, and a probe
	// or a browser hitting it should not be told JSON it never asked for.
	t.Run("path outside the api", func(t *testing.T) {
		resp := get(t, srv.Handler(), "/not-an-api-path")
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusNotFound {
			t.Fatalf("status = %d, want 404", resp.StatusCode)
		}
		if ct := resp.Header.Get("Content-Type"); strings.HasPrefix(ct, "application/json") {
			t.Errorf("Content-Type = %q; a non-API path should not claim to be JSON", ct)
		}
	})
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

// TestEmptyChatContentIsAccepted pins the asymmetry with the completions endpoint.
//
// A blank prompt is refused there because a completion with nothing to continue is not a
// request; a blank message here is accepted, because chat content is markup that may
// legitimately be empty -- an assistant turn carrying only a tool call is the usual case, and
// the neutral transcript renders it as "User: \n" rather than failing.
//
// The two are one line apart in the handlers, so without a test the difference reads as an
// oversight. Measured: {"messages":[{"role":"user","content":""}]} returns 200 with 8 prompt
// tokens against the test fixture, while {"prompt":""} returns 400.
func TestEmptyChatContentIsAccepted(t *testing.T) {
	srv, _ := newTestServer(t)

	// Chat: accepted.
	resp := post(t, srv.Handler(), "/v1/chat/completions",
		`{"messages":[{"role":"user","content":""}],"max_tokens":2}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("empty chat content = %d, want 200", resp.StatusCode)
	}
	body := decodeBody[chatResponse](t, resp)
	if len(body.Choices) != 1 {
		t.Errorf("got %d choices, want 1", len(body.Choices))
	}

	// Completions: refused, which is the half the existing table already covers -- asserted
	// here too so the two behaviours are visible together.
	resp = post(t, srv.Handler(), "/v1/completions", `{"prompt":"","max_tokens":2}`)
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("empty prompt = %d, want 400", resp.StatusCode)
	}
	_ = resp.Body.Close()
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

// TestStreamingUsageCountsEveryGeneratedToken covers the usage figures when a stop string
// ends the stream early.
//
// completion_tokens has to be the same number on both endpoints for the same request, because
// OpenAI defines it as every token the model generated rather than the text that reached the
// caller, and the blocking body counts them all. The streaming loop broke out at the stop and
// stopped counting, so one request reported 17 through /v1/completions and 2 through the same
// endpoint with stream: true -- measured against the fixture with stop "the" and a 60-token
// budget.
//
// The engine now reports the total on its terminal event, and the stream reads that event for
// the figure instead of tallying what it happened to see. Reading it costs one round, not the
// rest of the budget, which is why the two durations stay close.
func TestStreamingUsageCountsEveryGeneratedToken(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()

	// The fixture's own output contains NUL bytes, which cannot travel through a stop string,
	// so the stop text is taken from that output the way TestStreamingRespectsStopStrings does
	// it: pick a run of printable runes the model does produce.
	full := decodeBody[completionResponse](t, post(t, h, "/v1/completions",
		`{"prompt": "Once upon a time", "max_tokens": 20}`)).Choices[0].Text
	runes := []rune(full)
	start := -1
	for i := 0; i+2 <= len(runes); i++ {
		if runes[i] != 0 && runes[i+1] != 0 && i > 0 {
			start = i
			break
		}
	}
	if start < 0 {
		t.Skipf("continuation %q has no printable run of two runes", full)
	}
	stopText := string(runes[start : start+2])

	body := `"prompt": "Once upon a time", "max_tokens": 20, "stop": [` + quoteJSON(stopText) + `]`
	blocking := decodeBody[completionResponse](t, post(t, h, "/v1/completions", "{"+body+"}"))
	if blocking.Choices[0].FinishReason != "stop" {
		t.Fatalf("the blocking endpoint did not stop on %q: %+v", stopText, blocking.Choices[0])
	}

	raw := readBody(t, post(t, h, "/v1/completions",
		"{"+body+`, "stream": true, "stream_options": {"include_usage": true}}`))
	var streamed *usage
	var text strings.Builder
	for _, line := range strings.Split(raw, "\n") {
		payload, ok := strings.CutPrefix(line, "data: ")
		if !ok || payload == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Text string `json:"text"`
			} `json:"choices"`
			Usage *usage `json:"usage"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("chunk %q is not JSON: %v", payload, err)
		}
		for _, c := range chunk.Choices {
			text.WriteString(c.Text)
		}
		if chunk.Usage != nil {
			streamed = chunk.Usage
		}
	}
	if streamed == nil {
		t.Fatal("the stream carried no usage chunk")
	}
	if streamed.CompletionTokens != blocking.Usage.CompletionTokens {
		t.Errorf("completion_tokens = %d streaming and %d blocking for the same request",
			streamed.CompletionTokens, blocking.Usage.CompletionTokens)
	}
	if text.String() != blocking.Choices[0].Text {
		t.Errorf("streamed text %q, blocking text %q", text.String(), blocking.Choices[0].Text)
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

// TestNegativeMaxTokensIsRefused covers the request field that was silently replaced by the
// server's default.
//
// max_tokens is a plain int, so a zero is the only distinguishable "field not sent", and
// generate reaches the default with "<= 0". A negative budget therefore did not fail and did
// not clamp: it fell through to Config.MaxTokens, so a client that asked for -5 tokens got 256
// tokens of text with finish_reason "length" -- a reply it cannot have meant, reported as a
// normal one. Every command in cmd/orbix refuses a negative --max-tokens for the same reason.
//
// The status, the param and the code are all asserted, because the point of the error is that a
// client can find the field without parsing the sentence.
func TestNegativeMaxTokensIsRefused(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()

	cases := []struct{ name, path, body string }{
		{
			name: "completions",
			path: "/v1/completions",
			body: `{"prompt": "Once upon a time", "max_tokens": -5}`,
		},
		{
			name: "chat completions",
			path: "/v1/chat/completions",
			body: `{"messages": [{"role": "user", "content": "hi"}], "max_tokens": -5}`,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			resp := post(t, h, c.path, c.body)
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400; a negative budget is not a smaller budget",
					resp.StatusCode)
			}
			out := decodeBody[struct {
				Error struct {
					Type  string `json:"type"`
					Param string `json:"param"`
					Code  string `json:"code"`
				} `json:"error"`
			}](t, resp)
			if out.Error.Type != "invalid_request_error" {
				t.Errorf("type = %q, want invalid_request_error", out.Error.Type)
			}
			if out.Error.Param != "max_tokens" {
				t.Errorf("param = %q, want max_tokens", out.Error.Param)
			}
			if out.Error.Code != "invalid_value" {
				t.Errorf("code = %q, want invalid_value", out.Error.Code)
			}
		})
	}

	// The values that do work must keep working, so the check cannot pass by refusing every
	// budget. Zero means "use the server default" and a positive count is the normal case; both
	// have to reach generation rather than validation.
	for _, budget := range []int{0, 3} {
		body := fmt.Sprintf(`{"prompt": "Once upon a time", "max_tokens": %d}`, budget)
		resp := post(t, h, "/v1/completions", body)
		if resp.StatusCode != http.StatusOK {
			t.Errorf("max_tokens=%d: status = %d, want 200", budget, resp.StatusCode)
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
		ch, "cmpl-test", false, nil, false, 0)

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
		ch2, "cmpl-test", false, []string{"this stop never appears"}, false, 0)
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
		ch3, "cmpl-test", false, nil, false, 0)
	if got := collectStream(t, io.NopCloser(strings.NewReader(w3.Body.String()))); got != wantSingle {
		t.Errorf("one-byte stream gave %q, the same token decodes to %q", got, wantSingle)
	}
}

// TestSafePrefixLen covers the computation that decides how much streamed text is safe to send
// when a stop list is configured.
//
// The value is the length of text that cannot be the beginning of a stop string; everything
// after it is held back until the next chunk decides whether it is a match. Getting this wrong
// in either direction is a defect a client can see: too large and a stop string leaks a
// character before it truncates, too small and ordinary output arrives in more chunks than it
// needs to.
func TestSafePrefixLen(t *testing.T) {
	cases := []struct {
		text string
		stop []string
		want int
		why  string
	}{
		// The result is a byte offset, because it indexes a string: a three-byte character
		// counts as three. Written as byte counts here so the test states the unit rather
		// than leaving it to be inferred.
		{"北", []string{"北京"}, 0, "all of it may begin a match"},
		{"京", []string{"北京"}, 3, "京 alone does not begin the match, only continue it"},
		{"是京", []string{"北京"}, 6, "京 is not a prefix of 北京, so nothing is held"},
		{"北京是", []string{"北京"}, 9, "nothing here starts a match any more"},
		{"abc", []string{"xyz"}, 3, "no suffix can start a match, so all of it is safe"},
		{"ab", []string{"abc", "abd"}, 0, "ab begins both"},
		{"xab", []string{"abc"}, 1, "ab is held, x is safe"},
		{"", []string{"abc"}, 0, "empty text holds nothing back"},
		{"abc", nil, 3, "no stop list holds nothing back"},
		{"a中", []string{"a中X"}, 0, "a multi-byte suffix is held the same way"},
	}
	for _, c := range cases {
		if got := safePrefixLen(c.text, c.stop); got != c.want {
			t.Errorf("safePrefixLen(%q, %v) = %d, want %d (%s)", c.text, c.stop, got, c.want, c.why)
		}
	}
}

// TestStreamReleasesHeldTextWhenNoStopMatches covers the other half of the stop hold-back: text
// withheld because it might begin a stop string has to be sent anyway once generation ends
// without a match.
//
// Holding it back is what makes stop=["北京"] agree with the non-streaming body; releasing it
// is what keeps a stop list that never matches from silently truncating the response. Both
// need the same test to be honest about, since each looks correct alone.
func TestStreamReleasesHeldTextWhenNoStopMatches(t *testing.T) {
	srv, tok := newTestServerModel(t, gguftest.Model{
		Vocab:          []string{"<unk>", "<s>", "</s>", "ä", "¸", "Ń"},
		TokenizerModel: "gpt2",
		NoBytes:        false,
	})
	// 中 is e4 b8 ad: ids 3,4,5. A stop of "中X" holds back the whole character until
	// generation ends, where it must be flushed even though no match was found.
	ids := []int32{3, 4, 5}
	want := tok.DecodeCompletion(ids)

	ch := make(chan engine.Event, len(ids)+1)
	for _, id := range ids {
		ch <- engine.Event{Token: id, HasToken: true}
	}
	ch <- engine.Event{Done: true, FinishReason: "length"}
	close(ch)

	w := httptest.NewRecorder()
	srv.streamResponse(w, httptest.NewRequest(http.MethodPost, "/v1/completions", nil),
		ch, "cmpl-test", false, []string{"中X"}, false, 0)
	got := collectStream(t, io.NopCloser(strings.NewReader(w.Body.String())))
	if got != want {
		t.Errorf("streamed %q with an unmatched stop, want %q", got, want)
	}
}

// TestWriteErrorTypes pins the mapping from HTTP status to the error `type` field, which is what
// a client branches on.
//
// The 503 case was wrong: every status above 499 reported "server_error", so an engine saying
// "come back shortly" was indistinguishable from one that broke. OpenAI gives 503 its own type,
// and its SDKs map that type to a retryable error, which is the behaviour this server means.
func TestWriteErrorTypes(t *testing.T) {
	cases := []struct {
		status int
		want   string
		why    string
	}{
		{http.StatusBadRequest, "invalid_request_error", "a malformed request"},
		{http.StatusNotFound, "invalid_request_error", "a path or model that does not exist"},
		{http.StatusUnsupportedMediaType, "invalid_request_error", "an encoding this server will not decode"},
		{http.StatusInternalServerError, "server_error", "this server failed"},
		{http.StatusServiceUnavailable, "service_unavailable_error", "no capacity right now, retry later"},
	}
	for _, c := range cases {
		w := httptest.NewRecorder()
		writeError(w, c.status, "test")
		if w.Code != c.status {
			t.Errorf("status %d came back as %d", c.status, w.Code)
		}
		var body errorBody
		if err := json.NewDecoder(w.Body).Decode(&body); err != nil {
			t.Fatalf("status %d: decode body: %v", c.status, err)
		}
		if body.Error.Type != c.want {
			t.Errorf("status %d (%s) has type %q, want %q", c.status, c.why, body.Error.Type, c.want)
		}
	}
}

// TestContextOverflowCarriesItsCode is the regression for the one validation failure a
// client is most likely to branch on and least able to: OpenAI tells clients to match
// error.code rather than message text, and reports an oversized context as
// `context_length_exceeded` with a `param` naming the field. This server answered with a
// bare message -- no code, no param -- so a client following that guidance had nothing to
// match, and could not tell "trim the prompt and retry" from "your JSON was malformed".
//
// Both endpoints are checked because they name their input differently: a chat request
// overflows `messages`, a completion overflows `prompt`. Reporting the field the caller
// actually sent is the point of the param.
func TestContextOverflowCarriesItsCode(t *testing.T) {
	// A small window so a short prompt overflows it; the default fixture's 64 is reachable
	// but this keeps the test's intent obvious.
	srv, tok := newTestServerModel(t, gguftest.Model{ContextLen: 8, NoEOS: true})
	cfg := srv.eng.Model().Config()
	if cfg.ContextLen != 8 {
		t.Fatalf("context is %d, want the fixture's small window", cfg.ContextLen)
	}
	// Encode a prompt that certainly exceeds the window.
	prompt := strings.Repeat("word ", 64)
	if n := len(tok.Encode(prompt, true)); n < cfg.ContextLen {
		t.Fatalf("prompt is %d tokens against a %d window; raise the prompt", n, cfg.ContextLen)
	}

	cases := []struct {
		name  string
		path  string
		body  string
		param string
	}{
		{"completion overflow", "/v1/completions",
			fmt.Sprintf(`{"prompt": %q, "max_tokens": 1}`, prompt), "prompt"},
		{"chat overflow", "/v1/chat/completions",
			fmt.Sprintf(`{"messages": [{"role": "user", "content": %q}], "max_tokens": 1}`, prompt),
			"messages"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			resp := post(t, srv.Handler(), c.path, c.body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400", resp.StatusCode)
			}
			var body errorBody
			if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
				t.Fatalf("decode: %v", err)
			}
			if body.Error.Code != "context_length_exceeded" {
				t.Errorf("code = %q, want context_length_exceeded", body.Error.Code)
			}
			if body.Error.Param != c.param {
				t.Errorf("param = %q, want %q", body.Error.Param, c.param)
			}
			// The message still has to carry the numbers: a code tells a program what
			// happened, but the size is what a person uses to decide how much to trim.
			if !strings.Contains(body.Error.Message, "tokens") {
				t.Errorf("message %q does not state the sizes", body.Error.Message)
			}
		})
	}
}

// TestSamplerErrorsNameTheField covers the other two validation failures that used to be
// bare messages. A caller that sent a negative temperature and one that sent top_p = 2
// both got an uncoded 400, so a client could not tell which field to fix -- and both are
// per-field mistakes with an obvious answer.
func TestSamplerErrorsNameTheField(t *testing.T) {
	srv, _ := newTestServer(t)
	cases := []struct {
		name  string
		body  string
		param string
	}{
		{"negative temperature", `{"prompt": "Once upon a time", "max_tokens": 1, "temperature": -1}`,
			"temperature"},
		{"top_p above one", `{"prompt": "Once upon a time", "max_tokens": 1, "top_p": 2}`,
			"top_p"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			resp := post(t, srv.Handler(), "/v1/completions", c.body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400", resp.StatusCode)
			}
			var body errorBody
			if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
				t.Fatalf("decode: %v", err)
			}
			if body.Error.Param != c.param {
				t.Errorf("param = %q, want %q", body.Error.Param, c.param)
			}
			if body.Error.Code != "invalid_value" {
				t.Errorf("code = %q, want invalid_value", body.Error.Code)
			}
		})
	}
}

// TestStreamIncludeUsage covers the one streaming option this server honours.
//
// When stream_options.include_usage is set, OpenAI's contract has two halves and a client
// depends on both: one extra chunk is streamed before [DONE] whose choices is an *empty
// array* (that empty array is how a client recognises it) and whose usage holds the whole
// request's counts; and every ordinary chunk in the same stream carries "usage": null, so a
// client can read the field unconditionally.
//
// The counts must match the non-streaming body for the same request, or a caller that
// switches modes for the same work gets two different bills. Both endpoints are covered
// because they build their usage in different places.
func TestStreamIncludeUsage(t *testing.T) {
	srv, _ := newTestServer(t)

	// decodeChunks reads a stream body into the chunks before [DONE].
	decodeChunks := func(t *testing.T, body io.ReadCloser) []map[string]json.RawMessage {
		t.Helper()
		defer func() { _ = body.Close() }()
		var out []map[string]json.RawMessage
		sc := bufio.NewScanner(body)
		for sc.Scan() {
			line := sc.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			payload := strings.TrimPrefix(line, "data: ")
			if payload == "[DONE]" {
				break
			}
			var m map[string]json.RawMessage
			if err := json.Unmarshal([]byte(payload), &m); err != nil {
				t.Fatalf("chunk is not JSON: %v: %s", err, payload)
			}
			out = append(out, m)
		}
		return out
	}

	for _, tc := range []struct{ name, path, body string }{
		{"completions", "/v1/completions", `{"prompt": "Once upon a time", "max_tokens": 4}`},
		{"chat", "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "Once upon a time"}], "max_tokens": 4}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// The non-streaming body is the reference for the counts.
			var body map[string]any
			if err := json.Unmarshal([]byte(tc.body), &body); err != nil {
				t.Fatalf("test body: %v", err)
			}
			plain := post(t, srv.Handler(), tc.path, tc.body)
			wantUsage := decodeBody[struct {
				Usage map[string]int `json:"usage"`
			}](t, plain).Usage
			if wantUsage == nil {
				t.Fatal("the non-streaming body has no usage to compare against")
			}

			// The same request, streamed, asking for usage.
			body["stream"] = true
			body["stream_options"] = map[string]any{"include_usage": true}
			streamedBody, err := json.Marshal(body)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			resp := post(t, srv.Handler(), tc.path, string(streamedBody))
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("status = %d", resp.StatusCode)
			}
			chunks := decodeChunks(t, resp.Body)

			// Split into ordinary chunks and the usage chunk by how OpenAI marks it.
			var ordinary, withEmptyChoices []map[string]json.RawMessage
			for _, c := range chunks {
				var choices []json.RawMessage
				if err := json.Unmarshal(c["choices"], &choices); err != nil {
					t.Fatalf("choices: %v", err)
				}
				if len(choices) == 0 {
					withEmptyChoices = append(withEmptyChoices, c)
					continue
				}
				ordinary = append(ordinary, c)
			}
			if len(withEmptyChoices) != 1 {
				t.Fatalf("%d chunks with empty choices, want exactly 1 (the usage chunk)",
					len(withEmptyChoices))
			}
			// It must be the last chunk before [DONE], which is where a client looks for it.
			var lastChoices []json.RawMessage
			if err := json.Unmarshal(chunks[len(chunks)-1]["choices"], &lastChoices); err != nil {
				t.Fatalf("last chunk choices: %v", err)
			}
			if len(lastChoices) != 0 {
				t.Error("the usage chunk is not the last chunk before [DONE]")
			}

			// Every ordinary chunk carries usage:null, including the finish chunk. A client
			// reading the field unconditionally must not hit a missing key.
			for i, c := range ordinary {
				raw, ok := c["usage"]
				if !ok {
					t.Errorf("ordinary chunk %d has no usage key; the contract says it is null", i)
					continue
				}
				if string(raw) != "null" {
					t.Errorf("ordinary chunk %d usage = %s, want null", i, raw)
				}
			}

			var gotUsage map[string]int
			if err := json.Unmarshal(withEmptyChoices[0]["usage"], &gotUsage); err != nil {
				t.Fatalf("usage chunk: %v", err)
			}
			for _, k := range []string{"prompt_tokens", "completion_tokens", "total_tokens"} {
				if gotUsage[k] != wantUsage[k] {
					t.Errorf("streamed %s = %d, non-streaming = %d", k, gotUsage[k], wantUsage[k])
				}
			}
		})
	}

	// Without the option, no chunk carries usage and none has empty choices: a client that
	// did not ask for it must not receive a chunk whose shape it does not expect.
	t.Run("absent without the option", func(t *testing.T) {
		resp := post(t, srv.Handler(), "/v1/completions",
			`{"prompt": "hi", "max_tokens": 2, "stream": true}`)
		chunks := decodeChunks(t, resp.Body)
		for i, c := range chunks {
			if _, ok := c["usage"]; ok {
				t.Errorf("chunk %d has a usage key without stream_options", i)
			}
			var choices []json.RawMessage
			if err := json.Unmarshal(c["choices"], &choices); err != nil {
				t.Fatalf("choices: %v", err)
			}
			if len(choices) == 0 {
				t.Errorf("chunk %d has empty choices without stream_options", i)
			}
		}
	})
}

// TestResponseShapeFieldsAreRefused covers the one kind of unsupported request field this
// server cannot ignore.
//
// It ignores unknown fields on purpose -- an SDK sends logit_bias whether or not the engine
// can use it, and refusing it would make the server unusable from that SDK. A field that
// describes the *response* is different: a client that sent n=3 reads choices[2], and one
// that sent logprobs reads choice.logprobs. Answering with neither is not a smaller response,
// it is a different one, and the client fails with an IndexError or a KeyError far from the
// request that caused it.
//
// So these are refused with a 400 that names the field, which is the same fact said where the
// caller can act on it. The sampling knobs stay ignored, and that half is asserted too --
// otherwise "refuse unsupported fields" could drift into refusing everything, which is the
// behaviour the ignore-everything rule was chosen to avoid.
func TestResponseShapeFieldsAreRefused(t *testing.T) {
	srv, _ := newTestServer(t)

	// Each case names the field the error must carry. The last sends three at once: a caller
	// fixing these should not need one round trip per field.
	tooLoud := []struct {
		name  string
		path  string
		body  string
		field string
		all   []string
	}{
		{"n on completions", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "n": 3}`, "n", nil},
		{"logprobs on completions", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "logprobs": 1}`, "logprobs", nil},
		{"echo on completions", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "echo": true}`, "echo", nil},
		{"suffix on completions", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "suffix": "END"}`, "suffix", nil},
		{"best_of on completions", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "best_of": 2}`, "best_of", nil},
		{"n on chat", "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "n": 2}`, "n", nil},
		{"top_logprobs on chat", "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "top_logprobs": 2}`,
			"top_logprobs", nil},
		{"three at once", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "n": 2, "logprobs": 1, "echo": true}`,
			"n", []string{"n", "logprobs", "echo"}},
	}
	for _, c := range tooLoud {
		t.Run(c.name, func(t *testing.T) {
			resp := post(t, srv.Handler(), c.path, c.body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400", resp.StatusCode)
			}
			body := decodeBody[errorBody](t, resp)
			if body.Error.Param != c.field {
				t.Errorf("param = %q, want %q", body.Error.Param, c.field)
			}
			if body.Error.Code != "unsupported_parameter" {
				t.Errorf("code = %q, want unsupported_parameter", body.Error.Code)
			}
			// Every field is named in the message, so one round trip fixes the request.
			for _, want := range append([]string{c.field}, c.all...) {
				if !strings.Contains(body.Error.Message, want) {
					t.Errorf("message %q does not name %q", body.Error.Message, want)
				}
			}
		})
	}

	// The sampling knobs must stay ignored. If refusing grows to cover these, an SDK that
	// always sends them stops working, which is the outcome the ignore rule exists to prevent.
	quiet := []struct{ name, path, body string }{
		{"logit_bias", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "logit_bias": {"1": -100}}`},
		{"presence_penalty", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "presence_penalty": 0.5}`},
		{"frequency_penalty", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "frequency_penalty": 0.5}`},
		{"response_format", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "response_format": {"type": "json_object"}}`},
		{"logit_bias on chat", "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "logit_bias": {"1": -100}}`},
	}
	for _, c := range quiet {
		t.Run(c.name+" is ignored", func(t *testing.T) {
			resp := post(t, srv.Handler(), c.path, c.body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != http.StatusOK {
				body := decodeBody[errorBody](t, resp)
				t.Fatalf("status = %d, want 200: %s", resp.StatusCode, body.Error.Message)
			}
		})
	}
}

// TestValidationErrorsNameTheField covers error.param on the two required-field checks.
//
// OpenAI's error object documents `param` as the field at fault for a request-validation
// failure, and its clients surface it as e.param. Without it a caller has to parse the field
// name out of the message string, which is what this server was making them do.
func TestValidationErrorsNameTheField(t *testing.T) {
	srv, _ := newTestServer(t)

	cases := []struct {
		name  string
		path  string
		body  string
		param string
	}{
		{"completions without prompt", "/v1/completions",
			`{"max_tokens":1}`, "prompt"},
		{"chat without messages", "/v1/chat/completions",
			`{"max_tokens":1}`, "messages"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			resp := post(t, srv.Handler(), c.path, c.body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400", resp.StatusCode)
			}
			var body errorBody
			if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
				t.Fatalf("decode: %v", err)
			}
			if body.Error.Param != c.param {
				t.Errorf("param = %q, want %q", body.Error.Param, c.param)
			}
			if body.Error.Code != "missing_required_parameter" {
				t.Errorf("code = %q, want missing_required_parameter", body.Error.Code)
			}
		})
	}
}

// TestSingleTokenStopReachesTheEngine covers the half of stop-string handling the other two
// tests deliberately avoid.
//
// The engine can only early-out on a stop string that is exactly one token, so stopTokenIDs
// hands it those and drops the rest; the rest are found later in the decoded text by
// cutAtStop. Both existing stop tests cut inside the continuation, so the append in
// stopTokenIDs -- the one branch that produces a usable id -- never ran.
//
// Asserted on stopTokenIDs directly rather than through a request: a stop that fails to reach
// the engine still stops, just later, after the token was generated and cut out of the text,
// so the request path cannot tell the two apart. The fixture's random weights make the
// continuation unmatchable text, which is why the end-to-end version of this was skipped.
func TestSingleTokenStopReachesTheEngine(t *testing.T) {
	srv, tok := newTestServer(t)

	// A candidate the fixture's tokenizer maps to exactly one id. Checked rather than
	// assumed, so a tokenizer change fails loudly instead of silently making this pass for
	// the wrong reason.
	var stopText string
	var stopID int32
	for _, cand := range []string{"a", "e", "t", "o", "n", " ", ","} {
		if enc := tok.Encode(cand, false); len(enc) == 1 {
			stopText, stopID = cand, enc[0]
			break
		}
	}
	if stopText == "" {
		t.Skip("no single-token candidate in the fixture's vocabulary")
	}
	if got := srv.stopTokenIDs([]string{stopText}); len(got) != 1 || got[0] != stopID {
		t.Errorf("stopTokenIDs(%q) = %v, want [%d]", stopText, got, stopID)
	}

	// And the multi-token and empty cases still drop out, which is what keeps the engine from
	// being handed an id it cannot match.
	if got := srv.stopTokenIDs([]string{""}); got != nil {
		t.Errorf("stopTokenIDs([\"\"]) = %v, want nil", got)
	}
	if enc := tok.Encode(" the quick", false); len(enc) > 1 {
		if got := srv.stopTokenIDs([]string{" the quick"}); got != nil {
			t.Errorf("stopTokenIDs of a multi-token string = %v, want nil", got)
		}
	}
}

// TestWriteRequestErrorFallback covers the non-paramError path of writeRequestError.
//
// Every current call site passes a *paramError -- sampler's two validation failures and the
// two direct newParamError calls -- so the fallback has never run. It exists so a future call
// site that hands over an ordinary error still produces a well-formed OpenAI error body
// instead of losing the error, and that contract is worth stating in a test rather than
// leaving to the comment.
//
// Not a contrived state: writeRequestError takes an error, and an ordinary one is a value the
// function is documented to accept.
func TestWriteRequestErrorFallback(t *testing.T) {
	t.Run("paramError carries param and code", func(t *testing.T) {
		rec := httptest.NewRecorder()
		writeRequestError(rec, http.StatusBadRequest,
			newParamError("temperature", "invalid_value", "temperature must not be negative, got %v", -1))
		body := decodeBody[errorBody](t, rec.Result())
		if body.Error.Message != "temperature must not be negative, got -1" {
			t.Errorf("message = %q", body.Error.Message)
		}
		if body.Error.Param != "temperature" {
			t.Errorf("param = %q, want temperature", body.Error.Param)
		}
		if body.Error.Code != "invalid_value" {
			t.Errorf("code = %q, want invalid_value", body.Error.Code)
		}
	})

	t.Run("plain error degrades to message and type", func(t *testing.T) {
		rec := httptest.NewRecorder()
		writeRequestError(rec, http.StatusBadRequest, errors.New("something else went wrong"))
		body := decodeBody[errorBody](t, rec.Result())
		if body.Error.Message != "something else went wrong" {
			t.Errorf("message = %q, want the error's text", body.Error.Message)
		}
		if body.Error.Type != "invalid_request_error" {
			t.Errorf("type = %q, want invalid_request_error", body.Error.Type)
		}
		// Omitted rather than empty-but-present: omitempty is what tells a client the field
		// is not about a request parameter at all.
		if body.Error.Param != "" || body.Error.Code != "" {
			t.Errorf("param=%q code=%q, want both empty for a plain error",
				body.Error.Param, body.Error.Code)
		}
	})
}

// TestSeedZeroMeansRandom pins the branch TestSeedMakesSamplingReproducible cannot reach.
//
// That test always sends a non-zero seed, so seed == 0 -- the branch that substitutes the
// clock -- never ran. It is reachable with ordinary input: the field is a plain int64, so an
// omitted seed and an explicit 0 are the same value, and the OpenAI clients send 0 to mean
// "pick your own" often enough that it is the documented meaning of an absent seed.
//
// Two consequences worth asserting: the same request twice gives different text, and the
// branch does not leak into the non-sampling path (temperature 0 is greedy and ignores it).
func TestSeedZeroMeansRandom(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()
	post0 := func() string {
		t.Helper()
		resp := post(t, h, "/v1/completions", fmt.Sprintf(
			`{"model": %q, "prompt": "Once upon a time", "max_tokens": 16, "temperature": 1.5, "seed": 0}`,
			srv.cfg.ModelName))
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d", resp.StatusCode)
		}
		out := decodeBody[completionResponse](t, resp)
		if len(out.Choices) == 0 {
			t.Fatal("no choices")
		}
		return out.Choices[0].Text
	}
	// A population statement, like the reproducibility test's: one collision on one pair is
	// possible, five identical results from a clock-seeded sampler is not.
	seen := map[string]bool{}
	for range 5 {
		seen[post0()] = true
	}
	if len(seen) == 1 {
		t.Error("five seed=0 requests produced identical text: the clock branch did not run")
	}

	// Temperature 0 is greedy, so it must not depend on the seed at all.
	greedy := func(seed int) string {
		t.Helper()
		resp := post(t, h, "/v1/completions", fmt.Sprintf(
			`{"model": %q, "prompt": "Once upon a time", "max_tokens": 8, "temperature": 0, "seed": %d}`,
			srv.cfg.ModelName, seed))
		out := decodeBody[completionResponse](t, resp)
		return out.Choices[0].Text
	}
	if a, b := greedy(0), greedy(12345); a != b {
		t.Errorf("greedy output differs by seed: %q vs %q", a, b)
	}
}
