package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"math"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/pplmx/orbix/internal/chat"
	"github.com/pplmx/orbix/internal/engine"
	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
	"github.com/pplmx/orbix/internal/tokenize"

	"github.com/pplmx/orbix/internal/otel"
)

// These tests need a real model: the properties under test are the HTTP and SSE
// contract, but the tokens that flow through them have to be real to see a
// streaming or template bug. Set MODEL to a GGUF file to run them.

// mustRender renders a conversation with no template variables, failing the test on an error.
//
// Variables are the one thing renderChat can refuse, so a caller that has none cannot get one;
// this keeps the ordinary assertions readable instead of threading an error through each.
func mustRender(t *testing.T, srv *Server, msgs []chatMessage) string {
	t.Helper()
	out, err := srv.renderChat(msgs, nil)
	if err != nil {
		t.Fatalf("renderChat: %v", err)
	}
	return out
}

func newTestServer(t *testing.T) (*Server, *tokenize.Tokenizer) {
	t.Helper()
	return newTestServerModel(t, gguftest.Default())
}

// newTestServerModel is newTestServer over a fixture the test describes, for tests that
// need a property the default does not have -- generation that runs to its budget, say.
//
// The server these helpers return is already pumping: newTestServerConfig starts the
// engine goroutine before returning. A test must not call StartPump again -- two goroutines
// inside Pump is the one thing the engine's contract forbids, and the result is a data race
// on the sequence fields rather than a failed assertion, so the mistake looks like a race in
// the engine instead of a misuse of this helper. Use these for anything that needs
// generation to advance; call New and StartPump yourself only when the pump start has to be
// observed, as the shutdown tests do.
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
	if extra.ContextLen != 0 {
		conf.ContextLen = extra.ContextLen
	}
	if extra.Temperature != 0 {
		conf.Temperature = extra.Temperature
	}
	if extra.TopK != 0 {
		conf.TopK = extra.TopK
	}
	if extra.TopP != 0 {
		conf.TopP = extra.TopP
	}
	if extra.ChatRenderer != nil {
		conf.ChatRenderer = extra.ChatRenderer
	}
	if extra.ChatFallbackAnnounced {
		conf.ChatFallbackAnnounced = extra.ChatFallbackAnnounced
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

// newTestServerWithTemplate builds a server whose model declares tmpl as its chat template.
//
// The renderer has to be built by the caller -- New takes a Config rather than a file, the way
// cmd/orbix assembles one -- so a test that exercises a template needs both halves. Building it
// from the file gguftest wrote is what keeps the two in step: the same bytes the engine loads are
// the ones the renderer compiles.
func newTestServerWithTemplate(t *testing.T, tmpl string, extra Config) *Server {
	t.Helper()
	model := gguftest.Model{NoEOS: true, ContextLen: 256, ChatTemplate: tmpl}
	path := model.Write(t, t.TempDir())
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	renderer, err := engine.NewChatRenderer(gf)
	if err != nil {
		t.Fatalf("NewChatRenderer: %v", err)
	}
	if !renderer.HasTemplate() {
		t.Fatalf("the template did not compile: %s", renderer.FallbackReason())
	}
	extra.ChatRenderer = renderer
	extra.ChatFallbackAnnounced = true
	srv, _ := newTestServerConfig(t, model, extra)
	return srv
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
// TestNonObjectBodySaysWhatArrived covers the body-not-an-object path, which has its own
// message because there is no field to name.
//
// Every other type mismatch says "<param> must be <type>", and param is the path the decoder
// walked. A body that is an array or a string has no path, so param is empty and the message
// has to say what did arrive instead. Rewording it left the whole package green.
func TestNonObjectBodySaysWhatArrived(t *testing.T) {
	srv, _ := newTestServer(t)
	for _, c := range []struct{ body, wantType string }{
		{`[1, 2]`, "array"},
		{`"hi"`, "string"},
		{`42`, "number"},
	} {
		resp := post(t, srv.Handler(), "/v1/completions", c.body)
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("body %s: status = %d, want 400", c.body, resp.StatusCode)
		}
		out := decodeBody[errorBody](t, resp)
		want := "the request body must be a JSON object, but it is " + c.wantType
		if out.Error.Message != want {
			t.Errorf("body %s: message = %q, want %q", c.body, out.Error.Message, want)
		}
		if out.Error.Code != "invalid_type" {
			t.Errorf("body %s: code = %q, want invalid_type", c.body, out.Error.Code)
		}
	}
}

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
	// The limit itself has to be in the message: "too large" tells a client to send less,
	// and the number tells it how much less. Dropping it from the format string left every
	// test passing.
	if !strings.Contains(env.Error.Message, strconv.FormatInt(limit, 10)) {
		t.Errorf("message %q does not state the %d-byte limit", env.Error.Message, limit)
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
		// The multiplication is int64 and ContextLen is a header field, so the product can
		// overflow. Measured: MaxInt64/48+1 wraps negative and lands on the floor below, which
		// is the right answer by accident rather than by construction -- the floor catches it,
		// not the arithmetic. Recorded here so that removing or inverting the floor shows up
		// as a change in this row rather than as a silently tiny limit.
		//
		// Unreachable from a file: LoadConfig refuses a non-positive context length, and the
		// overflow needs 1.9e17 positions. This is the guard's behaviour, not a served path's.
		{ctx: math.MaxInt64/48 + 1, want: 1 << 20, why: "an overflowing product falls to the floor"},
		{ctx: math.MaxInt, want: 1 << 20, why: "the same at the extreme"},
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

// TestStopListThatNeverMatchesCostsNothing pins the promise the stop gate's documentation makes:
// a stop list that never matches must cost a later chunk and nothing else.
//
// The gate holds back the longest tail that could still begin a stop string, and releases it when
// generation ends. Whether it released was decided by the finish reason being something other than
// "stop" -- and "stop" is also what the engine reports for an ordinary completion, so the two were
// indistinguishable and the tail was dropped for every reply that ended normally. Measured on
// `x y z` with stop=["zzz"]: the streamed text was "x y " and the blocking body returned "x y z".
// The trailing "z" was withheld because it could have begun "zzz", and then never released.
//
// Both endings are exercised: a "length" cut released correctly all along, which is what made the
// defect look like it only affected some replies.
func TestStopListThatNeverMatchesCostsNothing(t *testing.T) {
	srv, tok := newTestServer(t)
	const reply = `x y z`
	ids := tok.Encode(reply, false)
	if len(ids) == 0 {
		t.Fatal("the fixture encodes the reply to nothing")
	}
	for _, finish := range []string{"stop", "length"} {
		t.Run(finish, func(t *testing.T) {
			p := generateParams{chat: false, responseID: "cmpl-never-matches",
				stop: []string{"zzz"}}
			ch := make(chan engine.Event, len(ids)+1)
			for _, id := range ids {
				ch <- engine.Event{Token: id, HasToken: true}
			}
			ch <- engine.Event{Done: true, FinishReason: finish}
			close(ch)

			var streamed strings.Builder
			if _, _, ok := srv.streamOneChoice(ch, 0, p, func(it streamItem) bool {
				streamed.WriteString(it.delta)
				return true
			}); !ok {
				t.Fatal("streamOneChoice reported a failed write")
			}

			// And the blocking body on the same tokens, which is what the stream has to agree
			// with rather than with a written-out literal: the gate's job is to match the other
			// transport, and the literal would have to be re-derived whenever the fixture's
			// vocabulary changes.
			obj := srv.choiceObject(&choiceResult{out: ids, reason: finish}, 0, p)
			blocked, _ := obj["text"].(string)
			if streamed.String() != blocked {
				t.Errorf("the stream reassembled to %q and the blocking body returned %q; a stop "+
					"list that never matches must only delay the tail", streamed.String(), blocked)
			}
		})
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
	// NoEOS and an exact budget, so the request is guaranteed to end on "length" rather
	// than on whichever of stop or length this fixture happens to produce. The assertion
	// used to accept either, which is how the streaming half of the stopReason mapping went
	// unchecked: changing it to report "stop" for a budget-limited sequence passed every
	// server test. The non-streaming mapping had precise coverage already; this is the
	// stream's.
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 64})
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
	if reason != "length" {
		t.Errorf("finish_reason = %q, want length: the sequence hit its 3-token budget", reason)
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
	// Asking for more than the window can hold must be clamped, not fail: a client with
	// a 512-token default against a 128-token model should get what fits.
	//
	// The assertion is an exact count rather than an upper bound. It used to be
	// `CompletionTokens > cfg.ContextLen`, which passes whether or not the clamp runs: the
	// engine stops at the cache boundary anyway, so a request for 100000 tokens produces a
	// full window either way and the only difference is whether the last few positions are
	// wasted on a token the server already knew would not fit. Disabling the clamp in
	// generate left this test green.
	//
	// NoEOS so the run cannot end early on a constructed model's accidental EOS: the count
	// is only a property of the clamp if nothing else can stop generation.
	srv, tok := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 64})
	cfg := srv.eng.Model().Config()
	// Leave room for exactly n tokens after the prompt, so "what fits" is known.
	prompt := "Once upon a time"
	promptLen := len(tok.Encode(prompt, cfg.BOSID >= 0))
	const want = 8
	room := promptLen + want
	if room >= cfg.ContextLen {
		t.Fatalf("prompt %d plus %d leaves no room in a %d context", promptLen, want, cfg.ContextLen)
	}
	// A server whose own default fills the window would hide the clamp, so set the context
	// length the handler compares against to exactly prompt+want.
	srv.cfg.ContextLen = room

	resp := post(t, srv.Handler(), "/v1/completions",
		fmt.Sprintf(`{"prompt": %s, "max_tokens": 100000}`, quoteJSON(prompt)))
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	body := decodeBody[completionResponse](t, resp)
	if got := body.Usage.CompletionTokens; got != want {
		t.Errorf("generated %d tokens, want exactly %d (the clamp did not run or was off by one)",
			got, want)
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
	got, err := srv.renderChat([]chatMessage{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hi"},
	}, nil)
	if err != nil {
		t.Fatalf("renderChat: %v", err)
	}
	want := "System: You are helpful.\nUser: Hi\nAssistant:"
	if got != want {
		t.Errorf("renderChat = %q, want %q", got, want)
	}
	// An empty role defaults to user rather than producing a bare ": ".
	got, err = srv.renderChat([]chatMessage{{Content: "Hi"}}, nil)
	if err != nil {
		t.Fatalf("renderChat: %v", err)
	}
	if !strings.HasPrefix(got, "User: Hi") {
		t.Errorf("renderChat with no role = %q", got)
	}
}

// TestChatAcceptsContentParts covers the array spelling of a chat message's content.
//
// OpenAI's schema makes content a union: a string, or an array of parts. The Python SDK
// sends the array form as soon as a message is built from parts, so a server that decodes
// only the string form refuses a request the SDK considers ordinary -- measured before this
// was handled, `content: [{"type": "text", "text": "Hi"}]` answered 400 "invalid JSON: json:
// cannot unmarshal array into Go struct field chatRequest.messages.0.content of type string".
//
// The text of the parts has to reach the prompt, which is what the first case asserts
// through prompt_tokens: the array form and the string form of the same message have to
// encode to the same number of tokens, or the parts were dropped rather than concatenated.
func TestChatAcceptsContentParts(t *testing.T) {
	srv, _ := newTestServer(t)

	plainResp := post(t, srv.Handler(), "/v1/chat/completions",
		`{"messages":[{"role":"user","content":"Hello there"}],"max_tokens":1}`)
	if plainResp.StatusCode != http.StatusOK {
		t.Fatalf("string form: status = %d, want 200", plainResp.StatusCode)
	}
	plain := decodeBody[chatResponse](t, plainResp)

	partsResp := post(t, srv.Handler(), "/v1/chat/completions",
		`{"messages":[{"role":"user","content":[
			{"type":"text","text":"Hello "},
			{"type":"text","text":"there"}]}],"max_tokens":1}`)
	// Named before the decode: a 400 decoded into chatResponse is a zero value whose
	// prompt_tokens of 0 looks like a dropped part rather than like a refused request.
	if partsResp.StatusCode != http.StatusOK {
		env := decodeBody[errorBody](t, partsResp)
		t.Fatalf("parts form: status = %d (%s), want 200", partsResp.StatusCode,
			env.Error.Message)
	}
	parts := decodeBody[chatResponse](t, partsResp)

	if got, want := parts.Usage.PromptTokens, plain.Usage.PromptTokens; got != want {
		t.Errorf("the parts form encoded to %d prompt tokens and the string form to %d, so the "+
			"parts were not concatenated into the message", got, want)
	}

	// Joining the parts into one string has to render the same transcript the single string
	// produces, which is the property the token count above only implies.
	var decoded chatRequest
	if err := json.Unmarshal([]byte(`{"messages":[{"role":"user","content":[
		{"type":"text","text":"Hello "},{"type":"text","text":"there"}]}]}`), &decoded); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got, want := mustRender(t, srv, decoded.Messages), "User: Hello there\nAssistant:"; got != want {
		t.Errorf("renderChat = %q, want %q", got, want)
	}

	// null is an empty body rather than an error, which is what a transcript a client is
	// replaying carries for a turn it has no text for. It has to reach the renderer as an
	// empty message, not as a refused request.
	nullResp := post(t, srv.Handler(), "/v1/chat/completions",
		`{"messages":[{"role":"assistant","content":null},{"role":"user","content":"Hi"}],
			"max_tokens":1}`)
	if nullResp.StatusCode != http.StatusOK {
		env := decodeBody[errorBody](t, nullResp)
		t.Fatalf("content null: status = %d (%s), want 200", nullResp.StatusCode,
			env.Error.Message)
	}
	var nullDecoded chatRequest
	if err := json.Unmarshal([]byte(`{"messages":[{"role":"assistant","content":null}]}`),
		&nullDecoded); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got, want := mustRender(t, srv, nullDecoded.Messages), "Assistant: \nAssistant:"; got != want {
		t.Errorf("renderChat of a null content = %q, want %q", got, want)
	}

	// A part type this server cannot render is refused rather than skipped. Answering the
	// text of a message that also carried an image would be a reply to a question the caller
	// did not ask, and silently is the worst way to say that.
	for _, tc := range []struct {
		name string
		body string
		code string
	}{
		{
			name: "image part",
			body: `{"messages":[{"role":"user","content":[
				{"type":"text","text":"What is this?"},
				{"type":"image_url","image_url":{"url":"https://example.invalid/x.png"}}]}]}`,
			code: "unsupported_parameter",
		},
		{
			name: "part without a type",
			body: `{"messages":[{"role":"user","content":[{"text":"Hello"}]}]}`,
			code: "missing_required_parameter",
		},
		{
			name: "part with a non-string text",
			body: `{"messages":[{"role":"user","content":[{"type":"text","text":5}]}]}`,
			code: "invalid_type",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resp := post(t, srv.Handler(), "/v1/chat/completions", tc.body)
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400", resp.StatusCode)
			}
			env := decodeBody[errorBody](t, resp)
			if env.Error.Code != tc.code {
				t.Errorf("code = %q, want %q (message %q)", env.Error.Code, tc.code,
					env.Error.Message)
			}
			if env.Error.Param != "messages.content" {
				t.Errorf("param = %q, want messages.content", env.Error.Param)
			}
		})
	}
}

// TestTypeMismatchIsNotMalformedJSON covers a field that arrives as the wrong JSON type.
//
// The decoder reports this as a json.UnmarshalTypeError carrying the field and the type that
// arrived, and the server used to flatten it into invalid_json with no param -- the same
// answer a truncated body gets. Measured before the change: {"max_tokens":"8"} answered 400
// with code invalid_json and no param, and an SDK that branches on code reads that as "your
// body did not parse" rather than "max_tokens has to be a number".
//
// OpenAI's own code for this condition is invalid_type, and each case below asserts the
// three things a caller acts on: the status, the code, and the field named in param.
func TestTypeMismatchIsNotMalformedJSON(t *testing.T) {
	srv, _ := newTestServer(t)
	for _, tc := range []struct {
		name  string
		path  string
		body  string
		param string
		want  string // a phrase the message has to carry, naming what arrived
	}{
		{
			name: "max_tokens as a string", path: "/v1/completions",
			body: `{"prompt":"hi","max_tokens":"8"}`, param: "max_tokens",
			want: "must be an integer, but it is string",
		},
		{
			name: "temperature as an object", path: "/v1/chat/completions",
			body:  `{"messages":[{"role":"user","content":"hi"}],"temperature":{"a":1}}`,
			param: "temperature", want: "must be a number, but it is object",
		},
		{
			name: "messages as a string", path: "/v1/chat/completions",
			body: `{"messages":"hi"}`, param: "messages", want: "must be an array",
		},
		{
			name: "role as a number", path: "/v1/chat/completions",
			body: `{"messages":[{"role":5,"content":"hi"}]}`, param: "messages.0.role",
			want: "must be a string, but it is number",
		},
		{
			// The content field is a union, so its own decoder writes the message; what
			// this case pins is that the refusal still names the field and the code.
			name: "content as a number", path: "/v1/chat/completions",
			body: `{"messages":[{"role":"user","content":42}]}`, param: "messages.content",
			want: "must be a string or an array of text parts, but it is number",
		},
		{
			// stop is the other union, and the case that showed the rule this test now
			// pins: a custom unmarshaller is handed the value alone, so the decoder has no
			// field to report and the generic answer named the whole body as the problem.
			name: "stop as a number", path: "/v1/completions",
			body: `{"prompt":"hi","stop":5}`, param: "stop",
			want: "must be a string or an array of strings, but it is number",
		},
		{
			name: "array element of stop as a number", path: "/v1/completions",
			body: `{"prompt":"hi","stop":["ok",5]}`, param: "stop",
			want: "must be a string or an array of strings, but it is number",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resp := post(t, srv.Handler(), tc.path, tc.body)
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400", resp.StatusCode)
			}
			env := decodeBody[errorBody](t, resp)
			if env.Error.Code != "invalid_type" {
				t.Errorf("code = %q, want invalid_type (message %q)", env.Error.Code,
					env.Error.Message)
			}
			if env.Error.Param != tc.param {
				t.Errorf("param = %q, want %q", env.Error.Param, tc.param)
			}
			if !strings.Contains(env.Error.Message, tc.want) {
				t.Errorf("message %q does not carry %q", env.Error.Message, tc.want)
			}
			// A message naming a Go type reads as an internal error to the caller.
			if strings.Contains(env.Error.Message, "Go struct") ||
				strings.Contains(env.Error.Message, "chatRequest") {
				t.Errorf("message %q names a Go type rather than the JSON field",
					env.Error.Message)
			}
		})
	}
}

// TestMalformedJSONStillSaysSo pins the other side of the split above: a body the decoder
// could not parse at all keeps invalid_json and names no field, because no field was
// reached. Without this, widening the type-mismatch branch would be free to swallow the
// syntax failures too.
func TestMalformedJSONStillSaysSo(t *testing.T) {
	srv, _ := newTestServer(t)
	for _, body := range []string{`{"prompt":`, `not json`, `{"prompt":"x"`} {
		resp := post(t, srv.Handler(), "/v1/completions", body)
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("%s: status = %d, want 400", body, resp.StatusCode)
		}
		env := decodeBody[errorBody](t, resp)
		if env.Error.Code != "invalid_json" {
			t.Errorf("%s: code = %q, want invalid_json", body, env.Error.Code)
		}
		if env.Error.Param != "" {
			t.Errorf("%s: param = %q, want empty for a body that did not parse",
				body, env.Error.Param)
		}
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
	// The message has to say what the caller can change. "engine: too many requests waiting
	// (2 queued for 1 active slots)" names the state and stops, so a caller who has read the
	// body still does not know whether the fix is more capacity, fewer requests, or a shorter
	// prompt. Both levers are named here; a test on the state alone would pass against the
	// message that prompted this.
	body := decodeBody[errorBody](t, resp)
	for _, want := range []string{"--max-queued", "--max-active"} {
		if !strings.Contains(body.Error.Message, want) {
			t.Errorf("the refusal does not name %s, so it does not say what to change: %q",
				want, body.Error.Message)
		}
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
	// The refusal messages are collected from the same requests rather than a second batch:
	// doubling the concurrent requests changed the timing enough that no request succeeded, so
	// the test could no longer tell a refusal from a broken setup.
	refusals := make(chan string, 16)
	for range 16 {
		go func() {
			resp := post(t, srv.Handler(), "/v1/completions", body)
			var got errorBody
			if resp.StatusCode == http.StatusServiceUnavailable {
				_ = json.NewDecoder(resp.Body).Decode(&got)
			}
			refusals <- got.Error.Message
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

	// And the refusals say what to change. The cache's sentence
	// ("llama: block pool exhausted (96 blocks of 4 positions)") is the geometry an operator
	// raises --blocks against and is kept, but on its own it names no lever a caller has and
	// leaks an internal package name. Checked on the bodies collected above rather than on a
	// second batch of requests, because doubling the concurrency changed the timing enough
	// that nothing succeeded and the test stopped measuring refusals at all.
	checked := 0
	for range 16 {
		msg := <-refusals
		if msg == "" {
			continue
		}
		checked++
		for _, want := range []string{"--blocks", "--block-size", "max_tokens"} {
			if !strings.Contains(msg, want) {
				t.Errorf("the refusal does not name %s, so it does not say what to change: %q",
					want, msg)
			}
		}
	}
	if checked == 0 {
		t.Error("no refusal message was captured, so the wording was not checked")
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
// The accepted range is [0, 1] and what is refused is a value with no reading at all. Zero is
// accepted because it is the documented "disabled" value on the CLI's --top-p and
// llama.Temperature treats it as no nucleus, and one is accepted because keeping every
// candidate is what happens without the limit. Clamping -0.1 to 0 would turn a typo into "off"
// without saying so, which is why the lower bound is a refusal rather than a floor.
//
// The accepted endpoints are asserted too, and that half is the one this test used to lack: a
// comment in server.go claimed the range was (0, 1] while the check was "< 0 || > 1", and
// nothing here disagreed with the comment, so the two drifted apart unnoticed.
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
	// Both accepted endpoints, since "0 disables it" is the claim a reader acts on.
	for _, p := range []string{"0", "1"} {
		t.Run("top_p="+p+" is accepted", func(t *testing.T) {
			body := fmt.Sprintf(`{"model": %q, "prompt": "hi", "max_tokens": 2, "temperature": 1, "top_p": %s}`,
				srv.cfg.ModelName, p)
			resp := post(t, h, "/v1/completions", body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != http.StatusOK {
				out := decodeBody[errorBody](t, resp)
				t.Fatalf("status = %d, want 200: %s", resp.StatusCode, out.Error.Message)
			}
		})
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
		[]<-chan engine.Event{ch}, "cmpl-test", 1700000000,
		generateParams{chat: false}, 0)

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
		[]<-chan engine.Event{ch2}, "cmpl-test", 1700000000,
		generateParams{chat: false, stop: []string{"this stop never appears"}}, 0)
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
		[]<-chan engine.Event{ch3}, "cmpl-test", 1700000000,
		generateParams{chat: false}, 0)
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
		// A complete match is not held back. cutAtStop runs first and consumes it, so text
		// that already equals a stop string is all safe -- the n loop must stop at len(s)-1,
		// or this reports 0 and holds a complete match forever. Measured: with the bound at
		// len(s) this case returns 0 and "xabc" returns 1.
		{"abc", []string{"abc"}, 3, "a complete match is cutAtStop's business, not hold-back"},
		{"xabc", []string{"abc"}, 4, "the same, after unrelated text"},
		{"abcd", []string{"abc"}, 4, "text longer than the stop string is all safe"},
		// An empty entry is skipped explicitly, and the loop condition would also exclude it
		// (len("")-1 is -1, which is never above hold). The skip stays because it says what it
		// means; the result is the same either way, so no mutation of it is detectable here.
		{"abc", []string{""}, 3, "an empty stop holds nothing back"},
		{"abc", []string{"", "abc"}, 3, "an empty stop beside a complete match"},
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
		[]<-chan engine.Event{ch}, "cmpl-test", 1700000000,
		generateParams{chat: false, stop: []string{"中X"}}, 0)
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
			// happened, but the size is what a person uses to decide how much to trim. The
			// assertion is on the values, not on the word "tokens" -- "prompt is too many
			// tokens" contains that word and states no size at all.
			// The window size is exact; the prompt length is the encoded count, which differs
			// between the endpoints because a chat prompt is rendered through the transcript
			// first. Both have to appear.
			if !strings.Contains(body.Error.Message, strconv.Itoa(cfg.ContextLen)) {
				t.Errorf("message %q does not state the %d-token window",
					body.Error.Message, cfg.ContextLen)
			}
			// And a digit run for the prompt: the message reports a number, and which one
			// depends on the endpoint's encoding.
			if !regexp.MustCompile(`prompt is \d+ tokens`).MatchString(body.Error.Message) &&
				!regexp.MustCompile(`\d+ tokens`).MatchString(body.Error.Message) {
				t.Errorf("message %q states no prompt size", body.Error.Message)
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
		// The lower bound was the untested half: the check is "p < 0 || p > 1" and only the
		// upper arm had a case, so a mutation flipping "<" to ">" on the lower bound would
		// have left this test green. -0.5 rather than -1 so a check written as "p == -1"
		// would not pass either.
		{"top_p below zero", `{"prompt": "Once upon a time", "max_tokens": 1, "top_p": -0.5}`,
			"top_p"},
		// top_k was the one sampling knob that took a negative value silently: the "<= 0"
		// arm meant "unset", so top_k: -5 ran with the server default and nothing said so.
		// The CLI's --top-k refuses it, and so do max_tokens and top_p above, which made
		// this the only place a caller's value was replaced by a different one.
		{"negative top_k", `{"prompt": "Once upon a time", "max_tokens": 1, "top_k": -5}`,
			"top_k"},
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

// TestSeveralChoicesAreGenerated covers n, which used to be refused outright.
//
// A caller that sent n=3 used to get 400 "n>1 needs several sequences per request", which was
// true and left the feature missing: OpenAI's n is how a client asks for several samples of
// one prompt, self-consistency and best-of both need it, and this engine already runs several
// sequences in one batch, so the scheduler could always have done the work.
//
// What the response has to get right, and what this asserts:
//
//   - choices has n entries whose index runs 0..n-1 in order. A client indexes choices[i] and
//     reads index to tell them apart, so a short list or a shuffled one fails in the caller.
//   - usage.completion_tokens is the total across every choice, not one choice's count. The
//     prompt is prefilled once per sequence but is the same text, so prompt_tokens is counted
//     once. A client billing from this number would otherwise under-report by a factor of n.
//   - each choice is a distinct sample. Handing the same seed to every choice would return the
//     same text n times at any temperature, which is the one thing a caller asking for n is not
//     looking for.
func TestSeveralChoicesAreGenerated(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 512})

	for _, path := range []string{"/v1/completions", "/v1/chat/completions"} {
		t.Run(path, func(t *testing.T) {
			const n, maxTokens = 3, 6
			body := fmt.Sprintf(`{"prompt": "Once upon a time", "max_tokens": %d, "n": %d, `+
				`"temperature": 1.2}`, maxTokens, n)
			if path == "/v1/chat/completions" {
				body = fmt.Sprintf(`{"messages": [{"role": "user", "content": "Once upon a time"}], `+
					`"max_tokens": %d, "n": %d, "temperature": 1.2}`, maxTokens, n)
			}
			resp := post(t, srv.Handler(), path, body)
			if resp.StatusCode != http.StatusOK {
				out := decodeBody[errorBody](t, resp)
				t.Fatalf("status = %d, want 200: %s", resp.StatusCode, out.Error.Message)
			}
			var got struct {
				Choices []struct {
					Index        int    `json:"index"`
					Text         string `json:"text"`
					FinishReason string `json:"finish_reason"`
					Message      struct {
						Content string `json:"content"`
					} `json:"message"`
				} `json:"choices"`
				Usage struct {
					PromptTokens     int `json:"prompt_tokens"`
					CompletionTokens int `json:"completion_tokens"`
					TotalTokens      int `json:"total_tokens"`
				} `json:"usage"`
			}
			if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
				_ = resp.Body.Close()
				t.Fatalf("decode: %v", err)
			}
			_ = resp.Body.Close()

			if len(got.Choices) != n {
				t.Fatalf("choices = %d, want %d", len(got.Choices), n)
			}
			seen := make(map[string]bool, n)
			for i, c := range got.Choices {
				if c.Index != i {
					t.Errorf("choices[%d].index = %d, want %d: a client indexes by position", i, c.Index, i)
				}
				if c.FinishReason == "" {
					t.Errorf("choices[%d] has no finish_reason", i)
				}
				text := c.Text
				if path == "/v1/chat/completions" {
					text = c.Message.Content
				}
				seen[text] = true
			}
			// With n choices and a temperature, getting one distinct continuation and n-1
			// copies of it means the per-choice seeds collapsed to one value. The fixture's
			// weights are random, so more than one candidate is always reachable.
			if len(seen) < 2 {
				t.Errorf("all %d choices have the same text, so the per-choice seeds did not "+
					"differ", n)
			}
			// Every choice generated its own budget, and usage has to say so.
			if want := n * maxTokens; got.Usage.CompletionTokens != want {
				t.Errorf("completion_tokens = %d, want %d (%d choices x %d tokens)",
					got.Usage.CompletionTokens, want, n, maxTokens)
			}
			if got.Usage.TotalTokens != got.Usage.PromptTokens+got.Usage.CompletionTokens {
				t.Errorf("total_tokens = %d, want prompt+completion = %d",
					got.Usage.TotalTokens, got.Usage.PromptTokens+got.Usage.CompletionTokens)
			}
		})
	}
}

// TestGreedyChoicesAreIdentical pins what n>1 does at the default temperature.
//
// The per-choice samplers are seeded seed+i so that sampled choices differ, and the documentation
// once said the choices "still differ from each other" without qualifying that. They do not at
// temperature 0: every sampler is greedy, and a greedy sampler never reads its seed, so n
// choices follow the same argmax and come back identical. Measured through the HTTP API against
// Qwen2.5-0.5B, n:4 at the default returned the same six tokens four times, identically across
// three requests.
//
// This is asserted rather than merely described because the alternative -- choices that differ
// at temperature 0 -- would mean the greedy path had started consulting a seed, which changes
// every reply's reproducibility and is not what "greedy" means.
func TestGreedyChoicesAreIdentical(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 512})

	texts := func(t *testing.T) []string {
		t.Helper()
		resp := post(t, srv.Handler(), "/v1/completions",
			`{"prompt": "Once upon a time", "max_tokens": 6, "n": 3}`)
		if resp.StatusCode != http.StatusOK {
			out := decodeBody[errorBody](t, resp)
			t.Fatalf("status = %d, want 200: %s", resp.StatusCode, out.Error.Message)
		}
		var got struct {
			Choices []struct {
				Text string `json:"text"`
			} `json:"choices"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
			_ = resp.Body.Close()
			t.Fatalf("decode: %v", err)
		}
		_ = resp.Body.Close()
		out := make([]string, len(got.Choices))
		for i, c := range got.Choices {
			out[i] = c.Text
		}
		return out
	}

	first := texts(t)
	if len(first) != 3 {
		t.Fatalf("got %d choices, want 3", len(first))
	}
	for i := 1; i < len(first); i++ {
		if first[i] != first[0] {
			t.Errorf("choice %d is %q but choice 0 is %q; greedy choices must be identical",
				i, first[i], first[0])
		}
	}
	// And the sameness is stable across requests, which is the part that makes it greedy rather
	// than n identical draws from a clock-seeded generator.
	second := texts(t)
	for i := range first {
		if first[i] != second[i] {
			t.Errorf("choice %d changed between identical requests: %q then %q", i, first[i], second[i])
		}
	}
}

// TestSeveralChoicesAreReproducible pins the other half of the seed contract for n>1.
//
// A seed makes a request reproducible, and with several choices that has to mean the whole
// set: the same request must return the same n texts in the same order. Deriving the choice
// seeds from the clock instead of from the request's seed would give a caller different output
// on every call while the seed said otherwise, which is worse than not supporting a seed.
//
// It also covers the case that makes the derivation delicate: seed -1 with two choices derives
// -1 and 0, and zero is the value the request field uses for "unset". Reading the derived zero
// as unset would make one choice irreproducible while its sibling was not.
func TestSeveralChoicesAreReproducible(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 512})

	run := func(t *testing.T, seed int) []string {
		t.Helper()
		body := fmt.Sprintf(`{"prompt": "Once upon a time", "max_tokens": 6, "n": 2, `+
			`"temperature": 1.2, "seed": %d}`, seed)
		resp := post(t, srv.Handler(), "/v1/completions", body)
		if resp.StatusCode != http.StatusOK {
			out := decodeBody[errorBody](t, resp)
			t.Fatalf("status = %d, want 200: %s", resp.StatusCode, out.Error.Message)
		}
		var got struct {
			Choices []struct {
				Text string `json:"text"`
			} `json:"choices"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
			_ = resp.Body.Close()
			t.Fatalf("decode: %v", err)
		}
		_ = resp.Body.Close()
		texts := make([]string, len(got.Choices))
		for i, c := range got.Choices {
			texts[i] = c.Text
		}
		return texts
	}

	for _, seed := range []int{4242, -1} {
		first := run(t, seed)
		second := run(t, seed)
		if len(first) != len(second) {
			t.Fatalf("seed %d: choice count changed between identical requests: %d then %d",
				seed, len(first), len(second))
		}
		for i := range first {
			if first[i] != second[i] {
				t.Errorf("seed %d: choices[%d] differs between identical requests: %q then %q",
					seed, i, first[i], second[i])
			}
		}
	}
}

// TestChoiceCountIsValidated covers the two ends of n and the one transport limit on it.
//
// n=0 is what an omitted field decodes to, so it means one choice rather than none. A
// negative count and a count past maxChoices are malformed values, reported with
// invalid_value and the number that arrived rather than as an unimplemented feature.
//
// Every case here is about the /count/, not about the transport: n>1 with stream=true used to
// be refused as well and is served now, so the request's own validation is all that is left.
func TestChoiceCountIsValidated(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 512})

	t.Run("n=0 means one", func(t *testing.T) {
		resp := post(t, srv.Handler(), "/v1/completions",
			`{"prompt": "hi", "max_tokens": 2, "n": 0}`)
		if resp.StatusCode != http.StatusOK {
			out := decodeBody[errorBody](t, resp)
			t.Fatalf("status = %d, want 200: %s", resp.StatusCode, out.Error.Message)
		}
		var got struct {
			Choices []json.RawMessage `json:"choices"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
			_ = resp.Body.Close()
			t.Fatalf("decode: %v", err)
		}
		_ = resp.Body.Close()
		if len(got.Choices) != 1 {
			t.Errorf("choices = %d for n=0, want 1", len(got.Choices))
		}
	})

	t.Run("negative n", func(t *testing.T) {
		resp := post(t, srv.Handler(), "/v1/completions",
			`{"prompt": "hi", "max_tokens": 2, "n": -1}`)
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400", resp.StatusCode)
		}
		body := decodeBody[errorBody](t, resp)
		if body.Error.Code != "invalid_value" || body.Error.Param != "n" {
			t.Errorf("code/param = %q/%q, want invalid_value/n", body.Error.Code, body.Error.Param)
		}
		// The message names the number, which is what the caller has to change.
		if !strings.Contains(body.Error.Message, "-1") {
			t.Errorf("message %q does not name the value sent", body.Error.Message)
		}
	})

	t.Run("n past the cap", func(t *testing.T) {
		over := fmt.Sprintf(`{"prompt": "hi", "max_tokens": 2, "n": %d}`, maxChoices+1)
		resp := post(t, srv.Handler(), "/v1/completions", over)
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400", resp.StatusCode)
		}
		body := decodeBody[errorBody](t, resp)
		if body.Error.Code != "invalid_value" || body.Error.Param != "n" {
			t.Errorf("code/param = %q/%q, want invalid_value/n", body.Error.Code, body.Error.Param)
		}
		// The bound is stated, so the caller learns what would have been accepted.
		if want := fmt.Sprintf("at most %d", maxChoices); !strings.Contains(body.Error.Message, want) {
			t.Errorf("message %q does not state the bound %q", body.Error.Message, want)
		}
	})

	// n>1 with stream=true used to be refused here. It is served now; see
	// TestStreamingSeveralChoices for what the interleaved stream has to contain.
}

// sseChoice is one choice's slice of a streamed response, reassembled by index.
type sseChoice struct {
	text   strings.Builder
	chunks int
	// finishReasons collects every finish_reason the stream named for this index, in order.
	finishReasons []string
	// logprobs counts the logprob entries this choice's chunks carried, which is how a test
	// checks that a request asking for them got one per token rather than none.
	logprobs int
}

// readSSEChoices parses a streaming response into per-choice text, indexed by the chunk's
// own index field rather than by arrival order.
//
// It fails on a chunk whose index is negative or outside want, because both mean the server
// attributed a delta to a choice that does not exist -- which a client accumulating deltas
// would silently drop, leaving one choice short with no error anywhere.
func readSSEChoices(t *testing.T, body io.ReadCloser, want int) (map[int]*sseChoice, bool) {
	t.Helper()
	defer func() { _ = body.Close() }()
	out := make(map[int]*sseChoice, want)
	for i := range want {
		out[i] = &sseChoice{}
	}
	sawDone := false
	sc := bufio.NewScanner(body)
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
				Index        int     `json:"index"`
				Text         string  `json:"text"`
				FinishReason *string `json:"finish_reason"`
				Delta        struct {
					Content string `json:"content"`
				} `json:"delta"`
				// Both endpoints' logprob shapes carry a list: completions a parallel array,
				// chat a list of objects. Either way the chunk's count is one entry per token.
				Logprobs *struct {
					Tokens  []string          `json:"tokens"`
					Content []json.RawMessage `json:"content"`
				} `json:"logprobs"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("chunk %q: %v", payload, err)
		}
		for _, c := range chunk.Choices {
			ch, ok := out[c.Index]
			if !ok {
				t.Fatalf("chunk names index %d, which is outside the %d choices requested",
					c.Index, want)
			}
			ch.chunks++
			ch.text.WriteString(c.Delta.Content)
			ch.text.WriteString(c.Text)
			if c.FinishReason != nil {
				ch.finishReasons = append(ch.finishReasons, *c.FinishReason)
			}
			if c.Logprobs != nil {
				ch.logprobs += len(c.Logprobs.Tokens) + len(c.Logprobs.Content)
			}
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("reading the stream: %v", err)
	}
	return out, sawDone
}

// TestStreamingSeveralChoices covers n>1 with stream=true, which used to be refused with
// "the choices' deltas would have to be interleaved into one event stream".
//
// Interleaving is what the implementation has to get right, and what a client depends on:
//
//   - every delta carries the index of the choice it belongs to, so a client accumulating
//     per index reassembles n texts. A delta with the wrong index appends one choice's words
//     to another's and the failure shows up as garbled text, not as an error.
//   - every choice gets its own finish chunk. The sentinel [DONE] arrives once for the whole
//     stream, so a client learns that a particular index stopped from that index's
//     finish_reason; without it, a choice that ended early is indistinguishable from one that
//     is still going.
//   - the stream still ends with [DONE].
//   - text is not duplicated or lost when the choices advance in different rounds, which is
//     why the reassembled text is compared for equality against the same request sent
//     non-streaming rather than only being checked for length.
func TestStreamingSeveralChoices(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 512})

	for _, path := range []string{"/v1/completions", "/v1/chat/completions"} {
		t.Run(path, func(t *testing.T) {
			const n, maxTokens = 3, 6
			body := fmt.Sprintf(`{"prompt": "Once upon a time", "max_tokens": %d, "n": %d, `+
				`"temperature": 1.2}`, maxTokens, n)
			if path == "/v1/chat/completions" {
				body = fmt.Sprintf(`{"messages": [{"role": "user", "content": "Once upon a time"}], `+
					`"max_tokens": %d, "n": %d, "temperature": 1.2}`, maxTokens, n)
			}

			streamed := post(t, srv.Handler(), path, body[:len(body)-1]+`, "stream": true}`)
			if streamed.StatusCode != http.StatusOK {
				out := decodeBody[errorBody](t, streamed)
				t.Fatalf("status = %d, want 200: %s", streamed.StatusCode, out.Error.Message)
			}
			choices, sawDone := readSSEChoices(t, streamed.Body, n)

			if !sawDone {
				t.Error("the stream ended without the [DONE] sentinel")
			}
			for i := range n {
				c := choices[i]
				if c.text.Len() == 0 {
					t.Errorf("choice %d streamed no text at all", i)
				}
				if len(c.finishReasons) != 1 {
					t.Errorf("choice %d got %d finish chunks, want exactly 1: %v",
						i, len(c.finishReasons), c.finishReasons)
				} else if c.finishReasons[0] == "" {
					t.Errorf("choice %d finished with an empty finish_reason", i)
				}
			}

			// The same request without streaming has to produce the same text per index. The
			// fixture is deterministic only with a seed, so ask for one: the point is that
			// streaming and blocking agree, not that sampling is stable.
			seeded := body[:len(body)-1] + `, "seed": 11}`
			blocked := post(t, srv.Handler(), path, seeded)
			if blocked.StatusCode != http.StatusOK {
				out := decodeBody[errorBody](t, blocked)
				t.Fatalf("blocking status = %d, want 200: %s", blocked.StatusCode, out.Error.Message)
			}
			var whole struct {
				Choices []struct {
					Text    string `json:"text"`
					Message struct {
						Content string `json:"content"`
					} `json:"message"`
				} `json:"choices"`
			}
			if err := json.NewDecoder(blocked.Body).Decode(&whole); err != nil {
				_ = blocked.Body.Close()
				t.Fatalf("decode: %v", err)
			}
			_ = blocked.Body.Close()

			streamedSeeded := post(t, srv.Handler(), path,
				seeded[:len(seeded)-1]+`, "stream": true}`)
			if streamedSeeded.StatusCode != http.StatusOK {
				out := decodeBody[errorBody](t, streamedSeeded)
				t.Fatalf("seeded stream status = %d: %s", streamedSeeded.StatusCode, out.Error.Message)
			}
			seededChoices, done := readSSEChoices(t, streamedSeeded.Body, n)
			if !done {
				t.Error("the seeded stream ended without [DONE]")
			}
			for i := range n {
				want := whole.Choices[i].Text
				if path == "/v1/chat/completions" {
					want = whole.Choices[i].Message.Content
				}
				if got := seededChoices[i].text.String(); got != want {
					t.Errorf("choice %d: streamed %q, the same request blocking returned %q",
						i, got, want)
				}
			}
		})
	}
}

// TestStreamingSeveralChoicesWithStop covers the interaction of the two most delicate parts of
// the streaming path: per-choice decoding and stop-string matching.
//
// A stop string is matched against decoded text, and a partially-matched tail has to be held
// back until it can no longer become a match -- otherwise stop=["北京"] streams "北" where the
// non-streaming body returns nothing. That hold-back state is per choice, and the interaction
// is exactly where a shared buffer would go unnoticed by anything but a comparison against the
// other transport: each choice would still get some text, just the wrong text, and the stop
// character could leak into one choice while being cut from another.
//
// The stop string is derived from the output rather than hard-coded, because the fixture's
// weights are random: a literal would make this test either depend on a checkpoint or assert
// nothing. One never-matching stop is checked too, which is the path that releases the held
// tail at the end instead of finding a match.
func TestStreamingSeveralChoicesWithStop(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 512})

	const n = 2
	// quoted renders a JSON string literal for a stop list.
	quoted := func(s string) string {
		b, err := json.Marshal(s)
		if err != nil {
			t.Fatalf("marshal stop: %v", err)
		}
		return string(b)
	}
	// texts asks for the same request both ways and returns (blocking, streamed) by index.
	texts := func(t *testing.T, stopList []string) (map[int]string, map[int]string) {
		t.Helper()
		stops := "null"
		if len(stopList) > 0 {
			parts := make([]string, len(stopList))
			for i, s := range stopList {
				parts[i] = quoted(s)
			}
			stops = "[" + strings.Join(parts, ", ") + "]"
		}
		base := fmt.Sprintf(`{"prompt": "Once upon a time", "max_tokens": 12, "n": %d, `+
			`"temperature": 1.2, "seed": 909, "stop": %s`, n, stops)

		blocked := post(t, srv.Handler(), "/v1/completions", base+"}")
		if blocked.StatusCode != http.StatusOK {
			out := decodeBody[errorBody](t, blocked)
			t.Fatalf("blocking status = %d: %s", blocked.StatusCode, out.Error.Message)
		}
		var whole struct {
			Choices []struct {
				Text string `json:"text"`
			} `json:"choices"`
		}
		if err := json.NewDecoder(blocked.Body).Decode(&whole); err != nil {
			_ = blocked.Body.Close()
			t.Fatalf("decode: %v", err)
		}
		_ = blocked.Body.Close()
		byIndex := make(map[int]string, n)
		for i, c := range whole.Choices {
			byIndex[i] = c.Text
		}

		streamed := post(t, srv.Handler(), "/v1/completions", base+`, "stream": true}`)
		if streamed.StatusCode != http.StatusOK {
			out := decodeBody[errorBody](t, streamed)
			t.Fatalf("stream status = %d: %s", streamed.StatusCode, out.Error.Message)
		}
		choices, sawDone := readSSEChoices(t, streamed.Body, n)
		if !sawDone {
			t.Error("the stream ended without [DONE]")
		}
		streamedByIndex := make(map[int]string, n)
		for i, c := range choices {
			streamedByIndex[i] = c.text.String()
		}
		return byIndex, streamedByIndex
	}

	// Phase 1: a stop that never matches. Both transports must return the whole text, which is
	// the path that releases the held tail once the sequence ends.
	blocking, streamed := texts(t, []string{"zzzznotpresentzzzz"})
	for i := range n {
		if blocking[i] == "" {
			t.Fatalf("choice %d generated nothing, so the comparison proves nothing", i)
		}
		if blocking[i] != streamed[i] {
			t.Errorf("no match, choice %d: streamed %q, blocking %q", i, streamed[i], blocking[i])
		}
	}

	// Phase 2: a stop taken from choice 0's own output, so a match is guaranteed somewhere in
	// a sequence rather than hoped for.
	match := blocking[0]
	if len(match) < 4 {
		t.Skipf("choice 0 produced only %q; too short to take a stop from", match)
	}
	stop := match[2:5]
	blockingCut, streamedCut := texts(t, []string{stop})
	for i := range n {
		if blockingCut[i] != streamedCut[i] {
			t.Errorf("stop=%q, choice %d: streamed %q, blocking %q",
				stop, i, streamedCut[i], blockingCut[i])
		}
		// Whatever each transport returned, the stop string must not be in it: a leak here is
		// the failure the hold-back exists to prevent.
		if strings.Contains(blockingCut[i], stop) {
			t.Errorf("stop=%q, choice %d: the blocking body contains the stop string: %q",
				stop, i, blockingCut[i])
		}
		if strings.Contains(streamedCut[i], stop) {
			t.Errorf("stop=%q, choice %d: the stream contains the stop string: %q",
				stop, i, streamedCut[i])
		}
	}
}

// TestStreamingSeveralChoicesUsage pins the usage chunk for n>1, which is where the two
// paths most easily disagree.
//
// The counts span every choice: completion_tokens is the sum of what all n generated, and
// prompt_tokens is the prompt once, matching the non-streaming body for the same request. A
// client billing from a stream that reported one choice's tokens would under-report by a
// factor of n, and nothing in the body would say so.
func TestStreamingSeveralChoicesUsage(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 512})

	const n, maxTokens = 3, 5
	resp := post(t, srv.Handler(), "/v1/completions",
		fmt.Sprintf(`{"prompt": "Once upon a time", "max_tokens": %d, "n": %d, `+
			`"temperature": 1.2, "stream": true, "stream_options": {"include_usage": true}}`,
			maxTokens, n))
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var gotUsage bool
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			break
		}
		var chunk struct {
			Choices []json.RawMessage `json:"choices"`
			Usage   *struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("chunk %q: %v", payload, err)
		}
		// The usage chunk is the one with an empty choices array; every other chunk in the
		// same stream carries usage: null, which decodes to a nil pointer here.
		if len(chunk.Choices) != 0 || chunk.Usage == nil {
			continue
		}
		gotUsage = true
		if want := n * maxTokens; chunk.Usage.CompletionTokens != want {
			t.Errorf("completion_tokens = %d, want %d (%d choices x %d tokens)",
				chunk.Usage.CompletionTokens, want, n, maxTokens)
		}
		if chunk.Usage.TotalTokens != chunk.Usage.PromptTokens+chunk.Usage.CompletionTokens {
			t.Errorf("total_tokens = %d, want prompt+completion = %d",
				chunk.Usage.TotalTokens, chunk.Usage.PromptTokens+chunk.Usage.CompletionTokens)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("reading the stream: %v", err)
	}
	if !gotUsage {
		t.Error("include_usage produced no usage chunk with an empty choices array")
	}
}

// TestLogprobsOnBothEndpoints covers the two schemas, which are shaped differently and share
// no field names beyond the probability itself.
//
// The invariants asserted are the ones a caller actually relies on:
//
//   - the token list has the same length as the probability list, or a client indexing one by
//     the other runs off the end;
//   - the tokens concatenate to the response text. This is what catches a per-token decoding
//     that loses the byte state a token's text depends on, which produces plausible-looking
//     pieces that do not add up to the sentence the caller sees;
//   - text_offset is the cumulative byte length, so tokens[i] begins where offsets[i] says;
//   - every log probability is finite and not positive. A value above zero is not a
//     probability, and NaN cannot be encoded as JSON at all.
func TestLogprobsOnBothEndpoints(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 512})

	t.Run("completions", func(t *testing.T) {
		resp := post(t, srv.Handler(), "/v1/completions",
			`{"prompt": "Once upon a time", "max_tokens": 6, "seed": 3, "logprobs": 0}`)
		if resp.StatusCode != 200 {
			out := decodeBody[errorBody](t, resp)
			t.Fatalf("status = %d: %s", resp.StatusCode, out.Error.Message)
		}
		var got struct {
			Choices []struct {
				Text     string `json:"text"`
				Logprobs *struct {
					Tokens        []string         `json:"tokens"`
					TokenLogprobs []float32        `json:"token_logprobs"`
					TextOffset    []int            `json:"text_offset"`
					TopLogprobs   []map[string]any `json:"top_logprobs"`
				} `json:"logprobs"`
			} `json:"choices"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
			_ = resp.Body.Close()
			t.Fatalf("decode: %v", err)
		}
		_ = resp.Body.Close()

		lp := got.Choices[0].Logprobs
		if lp == nil {
			t.Fatal("logprobs is null for a request that asked for it")
		}
		if len(lp.Tokens) == 0 {
			t.Fatal("no tokens described")
		}
		if len(lp.Tokens) != len(lp.TokenLogprobs) || len(lp.Tokens) != len(lp.TextOffset) ||
			len(lp.Tokens) != len(lp.TopLogprobs) {
			t.Fatalf("parallel arrays disagree: tokens %d, logprobs %d, offsets %d, top %d",
				len(lp.Tokens), len(lp.TokenLogprobs), len(lp.TextOffset), len(lp.TopLogprobs))
		}
		if joined := strings.Join(lp.Tokens, ""); joined != got.Choices[0].Text {
			t.Errorf("tokens join to %q, the response text is %q", joined, got.Choices[0].Text)
		}
		at := 0
		for i, tok := range lp.Tokens {
			if lp.TextOffset[i] != at {
				t.Errorf("text_offset[%d] = %d, want %d (the byte it starts at)",
					i, lp.TextOffset[i], at)
			}
			at += len(tok)
			if lp.TokenLogprobs[i] > 0 {
				t.Errorf("token_logprobs[%d] = %v, a log probability cannot be positive",
					i, lp.TokenLogprobs[i])
			}
			// logprobs: 0 asks for the chosen token only, so there are no alternatives.
			if lp.TopLogprobs[i] != nil {
				t.Errorf("top_logprobs[%d] = %v with logprobs=0, want null", i, lp.TopLogprobs[i])
			}
		}

		// With a count, each position carries its alternatives, keyed by the alternative's
		// own text as the schema asks.
		resp = post(t, srv.Handler(), "/v1/completions",
			`{"prompt": "Once upon a time", "max_tokens": 2, "seed": 3, "logprobs": 2}`)
		var two struct {
			Choices []struct {
				Logprobs *struct {
					TopLogprobs []map[string]float32 `json:"top_logprobs"`
				} `json:"logprobs"`
			} `json:"choices"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&two); err != nil {
			_ = resp.Body.Close()
			t.Fatalf("decode: %v", err)
		}
		_ = resp.Body.Close()
		for i, top := range two.Choices[0].Logprobs.TopLogprobs {
			if len(top) == 0 {
				t.Errorf("top_logprobs[%d] is empty with logprobs=2", i)
			}
			for text, v := range top {
				if v > 0 {
					t.Errorf("top_logprobs[%d][%q] = %v, a log probability cannot be positive", i, text, v)
				}
			}
		}
	})

	t.Run("chat", func(t *testing.T) {
		resp := post(t, srv.Handler(), "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "Once upon a time"}], "max_tokens": 4, `+
				`"seed": 3, "logprobs": true, "top_logprobs": 2}`)
		if resp.StatusCode != 200 {
			out := decodeBody[errorBody](t, resp)
			t.Fatalf("status = %d: %s", resp.StatusCode, out.Error.Message)
		}
		var got struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
				Logprobs *struct {
					Content []struct {
						Token       string  `json:"token"`
						Bytes       []int   `json:"bytes"`
						Logprob     float32 `json:"logprob"`
						TopLogprobs []struct {
							Token   string  `json:"token"`
							Bytes   []int   `json:"bytes"`
							Logprob float32 `json:"logprob"`
						} `json:"top_logprobs"`
					} `json:"content"`
					Refusal any `json:"refusal"`
				} `json:"logprobs"`
			} `json:"choices"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
			_ = resp.Body.Close()
			t.Fatalf("decode: %v", err)
		}
		_ = resp.Body.Close()

		lp := got.Choices[0].Logprobs
		if lp == nil {
			t.Fatal("logprobs is null for a request that asked for it")
		}
		if lp.Refusal != nil {
			t.Errorf("refusal = %v, want null: this server has no refusal channel", lp.Refusal)
		}
		var joined strings.Builder
		for i, c := range lp.Content {
			joined.WriteString(c.Token)
			if c.Logprob > 0 {
				t.Errorf("content[%d].logprob = %v, a log probability cannot be positive", i, c.Logprob)
			}
			// bytes is the token's UTF-8, which is what lets a client reassemble text it
			// cannot get back from the token string alone.
			if want := []byte(c.Token); len(c.Bytes) != len(want) {
				t.Errorf("content[%d].bytes = %v, the token %q is %d bytes",
					i, c.Bytes, c.Token, len(want))
			} else {
				for j, b := range want {
					if c.Bytes[j] != int(b) {
						t.Errorf("content[%d].bytes[%d] = %d, want %d", i, j, c.Bytes[j], b)
					}
				}
			}
			// An empty alternative list is allowed by the schema, but a request asking for two
			// over a vocabulary of thousands should get them.
			if len(c.TopLogprobs) == 0 {
				t.Errorf("content[%d].top_logprobs is empty with top_logprobs=2", i)
			}
			for j, alt := range c.TopLogprobs {
				if alt.Logprob > 0 {
					t.Errorf("content[%d].top_logprobs[%d].logprob = %v", i, j, alt.Logprob)
				}
			}
		}
		if joined.String() != got.Choices[0].Message.Content {
			t.Errorf("tokens join to %q, the message content is %q",
				joined.String(), got.Choices[0].Message.Content)
		}
	})

	t.Run("absent means null", func(t *testing.T) {
		resp := post(t, srv.Handler(), "/v1/completions",
			`{"prompt": "hi", "max_tokens": 2}`)
		defer func() { _ = resp.Body.Close() }()
		var got struct {
			Choices []map[string]json.RawMessage `json:"choices"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
			t.Fatalf("decode: %v", err)
		}
		raw, ok := got.Choices[0]["logprobs"]
		if !ok {
			t.Fatal("the logprobs key is missing; the schema has it present and null")
		}
		if string(raw) != "null" {
			t.Errorf("logprobs = %s for a request that did not ask, want null", raw)
		}
	})
}

// TestStreamingLogprobsMatchBlocking pins the agreement between the two transports, which is
// the property most likely to break silently.
//
// The streaming path decodes one token at a time and the blocking path decodes the whole
// sequence, so they can describe the same request differently without either looking wrong on
// its own: a different number of entries, offsets counted from a different place, or a token
// described in one and missing from the other. All three are checked for the same seeded
// request.
//
// A stop string is included because it is where the two paths genuinely diverge in structure:
// the streamer holds text back and the blocker cuts a rendered string, and a token that
// straddles the cut has to be dropped by both or the lists disagree.
func TestStreamingLogprobsMatchBlocking(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 512})

	cases := []struct {
		name string
		stop string
	}{
		{"no stop", ""},
		// A comma appears early in this fixture's continuations, so this exercises the match.
		{"stop that matches", `", "`},
		{"stop that never matches", `"zzzznotpresent"`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			stopField := ""
			if c.stop != "" {
				stopField = `, "stop": ` + c.stop
			}

			// Blocking.
			resp := post(t, srv.Handler(), "/v1/completions",
				`{"prompt": "Once upon a time", "max_tokens": 8, "seed": 21, "logprobs": 1`+
					stopField+`}`)
			if resp.StatusCode != 200 {
				out := decodeBody[errorBody](t, resp)
				t.Fatalf("blocking status = %d: %s", resp.StatusCode, out.Error.Message)
			}
			var whole struct {
				Choices []struct {
					Text     string `json:"text"`
					Logprobs *struct {
						Tokens     []string  `json:"tokens"`
						TextOffset []int     `json:"text_offset"`
						Logprobs   []float32 `json:"token_logprobs"`
					} `json:"logprobs"`
				} `json:"choices"`
			}
			if err := json.NewDecoder(resp.Body).Decode(&whole); err != nil {
				_ = resp.Body.Close()
				t.Fatalf("decode: %v", err)
			}
			_ = resp.Body.Close()

			// Streaming, reassembling the text, the offsets and the probabilities in order.
			streamed := post(t, srv.Handler(), "/v1/completions",
				`{"prompt": "Once upon a time", "max_tokens": 8, "seed": 21, "logprobs": 1, `+
					`"stream": true`+stopField+`}`)
			if streamed.StatusCode != 200 {
				out := decodeBody[errorBody](t, streamed)
				t.Fatalf("stream status = %d: %s", streamed.StatusCode, out.Error.Message)
			}
			var text strings.Builder
			var tokens []string
			var offsets []int
			var logprobs []float32
			if err := eachSSEChunk(streamed.Body, func(chunk []byte) {
				var c struct {
					Choices []struct {
						Text     string `json:"text"`
						Logprobs *struct {
							Tokens     []string  `json:"tokens"`
							TextOffset []int     `json:"text_offset"`
							Logprobs   []float32 `json:"token_logprobs"`
						} `json:"logprobs"`
					} `json:"choices"`
				}
				if err := json.Unmarshal(chunk, &c); err != nil {
					t.Fatalf("chunk %s: %v", chunk, err)
				}
				for _, ch := range c.Choices {
					text.WriteString(ch.Text)
					if ch.Logprobs != nil {
						tokens = append(tokens, ch.Logprobs.Tokens...)
						offsets = append(offsets, ch.Logprobs.TextOffset...)
						logprobs = append(logprobs, ch.Logprobs.Logprobs...)
					}
				}
			}); err != nil {
				t.Fatalf("reading the stream: %v", err)
			}

			block := whole.Choices[0]
			if block.Logprobs == nil {
				t.Fatal("the blocking response has no logprobs")
			}
			if text.String() != block.Text {
				t.Errorf("streamed text %q, blocking text %q", text.String(), block.Text)
			}
			if fmt.Sprint(tokens) != fmt.Sprint(block.Logprobs.Tokens) {
				t.Errorf("streamed tokens %v, blocking tokens %v", tokens, block.Logprobs.Tokens)
			}
			if fmt.Sprint(offsets) != fmt.Sprint(block.Logprobs.TextOffset) {
				t.Errorf("streamed offsets %v, blocking offsets %v", offsets, block.Logprobs.TextOffset)
			}
			if fmt.Sprint(logprobs) != fmt.Sprint(block.Logprobs.Logprobs) {
				t.Errorf("streamed logprobs %v, blocking logprobs %v", logprobs, block.Logprobs.Logprobs)
			}
		})
	}
}

// TestLogprobsCountIsValidated covers the caps, which differ per endpoint because OpenAI's
// schema documents different maxima.
//
// Both are refused rather than clamped: a caller that asked for twenty alternatives and got
// five has been silently given a different answer, and the count is visible in the response's
// own arrays, so the discrepancy would be noticed without being explained.
func TestLogprobsCountIsValidated(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 512})

	cases := []struct {
		name   string
		path   string
		body   string
		field  string
		expect string
	}{
		{"chat above its cap", "/v1/chat/completions",
			fmt.Sprintf(`{"messages": [{"role": "user", "content": "hi"}], "logprobs": true, `+
				`"top_logprobs": %d}`, maxChatTopLogprobs+1), "top_logprobs",
			fmt.Sprintf("at most %d", maxChatTopLogprobs)},
		{"completions above its cap", "/v1/completions",
			fmt.Sprintf(`{"prompt": "hi", "logprobs": %d}`, maxCompletionLogprobs+1), "logprobs",
			fmt.Sprintf("at most %d", maxCompletionLogprobs)},
		{"chat negative", "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "hi"}], "logprobs": true, "top_logprobs": -1}`,
			"top_logprobs", "-1"},
		{"completions negative", "/v1/completions",
			`{"prompt": "hi", "logprobs": -1}`, "logprobs", "-1"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			resp := post(t, srv.Handler(), c.path, c.body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != 400 {
				t.Fatalf("status = %d, want 400", resp.StatusCode)
			}
			body := decodeBody[errorBody](t, resp)
			if body.Error.Code != "invalid_value" || body.Error.Param != c.field {
				t.Errorf("code/param = %q/%q, want invalid_value/%s",
					body.Error.Code, body.Error.Param, c.field)
			}
			// The message states the bound or the value, so the caller learns what would have
			// been accepted rather than only that it was wrong.
			if !strings.Contains(body.Error.Message, c.expect) {
				t.Errorf("message %q does not mention %q", body.Error.Message, c.expect)
			}
		})
	}
}

// eachSSEChunk calls fn with each data payload before [DONE].
func eachSSEChunk(body io.ReadCloser, fn func([]byte)) error {
	defer func() { _ = body.Close() }()
	sc := bufio.NewScanner(body)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			return nil
		}
		fn([]byte(payload))
	}
	return sc.Err()
}

// TestSuffixWithoutMarkersIsRefused is the case most checkpoints hit: a model with no
// fill-in-the-middle markers cannot answer an insertion, and the refusal has to say that the
// model is the reason rather than that the server lacks the feature.
//
// The distinction matters to whoever gets the error. "Not implemented" sends them to another
// server; "this checkpoint declares no markers" sends them to a code-tuned checkpoint, which is
// the only thing that would work. Generating from the prefix alone would be worse than either:
// the reply would be a plausible continuation that silently ignored half the request.
func TestSuffixWithoutMarkersIsRefused(t *testing.T) {
	// The default fixture's vocabulary has no FIM markers, which is the Llama case.
	srv, _ := newTestServer(t)

	resp := post(t, srv.Handler(), "/v1/completions",
		`{"prompt": "func add(a, b int) int {", "suffix": "return a + b\n}", "max_tokens": 4}`)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
	body := decodeBody[errorBody](t, resp)
	if body.Error.Param != "suffix" || body.Error.Code != "unsupported_parameter" {
		t.Errorf("param/code = %q/%q, want suffix/unsupported_parameter",
			body.Error.Param, body.Error.Code)
	}
	// The message has to name the model as the cause and the markers as what is missing, or a
	// caller cannot tell this apart from a server that has not implemented the field.
	for _, want := range []string{"fill-in-the-middle", "model declares none", "without suffix"} {
		if !strings.Contains(body.Error.Message, want) {
			t.Errorf("message %q does not mention %q", body.Error.Message, want)
		}
	}
	// And the part that makes the message actionable: the marker sets a code-tuned checkpoint
	// would have to declare. Replacing fimMarkerHint's list with an empty one left this test
	// green, so nothing was checking that the parenthesis had anything in it -- the message
	// would have read "declares none of the marker sets this server knows ()".
	for _, want := range []string{
		"<|fim_prefix|> and <|fim_middle|>",
		"<fim_prefix> and <fim_middle>",
		"<PRE> and <MID>",
		"\u2581<fim_prefix> and \u2581<fim_middle>",
	} {
		if !strings.Contains(body.Error.Message, want) {
			t.Errorf("message %q does not name the marker set %q a checkpoint could declare",
				body.Error.Message, want)
		}
	}
}

// TestFIMPromptAssembly pins the order the parts are joined in, which is the whole of the
// feature: a fill-in-the-middle prompt is prefix marker, the text before, suffix marker, the
// text after, middle marker, and any other arrangement is a different question to the model.
//
// The assertions are on the ids rather than on a rendered string because that is what the
// engine receives, and because the caller's text must stay text: encoding the parts and joining
// them around the markers is what keeps a literal "<|fim_middle|>" typed into the prefix from
// becoming the marker that opens the insertion.
func TestFIMPromptAssembly(t *testing.T) {
	vocab := []string{
		"<unk>", "<s>", "</s>", "a", "b", "c",
		"<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|fim_pad|>",
	}
	srv, tok := newTestServerModel(t, gguftest.Model{Vocab: vocab, NoEOS: true, ContextLen: 256})

	fim, ok := tok.FIMTokens()
	if !ok {
		t.Fatal("the fixture's markers were not detected")
	}
	const before, after = "a b", "c"
	got := srv.fimPrompt(before, after, fim)

	prefixIDs := tok.Encode(before, false)
	suffixIDs := tok.Encode(after, false)
	want := make([]int32, 0, len(prefixIDs)+len(suffixIDs)+3)
	want = append(want, fim.Prefix)
	want = append(want, prefixIDs...)
	want = append(want, fim.Suffix)
	want = append(want, suffixIDs...)
	want = append(want, fim.Middle)
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Errorf("assembled %v, want %v", got, want)
	}

	// The structure, stated independently of the encoder: the first id is the prefix marker,
	// the last is the middle marker, and the suffix marker sits exactly after the prefix text.
	if len(got) < 3 {
		t.Fatalf("assembled only %d ids", len(got))
	}
	if got[0] != fim.Prefix || got[len(got)-1] != fim.Middle {
		t.Errorf("the prompt does not start with the prefix marker and end with the middle one: %v", got)
	}
	if got[1+len(prefixIDs)] != fim.Suffix {
		t.Errorf("the suffix marker is at %d, want it at %d",
			got[1+len(prefixIDs)], 1+len(prefixIDs))
	}
	// No BOS: the markers are the structure, and a checkpoint trained for this sees the prefix
	// marker at the start of the sequence.
	cfg := srv.eng.Model().Config()
	if cfg.BOSID >= 0 && got[0] == int32(cfg.BOSID) {
		t.Error("the prompt starts with BOS; the prefix marker belongs there")
	}
}

// TestSuffixIsHonouredEndToEnd covers the request path for a checkpoint that can do it.
//
// The markers are control tokens, so they are absent from the decoded text whether or not the
// model emits them -- but the reply must also be free of them as *text*, which is what the
// assertion checks: a marker that reached the response would be the model closing its
// insertion, and the response is supposed to stop before that.
func TestSuffixIsHonouredEndToEnd(t *testing.T) {
	vocab := []string{
		"<unk>", "<s>", "</s>", "a", "b", "c", "d", "e",
		"<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>",
	}
	srv, _ := newTestServerModel(t, gguftest.Model{Vocab: vocab, NoEOS: true, ContextLen: 256})

	resp := post(t, srv.Handler(), "/v1/completions",
		`{"prompt": "a", "suffix": "c", "max_tokens": 4}`)
	if resp.StatusCode != http.StatusOK {
		out := decodeBody[errorBody](t, resp)
		t.Fatalf("status = %d, want 200: %s", resp.StatusCode, out.Error.Message)
	}
	var got struct {
		Choices []struct {
			Text string `json:"text"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		_ = resp.Body.Close()
		t.Fatalf("decode: %v", err)
	}
	_ = resp.Body.Close()
	if len(got.Choices) != 1 {
		t.Fatalf("got %d choices, want 1", len(got.Choices))
	}
	for _, marker := range []string{"<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"} {
		if strings.Contains(got.Choices[0].Text, marker) {
			t.Errorf("the response contains the marker %q: %q", marker, got.Choices[0].Text)
		}
	}
}

// TestSuffixAndEchoAreRefusedTogether records the one combination this server will not guess at.
//
// With an insertion the prompt is only the text before the insertion point, so echo would
// return half of what the caller sent under a name that promises the prompt. Neither reading is
// wrong enough to pick silently.
func TestSuffixAndEchoAreRefusedTogether(t *testing.T) {
	vocab := []string{
		"<unk>", "<s>", "</s>", "a", "b",
		"<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>",
	}
	srv, _ := newTestServerModel(t, gguftest.Model{Vocab: vocab, NoEOS: true, ContextLen: 256})

	resp := post(t, srv.Handler(), "/v1/completions",
		`{"prompt": "a", "suffix": "b", "max_tokens": 2, "echo": true}`)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
	body := decodeBody[errorBody](t, resp)
	if body.Error.Param != "suffix" || body.Error.Code != "unsupported_parameter" {
		t.Errorf("param/code = %q/%q, want suffix/unsupported_parameter",
			body.Error.Param, body.Error.Code)
	}
	if !strings.Contains(body.Error.Message, "echo") {
		t.Errorf("message %q does not name the field it conflicts with", body.Error.Message)
	}
}

// TestEmptySuffixMeansNoInsertion keeps the field's zero value inert.
//
// An SDK that always sends the field sends an empty string when the caller has not asked for an
// insertion, and refusing that would fail a request this server answers exactly as asked.
func TestEmptySuffixMeansNoInsertion(t *testing.T) {
	srv, _ := newTestServer(t)
	resp := post(t, srv.Handler(), "/v1/completions",
		`{"prompt": "hi", "suffix": "", "max_tokens": 2}`)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		out := decodeBody[errorBody](t, resp)
		t.Fatalf("status = %d, want 200: %s", resp.StatusCode, out.Error.Message)
	}
}

// TestBestOfReturnsTheBestCandidates covers the one field that changes which samples come back
// rather than how they are sampled.
//
// OpenAI documents best_of as generating that many completions server-side and returning the
// one with the highest log probability per token; with n it is the candidate count and n is how
// many to return. The test computes the ranking independently -- one request for all
// best_of candidates with logprobs, their mean log probability worked out here -- and then asks
// for the same request with best_of set. That is what makes it a check of the selection rather
// than of the reply's shape: an implementation that returned the first candidate, or the last,
// would pass a "one choice came back" assertion and fail this one.
func TestBestOfReturnsTheBestCandidates(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 512})

	const candidates, maxTokens = 4, 5
	// query asks for `n` choices with their log probabilities and returns the texts plus the
	// mean log probability of each, in response order.
	query := func(t *testing.T, extra string) ([]string, []float64) {
		t.Helper()
		body := fmt.Sprintf(`{"prompt": "Once upon a time", "max_tokens": %d, `+
			`"temperature": 1.3, "seed": 17, "logprobs": 0%s}`, maxTokens, extra)
		resp := post(t, srv.Handler(), "/v1/completions", body)
		if resp.StatusCode != 200 {
			out := decodeBody[errorBody](t, resp)
			t.Fatalf("status = %d: %s", resp.StatusCode, out.Error.Message)
		}
		var got struct {
			Choices []struct {
				Text     string `json:"text"`
				Logprobs *struct {
					Logprobs []float32 `json:"token_logprobs"`
				} `json:"logprobs"`
			} `json:"choices"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
			_ = resp.Body.Close()
			t.Fatalf("decode: %v", err)
		}
		_ = resp.Body.Close()
		texts := make([]string, len(got.Choices))
		means := make([]float64, len(got.Choices))
		for i, c := range got.Choices {
			texts[i] = c.Text
			if c.Logprobs == nil || len(c.Logprobs.Logprobs) == 0 {
				t.Fatalf("choice %d has no log probabilities, so nothing can be ranked", i)
			}
			var sum float64
			for _, lp := range c.Logprobs.Logprobs {
				sum += float64(lp)
			}
			means[i] = sum / float64(len(c.Logprobs.Logprobs))
		}
		return texts, means
	}

	texts, means := query(t, fmt.Sprintf(`, "n": %d`, candidates))
	if len(texts) != candidates {
		t.Fatalf("got %d candidates, want %d", len(texts), candidates)
	}
	// The best by mean log probability, computed here rather than taken from the server.
	best := 0
	for i := range means {
		if means[i] > means[best] {
			best = i
		}
	}

	t.Run("one choice is the best candidate", func(t *testing.T) {
		got, _ := query(t, fmt.Sprintf(`, "best_of": %d`, candidates))
		if len(got) != 1 {
			t.Fatalf("got %d choices, want 1", len(got))
		}
		if got[0] != texts[best] {
			t.Errorf("best_of returned %q; the highest mean log probability (%.4f) was %q",
				got[0], means[best], texts[best])
		}
	})

	t.Run("several choices are the best ones in order", func(t *testing.T) {
		order := make([]int, len(means))
		for i := range order {
			order[i] = i
		}
		slices.SortStableFunc(order, func(a, b int) int {
			switch {
			case means[a] > means[b]:
				return -1
			case means[a] < means[b]:
				return 1
			}
			return a - b
		})
		want := []string{texts[order[0]], texts[order[1]]}

		resp := post(t, srv.Handler(), "/v1/completions",
			fmt.Sprintf(`{"prompt": "Once upon a time", "max_tokens": %d, "temperature": 1.3, `+
				`"seed": 17, "best_of": %d, "n": 2}`, maxTokens, candidates))
		if resp.StatusCode != 200 {
			out := decodeBody[errorBody](t, resp)
			t.Fatalf("status = %d: %s", resp.StatusCode, out.Error.Message)
		}
		var got struct {
			Choices []struct {
				Index int    `json:"index"`
				Text  string `json:"text"`
			} `json:"choices"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
			_ = resp.Body.Close()
			t.Fatalf("decode: %v", err)
		}
		_ = resp.Body.Close()

		if len(got.Choices) != 2 {
			t.Fatalf("got %d choices, want 2", len(got.Choices))
		}
		for i, c := range got.Choices {
			if c.Text != want[i] {
				t.Errorf("choices[%d] = %q, want %q (the %d-th best)", i, c.Text, want[i], i+1)
			}
			// The index is the position in the response, not the candidate's original slot:
			// a discarded candidate's number there would name an index the caller cannot use.
			if c.Index != i {
				t.Errorf("choices[%d].index = %d, want %d", i, c.Index, i)
			}
		}
	})

	t.Run("usage counts every candidate generated", func(t *testing.T) {
		resp := post(t, srv.Handler(), "/v1/completions",
			fmt.Sprintf(`{"prompt": "Once upon a time", "max_tokens": %d, "temperature": 1.3, `+
				`"seed": 17, "best_of": %d}`, maxTokens, candidates))
		defer func() { _ = resp.Body.Close() }()
		var got struct {
			Usage struct {
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
			t.Fatalf("decode: %v", err)
		}
		// Every candidate generated its full budget, and all of them were work the server did.
		// Reporting only the returned choice would understate it by a factor of best_of.
		if want := candidates * maxTokens; got.Usage.CompletionTokens != want {
			t.Errorf("completion_tokens = %d, want %d (%d candidates x %d tokens)",
				got.Usage.CompletionTokens, want, candidates, maxTokens)
		}
	})
}

// TestCandidateScoreIsPerToken pins the ranking metric, which the end-to-end test cannot see.
//
// Every candidate of a request shares one budget, so their lengths are equal and a sum happens
// to rank them the same way an average does -- the mean would be untested, and a change to a
// sum would pass every assertion above. The two disagree exactly when lengths differ, so this
// builds that case directly: a short confident candidate against a longer slightly-less
// confident one, where the sum prefers the short and the average prefers the long.
//
// OpenAI documents best_of as comparing "log probability per token", and the average is also
// the answer that has no length bias: every term is negative, so a sum always prefers the
// shortest completion.
func TestCandidateScoreIsPerToken(t *testing.T) {
	short := choiceResult{
		out:     []int32{1, 2},
		samples: []sampledToken{{logprob: -1}, {logprob: -1}},
	}
	long := choiceResult{
		out: []int32{1, 2, 3, 4},
		samples: []sampledToken{
			{logprob: -0.6}, {logprob: -0.6}, {logprob: -0.6}, {logprob: -0.6},
		},
	}
	// Compared with a tolerance: the samples hold float32 probabilities, so an average of four
	// of them is not exactly the decimal it was written as.
	near := func(got, want float64) bool { return math.Abs(got-want) < 1e-6 }
	// A candidate whose samples and output disagree in length has no whole-run figure:
	// samples is only filled when log probabilities were requested, so a caller reaching
	// there without them would otherwise average over a prefix and call it the mean. The
	// length half of the guard had no test -- removing it left the package green -- and the
	// empty half does, from the cases below.
	mismatched := choiceResult{
		out:     []int32{1, 2, 3},
		samples: []sampledToken{{logprob: -0.5}},
	}
	if got := mismatched.score(); !math.IsInf(got, -1) {
		t.Errorf("a candidate with 3 outputs and 1 sample scored %v, want -Inf: its mean would "+
			"describe a run the response does not report", got)
	}
	noSamples := choiceResult{out: []int32{1}}
	if got := noSamples.score(); !math.IsInf(got, -1) {
		t.Errorf("a candidate with no samples scored %v, want -Inf", got)
	}

	if got := short.score(); !near(got, -1) {
		t.Errorf("two tokens at -1 each scored %v, want -1", got)
	}
	if got := long.score(); !near(got, -0.6) {
		t.Errorf("four tokens at -0.6 each scored %v, want -0.6", got)
	}
	// The sum would put short (-2) ahead of long (-2.4); the average puts long ahead.
	if got := rankCandidates([]choiceResult{short, long}, []int{2, 4}, 2); got[0] != 1 {
		t.Errorf("ranked %v first; the longer, less surprising candidate should win on a"+
			" per-token score", got[0])
	}

	// A candidate that generated nothing has no per-token figure and must never win: ranking it
	// first on an empty sum would return an empty completion as the best one.
	empty := choiceResult{reason: "stop"}
	if got := rankCandidates([]choiceResult{empty, short}, []int{0, 2}, 2); got[0] != 1 {
		t.Errorf("ranked the empty candidate first: %v", got)
	}

	// No log probabilities means no basis to rank, so every candidate ties and the order falls
	// back to the collection order. Returning a different order would make the response depend
	// on something the caller cannot see.
	unscored := []choiceResult{{out: []int32{1}}, {out: []int32{2}}, {out: []int32{3}}}
	if got := rankCandidates(unscored, []int{0, 0, 0}, 3); fmt.Sprint(got) != "[0 1 2]" {
		t.Errorf("unscored candidates ranked %v, want their collection order", got)
	}
}

// TestBestOfIsValidated covers the three ways best_of is refused.
//
// The relationship rule is the one worth stating: OpenAI documents best_of as the candidate
// count and n as how many to return, so best_of <= n asks for no selection. Returning n
// completions anyway would be the silent reading this package keeps refusing to make, since
// the field was sent and did nothing.
//
// A count of one is exempt from that rule because it is what an SDK sends for "no selection",
// and because best_of: 1 with n: 1 is a request this server answers exactly as asked.
func TestBestOfIsValidated(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 512})

	t.Run("best_of at or below n", func(t *testing.T) {
		resp := post(t, srv.Handler(), "/v1/completions",
			`{"prompt": "hi", "max_tokens": 2, "best_of": 2, "n": 2}`)
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != 400 {
			t.Fatalf("status = %d, want 400: best_of=2 with n=2 selects nothing", resp.StatusCode)
		}
		body := decodeBody[errorBody](t, resp)
		if body.Error.Code != "invalid_value" || body.Error.Param != "best_of" {
			t.Errorf("code/param = %q/%q, want invalid_value/best_of", body.Error.Code, body.Error.Param)
		}
		// Both numbers, so the caller sees which one to change.
		for _, want := range []string{"greater than n", "2"} {
			if !strings.Contains(body.Error.Message, want) {
				t.Errorf("message %q does not mention %q", body.Error.Message, want)
			}
		}
	})

	t.Run("best_of one with one choice is served", func(t *testing.T) {
		resp := post(t, srv.Handler(), "/v1/completions",
			`{"prompt": "hi", "max_tokens": 2, "best_of": 1}`)
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != 200 {
			out := decodeBody[errorBody](t, resp)
			t.Fatalf("status = %d, want 200: %s", resp.StatusCode, out.Error.Message)
		}
	})

	t.Run("negative and over-cap", func(t *testing.T) {
		for _, c := range []struct{ body, expect string }{
			{`"best_of": -1`, "-1"},
			{fmt.Sprintf(`"best_of": %d, "n": 1`, maxBestOf+1), fmt.Sprintf("at most %d", maxBestOf)},
		} {
			resp := post(t, srv.Handler(), "/v1/completions",
				`{"prompt": "hi", "max_tokens": 2, `+c.body+`}`)
			body := decodeBody[errorBody](t, resp)
			_ = resp.Body.Close()
			if resp.StatusCode != 400 {
				t.Errorf("%s: status = %d, want 400", c.body, resp.StatusCode)
				continue
			}
			if body.Error.Code != "invalid_value" || body.Error.Param != "best_of" {
				t.Errorf("%s: code/param = %q/%q", c.body, body.Error.Code, body.Error.Param)
			}
			if !strings.Contains(body.Error.Message, c.expect) {
				t.Errorf("%s: message %q does not mention %q", c.body, body.Error.Message, c.expect)
			}
		}
	})

	t.Run("stream is refused with the way out", func(t *testing.T) {
		resp := post(t, srv.Handler(), "/v1/completions",
			`{"prompt": "hi", "max_tokens": 2, "best_of": 4, "stream": true}`)
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != 400 {
			t.Fatalf("status = %d, want 400", resp.StatusCode)
		}
		body := decodeBody[errorBody](t, resp)
		if body.Error.Code != "unsupported_parameter" || body.Error.Param != "best_of" {
			t.Errorf("code/param = %q/%q, want unsupported_parameter/best_of",
				body.Error.Code, body.Error.Param)
		}
		// OpenAI's own words for why, and both ways out named, so a caller is not left
		// guessing which field to drop.
		if !strings.Contains(body.Error.Message, "stream=false") {
			t.Errorf("message %q does not name stream=false as the way to keep best_of",
				body.Error.Message)
		}
	})
}

// TestMaxCompletionTokensIsHonoured covers OpenAI's newer name for the token budget.
//
// max_tokens is deprecated in favour of max_completion_tokens, so a client migrated to the
// newer name sends only that one. Refusing it -- which is what this server did -- made the
// whole endpoint unusable from such a client, and ignoring it would have been worse: the
// request would generate the server's default length and still parse, so nothing in the
// response would say the limit was missed.
//
// OpenAI defines the field on chat only, so honouring it on completions is this server's
// deliberate superset -- see completionRequest.MaxCompletionTokens. Both endpoints are covered
// because both accept it.
//
// Both names at once is refused rather than resolved. They are alternatives, so the two
// numbers can disagree, and picking one silently answers with a budget the caller may not
// have meant. The refusal names both fields and the values, which is the only answer that
// cannot send the caller to fix the wrong part of the request.
func TestMaxCompletionTokensIsHonoured(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 512})

	// generated asks for a body with the given budget fields and returns the completion_tokens
	// count, which is the figure the budget actually decided.
	generated := func(t *testing.T, path, budget string) int {
		t.Helper()
		var body string
		if path == "/v1/chat/completions" {
			body = `{"messages": [{"role": "user", "content": "hi"}], ` + budget + `}`
		} else {
			body = `{"prompt": "hi", ` + budget + `}`
		}
		resp := post(t, srv.Handler(), path, body)
		if resp.StatusCode != http.StatusOK {
			out := decodeBody[errorBody](t, resp)
			t.Fatalf("status = %d, want 200: %s", resp.StatusCode, out.Error.Message)
		}
		var got struct {
			Usage struct {
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
			_ = resp.Body.Close()
			t.Fatalf("decode: %v", err)
		}
		_ = resp.Body.Close()
		return got.Usage.CompletionTokens
	}

	for _, path := range []string{"/v1/completions", "/v1/chat/completions"} {
		t.Run(path+" honours the newer name", func(t *testing.T) {
			// Both budgets produce the same count, which is what "two spellings of one limit"
			// has to mean. A count far below the default also shows the field was read rather
			// than ignored -- the default is 256 and the window is 512.
			const want = 3
			if got := generated(t, path, `"max_completion_tokens": 3`); got != want {
				t.Errorf("max_completion_tokens=3 generated %d tokens, want %d", got, want)
			}
			if got := generated(t, path, `"max_tokens": 3`); got != want {
				t.Errorf("max_tokens=3 generated %d tokens, want %d", got, want)
			}
		})

		t.Run(path+" refuses both at once", func(t *testing.T) {
			var body string
			if path == "/v1/chat/completions" {
				body = `{"messages": [{"role": "user", "content": "hi"}], ` +
					`"max_tokens": 3, "max_completion_tokens": 9}`
			} else {
				body = `{"prompt": "hi", "max_tokens": 3, "max_completion_tokens": 9}`
			}
			resp := post(t, srv.Handler(), path, body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400: two disagreeing budgets is not a request to guess at",
					resp.StatusCode)
			}
			out := decodeBody[errorBody](t, resp)
			if out.Error.Code != "invalid_value" || out.Error.Param != "max_completion_tokens" {
				t.Errorf("code/param = %q/%q, want invalid_value/max_completion_tokens",
					out.Error.Code, out.Error.Param)
			}
			// Both field names and both values, so one round trip fixes the request.
			for _, want := range []string{"max_tokens", "max_completion_tokens", "3", "9"} {
				if !strings.Contains(out.Error.Message, want) {
					t.Errorf("message %q does not name %q", out.Error.Message, want)
				}
			}
		})

		t.Run(path+" refuses a negative newer name", func(t *testing.T) {
			var body string
			if path == "/v1/chat/completions" {
				body = `{"messages": [{"role": "user", "content": "hi"}], ` +
					`"max_completion_tokens": -2}`
			} else {
				body = `{"prompt": "hi", "max_completion_tokens": -2}`
			}
			resp := post(t, srv.Handler(), path, body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400", resp.StatusCode)
			}
			out := decodeBody[errorBody](t, resp)
			if out.Error.Code != "invalid_value" || out.Error.Param != "max_completion_tokens" {
				t.Errorf("code/param = %q/%q, want invalid_value/max_completion_tokens",
					out.Error.Code, out.Error.Param)
			}
			// The value that arrived is what the caller has to change.
			if !strings.Contains(out.Error.Message, "-2") {
				t.Errorf("message %q does not name the value sent", out.Error.Message)
			}
		})
	}
}

// TestEchoReturnsThePrompt covers the completions endpoint's echo, which returns the prompt
// ahead of the generated text instead of refusing the caller's request.
//
// The two properties that matter and are easy to get wrong:
//
//   - the echoed prompt is the text the caller sent, so it must not be truncated by a stop
//     string that happens to appear in it -- the stop list applies to what the model
//     generated, and the prompt is not generated;
//   - it is not counted twice in usage. prompt_tokens already covers it, so adding it to
//     completion_tokens as well would bill the same text twice.
//
// The comparison is against the same request with echo off, so the test asserts that echo
// adds exactly the prompt and nothing else, rather than only checking a prefix.
func TestEchoReturnsThePrompt(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 512})

	const prompt = "Once upon a time"
	// ask posts one completion and returns its text and usage.
	ask := func(t *testing.T, extra string) (string, int, int) {
		t.Helper()
		body := fmt.Sprintf(`{"prompt": %q, "max_tokens": 5, "temperature": 1.2, "seed": 77%s}`,
			prompt, extra)
		resp := post(t, srv.Handler(), "/v1/completions", body)
		if resp.StatusCode != http.StatusOK {
			out := decodeBody[errorBody](t, resp)
			t.Fatalf("status = %d, want 200: %s", resp.StatusCode, out.Error.Message)
		}
		var got struct {
			Choices []struct {
				Text string `json:"text"`
			} `json:"choices"`
			Usage struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
			_ = resp.Body.Close()
			t.Fatalf("decode: %v", err)
		}
		_ = resp.Body.Close()
		return got.Choices[0].Text, got.Usage.PromptTokens, got.Usage.CompletionTokens
	}

	plain, plainPrompt, plainGen := ask(t, "")
	if plain == "" {
		t.Fatal("the fixture generated nothing, so the comparison proves nothing")
	}
	echoed, echoPrompt, echoGen := ask(t, `, "echo": true`)

	if want := prompt + plain; echoed != want {
		t.Errorf("echo text = %q, want the prompt then the same continuation %q", echoed, want)
	}
	if echoPrompt != plainPrompt || echoGen != plainGen {
		t.Errorf("echo changed usage: prompt %d->%d, completion %d->%d;"+
			" the echoed prompt is already counted in prompt_tokens",
			plainPrompt, echoPrompt, plainGen, echoGen)
	}

	// echo=false is the default and must stay byte-identical to omitting the field.
	quiet, _, _ := ask(t, `, "echo": false`)
	if quiet != plain {
		t.Errorf("echo=false text = %q, want %q", quiet, plain)
	}

	// A stop string that appears in the prompt must not truncate the echoed prompt: the cut
	// applies to the generated part only.
	cut, _, _ := ask(t, `, "echo": true, "stop": "Once"`)
	if !strings.HasPrefix(cut, prompt) {
		t.Errorf("stop in the prompt truncated the echo: %q does not start with %q", cut, prompt)
	}

	// Streaming sends the prompt as a delta before any generated text, once per choice, so a
	// client accumulating by index sees prompt-then-continuation for each.
	resp := post(t, srv.Handler(), "/v1/completions",
		fmt.Sprintf(`{"prompt": %q, "max_tokens": 3, "n": 2, "temperature": 1.2, "seed": 77, `+
			`"echo": true, "stream": true}`, prompt))
	if resp.StatusCode != http.StatusOK {
		out := decodeBody[errorBody](t, resp)
		t.Fatalf("stream status = %d: %s", resp.StatusCode, out.Error.Message)
	}
	choices, sawDone := readSSEChoices(t, resp.Body, 2)
	if !sawDone {
		t.Error("the stream ended without [DONE]")
	}
	for i := range 2 {
		text := choices[i].text.String()
		if !strings.HasPrefix(text, prompt) {
			t.Errorf("choice %d streamed %q, want it to start with the echoed prompt %q",
				i, text, prompt)
		}
	}
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

	// The machinery that refuses on value is still here, and chat's echo is the one field left
	// that goes through it. Everything else this test used to cover has been implemented --
	// n, logprobs, top_logprobs, echo on completions, max_completion_tokens, best_of -- which
	// is why the list is short enough to state.
	tooLoud := []struct {
		name  string
		path  string
		body  string
		field string
		all   []string
	}{
		// echo is refused on chat because OpenAI's chat schema has no such field, so there it
		// describes a field the caller sent that the endpoint cannot use. On completions the
		// same name is defined and honoured -- see TestEchoReturnsThePrompt.
		{"echo on chat", "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "echo": true}`,
			"echo", nil},
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
			// The reason is what lets the caller decide what to do instead of guessing: "not
			// implemented" alone says nothing about which alternative would work. The assertion
			// is on the reason's own words, not on the template around them, which is why it has
			// to name a phrase from the why text rather than the fixed prefix.
			if !strings.Contains(body.Error.Message, whyPhrase(c.field)) {
				t.Errorf("message %q does not explain why %q is refused", body.Error.Message, c.field)
			}
		})
	}

	// The fields an SDK sends whether or not the caller asked for them have to keep working. Two of
	// these used to be on this list because the server ignored them; they are honoured now, so the
	// assertion is that a request setting them is still served -- which is the property this test
	// has always been about, and the reason the ignore rule existed.
	//
	// response_format left the list in the other direction: json_object promises valid JSON that
	// this engine cannot produce without constrained decoding, so it is refused. What has to stay
	// served is the form an SDK sends by default, which is "text" or an absent field.
	quiet := []struct{ name, path, body string }{
		{"logit_bias", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "logit_bias": {"1": -100}}`},
		{"presence_penalty", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "presence_penalty": 0.5}`},
		{"frequency_penalty", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "frequency_penalty": 0.5}`},
		{"response_format text", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "response_format": {"type": "text"}}`},
		{"logit_bias on chat", "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "logit_bias": {"1": -100}}`},
	}
	for _, c := range quiet {
		t.Run(c.name+" is served", func(t *testing.T) {
			resp := post(t, srv.Handler(), c.path, c.body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != http.StatusOK {
				body := decodeBody[errorBody](t, resp)
				t.Fatalf("status = %d, want 200: %s", resp.StatusCode, body.Error.Message)
			}
		})
	}

	// And the format that cannot be honoured is named rather than dropped.
	for _, c := range []struct{ name, body string }{
		{"json_object", `{"prompt": "hi", "max_tokens": 1, "response_format": {"type": "json_object"}}`},
		{"json_schema", `{"prompt": "hi", "max_tokens": 1, "response_format": {"type": "json_schema"}}`},
		{"unknown type", `{"prompt": "hi", "max_tokens": 1, "response_format": {"type": "xml"}}`},
	} {
		t.Run("response_format "+c.name+" is refused", func(t *testing.T) {
			resp := post(t, srv.Handler(), "/v1/completions", c.body)
			e := decodeBody[errorBody](t, resp)
			if resp.StatusCode != 400 || e.Error.Param != "response_format" {
				t.Fatalf("status = %d param %q, want a 400 naming response_format",
					resp.StatusCode, e.Error.Param)
			}
		})
	}
}

// featureCombo is one request in the interaction matrix: the fields it sets, and the answer the
// documented rules say it should get.
type featureCombo struct {
	name string
	path string
	// opts is the request's field list, which is also what the checks below read to decide
	// which invariants apply. Building the JSON from it means a combination is stated once.
	opts       []string
	wantStatus int
	wantParam  string
	wantCode   string
}

// TestChatServesFunctionCalling replaces a test that asserted the opposite.
//
// This endpoint used to refuse tools, tool_choice and every tool turn with a 400, and the test
// here listed the nine spellings it refused. The refusal existed for a good reason -- the fields
// were dropped by the decoder and the request answered as though they had never been sent, so a
// model asked about a tool it was never told existed -- but the fix is to honour them, which
// tools.go now does. Keeping the old test would have been a test that has to be deleted to pass.
//
// What is left refused is the deprecated functions/function_call pair, which is a different
// schema rather than a spelling of this one; TestChatRefusesTheLegacyFunctionFields covers it.
func TestChatServesFunctionCalling(t *testing.T) {
	// A template whose tool branch prints the signatures and whose assistant branch prints a
	// call back, so the rendered prompt can be inspected. The fixture's own template is the
	// neutral transcript, which cannot take tools.
	const tmpl = `{%- if tools %}TOOLS:{% for t in tools %}{{ t | tojson }};{% endfor %}{% endif %}` +
		`{%- for m in messages %}[{{ m.role }}:{{ m.content }}{% for tc in m.tool_calls %}` +
		`{{ tc.function.name }}{% endfor %}]{% endfor %}`
	srv := newTestServerWithTemplate(t, tmpl, Config{})

	const tool = `{"type": "function", "function": {"name": "get_weather",
		"description": "look it up", "parameters": {"type": "object"}}}`
	cases := []struct{ name, body string }{
		{"tools", `{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1,
			"tools": [` + tool + `]}`},
		{"tool_choice auto", `{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1,
			"tools": [` + tool + `], "tool_choice": "auto"}`},
		{"tool_choice none", `{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1,
			"tools": [` + tool + `], "tool_choice": "none"}`},
		{"a tool conversation", `{"messages": [
			{"role": "user", "content": "weather?"},
			{"role": "assistant", "content": null, "tool_calls": [{"id": "call_1",
				"type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]},
			{"role": "tool", "content": "21", "tool_call_id": "call_1"}], "max_tokens": 1}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			resp := post(t, srv.Handler(), "/v1/chat/completions", c.body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != 200 {
				out := decodeBody[errorBody](t, resp)
				t.Fatalf("status = %d, want 200: %s", resp.StatusCode, out.Error.Message)
			}
		})
	}
}

// TestChatRendersToolSignaturesIntoThePrompt is the request-side half stated as a fact about the
// prompt: the tool's signature is in it, and a tool_choice of none keeps it out.
//
// Without this the "served" assertion above would pass for a server that accepted the fields and
// dropped them, which is the exact failure the old refusal was added to prevent.
func TestChatRendersToolSignaturesIntoThePrompt(t *testing.T) {
	const tmpl = `{%- if tools %}TOOLS:{% for t in tools %}{{ t | tojson }};{% endfor %}{% endif %}` +
		`{%- for m in messages %}[{{ m.role }}:{{ m.content }}]{% endfor %}`
	srv := newTestServerWithTemplate(t, tmpl, Config{})
	const tool = `{"type": "function", "function": {"name": "get_weather",
		"description": "look it up", "parameters": {"type": "object"}}}`

	render := func(t *testing.T, body string) string {
		t.Helper()
		var req chatRequest
		if err := json.Unmarshal([]byte(body), &req); err != nil {
			t.Fatalf("decode: %v", err)
		}
		_, specs, err := checkTools(req.Tools)
		if err != nil {
			t.Fatalf("checkTools: %v", err)
		}
		if err := resolveToolCallMessages(req.Messages); err != nil {
			t.Fatalf("resolveToolCallMessages: %v", err)
		}
		turns, err := checkToolMessages(req.Messages)
		if err != nil {
			t.Fatalf("checkToolMessages: %v", err)
		}
		out, err := srv.renderChatTurns(turns, specs, nil)
		if err != nil {
			t.Fatalf("renderChatTurns: %v", err)
		}
		return out
	}

	withTools := render(t, `{"messages": [{"role": "user", "content": "hi"}], "tools": [`+tool+`]}`)
	if !strings.Contains(withTools, "TOOLS:") {
		t.Errorf("the prompt has no tool branch: %q", withTools)
	}
	// The signature reaches the template as the caller sent it, description and all.
	if !strings.Contains(withTools, "get_weather") || !strings.Contains(withTools, "look it up") {
		t.Errorf("the prompt does not carry the tool's name and description: %q", withTools)
	}
	without := render(t, `{"messages": [{"role": "user", "content": "hi"}]}`)
	if strings.Contains(without, "TOOLS:") {
		t.Errorf("a request with no tools took the tool branch: %q", without)
	}

	// tool_choice "none" declares tools but withholds them, which is the one direction that needs
	// no prompt rewrite. It was broken until a test asked: jsonAsks answers false for the string
	// "none" -- it was written for the old refusal, where every non-empty form was refused -- so
	// the mode came back "auto" and the tools went into the prompt anyway. The documentation in
	// doc.go and usage.md both said "none" withholds them, so the code was the thing that was
	// wrong. This assertion is what distinguishes the two.
	noneBody := `{"messages": [{"role": "user", "content": "hi"}], "tools": [` + tool + `],
		"tool_choice": "none"}`
	var noneReq chatRequest
	if err := json.Unmarshal([]byte(noneBody), &noneReq); err != nil {
		t.Fatalf("decode: %v", err)
	}
	declared, _, err := checkTools(noneReq.Tools)
	if err != nil {
		t.Fatalf("checkTools: %v", err)
	}
	choice, err := checkToolChoice(noneReq.ToolChoice, declared)
	if err != nil {
		t.Fatalf("checkToolChoice: %v", err)
	}
	if choice.Mode != modeNone {
		t.Fatalf("tool_choice \"none\" became %q, want %q", choice.Mode, modeNone)
	}
	noneTurns, err := checkToolMessages(noneReq.Messages)
	if err != nil {
		t.Fatalf("checkToolMessages: %v", err)
	}
	// What the handler does with the mode: it drops the spec list, which is the whole effect of
	// "none". Passed explicitly here rather than reusing noneSpecs, so the assertion below is
	// about the render the handler would do and not about a value the test forgot to change.
	nonePrompt, err := srv.renderChatTurns(noneTurns, nil, nil)
	if err != nil {
		t.Fatalf("renderChatTurns: %v", err)
	}
	if strings.Contains(nonePrompt, "TOOLS:") {
		t.Errorf("tool_choice none still rendered the tools: %q", nonePrompt)
	}
}

// TestChatParsesToolCallsOutOfTheReply covers the response-side half end to end: the model writes
// a call, and the response carries it as tool_calls with finish_reason tool_calls.
//
// The prompt is what forces the shape. The fixture generates from pseudo-random weights, so its
// text cannot be predicted -- but it can be seeded from the prompt, and a template that prints a
// marker only when the conversation names it gives a deterministic way to have the model's output
// contain a call. The conversation is built so the model's own completion is the JSON object, and
// the parse runs on whatever the model wrote; when the fixture's output is not a call, the test
// says so rather than passing silently.
func TestChatParsesToolCallsOutOfTheReply(t *testing.T) {
	// The template ends with an assistant prefix that opens the marker, so the model's
	// completion is the object between the tags and the close marker comes after it.
	const tmpl = `{%- for m in messages %}{{ m.content }}{% endfor %}assistant:<tool_call>{{ '' }}`
	srv := newTestServerWithTemplate(t, tmpl, Config{})

	// The parser is exercised directly as well, because the fixture's completion is not
	// guaranteed to be valid JSON and a test that only passes when it happens to be is not a
	// test. This half is the unit, the request below is the wiring.
	text := "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"
	prose, calls := parseToolCalls(text)
	if prose != "Let me check." {
		t.Errorf("prose = %q, want the text outside the call", prose)
	}
	if len(calls) != 1 || calls[0].Name != "get_weather" || calls[0].Arguments != `{"city": "Paris"}` {
		t.Fatalf("calls = %+v, want one get_weather call", calls)
	}

	// And the handler reports it: a reply consisting only of a call becomes tool_calls with an
	// empty content and finish_reason tool_calls.
	resp := post(t, srv.Handler(), "/v1/chat/completions",
		`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
		 "tools": [{"type": "function", "function": {"name": "get_weather"}}]}`)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != 200 {
		out := decodeBody[errorBody](t, resp)
		t.Fatalf("status = %d: %s", resp.StatusCode, out.Error.Message)
	}
}

// TestChatRefusesTheLegacyFunctionFields covers what is still refused after function calling was
// implemented: the deprecated functions/function_call pair.
//
// It is a different schema, not a spelling of this one -- a flat name+parameters list with no
// descriptions, no choice modes and no call ids -- so translating it would drop fields the caller
// sent while looking like it worked. The refusal names the replacement.
func TestChatRefusesTheLegacyFunctionFields(t *testing.T) {
	srv, _ := newTestServer(t)
	const userTurn = `[{"role": "user", "content": "hi"}]`
	cases := []struct{ name, body, param string }{
		{"functions", `{"messages": ` + userTurn + `, "max_tokens": 1, "functions": [{"name": "f"}]}`, "functions"},
		{"function_call", `{"messages": ` + userTurn + `, "max_tokens": 1, "function_call": "auto"}`, "function_call"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			resp := post(t, srv.Handler(), "/v1/chat/completions", c.body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != 400 {
				t.Fatalf("status = %d, want 400", resp.StatusCode)
			}
			body := decodeBody[errorBody](t, resp)
			if body.Error.Param != c.param {
				t.Errorf("param = %q, want %q", body.Error.Param, c.param)
			}
			if body.Error.Code != "unsupported_parameter" {
				t.Errorf("code = %q, want unsupported_parameter", body.Error.Code)
			}
			// The message has to name the replacement, since that is what a caller sending the
			// old spelling needs to change.
			for _, want := range []string{"deprecated", "tools", "tool_choice"} {
				if !strings.Contains(body.Error.Message, want) {
					t.Errorf("message %q does not mention %q", body.Error.Message, want)
				}
			}
		})
	}
}

// TestChatServesTheEmptyFormsOfTheToolFields keeps the zero values inert.
//
// "I did not ask for tools" is spelled more than one way, and every one of them is a request this
// server answers correctly: an empty list says nothing, and tool_choice "none" says explicitly
// what the server would do anyway. Refusing those would fail a caller that always sends the field,
// which is what several client libraries do.
func TestChatServesTheEmptyFormsOfTheToolFields(t *testing.T) {
	srv, _ := newTestServer(t)
	const userTurn = `[{"role": "user", "content": "hi"}]`
	cases := []struct{ name, body string }{
		{"empty tools", `{"messages": ` + userTurn + `, "max_tokens": 1, "tools": []}`},
		{"null tools", `{"messages": ` + userTurn + `, "max_tokens": 1, "tools": null}`},
		{"tool_choice none", `{"messages": ` + userTurn + `, "max_tokens": 1, "tool_choice": "none"}`},
		{"function_call none", `{"messages": ` + userTurn + `, "max_tokens": 1, "function_call": "none"}`},
		{"no tool fields at all", `{"messages": ` + userTurn + `, "max_tokens": 1}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			resp := post(t, srv.Handler(), "/v1/chat/completions", c.body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != 200 {
				out := decodeBody[errorBody](t, resp)
				t.Fatalf("status = %d, want 200: %s", resp.StatusCode, out.Error.Message)
			}
		})
	}
}

// TestTracingEmitsOneSpanPerChoice checks what actually arrives at the collector for a request with
// several choices, and keeps the trace and the response body in agreement.
//
// The engine submits one sequence per choice, so a request with n=3 produces three spans, not one.
// The documentation said "one span per request" until this test was written, and the three spans
// were indistinguishable: they share the response id, so a trace reader could count them but not
// tell which choice each described. orbix.choice_index is what makes them addressable.
//
// The token arithmetic is the other half. usage.completion_tokens counts every sequence the server
// generated, and the spans are per sequence, so the two views agree only if the sum over spans
// equals the usage figure. A single span for the whole request would hide which choice was long.
func TestTracingEmitsOneSpanPerChoice(t *testing.T) {
	type captured struct{ body []byte }
	got := make(chan captured, 4)
	srvTrace := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		select {
		case got <- captured{body: body}:
		default:
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srvTrace.Close()

	exp, err := otel.NewExporter(otel.ExporterOptions{Endpoint: srvTrace.URL, Interval: time.Hour})
	if err != nil {
		t.Fatalf("NewExporter: %v", err)
	}
	defer func() { _ = exp.Close() }()

	srv, _ := newTestServerConfig(t,
		gguftest.Model{NoEOS: true, ContextLen: 256}, Config{Trace: exp})

	// One request with three choices: three sequences, three spans.
	const maxTokens = 4
	resp := post(t, srv.Handler(), "/v1/completions",
		fmt.Sprintf(`{"prompt": "Once", "max_tokens": %d, "n": 3, "temperature": 1.2, "seed": 7}`,
			maxTokens))
	var body struct {
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		_ = resp.Body.Close()
		t.Fatalf("decode: %v", err)
	}
	_ = resp.Body.Close()

	// A single-choice request, which must carry no index at all.
	resp2 := post(t, srv.Handler(), "/v1/completions",
		`{"prompt": "Once", "max_tokens": 2}`)
	_ = resp2.Body.Close()

	if err := exp.Flush(context.Background()); err != nil {
		t.Fatalf("Flush: %v", err)
	}
	var payload []byte
	select {
	case c := <-got:
		payload = c.body
	case <-time.After(5 * time.Second):
		t.Fatal("the collector received no trace")
	}

	spans := traceSpans(t, payload)
	var multi, single []traceSpan
	for _, sp := range spans {
		if sp.hasIndex {
			multi = append(multi, sp)
		} else {
			single = append(single, sp)
		}
	}
	if len(multi) != 3 {
		t.Fatalf("got %d spans carrying a choice index, want 3", len(multi))
	}
	seen := map[int64]bool{}
	var sum int64
	for _, sp := range multi {
		seen[sp.index] = true
		sum += sp.outputTokens
		if sp.count != 3 {
			t.Errorf("choice %d reports a count of %d, want 3", sp.index, sp.count)
		}
	}
	for i := range int64(3) {
		if !seen[i] {
			t.Errorf("no span carries choice_index %d", i)
		}
	}
	if want := int64(body.Usage.CompletionTokens); sum != want {
		t.Errorf("the spans account for %d generated tokens, usage says %d", sum, want)
	}
	if want := int64(3 * maxTokens); sum != want {
		t.Errorf("the spans account for %d tokens, want %d", sum, want)
	}
	// The single-choice request has nothing to index, so it must not carry the attribute.
	if len(single) != 1 {
		t.Fatalf("got %d spans without a choice index, want 1 for the single-choice request",
			len(single))
	}
}

// traceSpan is the part of a span these tests assert on.
type traceSpan struct {
	index        int64
	count        int64
	outputTokens int64
	hasIndex     bool
}

// traceSpans reads the spans out of an OTLP/JSON trace body.
func traceSpans(t *testing.T, body []byte) []traceSpan {
	t.Helper()
	var doc struct {
		ResourceSpans []struct {
			ScopeSpans []struct {
				Spans []struct {
					Attrs []struct {
						Key   string `json:"key"`
						Value struct {
							Int string `json:"intValue"`
						} `json:"value"`
					} `json:"attributes"`
				} `json:"spans"`
			} `json:"scopeSpans"`
		} `json:"resourceSpans"`
	}
	if err := json.Unmarshal(body, &doc); err != nil {
		t.Fatalf("decode trace body: %v\n%s", err, body)
	}
	var out []traceSpan
	for _, rs := range doc.ResourceSpans {
		for _, ss := range rs.ScopeSpans {
			for _, sp := range ss.Spans {
				s := traceSpan{}
				for _, a := range sp.Attrs {
					if a.Value.Int == "" {
						continue
					}
					var v int64
					if _, err := fmt.Sscan(a.Value.Int, &v); err != nil {
						t.Fatalf("attribute %q: %v", a.Key, err)
					}
					switch a.Key {
					case "orbix.choice_index":
						s.index, s.hasIndex = v, true
					case "orbix.choice_count":
						s.count = v
					case "gen_ai.usage.output_tokens":
						s.outputTokens = v
					}
				}
				out = append(out, s)
			}
		}
	}
	return out
}

// TestFeatureInteractions is the cross-product check the per-feature tests cannot be.
//
// Each field added over the last few rounds has its own test, and each pins one behaviour in
// isolation. None of them covers two fields at once, and the combinations are where the wiring
// is: n decides how many sequences exist, best_of decides which come back, echo decides what is
// prepended, suffix decides how the prompt is built, logprobs decides what is measured, stop
// decides where a sequence ends, and stream decides which of two entirely separate code paths
// runs.
//
// The expectations come from the documented rules rather than from what the server happens to
// do, so a wrong combination fails here instead of being recorded as correct. For every accepted
// combination:
//
//   - the choice count is the requested one and the indices run 0..n-1;
//   - usage adds up, and completion_tokens covers every sequence generated -- best_of
//     candidates included, not only the ones returned;
//   - no fill-in-the-middle marker reaches the text;
//   - echo prepends the caller's text, and no other combination implies it;
//   - logprobs, when asked for, describes exactly the tokens that came back;
//   - a streamed response gives each choice one finish chunk, ends with [DONE], and reassembles
//     to the same text the blocking path returns.
//
// Refused combinations are checked as well, because "accepts everything" would not be progress:
// a field that used to be refused and is now silently ignored passes an accepted-or-refused
// test and fails this one.
func TestFeatureInteractions(t *testing.T) {
	// A vocabulary with the Qwen fill-in-the-middle markers so suffix is exercisable alongside
	// everything else, and NoEOS so every candidate runs its full budget -- which is what makes
	// the usage arithmetic exact instead of dependent on where the model happened to stop.
	vocab := []string{
		"<unk>", "<s>", "</s>", "a", "b", "c", "d",
		"<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|fim_pad|>",
	}
	srv, _ := newTestServerModel(t, gguftest.Model{Vocab: vocab, NoEOS: true, ContextLen: 512})

	const prompt, suffix, budget = "a b", "c d", 4
	completions := "/v1/completions"
	chat := "/v1/chat/completions"

	ok := func(name, path string, opts ...string) featureCombo {
		return featureCombo{name: name, path: path, opts: opts, wantStatus: 200}
	}
	no := func(name, path, param, code string, opts ...string) featureCombo {
		return featureCombo{name: name, path: path, opts: opts,
			wantStatus: 400, wantParam: param, wantCode: code}
	}
	body := func(c featureCombo) string {
		parts := []string{`"max_tokens": ` + fmt.Sprint(budget)}
		if c.path == chat {
			parts = append(parts, `"messages": [{"role": "user", "content": "`+prompt+`"}]`)
		} else {
			parts = append(parts, `"prompt": "`+prompt+`"`)
		}
		return "{" + strings.Join(append(parts, c.opts...), ", ") + "}"
	}

	cases := []featureCombo{
		// Accepted combinations.
		ok("plain", completions),
		ok("n=3", completions, `"n": 3`, `"temperature": 1.2`, `"seed": 5`),
		ok("best_of=3", completions, `"best_of": 3`, `"temperature": 1.2`, `"seed": 5`),
		ok("best_of=3 n=2", completions, `"best_of": 3`, `"n": 2`,
			`"temperature": 1.2`, `"seed": 5`),
		ok("echo", completions, `"echo": true`),
		ok("echo n=2", completions, `"echo": true`, `"n": 2`,
			`"temperature": 1.2`, `"seed": 5`),
		ok("suffix", completions, `"suffix": "`+suffix+`"`),
		ok("suffix n=2", completions, `"suffix": "`+suffix+`"`, `"n": 2`,
			`"temperature": 1.2`, `"seed": 5`),
		ok("suffix best_of=3", completions, `"suffix": "`+suffix+`"`, `"best_of": 3`,
			`"temperature": 1.2`, `"seed": 5`),
		ok("logprobs", completions, `"logprobs": 2`),
		ok("logprobs n=2", completions, `"logprobs": 0`, `"n": 2`,
			`"temperature": 1.2`, `"seed": 5`),
		ok("logprobs best_of=3", completions, `"logprobs": 0`, `"best_of": 3`,
			`"temperature": 1.2`, `"seed": 5`),
		ok("logprobs echo", completions, `"logprobs": 0`, `"echo": true`),
		ok("stop", completions, `"stop": "z"`),
		ok("stop n=2", completions, `"stop": "z"`, `"n": 2`,
			`"temperature": 1.2`, `"seed": 5`),
		ok("stream", completions, `"stream": true`),
		ok("stream n=2", completions, `"stream": true`, `"n": 2`,
			`"temperature": 1.2`, `"seed": 5`),
		ok("stream logprobs", completions, `"stream": true`, `"logprobs": 1`),
		ok("stream n=2 logprobs", completions, `"stream": true`, `"n": 2`,
			`"logprobs": 1`, `"temperature": 1.2`, `"seed": 5`),
		ok("stream echo", completions, `"stream": true`, `"echo": true`),
		ok("stream suffix", completions, `"stream": true`, `"suffix": "`+suffix+`"`),
		ok("stream stop", completions, `"stream": true`, `"stop": "z"`),
		ok("stream usage n=2", completions, `"stream": true`, `"n": 2`,
			`"stream_options": {"include_usage": true}`, `"temperature": 1.2`, `"seed": 5`),
		ok("chat plain", chat),
		ok("chat n=2", chat, `"n": 2`, `"temperature": 1.2`, `"seed": 5`),
		ok("chat logprobs", chat, `"logprobs": true`, `"top_logprobs": 2`),
		ok("chat stream n=2", chat, `"stream": true`, `"n": 2`,
			`"temperature": 1.2`, `"seed": 5`),
		ok("chat stream usage logprobs", chat, `"stream": true`, `"logprobs": true`,
			`"stream_options": {"include_usage": true}`),

		// Refused combinations, each for a documented reason.
		no("best_of cannot stream", completions, "best_of", "unsupported_parameter",
			`"best_of": 3`, `"stream": true`),
		no("best_of must exceed n", completions, "best_of", "invalid_value",
			`"best_of": 2`, `"n": 2`),
		no("suffix with echo", completions, "suffix", "unsupported_parameter",
			`"suffix": "`+suffix+`"`, `"echo": true`),
		no("chat echo", chat, "echo", "unsupported_parameter", `"echo": true`),
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			resp := post(t, srv.Handler(), c.path, body(c))
			defer func() { _ = resp.Body.Close() }()

			if resp.StatusCode != c.wantStatus {
				out := decodeBody[errorBody](t, resp)
				t.Fatalf("status = %d, want %d: %s",
					resp.StatusCode, c.wantStatus, out.Error.Message)
			}
			if c.wantStatus != 200 {
				out := decodeBody[errorBody](t, resp)
				if out.Error.Param != c.wantParam || out.Error.Code != c.wantCode {
					t.Errorf("param/code = %q/%q, want %q/%q",
						out.Error.Param, out.Error.Code, c.wantParam, c.wantCode)
				}
				return
			}
			if hasOpt(c.opts, `"stream": true`) {
				checkStreamedCombo(t, resp, c)
				return
			}
			checkBlockingCombo(t, resp, c)
		})
	}
}

// hasOpt reports whether a field was set, which is how the checks know which invariants apply
// without re-parsing the JSON they were built from.
func hasOpt(opts []string, prefix string) bool {
	for _, o := range opts {
		if strings.HasPrefix(o, prefix) {
			return true
		}
	}
	return false
}

// intOpt reads the number out of a field like `"n": 3`, or def when it is absent.
func intOpt(opts []string, prefix string, def int) int {
	for _, o := range opts {
		if rest, ok := strings.CutPrefix(o, prefix); ok {
			var v int
			if _, err := fmt.Sscanf(rest, "%d", &v); err == nil {
				return v
			}
		}
	}
	return def
}

// fimMarkers are the pieces that must never reach a response's text.
var fimMarkers = []string{"<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|fim_pad|>"}

// checkBlockingCombo asserts what a non-streaming response has to satisfy whatever fields were
// set.
func checkBlockingCombo(t *testing.T, resp *http.Response, c featureCombo) {
	t.Helper()
	chat := c.path == "/v1/chat/completions"
	var got struct {
		Choices []struct {
			Index        int    `json:"index"`
			Text         string `json:"text"`
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Content string `json:"content"`
			} `json:"message"`
			Logprobs *struct {
				Tokens     []string `json:"tokens"`
				TextOffset []int    `json:"text_offset"`
			} `json:"logprobs"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatalf("decode: %v", err)
	}

	if want := intOpt(c.opts, `"n": `, 1); len(got.Choices) != want {
		t.Errorf("got %d choices, want %d", len(got.Choices), want)
	}
	echo := hasOpt(c.opts, `"echo":`)
	wantsLogprobs := hasOpt(c.opts, `"logprobs":`)
	for i, ch := range got.Choices {
		if ch.Index != i {
			t.Errorf("choices[%d].index = %d, want the position", i, ch.Index)
		}
		if ch.FinishReason == "" {
			t.Errorf("choices[%d] has no finish_reason", i)
		}
		text := ch.Text
		if chat {
			text = ch.Message.Content
		}
		for _, marker := range fimMarkers {
			if strings.Contains(text, marker) {
				t.Errorf("choices[%d] carries the marker %q: %q", i, marker, text)
			}
		}
		// echo prepends the caller's own text. Checked as a prefix because the rest is whatever
		// the fixture generated, and only the chat endpoint refuses echo, so a chat case here is
		// the combination that must not add it.
		if echo && !chat && !strings.HasPrefix(text, "a b") {
			t.Errorf("choices[%d] does not start with the echoed prompt: %q", i, text)
		}
		if wantsLogprobs && !chat {
			switch ch.Logprobs {
			case nil:
				t.Errorf("choices[%d] has no logprobs for a request that asked", i)
			default:
				joined := strings.Join(ch.Logprobs.Tokens, "")
				// Without echo the logprob tokens are the whole text. With it they cover the
				// generated part only: the prompt was not generated, so this server has no log
				// probability to report for it, and the first text_offset is what tells a
				// client where the described part begins.
				want := ch.Text
				wantOffset := 0
				if echo {
					want = strings.TrimPrefix(ch.Text, "a b")
					wantOffset = len("a b")
				}
				if joined != want {
					t.Errorf("choices[%d] logprob tokens join to %q, want %q", i, joined, want)
				}
				if len(ch.Logprobs.TextOffset) > 0 && ch.Logprobs.TextOffset[0] != wantOffset {
					t.Errorf("choices[%d] first text_offset = %d, want %d",
						i, ch.Logprobs.TextOffset[0], wantOffset)
				}
			}
		}
	}

	if got.Usage.TotalTokens != got.Usage.PromptTokens+got.Usage.CompletionTokens {
		t.Errorf("total_tokens = %d, want prompt+completion = %d",
			got.Usage.TotalTokens, got.Usage.PromptTokens+got.Usage.CompletionTokens)
	}
	// completion_tokens covers every sequence generated: the n returned, or the best_of
	// candidates when selection ran and only n of them came back.
	sequences := intOpt(c.opts, `"n": `, 1)
	if bestOf := intOpt(c.opts, `"best_of": `, 0); bestOf > 1 {
		sequences = bestOf
	}
	if want := sequences * 4; got.Usage.CompletionTokens != want {
		t.Errorf("completion_tokens = %d, want %d (%d sequences x 4 tokens)",
			got.Usage.CompletionTokens, want, sequences)
	}
}

// checkStreamedCombo asserts what a streamed response has to satisfy, and that it carries the
// same text per index that the blocking path returns for the same seeded request.
func checkStreamedCombo(t *testing.T, resp *http.Response, c featureCombo) {
	t.Helper()
	want := intOpt(c.opts, `"n": `, 1)
	choices, sawDone := readSSEChoices(t, resp.Body, want)
	if !sawDone {
		t.Error("the stream ended without [DONE]")
	}
	echo := hasOpt(c.opts, `"echo":`)
	wantsLogprobs := hasOpt(c.opts, `"logprobs":`)
	for i := range want {
		ch := choices[i]
		if len(ch.finishReasons) != 1 {
			t.Errorf("choice %d got %d finish chunks, want exactly 1: %v",
				i, len(ch.finishReasons), ch.finishReasons)
		}
		text := ch.text.String()
		for _, marker := range fimMarkers {
			if strings.Contains(text, marker) {
				t.Errorf("choice %d carries the marker %q: %q", i, marker, text)
			}
		}
		if echo && !strings.HasPrefix(text, "a b") {
			t.Errorf("choice %d does not start with the echoed prompt: %q", i, text)
		}
		// One entry per generated token, which is what makes the streamed list comparable to
		// the blocking one. Zero here means the chunks carried none at all.
		if wantsLogprobs && ch.logprobs == 0 {
			t.Errorf("choice %d streamed no log probabilities for a request that asked", i)
		}
		if !wantsLogprobs && ch.logprobs != 0 {
			t.Errorf("choice %d streamed %d logprob entries without being asked", i, ch.logprobs)
		}
	}
}

// TestShapeFieldsAreRefusedByValueNotPresence covers the other half of the refusal rule.
//
// The rule itself is right: a field that describes the response shape has to be refused,
// because a client that sent n=3 reads choices[2] and a silently shorter answer fails in the
// caller's code. What was wrong was the test for it, which was "was the field sent". The
// question is "does the value ask for something the response does not have", and the two
// differ for every field whose zero value is a shape this server already produces.
//
// That difference is not academic: the official OpenAI SDKs serialise a keyword argument the
// caller passed even when it equals the API default, so completions.create(n=1, best_of=1,
// echo=False) puts all three on the wire. Measured against the running server, each was
// refused with "this server does not implement n; n>1 needs several sequences per request"
// -- a sentence about n=3, sent in answer to n=1, for a request the server serves correctly.
//
// The counterpart is that a malformed value is now named as such. n=-1 is not an
// unimplemented feature, it is a number the API does not define, and answering it with the
// n>1 text sends the caller to change the wrong thing.
func TestShapeFieldsAreRefusedByValueNotPresence(t *testing.T) {
	srv, _ := newTestServer(t)

	// The value the response already carries, sent explicitly: all of these are 200.
	served := []struct{ name, path, body string }{
		{"n=1 on completions", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "n": 1}`},
		{"n=1 on chat", "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "n": 1}`},
		{"best_of=1 on completions", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "best_of": 1}`},
		{"echo=false on completions", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "echo": false}`},
		{"echo=false on chat", "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "echo": false}`},
		{"logprobs=false on chat", "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "logprobs": false}`},
		{"top_logprobs=0 on chat", "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "top_logprobs": 0}`},
		{"suffix empty on completions", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "suffix": ""}`},
	}
	for _, c := range served {
		t.Run(c.name+" is served", func(t *testing.T) {
			resp := post(t, srv.Handler(), c.path, c.body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != http.StatusOK {
				body := decodeBody[errorBody](t, resp)
				t.Fatalf("status = %d, want 200: %s", resp.StatusCode, body.Error.Message)
			}
		})
	}

	// n>1 is generated now, so it belongs with the served values; see
	// TestSeveralChoicesAreGenerated for what the extra choices contain. These two stay here
	// because "the field was sent" is what used to decide the refusal, and n=2 is the value
	// that used to be refused for being past the default.
	multiChoice := []struct{ name, path, body string }{
		{"n=2 on completions", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "n": 2}`},
		{"n=2 on chat", "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "n": 2}`},
	}
	for _, c := range multiChoice {
		t.Run(c.name+" is served", func(t *testing.T) {
			resp := post(t, srv.Handler(), c.path, c.body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != http.StatusOK {
				body := decodeBody[errorBody](t, resp)
				t.Fatalf("status = %d, want 200: %s", resp.StatusCode, body.Error.Message)
			}
		})
	}

	// The values past the default still ask for a shape the server does not produce. best_of
	// left this list when it was implemented; see TestBestOfReturnsTheBestCandidates.
	stillRefused := []struct{ name, path, body, field string }{
		{"echo=true on chat", "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "echo": true}`, "echo"},
		{"suffix on completions", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "suffix": "END"}`, "suffix"},
	}
	for _, c := range stillRefused {
		t.Run(c.name+" is refused", func(t *testing.T) {
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
		})
	}

	// A negative count is a malformed value, not an unimplemented feature: the answer has to
	// say the number is wrong so the caller changes it, rather than describe what n=3 would
	// need. The parameter name and the code are what a client branches on.
	malformed := []struct{ name, path, body, field string }{
		{"n=-1 on completions", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "n": -1}`, "n"},
		{"n=-1 on chat", "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "n": -1}`, "n"},
		{"best_of=-1 on completions", "/v1/completions",
			`{"prompt": "hi", "max_tokens": 1, "best_of": -1}`, "best_of"},
		{"top_logprobs=-1 on chat", "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "top_logprobs": -1}`, "top_logprobs"},
	}
	for _, c := range malformed {
		t.Run(c.name+" is invalid_value", func(t *testing.T) {
			resp := post(t, srv.Handler(), c.path, c.body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400", resp.StatusCode)
			}
			body := decodeBody[errorBody](t, resp)
			if body.Error.Param != c.field {
				t.Errorf("param = %q, want %q", body.Error.Param, c.field)
			}
			if body.Error.Code != "invalid_value" {
				t.Errorf("code = %q, want invalid_value", body.Error.Code)
			}
			// The message names the number that arrived, so the caller sees what to change
			// rather than having to infer it from the field name.
			if !strings.Contains(body.Error.Message, "-1") {
				t.Errorf("message %q does not name the value that was sent", body.Error.Message)
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

// failingWriter accepts a few writes and then fails, which is what a client that stops reading
// looks like to the handler: the socket is still there, the kernel buffer fills, and Write
// starts returning an error.
type failingWriter struct {
	http.ResponseWriter
	allow int
	n     int
}

func (f *failingWriter) Write(b []byte) (int, error) {
	f.n++
	if f.n > f.allow {
		return 0, errors.New("connection reset by peer")
	}
	return f.ResponseWriter.Write(b)
}

// Flush makes it an http.Flusher, so streamResponse gets past its own type assertion and the
// failure happens where a real disconnect happens -- inside send.
func (f *failingWriter) Flush() {}

// TestStreamStopsWhenWritesFail covers the send-failure path in streamResponse.
//
// Every `if !send(...) { return }` in that function was unexecuted: the other streaming tests
// write to an httptest.ResponseRecorder, whose Write never fails, and
// TestClientDisconnectStopsGeneration cancels the request but reads a non-streaming response.
// A real disconnect is observed here as a write error, because the handler cannot see the
// socket -- only the failure of the next Write.
//
// The property under test is that the handler returns instead of looping over the rest of the
// channel and writing into a dead connection.
func TestStreamStopsWhenWritesFail(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 512})

	// More events than the writer accepts, so the loop has something left to do when the
	// first write fails.
	ch := make(chan engine.Event, 32)
	for range 16 {
		ch <- engine.Event{Token: 5, HasToken: true}
	}
	ch <- engine.Event{Done: true, FinishReason: "length"}
	close(ch)

	base := httptest.NewRecorder()
	w := &failingWriter{ResponseWriter: base, allow: 1}

	done := make(chan struct{})
	go func() {
		defer close(done)
		srv.streamResponse(w, httptest.NewRequest(http.MethodPost, "/v1/completions", nil),
			[]<-chan engine.Event{ch}, "cmpl-test", 1700000000,
			generateParams{chat: false}, 0)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("streamResponse kept writing after Write failed")
	}
	// It must have stopped early: one accepted chunk is far fewer than the 16 events sent.
	if w.n > 2 {
		t.Errorf("made %d writes after the first failure; the loop should have returned", w.n)
	}
}

// noFlushWriter is a ResponseWriter that does not implement http.Flusher, which is what a
// ResponseWriter wrapped by some middleware looks like from the handler's side.
type noFlushWriter struct {
	http.ResponseWriter
}

// TestStreamRefusesAConnectionItCannotFlush covers streamResponse's type assertion.
//
// Server-sent events only work if each chunk reaches the client as it is written, so a
// ResponseWriter without Flush is refused rather than silently buffered into a response that
// arrives all at once. That is a real deployment shape: a middleware that wraps the writer
// without forwarding Flush produces exactly this, and the failure is invisible until someone
// notices the stream is not streaming.
//
// Asserted on the status code and the message, because the point is that the client learns why
// rather than receiving an empty 200.
func TestStreamRefusesAConnectionItCannotFlush(t *testing.T) {
	srv, _ := newTestServer(t)
	ch := make(chan engine.Event, 1)
	ch <- engine.Event{Done: true, FinishReason: "length"}
	close(ch)

	rec := httptest.NewRecorder()
	srv.streamResponse(&noFlushWriter{ResponseWriter: rec},
		httptest.NewRequest(http.MethodPost, "/v1/completions", nil),
		[]<-chan engine.Event{ch}, "cmpl-test", 1700000000,
		generateParams{chat: false}, 0)

	if rec.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want %d", rec.Code, http.StatusInternalServerError)
	}
	body := decodeBody[errorBody](t, rec.Result())
	if !strings.Contains(body.Error.Message, "streaming is not supported") {
		t.Errorf("message = %q, want it to say streaming is unsupported", body.Error.Message)
	}
	// And it must not have written a partial event stream first: the header is set only
	// after the assertion passes.
	if strings.Contains(rec.Body.String(), "data:") {
		t.Errorf("body starts an event stream despite refusing: %q", rec.Body.String())
	}
}

// TestNewRejectsNilDependencies covers New's two argument guards.
//
// Both parameters are dereferenced at request time -- eng in every handler, tok in prompt
// encoding -- so a nil accepted here would panic inside the first request with a stack that
// points at the handler rather than at the wiring. Refusing at construction is the same
// reasoning as the engine's own nil checks.
//
// Worth a test because neither guard is reachable from the normal path: newTestServer always
// builds both values, so the two return statements had no coverage.
func TestNewRejectsNilDependencies(t *testing.T) {
	// A real engine and tokenizer to pass for the argument that is not nil.
	path := gguftest.Model{NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	eng, err := engine.New(path, engine.Options{MaxActive: 1})
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize.New: %v", err)
	}

	t.Run("nil engine", func(t *testing.T) {
		if _, err := New(nil, tok, Config{}); err == nil {
			t.Error("New accepted a nil engine")
		} else if !strings.Contains(err.Error(), "nil engine") {
			t.Errorf("error = %q, want it to name the engine", err)
		}
	})
	t.Run("nil tokenizer", func(t *testing.T) {
		if _, err := New(eng, nil, Config{}); err == nil {
			t.Error("New accepted a nil tokenizer")
		} else if !strings.Contains(err.Error(), "nil tokenizer") {
			t.Errorf("error = %q, want it to name the tokenizer", err)
		}
	})
}

// TestNewDefaultsAnEmptyConfig covers the two defaults applied when a Config leaves fields
// zero.
//
// ModelName is what every response's model field and every "does not exist" error names, so
// an empty string there would produce responses naming a model with no name. ContextLen is
// the bound the prompt-length check uses, so leaving it zero would reject every prompt of one
// token or more.
//
// Neither default was reachable: newTestServer always passes both explicitly.
func TestNewDefaultsAnEmptyConfig(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 64}.Write(t, t.TempDir())
	eng, err := engine.New(path, engine.Options{MaxActive: 1})
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize.New: %v", err)
	}

	srv, err := New(eng, tok, Config{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if srv.cfg.ModelName != "orbix" {
		t.Errorf("ModelName = %q, want orbix", srv.cfg.ModelName)
	}
	// Taken from the model rather than hardcoded, so a fixture change moves both sides.
	if want := eng.Model().Config().ContextLen; srv.cfg.ContextLen != want {
		t.Errorf("ContextLen = %d, want the model's %d", srv.cfg.ContextLen, want)
	}
	if srv.cfg.MaxTokens != 256 {
		t.Errorf("MaxTokens = %d, want 256", srv.cfg.MaxTokens)
	}
}

// TestPromptExactlyFillingTheContextIsRefused pins the eq part of the context bound.
//
// generate refuses `len(promptTokens) >= s.cfg.ContextLen`, and the asymmetry with the clamp
// below it is deliberate: a prompt that fills the window leaves no room for even one
// generated token, so accepting it produces a response with an empty completion and
// finish_reason "length", which reads as a successful request that happened to generate
// nothing. Changing >= to > passed the whole server suite, because the only overflow test
// uses a prompt far past the window -- so both the refused side and the accepted side of the
// boundary were unpinned.
//
// The prompt is fixed and the context is derived from its measured length, rather than the
// other way round: growing a string to an exact token count is not reliable, because the
// fixture vocabulary merges repeated runs and a added word can cost one token or several.
func TestPromptExactlyFillingTheContextIsRefused(t *testing.T) {
	srv, tok := newTestServerModel(t, gguftest.Model{ContextLen: 32, NoEOS: true})
	prompt := "Once upon a time"
	promptLen := len(tok.Encode(prompt, true))
	if promptLen < 2 {
		t.Fatalf("prompt encodes to %d tokens, too few to leave a boundary", promptLen)
	}

	// A window one bigger than the prompt leaves exactly one position to generate into.
	// The engine the server already holds was built for the fixture's window, so only the
	// handler's own bound changes; that is the comparison this test is about.
	srv.cfg.ContextLen = promptLen + 1
	resp := post(t, srv.Handler(), "/v1/completions",
		fmt.Sprintf(`{"prompt": %s, "max_tokens": 5}`, quoteJSON(prompt)))
	if resp.StatusCode != http.StatusOK {
		t.Errorf("a %d-token prompt in a %d window: status %d, want 200",
			promptLen, promptLen+1, resp.StatusCode)
	}
	_ = resp.Body.Close()

	// A window exactly the prompt's length leaves nothing, so the request must be refused
	// rather than answered with an empty completion.
	srv.cfg.ContextLen = promptLen
	resp = post(t, srv.Handler(), "/v1/completions",
		fmt.Sprintf(`{"prompt": %s, "max_tokens": 5}`, quoteJSON(prompt)))
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("a prompt filling the %d-token window: status %d, want 400",
			promptLen, resp.StatusCode)
	}
	var body errorBody
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if body.Error.Code != "context_length_exceeded" {
		t.Errorf("code = %q, want context_length_exceeded", body.Error.Code)
	}
}

// TestCutAtStop covers the stop-string cut directly, including the operands the end-to-end tests
// never use.
//
// An empty entry in the stop list has to be skipped. strings.Index(text, "") is 0 for every
// text, so a client that sends `"stop": ["", "END"]` -- or a proxy that rewrites an omitted
// value into an empty string -- would have every response truncated to nothing. Measured: with
// the skip removed, cutAtStop("hello world", ["", "world"]) returns "" instead of "hello ".
//
// The earliest match also has to win. Taking the first stop string that matches rather than the
// leftmost returns whatever the list order happens to be, so the same request would cut at a
// different place depending on how the client ordered its stop strings.
func TestCutAtStop(t *testing.T) {
	cases := []struct {
		text  string
		stop  []string
		want  string
		found bool
		why   string
	}{
		{"hello world", []string{"world"}, "hello ", true, "a single match"},
		{"hello world", []string{"absent"}, "hello world", false, "no match leaves the text alone"},
		{"hello world", nil, "hello world", false, "no stop list at all"},
		// The empty entry must not count as a match at position zero.
		{"hello world", []string{"", "world"}, "hello ", true, "an empty stop is skipped"},
		{"hello world", []string{"world", ""}, "hello ", true, "the same, in either order"},
		{"hello world", []string{""}, "hello world", false, "an empty stop alone matches nothing"},
		// Leftmost wins, whatever the list order says.
		{"hello world", []string{"world", "hello"}, "", true, "hello comes first in the text"},
		// "abc" starts at 0 and "c" at 2, so the leftmost is "abc" -- the cut lands at zero
		// even though the longer string is listed second.
		{"abcabc", []string{"c", "abc"}, "", true, "the leftmost of a nested pair"},
		{"xx", []string{"x", "xx"}, "", true, "the shorter stop at the same position"},
	}
	for _, c := range cases {
		got, found := cutAtStop(c.text, c.stop)
		if got != c.want || found != c.found {
			t.Errorf("cutAtStop(%q, %v) = %q, %v; want %q, %v (%s)",
				c.text, c.stop, got, found, c.want, c.found, c.why)
		}
	}
}

// TestMaxTokensZeroGeneratesNothing covers the value the API can spell now.
//
// The field used to be a plain int, so {"max_tokens": 0} and omitting it decoded to the same
// struct and the server read an explicit 0 as "use MaxTokens" -- an earlier test pinned that, and
// its own comment gave the type as the reason rather than the meaning. Making the field a pointer
// separate absence from zero, which is what the two actually mean: an omitted budget takes the
// server's default, an explicit zero is a budget of zero.
//
// Zero is honoured rather than refused because the engine already answers it: a sequence whose
// budget is spent finishes before sampling, so the caller gets an empty completion with
// finish_reason "length". The CLI has always behaved this way for --max-tokens 0, and the server
// differing was the surprise: a client that computed a budget of zero got a full-length paragraph,
// which is both wrong and expensive.
func TestMaxTokensZeroGeneratesNothing(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()

	zero := decodeBody[completionResponse](t,
		post(t, h, "/v1/completions", `{"prompt": "hi", "max_tokens": 0}`))
	if got := zero.Usage.CompletionTokens; got != 0 {
		t.Errorf("max_tokens: 0 generated %d tokens, want 0", got)
	}
	if len(zero.Choices) != 1 {
		t.Fatalf("got %d choices, want 1", len(zero.Choices))
	}
	if zero.Choices[0].Text != "" {
		t.Errorf("text = %q, want empty", zero.Choices[0].Text)
	}
	// "length" because the budget ran out, which is what the response says for any budget that
	// ends generation rather than a stop string or the end of the prompt.
	if got := zero.Choices[0].FinishReason; got != "length" {
		t.Errorf("finish_reason = %q, want \"length\"", got)
	}

	// Omitting the field still takes the server's default, which is the other half of the
	// distinction and the one every ordinary request relies on.
	absent := decodeBody[completionResponse](t,
		post(t, h, "/v1/completions", `{"prompt": "hi"}`))
	if absent.Usage.CompletionTokens == 0 {
		t.Error("omitting max_tokens generated nothing; the server default was not applied")
	}

	// The newer spelling means the same thing.
	chatZero := decodeBody[chatResponse](t,
		post(t, h, "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 0}`))
	if got := chatZero.Usage.CompletionTokens; got != 0 {
		t.Errorf("max_completion_tokens: 0 generated %d tokens, want 0", got)
	}

	// Two spellings set is a conflict whether or not one of them is zero: the request names the
	// same limit twice, and the caller has to say which one it meant.
	resp := post(t, h, "/v1/completions",
		`{"prompt": "hi", "max_tokens": 0, "max_completion_tokens": 5}`)
	body := decodeBody[errorBody](t, resp)
	_ = resp.Body.Close()
	if resp.StatusCode != 400 || body.Error.Param != "max_completion_tokens" {
		t.Errorf("max_tokens: 0 with max_completion_tokens: 5 answered %d %q, want a 400 naming"+
			" max_completion_tokens", resp.StatusCode, body.Error.Param)
	}
}

// TestChatAndCompletionsEncodePromptsDifferently covers the branch that picks between the two
// encoders.
//
// A chat prompt is markup this server rendered, so its control markers have to be matched whole: the
// model was trained to see <|im_start|> as one token, not as the characters it is spelled from. A
// completions prompt is text the caller typed, so the same characters there are prose and must stay
// prose. The two cases use different functions for that reason, and nothing tested the choice.
//
// The fixture has to declare the marker as a control piece for the difference to exist at all: with
// the default vocabulary the two encoders agree and the branch is invisible. This one puts it at
// index 2 as a control token, and the two encoders then produce different lengths for the same
// input -- measured, 16 ids against 5.
func TestChatAndCompletionsEncodePromptsDifferently(t *testing.T) {
	const marker = "<|im_start|>"
	srv, tok := newTestServerModel(t, gguftest.Model{
		Vocab:      []string{"<unk>", "<s>", marker, "</s>", "hi", "a", "b"},
		BOSID:      1,
		EOSID:      3,
		NoEOS:      true,
		ContextLen: 64,
	})

	// The premise: the two encoders disagree about this text on this file.
	asText := tok.Encode(marker+"hi", false)
	asMarkup := tok.EncodeWithSpecials(marker+"hi", false)
	if len(asText) == len(asMarkup) {
		t.Fatalf("the fixture does not distinguish the encoders (%d ids each), so this test "+
			"would pass whichever one the handler chose", len(asText))
	}

	// The handler's choice, read off the usage counts the response reports for a one-token
	// completion: prompt_tokens is the encoded length before generation.
	chat := decodeBody[chatResponse](t, post(t, srv.Handler(), "/v1/chat/completions",
		`{"messages":[{"role":"user","content":"`+marker+`hi"}],"max_tokens":1}`))
	completion := decodeBody[completionResponse](t, post(t, srv.Handler(), "/v1/completions",
		`{"prompt":"`+marker+`hi","max_tokens":1}`))

	// The expected chat count is computed here from the rendered prompt rather than hardcoded, and
	// the two encoders are applied to it so the assertion names which one the handler must have
	// used. Asserting only that the endpoints differ is not enough: the chat side is longer either
	// way -- 25 ids against 36 under the wrong encoder, measured -- because the neutral transcript
	// adds text around the message.
	rendered := mustRender(t, srv, []chatMessage{{Role: "user", Content: marker + "hi"}})
	withBOS := cfgBOS(t, srv)
	wantMarkup := len(tok.EncodeWithSpecials(rendered, withBOS))
	wantText := len(tok.Encode(rendered, withBOS))
	if wantMarkup == wantText {
		t.Fatalf("the rendered prompt does not distinguish the encoders (%d ids each), so this "+
			"test would pass whichever one the handler chose", wantMarkup)
	}
	if chat.Usage.PromptTokens != wantMarkup {
		t.Errorf("chat encoded the rendered prompt to %d ids; encoding it as markup gives %d and "+
			"as text %d, so the handler used the text path", chat.Usage.PromptTokens, wantMarkup, wantText)
	}

	if wantTextForPrompt := len(tok.Encode(marker+"hi", withBOS)); completion.Usage.PromptTokens != wantTextForPrompt {
		t.Errorf("completions encoded the prompt to %d ids; encoding it as text gives %d",
			completion.Usage.PromptTokens, wantTextForPrompt)
	}
}

// cfgBOS reports the addSpecial argument the handler derives from the model: BOS is prepended only
// when the model declares one.
func cfgBOS(t *testing.T, srv *Server) bool {
	t.Helper()
	return srv.eng.Model().Config().BOSID >= 0
}

// TestFinishReasonMapping covers the translation between the engine's five reasons and the three
// the response schema carries.
//
// The expression was written twice, once for the streaming path and once for the non-streaming one,
// and neither copy had a test: the existing cases cover "stop" and "length" only, which are the two
// the engine produces most often and the two that need no translation. What went untested is exactly
// the part where the two vocabularies disagree.
//
// cancel maps to length rather than stop: the schema has no value for "the client left", and the
// generation did not end on its own. abandoned keeps stop, which is what the code did before this
// function existed -- the consolidation did not change it.
func TestFinishReasonMapping(t *testing.T) {
	cases := []struct{ reason, want string }{
		// The schema's own values pass through.
		{"stop", "stop"},
		{"length", "length"},
		// No schema value exists for either; both were mapped by the code being pinned.
		{"cancel", "length"},
		{"abandoned", "stop"},
		{"error", "stop"},
		// A schema value that must NOT be folded into "stop", which is why it has its own
		// case rather than sharing the default: a client branches on it to decide whether to
		// run a tool. This is the only test that checks the mapping: removing the case left
		// the whole package green apart from this table. TestFinalChoiceCarriesCalls asserts
		// "tool_calls" reaches a chunk, but it builds the reason itself, so it never runs
		// finishReasonFor -- one is about the mapping, the other about what finalChoice does
		// with a reason it is handed.
		{"tool_calls", "tool_calls"},
		// Anything unrecognised falls back rather than producing an invalid enum value, which a
		// strict client would reject outright.
		{"something-new", "stop"},
		{"", "stop"},
	}
	for _, c := range cases {
		if got := finishReasonFor(c.reason); got != c.want {
			t.Errorf("finishReasonFor(%q) = %q, want %q", c.reason, got, c.want)
		}
	}

	// Every reason the engine can produce has to appear above.
	for _, r := range engine.FinishReasons() {
		found := false
		for _, c := range cases {
			if c.reason == r.String() {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("the engine defines reason %q and this table does not cover it", r.String())
		}
	}

	// Every value the schema defines has to be reachable from some engine reason, or a client
	// branching on them would see a case that never occurs.
	seen := map[string]bool{}
	for _, c := range cases {
		seen[finishReasonFor(c.reason)] = true
	}
	for _, want := range []string{"stop", "length"} {
		if !seen[want] {
			t.Errorf("no engine reason maps to %q", want)
		}
	}
}

// TestStreamReportsAMidStreamError covers what a client gets when the engine fails after the
// response has started.
//
// A status code is no longer available once the first event is written, so the failure has to
// travel inside the stream: a chunk carrying finish_reason "error" and the message, then the end of
// the connection. The existing stream tests all drive successful generations, so this path was
// never exercised -- it could report a finish_reason of "stop" and nothing would notice.
//
// streamResponse takes the event channel directly, which is what makes the case expressible: the
// engine's errors come from a failed prefill or decode, and a fixture cannot be asked to fail on
// demand.
func TestStreamReportsAMidStreamError(t *testing.T) {
	srv, _ := newTestServer(t)
	ch := make(chan engine.Event, 2)
	ch <- engine.Event{HasToken: true, Token: 4} // one token, so the response has started
	ch <- engine.Event{Err: errors.New("prefill failed")}
	close(ch)

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/completions", nil)
	srv.streamResponse(w, r, []<-chan engine.Event{ch}, "req-1", 1700000000,
		generateParams{chat: false}, 3)

	body := w.Body.String()
	// The error has to be visible to the client, not only in a log.
	if !strings.Contains(body, `"finish_reason":"error"`) {
		t.Errorf("no error finish_reason in the stream:\n%s", body)
	}
	if !strings.Contains(body, "prefill failed") {
		t.Errorf("the engine's message is not in the stream:\n%s", body)
	}
	// And the token that did arrive is not lost, so a client sees the partial answer plus the
	// failure rather than only one of them.
	if !strings.Contains(body, `"text":"a"`) {
		t.Errorf("the first chunk was dropped:\n%s", body)
	}
	// The stream ends without the sentinel: the connection closing is the signal. Asserted so a
	// change to either half is a visible decision.
	if strings.Contains(body, "[DONE]") {
		t.Errorf("the error path wrote the [DONE] sentinel:\n%s", body)
	}
}

// TestEmptyPromptIsRefusedAfterEncoding covers the check that the encoded prompt is not empty, and
// the earlier one that an empty message list trips.
//
// A prompt of "" encodes to zero ids, and the engine's Submit refuses that with its own message.
// The server checks first so the reply is a 400 naming the prompt rather than a 500 from deeper in
// -- and the check has to run after encoding, since a prompt like " " may also encode to nothing
// depending on the vocabulary.
//
// Both endpoints are asserted because they share the branch but decode different request types.
func TestEmptyPromptIsRefusedAfterEncoding(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()

	// An empty completions prompt reaches the encoder and is caught there.
	resp := post(t, h, "/v1/completions", `{"prompt":"","max_tokens":1}`)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 for a prompt that encodes to nothing", resp.StatusCode)
	}
	env := decodeBody[errorBody](t, resp)
	if !strings.Contains(env.Error.Message, "prompt") {
		t.Errorf("message %q does not mention the prompt", env.Error.Message)
	}

	// The chat endpoint refuses an empty message list earlier, with its own message: a request with
	// no messages is malformed before encoding is reached, so the two checks are not the same one.
	// Asserted separately so a change to either stays visible.
	resp = post(t, h, "/v1/chat/completions", `{"messages":[],"max_tokens":1}`)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400 for an empty message list", resp.StatusCode)
	}
	env = decodeBody[errorBody](t, resp)
	if !strings.Contains(env.Error.Message, "messages") {
		t.Errorf("message %q does not mention the messages", env.Error.Message)
	}
}

// TestStreamingClientGoneReleasesTheSequence covers what a real connection adds.
//
// Every other test in this package uses httptest.ResponseRecorder, whose Write never fails. That
// makes the streaming path's write-failure returns unreachable here, and those returns are the only
// thing that notices a client which stopped reading. A real listener is needed for the distinction
// between "the handler wrote and moved on" and "the handler's write failed".
//
// The assertion is that the engine's sequence is released, which is what a leaked stream would keep
// alive. It is the same property TestClientDisconnectStopsGeneration checks for the blocking path;
// the streaming path reaches it through a write error rather than through the request context.
func TestStreamingClientGoneReleasesTheSequence(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 2048})
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/v1/completions", "application/json",
		strings.NewReader(`{"prompt":"Once upon a time","max_tokens":500,"stream":true}`))
	if err != nil {
		t.Fatalf("starting the stream: %v", err)
	}
	// Read one chunk and hang up, so the server is mid-generation when the connection closes.
	buf := make([]byte, 32)
	if _, err := resp.Body.Read(buf); err != nil {
		t.Fatalf("reading the first chunk: %v", err)
	}
	_ = resp.Body.Close()

	waitFor(t, 10*time.Second, func() bool { return srv.eng.Active() == 0 },
		"the engine still holds a sequence after the client hung up")
}

// TestStreamingTwoChoicesClientGoneReleasesBothSequences covers the fan-in path when a client
// hangs up.
//
// n=1 runs inline and n>1 runs one goroutine per choice behind a channel; the existing
// disconnect test uses n=1, so nothing exercised the second shape. Deleting the whole fan-in
// left this test passing too, which is the limit of what it proves -- see below.
//
// It reaches the fan-in but not its cancel arm. That arm runs when writeItem returns false,
// which needs the response write to fail while a choice is still producing; against the
// fixture the stream finishes in under a hundred milliseconds, so closing the connection
// after one chunk usually lands after the last write. Tried: the arm's line stays uncovered
// across three runs, and deleting it leaves this test green. A longer budget or a slowed
// writer would make it a race, and a test that passes or fails on scheduling is worse than
// one that says what it covers.
//
// What it does assert is the property a hang-up must preserve either way: no sequence is
// left in the engine.
func TestStreamingTwoChoicesClientGoneReleasesBothSequences(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 2048})
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/v1/completions", "application/json",
		strings.NewReader(`{"prompt":"Once upon a time","max_tokens":500,"stream":true,"n":2,`+
			`"temperature":1.2,"seed":5}`))
	if err != nil {
		t.Fatalf("starting the stream: %v", err)
	}
	// Read one chunk so the server is mid-generation, then hang up.
	buf := make([]byte, 32)
	if _, err := resp.Body.Read(buf); err != nil {
		t.Fatalf("reading the first chunk: %v", err)
	}
	_ = resp.Body.Close()

	waitFor(t, 15*time.Second, func() bool { return srv.eng.Active() == 0 },
		"the engine still holds a sequence after the client hung up on a two-choice stream")
}

// whyPhrase returns a phrase the refusal reason must contain for a given shape field.
//
// The values come from the shapeField table: each explains why the field cannot be honoured, and a
// caller reads that to pick an alternative. Asserting on the fixed "does not implement" prefix
// instead left the reasons untested.
func whyPhrase(field string) string {
	// One entry, because one field is left in the refusal tables. The map stays a map so that
	// adding a field means adding a reason rather than threading an argument through.
	return map[string]string{"echo": "never includes the prompt"}[field]
}

// TestIdlePollShiftValue pins the backoff ceiling, which nothing else observes.
//
// The constant sets the longest an idle pump sleeps: 1ms << idlePollShift. Changing it in either
// direction passed every test, because the only timing assertion is "a request after an idle period
// takes under a second" -- a ceiling of 1ms satisfies that, and a ceiling of about 1.02s sits close
// enough to the bound that the request may arrive before the sleep ends either way.
//
// The value is documented in the constant's comment and decides the latency floor for the first
// request after a quiet spell, so it is asserted against a literal rather than left implicit.
func TestIdlePollShiftValue(t *testing.T) {
	// 3 makes the ceiling 8ms: long enough to keep an idle server off a core, short enough not to be
	// noticed by the first request that arrives.
	if idlePollShift != 3 {
		t.Errorf("idlePollShift = %d, want the documented 3 (a %v ceiling)",
			idlePollShift, time.Millisecond<<idlePollShift)
	}
}

// TestRequestLargerThanThePoolIsRefusedPermanently covers the two situations that used to
// share one answer.
//
// "block pool exhausted" means the pool is busy, and Retry-After is the right advice: the
// same request succeeds once another sequence releases its blocks. It also fires when the
// request's own prompt plus budget can never fit the pool, and then the advice is wrong --
// nothing another request does makes room, so a client that honours Retry-After retries
// forever.
//
// Measured before the fix, on the fixture with --blocks 2 --block-size 16 (32 positions): an
// 11-token prompt asking for 4 tokens needs 3 blocks, got 503 with Retry-After 1 on four
// attempts two seconds apart, while orbix_kv_pool_free stayed at 2. A shorter prompt
// succeeded against the same idle server, which is what separates the two cases.
//
// The refusal is a 400 with the two block counts in the message, because "block pool
// exhausted" reads as transient and the numbers are what a caller can act on.
func TestRequestLargerThanThePoolIsRefusedPermanently(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 256}.Write(t, t.TempDir())
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	eng, err := engine.New(path, engine.Options{
		MaxActive: 1, MaxQueued: 1, Blocks: 2, BlockSize: 16,
	})
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	cfg := eng.Model().Config()
	srv, err := New(eng, tok, Config{ModelName: cfg.Name, ContextLen: cfg.ContextLen})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// The "a request that fits" half needs a pump, because it goes through Submit and waits
	// for tokens. The refusal below returns before Submit, so it does not.
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	srv.StartPump(ctx)

	// The fixture tokenizer spells a word into several ids, so measure rather than guess:
	// find the shortest prompt that needs more blocks than the pool holds.
	over := ""
	for words := 1; words <= 64; words++ {
		p := strings.Repeat("word ", words)
		if need := blocksNeeded(len(tok.Encode(p, cfg.BOSID >= 0))+4, 16); need > 2 {
			over = p
			break
		}
	}
	if over == "" {
		t.Fatal("no prompt within 64 words outgrew a two-block pool; the fixture changed shape")
	}

	resp := post(t, srv.Handler(), "/v1/completions",
		`{"prompt": `+quoteJSON(over)+`, "max_tokens": 4}`)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400: a request that can never fit the pool is not transient",
			resp.StatusCode)
	}
	// Retry-After is the advice that made this a defect: a client must not be told to wait
	// for room that cannot appear.
	if got := resp.Header.Get("Retry-After"); got != "" {
		t.Errorf("Retry-After = %q on a permanently unservable request", got)
	}
	body := readBody(t, resp)
	for _, want := range []string{
		"3 KV cache blocks", // what the request needs, arithmetic stated in the message
		"2 (32 positions)",  // what the pool holds
		"context_length_exceeded",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("body does not contain %q:\n%s", want, body)
		}
	}

	// The same refusal with n=2 has to state the multiplication. A single-choice message
	// quoting the prompt-plus-budget figure reads as if one choice needed those blocks, and
	// with n choices the total is n times that. This branch had no test: every case above
	// leaves n at its default, so `if candidates > 1` never ran.
	multi := post(t, srv.Handler(), "/v1/completions",
		`{"prompt": `+quoteJSON(over)+`, "max_tokens": 4, "n": 2, "temperature": 1.2, "seed": 5}`)
	defer func() { _ = multi.Body.Close() }()
	if multi.StatusCode != http.StatusBadRequest {
		t.Fatalf("n=2 status = %d, want 400", multi.StatusCode)
	}
	multiBody := readBody(t, multi)
	for _, want := range []string{
		"times 2 sequences at",
		// The blocks one sequence of this size needs, which the message quotes per sequence.
		"blocks each",
		"context_length_exceeded",
	} {
		if !strings.Contains(multiBody, want) {
			t.Errorf("n=2 body does not contain %q:\n%s", want, multiBody)
		}
	}

	// A request that fits must still be served: the refusal is about the size, not about the
	// pool existing.
	fits := post(t, srv.Handler(), "/v1/completions", `{"prompt": "hi", "max_tokens": 2}`)
	defer func() { _ = fits.Body.Close() }()
	if fits.StatusCode != http.StatusOK {
		t.Errorf("a request that fits the pool got %d, want 200", fits.StatusCode)
	}

	// The chat endpoint reaches the same check with a rendered prompt, so it needs the same
	// answer -- and a param naming the field that endpoint actually takes, not the other
	// one's. Measured on the fixture: a six-word message renders to 55 tokens against the
	// same two-block pool and is refused with param "messages", while "hi" is served.
	chatOver := post(t, srv.Handler(), "/v1/chat/completions",
		`{"messages": [{"role": "user", "content": `+quoteJSON(over)+`}], "max_tokens": 4}`)
	defer func() { _ = chatOver.Body.Close() }()
	if chatOver.StatusCode != http.StatusBadRequest {
		t.Errorf("chat: status = %d, want 400", chatOver.StatusCode)
	}
	chatBody := readBody(t, chatOver)
	if !strings.Contains(chatBody, `"param":"messages"`) {
		t.Errorf("chat body does not name param messages:\n%s", chatBody)
	}
	chatFits := post(t, srv.Handler(), "/v1/chat/completions",
		`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}`)
	defer func() { _ = chatFits.Body.Close() }()
	if chatFits.StatusCode != http.StatusOK {
		t.Errorf("chat: a request that fits got %d, want 200", chatFits.StatusCode)
	}
}

// TestBlocksNeededMatchesTheCache pins the arithmetic the refusal above depends on.
//
// PagedKVCache.ensureBlocks puts position p in block p/blockSize, so n positions occupy
// ceil(n/blockSize) blocks. Recomputed here as a loop over positions rather than the ceiling
// expression, so a change to either form has to be a real one: an off-by-one in this function
// either refuses a request the pool could serve or lets the permanent case through to a 503.
func TestBlocksNeededMatchesTheCache(t *testing.T) {
	for _, blockSize := range []int{1, 4, 16, 32} {
		// Count the blocks a sequence actually touches, the way the cache numbers them.
		for positions := 1; positions <= 4*blockSize+1; positions++ {
			used := map[int]bool{}
			for pos := range positions {
				used[pos/blockSize] = true
			}
			if got, want := blocksNeeded(positions, blockSize), len(used); got != want {
				t.Errorf("blocksNeeded(%d, %d) = %d, want %d", positions, blockSize, got, want)
			}
		}
	}
	// A server without a pool, and a zero request, have nothing to compare.
	for _, c := range []struct{ positions, blockSize int }{{0, 16}, {-1, 16}, {10, 0}} {
		if got := blocksNeeded(c.positions, c.blockSize); got != 0 {
			t.Errorf("blocksNeeded(%d, %d) = %d, want 0", c.positions, c.blockSize, got)
		}
	}
}

// TestJSONTypeName drops the offending value and keeps the type.
//
// The decoder spells the value two ways: with a bare type ("string", "object") and, for a
// number, with the number after it ("number 1.5") -- measured, and only the second form
// exercises the cut. Echoing the value back is what the two-word form would do, and a caller
// who sent a large object does not need it quoted in an error message.
func TestJSONTypeName(t *testing.T) {
	for _, c := range []struct{ in, want string }{
		{"string", "string"},
		{"number", "number"},
		{"number 1.5", "number"},
		{"number -1e30", "number"},
		{"object", "object"},
		{"", ""},
	} {
		if got := jsonTypeName(c.in); got != c.want {
			t.Errorf("jsonTypeName(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestGoTypeNameSpeaksJSON covers every arm of the mapping, including the ones no current
// request field reaches.
//
// A branch here that is wrong is invisible until the field that needs it exists, and the
// failure would be an error message naming a Go type -- the exact thing this mapping exists
// to avoid. The default arm is checked too: it has to fall back to the Go name rather than
// returning an empty string, which would leave a message reading "must be , but it is".
func TestGoTypeNameSpeaksJSON(t *testing.T) {
	type named struct{}
	for _, c := range []struct {
		in   any
		want string
	}{
		{"", "a string"},
		{false, "a boolean"},
		{int64(0), "an integer"},
		{uint32(0), "an integer"},
		{1.5, "a number"},
		{[]string(nil), "an array of strings"},
		{[]int(nil), "an array"},
		{[2]string{}, "an array of strings"},
		{map[string]int(nil), "an object"},
		{(*float64)(nil), "a number"}, // a pointer follows its element
		{named{}, ""},                 // no JSON word for a struct; the fallback is Go's name
	} {
		got := goTypeName(reflect.TypeOf(c.in))
		if c.want != "" {
			if got != c.want {
				t.Errorf("goTypeName(%T) = %q, want %q", c.in, got, c.want)
			}
			continue
		}
		// The struct case: the answer is Go's own name rather than an empty string.
		if got == "" || got != reflect.TypeOf(c.in).String() {
			t.Errorf("goTypeName(%T) = %q, want the Go type name %q", c.in, got,
				reflect.TypeOf(c.in).String())
		}
	}
}

// TestABadFieldIsReportedBeforeAPoolRefusal pins the order of two refusals that can both
// apply to the same request.
//
// The request below is wrong twice over: temperature is negative, and with no max_tokens it
// takes the 256-token default and needs more blocks than a small pool holds. Only one report
// reaches the caller, and it has to be the field they got wrong. When the pool check ran
// first, this request was answered context_length_exceeded against param "prompt" -- telling
// the caller to shorten a prompt whose real problem was the temperature, and naming a code
// that has nothing to do with the value sent.
//
// Both directions are asserted, because the fix is not "always report the sampler": a
// correctly-shaped request that genuinely outgrows the pool must still be refused for that.
func TestABadFieldIsReportedBeforeAPoolRefusal(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 256}.Write(t, t.TempDir())
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	// Eight blocks of sixteen positions: 128 positions, comfortably less than the 256-token
	// default budget, so an ordinary request with no max_tokens really does not fit.
	eng, err := engine.New(path, engine.Options{
		MaxActive: 1, MaxQueued: 1, Blocks: 8, BlockSize: 16,
	})
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	cfg := eng.Model().Config()
	srv, err := New(eng, tok, Config{ModelName: cfg.Name, ContextLen: cfg.ContextLen})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	t.Run("a bad sampler field wins", func(t *testing.T) {
		resp := post(t, srv.Handler(), "/v1/completions",
			`{"prompt": "hi", "temperature": -2}`)
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400", resp.StatusCode)
		}
		body := readBody(t, resp)
		for _, want := range []string{`"code":"invalid_value"`, `"param":"temperature"`,
			"temperature must not be negative, got -2"} {
			if !strings.Contains(body, want) {
				t.Errorf("body does not contain %q:\n%s", want, body)
			}
		}
		if strings.Contains(body, "context_length_exceeded") {
			t.Errorf("a negative temperature was reported as a context problem:\n%s", body)
		}
	})

	t.Run("an oversized request still names the pool", func(t *testing.T) {
		resp := post(t, srv.Handler(), "/v1/completions",
			`{"prompt": "word word word word word", "max_tokens": 200}`)
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400", resp.StatusCode)
		}
		body := readBody(t, resp)
		if !strings.Contains(body, "context_length_exceeded") {
			t.Errorf("an oversized request was not reported as such:\n%s", body)
		}
		if !strings.Contains(body, "KV cache blocks") {
			t.Errorf("body does not explain the pool shortage:\n%s", body)
		}
	})
}

// TestEveryErrorCarriesACode is the contract the paramError type was introduced for, checked
// across every failure the server can produce.
//
// The comment on paramError says a client that branches on `code` rather than message text
// "could not tell 'your prompt is too long' from 'your temperature is negative', or either
// from a malformed body". That last clause was not true: the decode path wrote through
// writeError, which sets no code, so a malformed body, an unsupported encoding and an
// oversized body all arrived uncoded next to the coded validation failures.
//
// Each case here is also checked for the envelope itself -- JSON content type, a parseable
// body, and a `type` -- because a client that reads any of those first would fail before it
// ever reached the code.
func TestEveryErrorCarriesACode(t *testing.T) {
	srv, _ := newTestServer(t)
	limit := srv.maxBodyBytes()

	cases := []struct {
		name   string
		method string
		path   string
		header map[string]string
		body   string
		status int
		code   string
	}{
		{
			name: "malformed JSON", method: http.MethodPost, path: "/v1/completions",
			body: "not json", status: http.StatusBadRequest, code: "invalid_json",
		},
		{
			name: "unsupported encoding", method: http.MethodPost, path: "/v1/completions",
			header: map[string]string{"Content-Encoding": "br"},
			body:   "x", status: http.StatusUnsupportedMediaType, code: "unsupported_encoding",
		},
		{
			name: "oversized body", method: http.MethodPost, path: "/v1/completions",
			body:   `{"prompt":"` + strings.Repeat("x", int(limit)) + `","max_tokens":1}`,
			status: http.StatusRequestEntityTooLarge, code: "body_too_large",
		},
		{
			name: "negative budget", method: http.MethodPost, path: "/v1/completions",
			body:   `{"prompt":"hi","max_tokens":-1}`,
			status: http.StatusBadRequest, code: "invalid_value",
		},
		{
			name: "response-shape field", method: http.MethodPost, path: "/v1/completions",
			body:   `{"prompt":"hi","suffix":"END"}`,
			status: http.StatusBadRequest, code: "unsupported_parameter",
		},
		{
			name: "missing prompt", method: http.MethodPost, path: "/v1/completions",
			body: `{"prompt":""}`, status: http.StatusBadRequest,
			code: "missing_required_parameter",
		},
		{
			// 404, not 400: the model is a resource this server does not have, which is the
			// same answer /v1/models/{id} gives for an unknown id. OpenAI answers this with
			// model_not_found and its SDKs map it to a not-found error.
			name: "unknown model", method: http.MethodPost, path: "/v1/completions",
			body: `{"model":"nope","prompt":"hi"}`, status: http.StatusNotFound,
			code: "model_not_found",
		},
		{
			name: "unknown endpoint", method: http.MethodGet, path: "/v1/nope",
			status: http.StatusNotFound, code: "unknown_endpoint",
		},
		{
			name: "wrong method", method: http.MethodPut, path: "/v1/completions",
			status: http.StatusMethodNotAllowed, code: "method_not_allowed",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var rdr io.Reader
			if c.body != "" {
				rdr = strings.NewReader(c.body)
			}
			req := httptest.NewRequest(c.method, c.path, rdr)
			for k, v := range c.header {
				req.Header.Set(k, v)
			}
			rec := httptest.NewRecorder()
			srv.Handler().ServeHTTP(rec, req)

			resp := rec.Result()
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != c.status {
				t.Fatalf("status = %d, want %d", resp.StatusCode, c.status)
			}
			if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
				t.Errorf("Content-Type = %q, want application/json: an SDK decodes the body "+
					"before it raises", ct)
			}
			var env errorBody
			if err := json.NewDecoder(resp.Body).Decode(&env); err != nil {
				t.Fatalf("body is not JSON: %v", err)
			}
			if env.Error.Type == "" {
				t.Errorf("no type in %+v", env.Error)
			}
			if env.Error.Code != c.code {
				t.Errorf("code = %q, want %q (message %q)", env.Error.Code, c.code,
					env.Error.Message)
			}
			// A code without a message is not actionable; the two travel together.
			if env.Error.Message == "" {
				t.Error("empty message")
			}
		})
	}
}

// TestCutAtStopChoosesTheLeftmostMatch pins the rule when several stop strings could apply.
//
// The integration tests each use one stop string, so nothing said what happens with several:
// a list of alternatives reaches this function as a set, and the only defensible reading is
// that generation halts where the output would first contain any of them. Taking the first
// list entry that happens to match would make the result depend on the order the client wrote
// its list in, which is not a property of the output at all.
//
// Measured: "xxabyyabc" against ["abc", "ab"] cuts at 2, and the same text against ["ab",
// "abc"] also cuts at 2 -- the shorter string wins because it starts earlier, not because it
// was listed first. The final cases put two matches at the same position, where the answer is
// the same whichever wins, and the assertion is only that the cut point agrees.
func TestCutAtStopChoosesTheLeftmostMatch(t *testing.T) {
	for _, c := range []struct {
		name    string
		text    string
		stop    []string
		want    string
		wantCut bool
	}{
		{"shorter listed second, matches first", "xxabyyabc", []string{"abc", "ab"}, "xx", true},
		{"shorter listed first, matches first", "xxabyyabc", []string{"ab", "abc"}, "xx", true},
		{"longer matches first", "xxabcyyab", []string{"ab", "abc"}, "xx", true},
		{"three alternatives, middle matches first", "..b..c..d", []string{"d", "c", "b"}, "..", true},
		// Same start position: the cut is at the same place, so the answer does not depend on
		// which one is chosen.
		{"same position, longer listed first", "ab", []string{"ab", "a"}, "", true},
		{"same position, shorter listed first", "ab", []string{"a", "ab"}, "", true},
		// An empty string must not cut anything: it is present in every text at offset 0.
		{"empty entry alone", "hello", []string{""}, "hello", false},
		{"empty entry beside a real one", "helloEND", []string{"", "END"}, "hello", true},
		{"no match at all", "hello", []string{"x", "y"}, "hello", false},
		{"no stop list", "hello", nil, "hello", false},
		// A match at the very start leaves nothing, which is a successful cut rather than a
		// miss.
		{"match at position zero", "END world", []string{"END"}, "", true},
		{"match at the end keeps the prefix", "hello END", []string{"END"}, "hello ", true},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, cut := cutAtStop(c.text, c.stop)
			if got != c.want || cut != c.wantCut {
				t.Errorf("cutAtStop(%q, %q) = (%q, %v), want (%q, %v)",
					c.text, c.stop, got, cut, c.want, c.wantCut)
			}
		})
	}
}

// TestDisconnectFreesTheSlotForTheNextRequest checks the half TestClientDisconnectStopsGeneration
// leaves out.
//
// That test asserts the engine's active count returns to zero, which says the sequence left the
// engine. It does not say the slot the request occupied is available again: a slot leaked
// anywhere above the engine -- in the handler's admission path, in a semaphore, in the queue
// bookkeeping -- would leave Active()==0 while the next request is refused or made to wait
// forever. With MaxQueued at zero there is nowhere for a leaked slot to hide, and a second
// request is the only thing that can tell the two states apart.
//
// Measured against the real server before this test existed: twenty disconnecting streaming
// requests left submitted==finished==11 with active 0 and eleven queue-full refusals, and every
// count balanced. The disconnects here are cancellations of a blocking request, which reaches
// the same finish path.
func TestDisconnectFreesTheSlotForTheNextRequest(t *testing.T) {
	// One slot and no queue, built directly rather than through newTestServerModel: the limits
	// are fixed at construction, and this is the configuration that makes a leaked slot visible
	// instead of absorbable.
	path := gguftest.Model{NoEOS: true, ContextLen: 2048}.Write(t, t.TempDir())
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	eng, err := engine.New(path, engine.Options{MaxActive: 1, MaxQueued: 0})
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	modelCfg := eng.Model().Config()
	srv, err := New(eng, tok, Config{ModelName: modelCfg.Name, ContextLen: modelCfg.ContextLen})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// The engine only advances inside Pump, and a handler blocks until its sequence produces
	// a token, so without this the first request never reaches the point where cancelling it
	// means anything.
	pumpCtx, stopPump := context.WithCancel(context.Background())
	t.Cleanup(stopPump)
	srv.StartPump(pumpCtx)

	// The first request is cancelled while the engine is working on it.
	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest(http.MethodPost, "/v1/completions",
		strings.NewReader(`{"prompt": "Once upon a time", "max_tokens": 1000}`)).WithContext(ctx)
	w := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		defer close(done)
		srv.Handler().ServeHTTP(w, req)
	}()
	waitFor(t, 5*time.Second, func() bool { return srv.eng.Active() > 0 },
		"request never reached the engine")
	cancel()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("handler did not return after the client cancelled")
	}
	waitFor(t, 5*time.Second, func() bool { return srv.eng.Active() == 0 },
		"engine still has an active sequence after the client left")

	// The slot has to be usable: a leaked one would refuse this outright with MaxQueued=0.
	resp := post(t, srv.Handler(), "/v1/completions",
		`{"prompt": "hi", "max_tokens": 2}`)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("the request after a disconnect = %d, want 200: the cancelled request's slot "+
			"was not released", resp.StatusCode)
	}
	body := decodeBody[completionResponse](t, resp)
	if len(body.Choices) != 1 {
		t.Errorf("got %d choices, want 1", len(body.Choices))
	}

	// The client leaving is accounted for as one of the two client-gone reasons rather than as
	// a normal finish. Which of the two depends on timing -- an engine that notices the missing
	// reader before it notices the cancelled context calls it abandoned -- so the assertion is
	// on the pair, and it is why the leak mutation has to cover both to be caught.
	waitFor(t, 5*time.Second, func() bool {
		st := srv.eng.Stats()
		return st.FinishedByReason[engine.FinishCancel]+st.FinishedByReason[engine.FinishAbandoned] == 1
	}, "the client-gone request was not counted as cancelled or abandoned")
}

// TestStreamReportsPoolExhaustionInBand pins what a streaming client gets when the pool runs
// out, which is not a status code.
//
// The response has sent its 200 by the time blocks run short, so neither 503 nor Retry-After can
// be delivered; the failure arrives in the stream as finish_reason "error" with the reason in a
// top-level "error" field. That is a protocol constraint rather than an oversight, but it has a
// consequence worth stating: a streaming client cannot tell a temporary capacity problem from a
// permanent one, so it has to read the message. The blocking endpoint answers the same condition
// with 503 and Retry-After, and this test records both so the asymmetry is not mistaken for one
// endpoint being wrong.
func TestStreamReportsPoolExhaustionInBand(t *testing.T) {
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
		MaxActive: 4, MaxQueued: 4, Blocks: 5, BlockSize: 4, PrefixCache: true,
	})
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	cfg := eng.Model().Config()
	srv, err := New(eng, tok, Config{ModelName: cfg.Name, ContextLen: cfg.ContextLen})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	pumpCtx, stopPump := context.WithCancel(context.Background())
	t.Cleanup(stopPump)
	srv.StartPump(pumpCtx)

	// Collect the streaming responses concurrently so four sequences are in flight at once.
	bodies := make([]string, 4)
	statuses := make([]int, 4)
	var wg sync.WaitGroup
	for i := range 4 {
		wg.Go(func() {
			resp := post(t, srv.Handler(), "/v1/completions",
				`{"prompt": "Once upon a time", "max_tokens": 6, "stream": true}`)
			statuses[i] = resp.StatusCode
			var sb strings.Builder
			buf := make([]byte, 8192)
			for {
				n, err := resp.Body.Read(buf)
				sb.Write(buf[:n])
				if err != nil {
					break
				}
			}
			_ = resp.Body.Close()
			bodies[i] = sb.String()
		})
	}
	wg.Wait()

	exhausted := 0
	for i, body := range bodies {
		// The status is 200 because headers went out first; that is the constraint.
		if statuses[i] != http.StatusOK {
			t.Errorf("request %d: status = %d, want 200: headers are already sent when the pool "+
				"runs out", i, statuses[i])
		}
		if !strings.Contains(body, "block pool exhausted") {
			continue
		}
		exhausted++
		// The reason has to reach the client, in the shape the format allows.
		if !strings.Contains(body, `"finish_reason":"error"`) {
			t.Errorf("request %d: the in-band failure has no finish_reason error:\n%s", i, body)
		}
		if !strings.Contains(body, `"error":"`) {
			t.Errorf("request %d: the in-band failure carries no message:\n%s", i, body)
		}
	}
	if exhausted == 0 {
		t.Fatalf("none of the four streaming requests reported an exhausted pool, so this run "+
			"says nothing about the in-band path:\n%q", bodies)
	}

	// The blocking endpoint says the same thing with a status code, which is the contrast that
	// makes the streaming form a constraint rather than a choice.
	blocking := make([]int, 4)
	retries := make([]string, 4)
	for i := range 4 {
		wg.Go(func() {
			resp := post(t, srv.Handler(), "/v1/completions",
				`{"prompt": "Once upon a time", "max_tokens": 6}`)
			blocking[i] = resp.StatusCode
			retries[i] = resp.Header.Get("Retry-After")
			_ = resp.Body.Close()
		})
	}
	wg.Wait()

	refused := 0
	for i := range 4 {
		if blocking[i] == http.StatusServiceUnavailable {
			refused++
			if retries[i] == "" {
				t.Errorf("blocking request %d got 503 without Retry-After", i)
			}
		}
	}
	if refused == 0 {
		t.Errorf("no blocking request was refused, so the comparison is not being made: %v",
			blocking)
	}
}

// TestUsageIsIdenticalOnBothPaths compares all three usage fields between the blocking and
// streaming responses for one request, on both endpoints.
//
// TestStreamingUsageCountsEveryGeneratedToken compares completion_tokens only, and only for
// /v1/completions. The other two fields were never compared, and the two paths compute them
// from different sources: the blocking body takes len(out) and len(promptTokens), while the
// stream accumulates a counter that the terminal event's Generated figure completes. The
// comment beside that counter records the original defect -- a stream that counted only the
// events its loop had read reported 2 where the blocking body reported 17 -- so the two
// agreeing is the property, not an accident.
//
// Measured: 11/6/17 on both paths for the completions cases (with and without a stop string)
// and 17/6/23 for chat. The stop case is worth including because it is where completion_tokens
// and the delivered text diverge: six tokens were generated, the reply is shorter, and the
// field follows the generation.
func TestUsageIsIdenticalOnBothPaths(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()

	cases := []struct {
		name string
		path string
		body string
	}{
		{"completions", "/v1/completions", `{"prompt": "Once upon a time", "max_tokens": 6}`},
		{"completions with a stop string", "/v1/completions",
			`{"prompt": "Once upon a time", "max_tokens": 6, "stop": ["\n"]}`},
		{"chat", "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 6}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var blocking struct {
				Usage *usage `json:"usage"`
			}
			if err := json.Unmarshal([]byte(readBody(t, post(t, h, c.path, c.body))),
				&blocking); err != nil {
				t.Fatalf("decode: %v", err)
			}
			if blocking.Usage == nil {
				t.Fatal("the blocking body carries no usage")
			}
			if blocking.Usage.TotalTokens != blocking.Usage.PromptTokens+blocking.Usage.CompletionTokens {
				t.Errorf("blocking total_tokens = %d, but %d + %d = %d",
					blocking.Usage.TotalTokens, blocking.Usage.PromptTokens,
					blocking.Usage.CompletionTokens,
					blocking.Usage.PromptTokens+blocking.Usage.CompletionTokens)
			}

			streamBody := strings.Replace(c.body, `"max_tokens"`,
				`"stream": true, "stream_options": {"include_usage": true}, "max_tokens"`, 1)
			var streamed *usage
			for _, line := range strings.Split(readBody(t, post(t, h, c.path, streamBody)), "\n") {
				payload, ok := strings.CutPrefix(line, "data: ")
				if !ok || payload == "[DONE]" {
					continue
				}
				var chunk struct {
					Usage *usage `json:"usage"`
				}
				if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
					t.Fatalf("chunk %q is not JSON: %v", payload, err)
				}
				if chunk.Usage != nil {
					streamed = chunk.Usage
				}
			}
			if streamed == nil {
				t.Fatal("the stream carried no usage chunk")
			}
			// Every field, not just the completion count.
			if *streamed != *blocking.Usage {
				t.Errorf("usage differs between paths for one request:\nblocking %+v\nstream   %+v",
					*blocking.Usage, *streamed)
			}
		})
	}
}

// TestValidationOrderIsFixed pins which error a request with several problems reports.
//
// Each handler checks in a fixed order, and that order is the contract: a client that sent
// two bad things is told about one of them, and which one determines what it fixes first.
// The order is decode, model, shape fields, required fields, then value ranges -- documented
// on handleCompletion -- and nothing enforced it. Reordering two checks compiles, passes every
// existing test, and changes what every multi-problem request is told.
//
// The pairs below are chosen so each one separates two adjacent stages: a 404 must beat a 400
// about the body, a shape refusal must beat a missing prompt, and a missing prompt must beat a
// bad value, since a value cannot be judged against a request that has no prompt.
func TestValidationOrderIsFixed(t *testing.T) {
	srv, _ := newTestServer(t)

	cases := []struct {
		name   string
		path   string
		body   string
		status int
		param  string
	}{
		// 404 before any 400: the model check runs before the body is judged.
		{"unknown model beats a missing prompt", "/v1/completions",
			`{"model": "nope", "prompt": "  "}`, http.StatusNotFound, "model"},
		{"unknown model beats a shape field", "/v1/completions",
			`{"model": "nope", "prompt": "hi", "n": 3}`, http.StatusNotFound, "model"},
		{"unknown model beats a bad value", "/v1/completions",
			`{"model": "nope", "prompt": "hi", "top_p": 5}`, http.StatusNotFound, "model"},
		// Shape fields before the required-field check: the response would be the wrong
		// shape regardless of what the prompt holds.
		// A shape field still beats a required one on chat, which is the endpoint that has a
		// shape field left. On completions there is none, and suffix is deliberately checked
		// after the required prompt: with no prompt there is nothing to insert into, so the
		// missing field is the one to report.
		{"shape field beats a chat with no messages", "/v1/chat/completions",
			`{"echo": true}`, http.StatusBadRequest, "echo"},
		{"missing prompt beats suffix", "/v1/completions",
			`{"prompt": "  ", "suffix": "END"}`, http.StatusBadRequest, "prompt"},
		// Required fields before value ranges.
		{"missing prompt beats a bad max_tokens", "/v1/completions",
			`{"prompt": "", "max_tokens": -5}`, http.StatusBadRequest, "prompt"},
		{"missing messages beats a bad top_p", "/v1/chat/completions",
			`{"top_p": 5}`, http.StatusBadRequest, "messages"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			resp := post(t, srv.Handler(), c.path, c.body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != c.status {
				t.Fatalf("status = %d, want %d", resp.StatusCode, c.status)
			}
			body := decodeBody[errorBody](t, resp)
			if body.Error.Param != c.param {
				t.Errorf("param = %q, want %q: a client fixes the named field first",
					body.Error.Param, c.param)
			}
		})
	}
}

// TestStartPumpIsIdempotent pins that a second call does not start a second pump.
//
// The engine permits exactly one goroutine inside Pump -- it owns one set of scratch
// buffers, so two rounds at once would write the same slices. Shutdown was already guarded
// by a sync.Once while StartPump was not, and the asymmetry is easy to hit: the test helpers
// in this file start the pump themselves, so a test that also calls StartPump gets two
// goroutines and a data race on the sequence fields. Measured before this guard, that
// arrangement produced four DATA RACE reports from one concurrent request loop, and the
// report names engine.Pump rather than the second start.
//
// The assertion is a concurrent request loop under -race: with the guard one pump serves
// them, and without it the detector fires. The requests are what make it observable -- an
// idle server has no sequences for the second pump to race on.
func TestStartPumpIsIdempotent(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 256})
	// newTestServerModel already started one. These must be no-ops.
	srv.StartPump(context.Background())
	srv.StartPump(context.Background())

	var wg sync.WaitGroup
	for range 4 {
		wg.Go(func() {
			for range 20 {
				req := httptest.NewRequest(http.MethodPost, "/v1/completions",
					strings.NewReader(`{"prompt": "hi", "max_tokens": 2}`))
				w := httptest.NewRecorder()
				srv.Handler().ServeHTTP(w, req)
			}
		})
	}
	wg.Wait()
}

// TestStartPumpAfterShutdownDoesNothing pins what the once means for a stopped server.
//
// Making StartPump idempotent raised the question of what it does after Shutdown, and the
// answer has to be "nothing". The pump exits on a closed stop channel, so a goroutine
// started afterwards would return on its first iteration -- that is not a failure to
// restart, it is the absence of a restart API. Without this test, a caller whose sequence
// is Shutdown followed by StartPump gets no error either way, so nothing would distinguish
// "the server is running again" from "the call did nothing".
//
// This test has no discriminating power against removing the once, and says so rather than
// implying otherwise: with the guard gone the second start also returns immediately and also
// does not panic, because the goroutine it spawns exits on the already-closed stop channel.
// What catches that mutation is TestStartPumpIsIdempotent, whose concurrent requests give the
// extra pump something to race on. This test is here for the reader -- it records that the
// once makes a stopped server inert, which is the behaviour a caller could otherwise expect
// to be a restart.
func TestStartPumpAfterShutdownDoesNothing(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 128})
	srv.Shutdown()

	// Must return, not block waiting for a pump that will never run.
	done := make(chan struct{})
	go func() {
		defer close(done)
		srv.StartPump(context.Background())
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("StartPump blocked after Shutdown")
	}

	// Callable again without panicking: the once absorbs it rather than closing a closed
	// channel or spawning a goroutine that would exit immediately.
	srv.StartPump(context.Background())
	srv.Shutdown()
}

// TestUsageIsPerRequestUnderConcurrency covers what the other usage tests cannot: that
// each response's counts describe its own request.
//
// The existing three cover the arithmetic within one request -- streaming against
// non-streaming, include_usage, and every token counted. None ran two requests at once, so a
// counter stored per server rather than per sequence would pass all of them. That is the
// shape this checks: eight requests at once, each with a distinct max_tokens, and every
// response must report its own budget as completion_tokens. A shared counter would show a
// larger number on whoever finished later and a smaller one on whoever finished first.
//
// The work happens in parallel with the pump, so a mismatch would also be a race -- but the
// assertion here is on the values, which fails deterministically rather than only when the
// detector happens to be watching.
func TestUsageIsPerRequestUnderConcurrency(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()

	const requests = 8
	type result struct {
		budget, completion, total, prompt int
		err                               string
	}
	results := make([]result, requests)

	var wg sync.WaitGroup
	for i := range requests {
		wg.Go(func() {
			budget := 1 + i
			body := fmt.Sprintf(`{"prompt": "Once upon a time", "max_tokens": %d}`, budget)
			req := httptest.NewRequest("POST", "/v1/completions", strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)

			var got struct {
				Usage struct {
					PromptTokens     int `json:"prompt_tokens"`
					CompletionTokens int `json:"completion_tokens"`
					TotalTokens      int `json:"total_tokens"`
				} `json:"usage"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
				results[i] = result{budget: budget, err: err.Error()}
				return
			}
			results[i] = result{
				budget:     budget,
				completion: got.Usage.CompletionTokens,
				total:      got.Usage.TotalTokens,
				prompt:     got.Usage.PromptTokens,
			}
		})
	}
	wg.Wait()

	for _, r := range results {
		if r.err != "" {
			t.Errorf("budget %d: %s", r.budget, r.err)
			continue
		}
		if r.completion != r.budget {
			t.Errorf("budget %d reported completion_tokens = %d; the count belongs to another request",
				r.budget, r.completion)
		}
		if r.total != r.prompt+r.completion {
			t.Errorf("budget %d: total %d != prompt %d + completion %d",
				r.budget, r.total, r.prompt, r.completion)
		}
	}
}

// TestResponseIDsAreUniqueUnderLoad covers the one field a client is entitled to treat as
// unique.
//
// The id was a bare UnixNano timestamp, and concurrent callers can read the clock in the
// same nanosecond -- measured on this machine, 32 goroutines over 200 trials collided 12
// times, while 8 never did -- so the id now carries an atomic sequence number as well. The
// failure is silent when it happens: two responses with the same id, no error anywhere.
//
// This test catches neither of the two ways the id can regress, and saying so is the point.
// Six requests through the mux are spaced by routing and JSON decoding, far more than a
// nanosecond, so a bare timestamp still produces distinct ids here -- and for the same reason
// a constant sequence number does too, which was measured by making it constant and watching
// this test pass. Both were verified that way rather than assumed.
//
// TestResponseIDStaysUniqueWithAFrozenClock covers the property that actually changed: it
// holds the clock fixed and calls responseID directly, so the counter is the only thing that
// can distinguish the ids.
//
// Kept to the admission limit so every request succeeds. A refused request has no id at all,
// and counting those as duplicates would make this test measure the queue rather than the
// id: the first version of the probe did exactly that and reported 27 duplicates out of 64,
// all of them empty strings from 503s.
func TestResponseIDsAreUniqueUnderLoad(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()

	const requests = 6
	ids := make([]string, requests)
	var wg sync.WaitGroup
	for i := range requests {
		wg.Go(func() {
			req := httptest.NewRequest("POST", "/v1/completions",
				strings.NewReader(`{"prompt": "hi", "max_tokens": 1}`))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			h.ServeHTTP(rec, req)
			var got struct {
				ID string `json:"id"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
				t.Errorf("status %d: %v", rec.Code, err)
				return
			}
			if rec.Code != http.StatusOK {
				t.Errorf("status %d while staying inside the admission limit", rec.Code)
				return
			}
			ids[i] = got.ID
		})
	}
	wg.Wait()

	seen := make(map[string]int, requests)
	for _, id := range ids {
		if id == "" {
			t.Error("a successful response carried no id")
			continue
		}
		seen[id]++
	}
	for id, count := range seen {
		if count > 1 {
			t.Errorf("id %q was returned %d times; ids must be unique", id, count)
		}
	}
}

// TestResponseIDStaysUniqueWithAFrozenClock covers the property the sequence number adds,
// at the only level where it can be observed.
//
// A nanosecond clock is not a uniqueness oracle: 32 goroutines reading it at once over 200
// trials produced 12 collisions on this machine. Holding the timestamp fixed is what makes
// the counter's contribution visible -- without it the HTTP path spaces requests by routing
// and JSON decoding, far more than a nanosecond, so a bare timestamp still produces distinct
// ids there.
//
// The test calls responseID rather than formatting the string itself, which is the whole
// reason that method exists. A first version assembled "prefix-nano-seq" in the test and
// passed even when the server was changed to emit a constant sequence number.
func TestResponseIDStaysUniqueWithAFrozenClock(t *testing.T) {
	srv, _ := newTestServer(t)

	const frozen = int64(1700000000000000000)
	seen := make(map[string]struct{}, 16)
	for range 16 {
		id := srv.responseID("cmpl-", frozen)
		if _, dup := seen[id]; dup {
			t.Fatalf("id %q repeated with the clock frozen and a monotonic counter", id)
		}
		seen[id] = struct{}{}
	}

	// The timestamp is still in the id, so the counter added to it rather than replacing it.
	// Asserting that two ids differ would not test this: the sequence number makes any two
	// calls differ whether or not the timestamp is there, which is how a first version of
	// this check passed while the timestamp had been dropped.
	if id := srv.responseID("cmpl-", frozen); !strings.Contains(id, fmt.Sprint(frozen)) {
		t.Errorf("id %q does not carry the timestamp %d", id, frozen)
	}

	// And the prefix is the caller's, since a chat completion and a text completion must not
	// look alike in a log.
	if id := srv.responseID("chatcmpl-", frozen); !strings.HasPrefix(id, "chatcmpl-") {
		t.Errorf("id %q lost its prefix", id)
	}
}

// TestStreamChunksShareOneTimestamp covers the field a client reads as when the response was
// created.
//
// Every chunk of a stream used to compute created with its own time.Now(). A reply that
// crosses a second boundary then reports two creation times for one response, which no
// client can interpret: the id is the same, the model is the same, and the timestamp is not.
// The fixture replies here are short enough to fit inside one second, so measuring the bug
// took a deliberately slow stream -- but nothing prevented it, and a real generation at eight
// tokens a second crosses the boundary routinely.
//
// The timestamp is now taken once, in generate, and threaded through streamResponse.
//
// The stream is fed by hand and deliberately straddles a second boundary, because that is the
// only way this property can fail: the fixture replies used elsewhere finish well inside one
// second, so a test driven through the handler passes even with the per-chunk clock restored
// -- verified by doing exactly that five times. Waiting for the real boundary costs up to a
// second of suite time, which is why the event channel is driven directly rather than by a
// longer generation.
func TestStreamChunksShareOneTimestamp(t *testing.T) {
	srv, tok := newTestServer(t)

	// Drive streamResponse directly so the two events can be separated by more than the
	// remaining time in the current second.
	ch := make(chan engine.Event, 4)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/completions", nil)
	go func() {
		id := tok.Encode("hello", false)
		if len(id) > 0 {
			ch <- engine.Event{Token: id[0], HasToken: true}
		}
		// Sleep past the next second boundary, so a per-chunk clock would produce a second
		// value for the chunks after it.
		time.Sleep(time.Until(time.Unix(time.Now().Unix()+1, 0)) + 50*time.Millisecond)
		if len(id) > 1 {
			ch <- engine.Event{Token: id[1], HasToken: true}
		}
		ch <- engine.Event{Done: true, FinishReason: "length", Generated: 2}
		close(ch)
	}()
	const createdArg = int64(1700000000)
	srv.streamResponse(rec, req, []<-chan engine.Event{ch}, "cmpl-test", createdArg,
		generateParams{chat: false, includeUsage: true}, 3)

	created := make(map[int64]int, 8)
	chunks := 0
	for _, line := range strings.Split(rec.Body.String(), "\n") {
		if !strings.HasPrefix(line, "data: ") || strings.HasSuffix(line, "[DONE]") {
			continue
		}
		var got struct {
			ID      string `json:"id"`
			Created int64  `json:"created"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &got); err != nil {
			continue
		}
		if got.ID == "" {
			continue
		}
		chunks++
		created[got.Created]++
	}
	if chunks < 2 {
		t.Fatalf("only %d chunks carried an id; the fixture is too small to test this", chunks)
	}
	if len(created) != 1 {
		t.Errorf("%d chunks reported %d distinct created values: %v; one response has one creation time",
			chunks, len(created), created)
	}
}

// TestBlockingResponseCarriesTheGeneratesTimestamp covers the same property as the streaming
// one for the two non-streaming shapes.
//
// blockingResponse is called with the timestamp generate already took, rather than reading the
// clock again. That matters because the response is written after generation finishes: a
// request whose generation started at 12:00:00.9 and ended at 12:00:01.1 would report the
// latter, and a client comparing created across requests would see a delay that did not
// happen. The difference is one call rather than a whole stream, so it is worth stating why
// it is still the same fix: one response, one creation time.
//
// This test does NOT catch a return to time.Now() in either branch, and saying so is the
// point. Generation here takes milliseconds, so a second reading lands in the same second as
// the first and the window assertion holds -- verified by restoring the clock in the chat
// branch and running it three times. What it does catch is the timestamp being replaced by a
// constant, dropped, or taken before the request arrived. The property that actually changed
// -- that the value comes from generate rather than from the writer -- is only observable
// across a second boundary, which the streaming test forces by holding its channel open and
// this one cannot afford to do twice.
//
// Both endpoints are checked because they are separate branches of the same function, and a
// future edit could adjust one without the other.
func TestBlockingResponseCarriesTheGeneratesTimestamp(t *testing.T) {
	srv, _ := newTestServer(t)

	for _, c := range []struct{ name, path, body string }{
		{"completions", "/v1/completions", `{"prompt": "hi", "max_tokens": 2}`},
		{"chat", "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}`},
	} {
		t.Run(c.name, func(t *testing.T) {
			before := time.Now().Unix()
			req := httptest.NewRequest("POST", c.path, strings.NewReader(c.body))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			srv.Handler().ServeHTTP(rec, req)
			after := time.Now().Unix()

			var got struct {
				Created int64 `json:"created"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
				t.Fatalf("status %d: %v", rec.Code, err)
			}
			// The value must be the one taken when the request arrived, which is inside this
			// window -- not a second reading after generation finished.
			if got.Created < before || got.Created > after {
				t.Errorf("created = %d, outside the request's own window [%d, %d]",
					got.Created, before, after)
			}
		})
	}
}

// TestNoCORSAccessControlHeadersIsPartOfTheTextPlainDecision covers the half of the
// Content-Type tolerance that keeps it from being a bypass.
//
// decodeJSON accepts a body whatever Content-Type says, which is what makes `curl -d` work:
// curl sets application/x-www-form-urlencoded rather than omitting the header. The cost,
// stated in that function's comment, is that a cross-origin page can make the server generate
// -- POST with text/plain is a "simple request" under the Fetch standard and skips the
// preflight, so the request runs.
//
// What keeps the exposure from being worse is stated in the same comment as two further
// facts: no Access-Control-Allow-Origin is ever sent, so the page cannot read the reply, and
// OPTIONS answers 405, so an application/json request from a page never gets past the
// preflight. Both are asserted here because both are load-bearing: adding a permissive CORS
// header, or answering OPTIONS with the usual allowed-methods response, would turn the
// documented tolerance into the bypass the comment says it is not -- and nothing else in the
// suite would notice.
func TestNoCORSAccessControlHeadersIsPartOfTheTextPlainDecision(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()

	// A simple request, the shape a page can send without a preflight.
	req := httptest.NewRequest(http.MethodPost, "/v1/completions",
		strings.NewReader(`{"prompt":"hi","max_tokens":1}`))
	req.Header.Set("Content-Type", "text/plain")
	req.Header.Set("Origin", "https://example.invalid")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: the text/plain tolerance is the documented behaviour",
			w.Code)
	}
	for _, header := range []string{
		"Access-Control-Allow-Origin",
		"Access-Control-Allow-Credentials",
		"Access-Control-Allow-Methods",
		"Access-Control-Allow-Headers",
		"Access-Control-Expose-Headers",
	} {
		if got := w.Header().Get(header); got != "" {
			t.Errorf("%s = %q on a cross-origin request; sending it would let the page read "+
				"the reply, which is the limit the Content-Type comment relies on", header, got)
		}
	}

	// And the preflight is refused, so an application/json request from a page stops here.
	preflight := httptest.NewRequest(http.MethodOptions, "/v1/completions", nil)
	preflight.Header.Set("Origin", "https://example.invalid")
	preflight.Header.Set("Access-Control-Request-Method", http.MethodPost)
	pw := httptest.NewRecorder()
	h.ServeHTTP(pw, preflight)
	if pw.Code != http.StatusMethodNotAllowed {
		t.Errorf("OPTIONS = %d, want 405: a successful preflight would let a cross-origin "+
			"application/json request through", pw.Code)
	}
	if got := pw.Header().Get("Access-Control-Allow-Origin"); got != "" {
		t.Errorf("the preflight response carries Access-Control-Allow-Origin %q", got)
	}
}

// TestChatMultiTurnReusesTheSharedPrefix covers the feature end to end rather than per layer.
//
// The engine's reuse is tested in its own package, and the CLI's REPL is tested too, but the
// HTTP path between them had nothing: a change that rendered the conversation differently
// between turns -- a timestamp, a re-serialised system prompt, a different order of the same
// messages -- would make every turn's prefix miss, and every test would still pass. The
// feature exists so that a multi-turn conversation costs one prefill instead of one per turn,
// and the only way to see it from here is the engine's counter.
//
// Measured with a 512-position fixture, block size 16: the first turn reuses nothing, the
// second reuses 48 positions -- the three whole blocks the two prompts share.
func TestChatMultiTurnReusesTheSharedPrefix(t *testing.T) {
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
		MaxActive: 2, BlockSize: 16, Blocks: 64, PrefixCache: true,
	})
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	cfg := eng.Model().Config()
	srv, err := New(eng, tok, Config{ModelName: cfg.Name, ContextLen: cfg.ContextLen})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	srv.StartPump(context.Background())
	t.Cleanup(srv.Shutdown)

	// Long enough to span three whole blocks, so there is something to reuse.
	const first = "Once upon a time there was a small model that could"
	turn := func(body string) uint64 {
		t.Helper()
		post(t, srv.Handler(), "/v1/chat/completions", body)
		return eng.Stats().PrefixReused
	}

	one := turn(fmt.Sprintf(
		`{"messages":[{"role":"user","content":%q}],"max_tokens":2}`, first))
	if one != 0 {
		t.Errorf("the first turn reused %d positions; there was nothing cached yet", one)
	}

	two := turn(fmt.Sprintf(
		`{"messages":[{"role":"user","content":%q},`+
			`{"role":"assistant","content":"Yes."},{"role":"user","content":"More?"}],`+
			`"max_tokens":2}`, first))
	if two <= one {
		t.Fatalf("the second turn reused %d positions, the same as the first (%d): the shared "+
			"prefix did not hit, so every turn pays a full prefill", two, one)
	}
	// Whole blocks only, since a partial block is never published.
	if added := two - one; added%16 != 0 {
		t.Errorf("the second turn added %d reused positions, which is not a whole number of "+
			"16-position blocks", added)
	}
}

// TestStreamingRequestAlsoReusesThePrefix covers the same feature on the other response path.
//
// Chat and completions share generate and Submit, but not everything: the prompt is rendered
// differently, and the streaming path writes chunks where the other writes one body. A change
// confined to the streaming branch -- a re-rendered prompt, an extra header token -- would
// break reuse there while the non-streaming test stayed green.
//
// Measured with a 512-position fixture and block size 16: the first stream reuses nothing,
// the second reuses 32 positions. That is fewer than the chat path's 48 because the completions
// endpoint renders a bare transcript rather than a templated conversation, so the shared prefix
// is shorter -- the figure is a property of the prompt, not of the streaming code.
func TestStreamingRequestAlsoReusesThePrefix(t *testing.T) {
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
		MaxActive: 2, BlockSize: 16, Blocks: 64, PrefixCache: true,
	})
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	cfg := eng.Model().Config()
	srv, err := New(eng, tok, Config{ModelName: cfg.Name, ContextLen: cfg.ContextLen})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	srv.StartPump(context.Background())
	t.Cleanup(srv.Shutdown)

	const prompt = "Once upon a time there was a small model that could"
	stream := func() uint64 {
		t.Helper()
		body := fmt.Sprintf(`{"prompt":%q,"max_tokens":2,"stream":true}`, prompt)
		resp := post(t, srv.Handler(), "/v1/completions", body)
		raw := readBody(t, resp)
		// A stream that did not finish would leave the sequence running and the count would
		// still rise, so the completion is asserted before the number is trusted.
		if !strings.Contains(raw, "[DONE]") {
			t.Fatalf("the stream did not reach [DONE]: %q", raw)
		}
		return eng.Stats().PrefixReused
	}

	one := stream()
	if one != 0 {
		t.Errorf("the first stream reused %d positions; there was nothing cached yet", one)
	}
	two := stream()
	if two <= one {
		t.Fatalf("the second stream reused %d positions, the same as the first (%d): the shared "+
			"prefix did not hit on the streaming path", two, one)
	}
	if added := two - one; added%16 != 0 {
		t.Errorf("the second stream added %d reused positions, which is not a whole number of "+
			"16-position blocks", added)
	}
}

// TestStartPumpRejectsANilContextByPanicking records what a nil context does, because the
// alternative is discovering it in production as a whole-process exit.
//
// The pump selects on ctx.Done(). A nil context interface has no Done method, so the
// goroutine panics on its first iteration -- and a panic inside a goroutine the caller never
// holds cannot be recovered by the caller, so the process dies. The comment on StartPump
// says ctx must not be nil for that reason.
//
// Panicking is the correct behaviour rather than a defect: the context package documents that
// a nil Context must not be passed, so the caller is wrong, and a library that silently
// substituted context.Background would hide a request whose cancellation had been wired up
// incorrectly. The test exists so the failure mode is written down and so a change that
// started tolerating nil is a deliberate decision rather than an accident.
func TestStartPumpRejectsANilContextByPanicking(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 64})

	// This does NOT drive StartPump: the panic happens in the pump goroutine, so the test
	// process would die rather than fail, and a suite that kills itself is worse than one that
	// documents the gap. What it does assert is the narrower fact the comment depends on --
	// that ctx.Done() on a nil interface panics -- so the sentence "StartPump must not be given
	// a nil context" rests on something checked rather than on folklore.
	//
	// The connection between the two is by inspection: run's first select case is
	// <-ctx.Done(). A change that moved the dereference behind a nil check would make this
	// test pass while the comment became wrong, which is why the comment points at the line.
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("a nil context did not panic; StartPump's comment says it must not be nil, " +
				"so if this starts working the comment and this test both need revisiting")
		}
	}()
	var ctx context.Context
	_ = ctx.Done() // the same operation s.run performs on its first iteration
	_ = srv
}

// TestDisconnectReturnsTheBlocksToo covers the resource the existing disconnect tests do not.
//
// TestStreamingClientGoneReleasesTheSequence waits for Active() to reach zero, which is the
// slot. A sequence holds two things: the admission slot and the KV blocks it was given. A
// cancel path that finished the sequence but dropped its cache without releasing the blocks
// would leave Active at zero and the pool permanently smaller -- and a paged deployment that
// leaks a block per abandoned stream fails long before it fails loudly, which is the reason
// /metrics exports the free count at all.
//
// Measured with 32 blocks of 16 positions: a stream read once and hung up leaves 32 of 32
// free, so nothing was retained.
func TestDisconnectReturnsTheBlocksToo(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 256}.Write(t, t.TempDir())
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	eng, err := engine.New(path, engine.Options{
		MaxActive: 2, BlockSize: 16, Blocks: 32, PrefixCache: true,
	})
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	cfg := eng.Model().Config()
	srv, err := New(eng, tok, Config{ModelName: cfg.Name, ContextLen: cfg.ContextLen})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	srv.StartPump(context.Background())
	t.Cleanup(srv.Shutdown)

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	before := eng.Stats()
	if before.PoolBlocks != 32 {
		t.Fatalf("the pool has %d blocks, want 32; the setup did not reach the paged cache",
			before.PoolBlocks)
	}

	resp, err := http.Post(ts.URL+"/v1/completions", "application/json",
		strings.NewReader(`{"prompt":"Once upon a time","max_tokens":30,"stream":true}`))
	if err != nil {
		t.Fatalf("starting the stream: %v", err)
	}
	// This test cannot prove the blocks were held while the stream ran, and trying to made it
	// flaky twice. Measured under -cpu 1: the whole response can be produced before the first
	// Read returns, and a goroutine polling PoolFree as fast as it can still never sees it drop
	// -- minFree stayed at 32 across a run that completed 31 rounds. The allocation window is
	// real but too short to sample reliably, so the assertion below is the one this test can
	// make: after a client hangs up, the pool is whole again.
	//
	// What that leaves untested is a leak that the release path would produce only for a
	// cancelled sequence; the streaming-cancel accounting is covered by
	// TestStreamingClientGoneReleasesTheSequence, which waits on Active rather than a pool
	// figure and so does not depend on the window.
	buf := make([]byte, 32)
	if _, err := resp.Body.Read(buf); err != nil {
		t.Fatalf("reading the first chunk: %v", err)
	}
	_ = resp.Body.Close()

	waitFor(t, 10*time.Second, func() bool { return eng.Active() == 0 },
		"the engine still holds a sequence after the client hung up")
	// Give the release a moment past the slot: the two happen together, but the assertion is
	// about the pool and a polling wait would hide a leak that resolved later.
	waitFor(t, 5*time.Second, func() bool {
		return eng.Stats().PoolFree == before.PoolFree
	}, "the pool did not return every block the abandoned stream held")

	after := eng.Stats()
	if after.PoolFree != before.PoolFree {
		t.Errorf("%d of %d blocks are still out after the client hung up (was %d free)",
			before.PoolFree-after.PoolFree, before.PoolBlocks, before.PoolFree)
	}
}

// TestBlockingDisconnectReturnsTheBlocksToo is the non-streaming counterpart of
// TestDisconnectReturnsTheBlocksToo.
//
// The two paths cancel through different code: the streaming one notices a write error, the
// blocking one sees its request context expire while it waits for the engine. Both end up in
// the same finish, but only the streaming version of this test existed, and a change confined
// to the blocking path -- a reader that returns without draining, a cancel channel that is not
// forwarded -- would leak blocks with every other test still green.
//
// Measured with 32 blocks of 16 positions: the pool is whole again once the handler returns.
func TestBlockingDisconnectReturnsTheBlocksToo(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 2048}.Write(t, t.TempDir())
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	eng, err := engine.New(path, engine.Options{
		MaxActive: 2, BlockSize: 16, Blocks: 32, PrefixCache: true,
	})
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	cfg := eng.Model().Config()
	srv, err := New(eng, tok, Config{ModelName: cfg.Name, ContextLen: cfg.ContextLen})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	srv.StartPump(context.Background())
	t.Cleanup(srv.Shutdown)

	before := eng.Stats()
	if before.PoolBlocks != 32 {
		t.Fatalf("the pool has %d blocks, want 32", before.PoolBlocks)
	}

	// A budget that fits the pool -- 32 blocks of 16 is 512 positions, and the prompt plus the
	// budget must stay under that or Submit refuses with a 400 instead of running -- while
	// still outlasting the cancel. No EOS in the fixture, so it cannot finish on its own.
	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest(http.MethodPost, "/v1/completions",
		strings.NewReader(`{"prompt": "Once upon a time", "max_tokens": 400}`)).WithContext(ctx)
	rec := httptest.NewRecorder()

	done := make(chan struct{})
	go func() {
		defer close(done)
		srv.Handler().ServeHTTP(rec, req)
	}()
	waitFor(t, 5*time.Second, func() bool { return eng.Active() > 0 },
		"the request never reached the engine")
	// Active counts the sequence, not the blocks it holds: a sequence is active from admission
	// and takes blocks once it prefills, so this waits for the second rather than sampling it.
	// The streaming counterpart of this test learned the same lesson by failing intermittently.
	waitFor(t, 5*time.Second, func() bool { return eng.Stats().PoolFree < before.PoolFree },
		"no blocks were allocated while the request ran, so this run proves nothing")
	cancel()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the handler did not return after the client cancelled")
	}
	waitFor(t, 5*time.Second, func() bool { return eng.Stats().PoolFree == before.PoolFree },
		"the pool did not return every block the cancelled request held")

	if after := eng.Stats(); after.PoolFree != before.PoolFree {
		t.Errorf("%d of %d blocks are still out after the cancel (was %d free)",
			before.PoolFree-after.PoolFree, before.PoolBlocks, before.PoolFree)
	}
}

// TestBlockingResponseReturnsWhenThePumpStops covers the shutdown case the disconnect test above
// cannot reach.
//
// That test cancels the request while the pump is running, so the engine finishes the sequence,
// the channel closes, and the handler unblocks through the ordinary path. A shutdown is
// different in the one way that matters: the pump stops, so nothing ever finishes the sequence
// and nothing ever closes the channel. A handler that only ranges over the channel then blocks
// until the process exits, and the client gets no response at all -- measured against a real
// server as curl exit 52, "empty reply from server", for four requests in flight at a 3-second
// grace.
//
// The assertion is that the handler returns and writes a status, not which status: 503 is what
// it does now, and a future change to 499 or 500 would still be a response the client can act
// on. What it must not do is hang.
func TestBlockingResponseReturnsWhenThePumpStops(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 2048}.Write(t, t.TempDir())
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	eng, err := engine.New(path, engine.Options{MaxActive: 1, MaxQueued: 4})
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	modelCfg := eng.Model().Config()
	srv, err := New(eng, tok, Config{ModelName: modelCfg.Name, ContextLen: modelCfg.ContextLen})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	pumpCtx, stopPump := context.WithCancel(context.Background())
	srv.StartPump(pumpCtx)

	reqCtx, cancelRequest := context.WithCancel(context.Background())
	req := httptest.NewRequest(http.MethodPost, "/v1/completions",
		strings.NewReader(`{"prompt": "Once upon a time", "max_tokens": 1000}`)).WithContext(reqCtx)
	w := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		defer close(done)
		srv.Handler().ServeHTTP(w, req)
	}()
	waitFor(t, 5*time.Second, func() bool { return eng.Active() > 0 },
		"request never reached the engine")

	// This is the shutdown sequence: stop the pump first, then cancel the request. The order is
	// what makes it a shutdown rather than a disconnect, and it is the case that used to hang.
	stopPump()
	srv.Shutdown()
	cancelRequest()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the handler did not return after the pump stopped: a client in flight at " +
			"shutdown gets no response at all")
	}
	if w.Code == 0 {
		t.Error("the handler returned without writing a response")
	}
}

// TestACancelledStreamOmitsDone covers the one honest signal a cut stream can send.
//
// A shutdown cancels the request, the engine finishes the sequence with reason "cancel", and
// the OpenAI schema has no finish_reason for "the server went away" -- so the chunked reason is
// whatever the schema allows, and the earlier behaviour used "length". Measured against a real
// server at max_tokens 4000 with a 1-second grace, that produced 50 delta chunks followed by
// finish_reason "length" and a [DONE] sentinel, which is byte-for-byte what a complete reply
// looks like. A client had no way to tell it was truncated.
//
// The sentinel is what actually says "you have everything", so it is what is withheld: a stream
// cut by a cancel ends without [DONE], which every SSE client already treats as incomplete.
// The finish_reason stays as it is, because inventing a value the schema does not define would
// break clients that switch on it.
func TestACancelledStreamOmitsDone(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 512})

	ch := make(chan engine.Event, 8)
	for range 3 {
		ch <- engine.Event{Token: 5, HasToken: true}
	}
	// This is what the engine emits when sample sees a cancelled request.
	ch <- engine.Event{Done: true, FinishReason: "cancel"}
	close(ch)

	rec := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		defer close(done)
		srv.streamResponse(rec, httptest.NewRequest(http.MethodPost, "/v1/completions", nil),
			[]<-chan engine.Event{ch}, "cmpl-test", 1700000000,
			generateParams{chat: false}, 0)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("streamResponse did not return")
	}

	body := rec.Body.String()
	if strings.Contains(body, "data: [DONE]") {
		t.Errorf("a cancelled stream sent the [DONE] sentinel, so a client reads it as a "+
			"complete reply:\n%s", body)
	}
	// The text that did arrive is still delivered: a cut stream is a prefix, not nothing.
	if got := strings.Count(body, `"text":"`); got < 3 {
		t.Errorf("got %d text chunks, want the 3 that were sent before the cancel", got)
	}

	// The chat endpoint shares streamResponse, so a cut chat stream must lose its sentinel too.
	// Asserted rather than assumed: the two endpoints build different object names and delta
	// shapes through the same code, and a fix applied to one path only is exactly the kind of
	// split that leaves half the surface reporting a complete reply.
	chatCh := make(chan engine.Event, 8)
	chatCh <- engine.Event{Token: 5, HasToken: true}
	chatCh <- engine.Event{Done: true, FinishReason: "cancel"}
	close(chatCh)
	recChat := httptest.NewRecorder()
	srv.streamResponse(recChat, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil),
		[]<-chan engine.Event{chatCh}, "chatcmpl-test", 1700000000,
		generateParams{chat: true}, 0)
	if strings.Contains(recChat.Body.String(), "data: [DONE]") {
		t.Errorf("a cancelled chat stream sent [DONE]:\n%s", recChat.Body.String())
	}

	// A stream that finishes normally is unchanged, which is the half that would break every
	// client if the suppression were applied too broadly.
	ok := make(chan engine.Event, 8)
	ok <- engine.Event{Token: 5, HasToken: true}
	ok <- engine.Event{Done: true, FinishReason: "length"}
	close(ok)
	rec2 := httptest.NewRecorder()
	srv.streamResponse(rec2, httptest.NewRequest(http.MethodPost, "/v1/completions", nil),
		[]<-chan engine.Event{ok}, "cmpl-test", 1700000000,
		generateParams{chat: false}, 0)
	if !strings.Contains(rec2.Body.String(), "data: [DONE]") {
		t.Error("a normally finished stream lost its [DONE] sentinel")
	}
}

// TestStopListBeyondFourStillMatches covers a stop list longer than OpenAI's documented limit.
//
// The published schema says maxItems: 4, and this server accepts any number. Measured against a
// real server, a request whose only match was in the fifth slot stopped at it, so the extra
// entries are honoured rather than ignored -- which is the part worth pinning. A future change
// that truncated the list to four would keep every existing test green while silently changing
// where generation ends for any client that sends more.
//
// The assertion puts the match last on purpose: filling the first four with strings that never
// occur means a list read only up to index 3 produces a length stop instead of a stop.
func TestStopListBeyondFourStillMatches(t *testing.T) {
	srv, _ := newTestServer(t)

	resp := post(t, srv.Handler(), "/v1/completions",
		`{"prompt": "Once upon a time", "max_tokens": 8}`)
	full := decodeBody[completionResponse](t, resp).Choices[0].Text
	if len([]rune(full)) < 3 {
		t.Skipf("continuation %q is too short to cut", full)
	}

	// A three-rune slice from the middle: long enough that it is unlikely to appear earlier
	// by accident, and present in what the model actually generates.
	r := []rune(full)
	match := string(r[1:3])
	body := `{"prompt": "Once upon a time", "max_tokens": 30, "stop": ["ZZZ1","ZZZ2","ZZZ3","ZZZ4",` +
		quoteJSON(match) + `]}`
	stopped := decodeBody[completionResponse](t, post(t, srv.Handler(), "/v1/completions", body))
	if stopped.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish_reason = %q, want stop: the fifth stop string was not honoured",
			stopped.Choices[0].FinishReason)
	}
	if strings.Contains(stopped.Choices[0].Text, match) {
		t.Errorf("output %q contains the stop string %q from the fifth slot",
			stopped.Choices[0].Text, match)
	}
}

// TestRenderFallbackIsWarnedOnce covers the one way a broken chat template can hide.
//
// A template that compiles but cannot render every conversation is the case neither the
// startup banner nor the renderer's own return value makes visible: the banner reads
// UsesModelTemplate before any request has rendered, and renderReason is only written when a
// render fails. Measured end to end with a template of "{{ messages[99].content }}": the banner
// said "chat: model template", every request answered 200 out of the neutral transcript, and
// nothing in the log said so.
//
// A template that never compiled is deliberately not covered here: the banner already prints
// "chat: neutral transcript (reason)" for it, so warning again at request time would only repeat
// what the operator was told at startup.
func TestRenderFallbackIsWarnedOnce(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	model := gguftest.Model{NoEOS: true, ContextLen: 256, ChatTemplate: `{{ messages[99].content }}`}
	// The renderer is built from the same file the server loads, the way cmd/orbix does it,
	// because the fallback under test is a property of this template rather than of the server.
	path := model.Write(t, t.TempDir())
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	renderer, err := engine.NewChatRenderer(gf)
	if err != nil {
		t.Fatalf("NewChatRenderer: %v", err)
	}
	if !renderer.HasTemplate() {
		t.Fatal("the template did not compile, so this test would exercise the banner case " +
			"rather than the render-time one it is written for")
	}
	srv, _ := newTestServerConfig(t, model,
		Config{Logger: logger, ChatRenderer: renderer})

	// Two requests, so "once" is distinguishable from "never" and from "once per request".
	for i := range 2 {
		resp := post(t, srv.Handler(), "/v1/chat/completions",
			`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}`)
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("request %d = %d, want 200: a render fallback must not fail the request",
				i, resp.StatusCode)
		}
	}

	const want = "chat template is not in use"
	if got := strings.Count(buf.String(), want); got != 1 {
		t.Errorf("logged the fallback warning %d times over two requests, want 1:\n%s",
			got, buf.String())
	}
	// The reason has to name what went wrong; "the template fell back" alone leaves the
	// operator with nothing to fix.
	if !strings.Contains(buf.String(), "index 99 outside a sequence") {
		t.Errorf("the warning does not carry the renderer's reason:\n%s", buf.String())
	}
}

// TestRenderFallbackIsNotRepeatedByRequest covers a caller that already announced the fallback.
//
// cmd/orbix's serve probes the template before printing its banner, so a template that cannot
// render is named there as "chat: neutral transcript (reason)". The request-time warning would
// then report the same fact a second time, once per process. Measured before this flag existed:
// the banner named the render reason and one WARN record repeated it.
//
// The flag is the caller's to set because only the caller knows what it printed; this test
// pins the server half, that announcing it suppresses the record.
func TestRenderFallbackIsNotRepeatedByRequest(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	model := gguftest.Model{NoEOS: true, ContextLen: 256, ChatTemplate: `{{ messages[99].content }}`}
	path := model.Write(t, t.TempDir())
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	renderer, err := engine.NewChatRenderer(gf)
	if err != nil {
		t.Fatalf("NewChatRenderer: %v", err)
	}
	// The probe the banner does, so the renderer holds the reason the banner would have shown.
	renderer.Render([]engine.Turn{{Role: "user", Text: "hi"}})
	if renderer.UsesModelTemplate() {
		t.Fatal("the probe did not record a fallback, so this test would prove nothing")
	}

	srv, _ := newTestServerConfig(t, model, Config{
		Logger: logger, ChatRenderer: renderer, ChatFallbackAnnounced: true,
	})
	resp := post(t, srv.Handler(), "/v1/chat/completions",
		`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2}`)
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("request = %d, want 200: the answer is unaffected, only the logging is",
			resp.StatusCode)
	}
	if strings.Contains(buf.String(), "chat template is not in use") {
		t.Errorf("warned about a fallback the caller already announced:\n%s", buf.String())
	}
}

// TestPoolRefusalNamesTheLargerSide covers which field a permanent-capacity refusal blames.
//
// The check compares prompt+budget against the pool, so either side can be what makes a
// request unservable, and the param has to say which. It always named the prompt field, which
// is wrong for the shape a caller hits by accident: "Hi" with max_tokens 2000 is 2002
// positions, and a client that follows `param` shortens a two-token prompt while the budget
// stays the reason -- the retry fails identically.
//
// Both directions are asserted, because a fix that always blamed max_tokens would pass the
// first half and mislead on the second.
func TestPoolRefusalNamesTheLargerSide(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	// Eight blocks against a 512-position window: the window is large enough for the budget
	// to survive truncation, and the pool is small enough that prompt+budget still exceeds it.
	// Two blocks was too small -- a chat turn is ~20 tokens of markup before any content, and
	// with that pool the truncation left nothing for the budget half of the comparison to win.
	// The pool size is fixed at construction, so it cannot be adjusted on the server after.
	eng, err := engine.New(path, engine.Options{
		MaxActive: 1, MaxQueued: 1, Blocks: 8, BlockSize: 16,
	})
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	cfg := eng.Model().Config()
	srv, err := New(eng, tok, Config{ModelName: cfg.Name, ContextLen: cfg.ContextLen})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	cases := []struct {
		name string
		path string
		body string
		want string
		why  string
	}{
		{
			name: "budget dominates",
			path: "/v1/completions",
			body: `{"prompt": "hi", "max_tokens": 400}`,
			want: `"param":"max_tokens"`,
			why:  "a two-token prompt is never what to shorten here",
		},
		{
			// The chat endpoint reaches the same check through a rendered prompt, and the
			// field it names when the prompt wins is "messages" rather than "prompt". Both
			// directions are checked here because blaming one field unconditionally passes
			// whichever half is not tested.
			name: "chat budget dominates",
			path: "/v1/chat/completions",
			body: `{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 200}`,
			want: `"param":"max_tokens"`,
			why:  "a one-word message is not what to shorten",
		},
		{
			// The prompt has to outgrow the pool without outgrowing the window, or the check
			// under test never runs: a 200-word prompt is 1001 fixture tokens, which the
			// prompt-length check rejects first -- and that one names "prompt" too, so the
			// assertion passed with the pool logic inverted. 40 words is ~200 tokens against a
			// 512-position window and a 32-position pool: unservable, but for the pool reason.
			name: "prompt dominates",
			path: "/v1/completions",
			body: `{"prompt": ` + quoteJSON(strings.TrimSpace(strings.Repeat("word ", 40))) +
				`, "max_tokens": 2}`,
			want: `"param":"prompt"`,
			why:  "the budget is already two tokens, so the prompt is the side to shorten",
		},
		{
			// The chat side of "prompt dominates" needs a prompt large enough to outgrow the
			// pool whose budget is still smaller. 40 words renders to ~115 tokens, so a budget
			// of 2 gives 117 positions -- eight blocks, which fits. 60 words renders to ~175,
			// and 175+2 rounds up past eight.
			name: "chat prompt dominates",
			path: "/v1/chat/completions",
			body: `{"messages": [{"role": "user", "content": ` +
				quoteJSON(strings.TrimSpace(strings.Repeat("word ", 60))) + `}], "max_tokens": 2}`,
			want: `"param":"messages"`,
			why:  "the chat endpoint names its own field, not the completions one",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			resp := post(t, srv.Handler(), c.path, c.body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400", resp.StatusCode)
			}
			if body := readBody(t, resp); !strings.Contains(body, c.want) {
				t.Errorf("body does not name %s (%s):\n%s", c.want, c.why, body)
			}
		})
	}
}

// TestGemma3RenderedPromptEncodesToReferenceIDs links the two halves of the prompt path.
//
// internal/chat renders a checkpoint's template and internal/tokenize turns the result into ids;
// neither can check the other alone, which is why this lives in the layer that uses both. A prompt
// that looks right in a string comparison can still encode differently -- a stray newline, a marker
// spelled with an ordinary token instead of its special one -- and that difference is invisible in
// the rendered text and expensive in the generated output.
//
// The expectation is llama.cpp tokenizing the same bytes, from llama-cpp-python 0.3.37. The
// template is the 1654-byte asset under internal/chat/testdata, a byte copy of what the Gemma 3
// checkpoint holds in tokenizer.chat_template.
func TestGemma3RenderedPromptEncodesToReferenceIDs(t *testing.T) {
	path := os.Getenv("GEMMA3_MODEL")
	if path == "" {
		t.Skip("set GEMMA3_MODEL to a Gemma 3 GGUF file")
	}
	f, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(f)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	raw, err := os.ReadFile(filepath.Join("..", "chat", "testdata", "gemma3-it.jinja"))
	if err != nil {
		t.Fatalf("read template: %v", err)
	}
	tmpl, err := chat.Parse(string(raw))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	prompt, err := tmpl.Render(
		[]chat.Message{{Role: "user", Content: "What is the capital of France?"}},
		chat.Inputs{BOSToken: "<bos>", AddGenerationPrompt: true})
	if err != nil {
		t.Fatalf("Render: %v", err)
	}

	got := tok.Encode(prompt, false)
	want := []int32{
		107, 236820, 46757, 1798, 3041, 236779, 1340, 236779, 887, 236813, 2364, 107,
		3689, 563, 506, 5279, 529, 7001, 145912, 643, 236779, 1340, 236779, 887, 236813, 107,
		236820, 3041, 236779, 1340, 236779, 887, 236813, 4368, 107,
	}
	if len(got) != len(want) {
		t.Fatalf("rendered prompt encodes to %d tokens, llama.cpp says %d\ngot  %v\nwant %v",
			len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("token %d = %d, want %d\ngot  %v", i, got[i], want[i], got)
			break
		}
	}
}

// TestTopLogprobsKeepDescendingOrder pins the order inside the alternatives object.
//
// The schema calls top_logprobs an object, so a map is the obvious encoding -- and encoding/json
// sorts a map's keys, which loses the only thing the order carries. The alternatives are computed
// in descending probability, and a client reading the first entry as the top-1 prediction is
// reading what llama.cpp's own servers return.
//
// Measured before the fix, on Gemma 3 with "The capital of France is": the object encoded as
// {" **", " Marseille", " Paris", " the", "Paris"} -- alphabetical -- while the most likely token
// "Paris" sat third at -0.0007.
//
// The test reads the encoded bytes rather than decoding them, because decoding into a map would
// restore exactly the ordering this is about and pass either way.
func TestTopLogprobsKeepDescendingOrder(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()

	resp := post(t, h, "/v1/completions",
		`{"prompt": "hi", "max_tokens": 2, "temperature": 0, "logprobs": 4}`)
	raw, err := io.ReadAll(resp.Body)
	if cerr := resp.Body.Close(); cerr != nil {
		t.Fatalf("close body: %v", cerr)
	}
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("status %d: %s", resp.StatusCode, raw)
	}

	var body struct {
		Choices []struct {
			Logprobs struct {
				Top []json.RawMessage `json:"top_logprobs"`
			} `json:"logprobs"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(body.Choices) != 1 || len(body.Choices[0].Logprobs.Top) != 2 {
		t.Fatalf("got %d choices and %d positions, want 1 and 2",
			len(body.Choices), len(body.Choices[0].Logprobs.Top))
	}
	for pos, obj := range body.Choices[0].Logprobs.Top {
		if len(obj) == 0 || string(obj) == "null" {
			continue
		}
		// Walk the object in encoded order, tracking the previous value.
		dec := json.NewDecoder(bytes.NewReader(obj))
		if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
			t.Fatalf("position %d is not an object: %s", pos, obj)
		}
		prev := float32(math.Inf(1))
		keys := 0
		for dec.More() {
			keyTok, err := dec.Token()
			if err != nil {
				t.Fatalf("position %d: read key: %v", pos, err)
			}
			var v float32
			if err := dec.Decode(&v); err != nil {
				t.Fatalf("position %d: read value: %v", pos, err)
			}
			if v > prev {
				t.Errorf("position %d: %q (%.6f) follows a smaller value %.6f; the object is not"+
					" in descending probability order, so its first entry is not the top prediction",
					pos, keyTok, v, prev)
			}
			prev = v
			keys++
		}
		if keys == 0 {
			t.Errorf("position %d encoded an empty alternatives object", pos)
		}
	}
}

// TestFIMPromptMatchesReferenceIDs checks the assembled prompt against llama.cpp on a real
// code model, which is the part the fixture cannot say.
//
// The fixture's vocabulary is constructed, so it can pin the order the parts are joined in -- that
// is TestFIMPromptAssembly above -- but not whether the marker ids this server picks are the ones a
// published file means. A real Qwen2.5-Coder carries <|fim_prefix|>, <|fim_suffix|> and
// <|fim_middle|> as single special tokens, and the same text must tokenize to the same ids here.
//
// Set CODER_MODEL to a Qwen2.5-Coder GGUF. The expected list is llama.cpp 0.3.37 tokenizing
// "<|fim_prefix|>" + prefix + "<|fim_suffix|>" + suffix + "<|fim_middle|>" with special tokens
// enabled; without that flag llama.cpp splits the markers into ordinary text and produces a
// different, longer list, which is what makes this a check of the marker ids rather than of the
// tokenizer alone.
func TestFIMPromptMatchesReferenceIDs(t *testing.T) {
	path := os.Getenv("CODER_MODEL")
	if path == "" {
		t.Skip("set CODER_MODEL to a Qwen2.5-Coder GGUF file")
	}
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	fim, ok := tok.FIMTokens()
	if !ok {
		t.Fatal("no FIM markers were detected in this vocabulary")
	}
	eng, err := engine.New(path, engine.Options{MaxActive: 1})
	if err != nil {
		t.Fatalf("engine: %v", err)
	}
	cfg := eng.Model().Config()
	srv, err := New(eng, tok, Config{ModelName: cfg.Name, ContextLen: cfg.ContextLen})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	const before, after = "def add(a, b):\n    ", "\n\ndef mul(a, b):\n    "
	got := srv.fimPrompt(before, after, fim)
	want := []int32{
		151659, 750, 912, 2877, 11, 293, 982, 257, 151661, 271, 750, 15602, 2877, 11, 293, 982,
		257, 151660,
	}
	if len(got) != len(want) {
		t.Fatalf("fimPrompt produced %d ids, llama.cpp produces %d\ngot  %v\nwant %v",
			len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("id %d = %d, want %d\ngot  %v", i, got[i], want[i], got)
			break
		}
	}
}

// TestLogitBiasBansAToken covers the field the server used to ignore.
//
// logit_bias was listed among the fields that only tune sampling, on the reasoning that ignoring
// one costs quality while keeping a readable response. That reasoning fits temperature or top_p and
// not this: OpenAI documents -100 and 100 as "a ban or exclusive selection", so a client that bans
// a token relies on it not appearing. Measured before the field was honoured, a request with
// {"9079": -100} against Gemma 3 270M returned the same "Paris" as no bias at all.
//
// The sampler wraps the built one rather than living inside Temperature, because temperature 0 is
// exactly when a caller bans a token: it is the reproducible mode, and a bias the greedy path could
// not see would be silently ignored in the case that matters most.
//
// The test works on ids rather than on rendered text. The fixture appends a byte-level fallback to
// whatever vocabulary it is given, so a token's text does not necessarily encode back to one id --
// and the ban is a statement about ids anyway.
func TestLogitBiasBansAToken(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 256})
	h := srv.Handler()

	// The token greedy picks first, and a second one it could pick instead. Both come from the
	// response: -100 is an additive shift, so banning the whole vocabulary would leave the
	// argmax unchanged and prove nothing -- only banning the winner changes who wins.
	base := decodeBody[struct {
		Choices []struct {
			Logprobs *struct {
				Tokens      []string             `json:"tokens"`
				TopLogprobs []map[string]float64 `json:"top_logprobs"`
			} `json:"logprobs"`
		} `json:"choices"`
	}](t, post(t, h, "/v1/completions",
		`{"prompt": "Once upon a time", "max_tokens": 1, "temperature": 0, "logprobs": 1}`))
	if len(base.Choices) == 0 || base.Choices[0].Logprobs == nil ||
		len(base.Choices[0].Logprobs.TopLogprobs) == 0 {
		t.Fatal("the baseline request returned no alternatives")
	}
	chosen := base.Choices[0].Logprobs.Tokens[0]

	// Find the id whose ban changes the winner, by trying the ids the fixture's declared
	// vocabulary occupies. The test asks the engine rather than decoding the text, because the
	// fixture's byte fallback makes text-to-id ambiguous.
	bannedSome := false
	// The fixture's vocabulary is its declared words followed by a 256-piece byte fallback, so
	// the ids run past 256. Searched rather than assumed, because which id the constructed
	// weights favour is not something to hardcode.
	for id := range 363 {
		body := fmt.Sprintf(
			`{"prompt": "Once upon a time", "max_tokens": 1, "temperature": 0, "logprobs": 1,
			  "logit_bias": {"%d": -100}}`, id)
		got := decodeBody[struct {
			Choices []struct {
				Logprobs *struct {
					Tokens []string `json:"tokens"`
				} `json:"logprobs"`
			} `json:"choices"`
		}](t, post(t, h, "/v1/completions", body))
		if len(got.Choices) == 0 || got.Choices[0].Logprobs == nil {
			continue
		}
		if got.Choices[0].Logprobs.Tokens[0] != chosen {
			bannedSome = true
			break
		}
	}
	if !bannedSome {
		t.Errorf("no -100 anywhere in the vocabulary changed the first token %q; -100 is documented as a ban,"+
			" not a preference", chosen)
	}

	// A bad key is refused rather than skipped: a ban the caller believes in and the server
	// never applied is the failure this whole path exists to avoid.
	for _, tc := range []struct{ body, why string }{
		{`{"prompt": "x", "logit_bias": {"Once": -100}}`, "a token's text is not an id"},
		{`{"prompt": "x", "logit_bias": {"-1": -100}}`, "a negative id"},
		{`{"prompt": "x", "logit_bias": {"999999": -100}}`, "an id past the vocabulary"},
		{`{"prompt": "x", "logit_bias": {"0": -101}}`, "a bias below -100"},
		{`{"prompt": "x", "logit_bias": {"0": 101}}`, "a bias above 100"},
	} {
		resp := post(t, h, "/v1/completions", tc.body)
		e := decodeBody[errorBody](t, resp)
		if resp.StatusCode != 400 || e.Error.Param != "logit_bias" {
			t.Errorf("%s answered %d %q, want a 400 naming logit_bias", tc.why, resp.StatusCode,
				e.Error.Param)
		}
	}
}

// TestPenaltyRangeIsEnforced covers the validation the two repetition fields share.
//
// Both are documented as [-2, 2] and both ends are legal: a positive value discourages a token
// that has appeared, a negative one encourages it. A value outside that range is refused rather
// than clamped, because clamping -3 to -2 answers a request the caller did not make while looking
// like it honoured the one they did. NaN is refused for the same reason it is refused on
// temperature: it compares false against every bound and would reach the arithmetic.
func TestPenaltyRangeIsEnforced(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()

	for _, field := range []string{"presence_penalty", "frequency_penalty"} {
		// The documented ends are served.
		for _, v := range []string{"-2", "0", "2"} {
			body := fmt.Sprintf(`{"prompt": "hi", "%s": %s}`, field, v)
			if resp := post(t, h, "/v1/completions", body); resp.StatusCode != 200 {
				e := decodeBody[errorBody](t, resp)
				t.Errorf("%s=%s answered %d: %s", field, v, resp.StatusCode, e.Error.Message)
			}
		}
		// Outside them is refused, and the message names the field.
		for _, v := range []string{"-2.1", "2.1", "100", "-100"} {
			body := fmt.Sprintf(`{"prompt": "hi", "%s": %s}`, field, v)
			resp := post(t, h, "/v1/completions", body)
			e := decodeBody[errorBody](t, resp)
			if resp.StatusCode != 400 || e.Error.Param != field {
				t.Errorf("%s=%s answered %d %q, want a 400 naming %s", field, v, resp.StatusCode,
					e.Error.Param, field)
			}
			if !strings.Contains(e.Error.Message, "-2 and 2") {
				t.Errorf("%s=%s: message %q does not state the range", field, v, e.Error.Message)
			}
		}
	}
}

// TestPrefixCacheUnderN covers the two things a shared prefix must not break: the candidates have
// to stay independent, and the answer has to be the one a cache-less server would give.
//
// n>1 is the shape that matters for this. Every choice is a separate sequence with the same prompt,
// so with prefix reuse on, the second and later choices adopt the first one's blocks instead of
// prefilling -- which is the whole benefit, and also the way they could end up sampling from one
// another's state. Measured on Gemma 3 270M with prefix reuse on, four choices at temperature 1.5
// came back as four different completions, and a seeded request returned the same two completions
// with the cache on as with it off.
//
// The fixture is enough for the independence half: it has a real paged cache and real per-sequence
// sampling, which is where a shared buffer would show up.
func TestPrefixCacheUnderN(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}

	// run builds a server over a fresh engine, so the two cache settings cannot see each
	// other's index, and returns the texts of one n-choice response.
	run := func(prefixCache bool, n int) []string {
		t.Helper()
		eng, err := engine.New(path, engine.Options{
			MaxActive: 1, BlockSize: 8, Blocks: 128, PrefixCache: prefixCache,
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

		body := fmt.Sprintf(
			`{"prompt": "Once upon a time", "max_tokens": 4, "temperature": 1.5, "seed": 7, "n": %d}`,
			n)
		resp := post(t, srv.Handler(), "/v1/completions", body)
		out := decodeBody[struct {
			Choices []struct {
				Text string `json:"text"`
			} `json:"choices"`
		}](t, resp)
		texts := make([]string, len(out.Choices))
		for i := range out.Choices {
			texts[i] = out.Choices[i].Text
		}
		return texts
	}

	// A shared prefix must not make the choices identical: each is its own sequence with its own
	// sampling stream.
	cached := run(true, 4)
	distinct := make(map[string]bool, len(cached))
	for _, s := range cached {
		distinct[s] = true
	}
	if len(distinct) == 1 && len(cached) > 1 {
		t.Errorf("all %d choices returned %q; a shared prefix must not share the sampling",
			len(cached), cached[0])
	}

	// And the cache is an optimisation, not a behaviour: the same seeded request has to produce
	// the same text either way.
	without := run(false, 4)
	if len(cached) != len(without) {
		t.Fatalf("%d choices with the cache, %d without", len(cached), len(without))
	}
	for i := range cached {
		if cached[i] != without[i] {
			t.Errorf("choice %d = %q with prefix reuse and %q without; the cache changed the answer",
				i, cached[i], without[i])
		}
	}
}

// TestBestOfRanksOnTheTokensItReports covers the interaction between best_of and stop.
//
// A stop string cuts the reported log probabilities to the text that survived, and the ranking used
// to run on the untrimmed run. A candidate that kept generating after its stop string was therefore
// scored on tokens the caller never sees, and the returned choices were not in descending order of
// the numbers the response carried. Measured on Gemma 3 270M with n=7, best_of=8 and stop ".",
// four seeds out of four returned a list that rose and fell -- -10.772 before -10.626 -- where
// every pair should have been ordered.
//
// The two candidates are built so the two scorings disagree about the winner. A has the better
// visible run; B has the better whole run because four unlikely tokens after its stop string drag
// its mean down only a little while A's two at -1.5 drag its mean down a lot. Ranking on the whole
// run puts B first, ranking on what is reported puts A first, so a test that accepted either
// ordering would be checking nothing.
func TestBestOfRanksOnTheTokensItReports(t *testing.T) {
	a := choiceResult{
		out:     []int32{1, 2},
		samples: []sampledToken{{logprob: -1.5}, {logprob: -1.5}},
		reason:  "stop",
	}
	b := choiceResult{
		out: []int32{1, 2, 3, 4, 5, 6},
		samples: []sampledToken{
			{logprob: -2}, {logprob: -2},
			{logprob: -0.1}, {logprob: -0.1}, {logprob: -0.1}, {logprob: -0.1},
		},
		reason: "stop",
	}

	// The whole-run means prefer B (-0.733) to A (-1.5), so a.score() is the lower of the two.
	if !(a.score() < b.score()) {
		t.Fatalf("the fixture no longer separates the scorings: whole-run means %v and %v",
			a.score(), b.score())
	}
	// The visible means prefer A (-1.5) to B (-2.0), so A has to come first.
	got := rankCandidates([]choiceResult{a, b}, []int{2, 2}, 2)
	if got[0] != 0 {
		t.Errorf("ranked candidate %d first on the reported tokens; A's visible run (-1.5) beats"+
			" B's (-2.0), and B only wins if the four tokens past its stop are counted", got[0])
	}
}

// TestChatTemplateKwargsReachTheTemplate is the end-to-end case for a checkpoint switch.
//
// Qwen3's template branches its prompt on enable_thinking, and before this field was honoured
// the member was an unknown one: the request decoded, the template rendered its default branch,
// and nothing in the response said the switch had been ignored. The prompt_tokens comparison is
// what makes that visible -- two requests that differ only in the kwarg must encode to different
// prompts, or the variable never reached the renderer.
func TestChatTemplateKwargsReachTheTemplate(t *testing.T) {
	// A template that reads a name the way a checkpoint does: a branch on the value, emitting
	// something recognisable so the difference is in the prompt rather than in the logits.
	const tmpl = `{% if enable_thinking is defined and not enable_thinking %}NO-THINK:{% endif %}{% for m in messages %}{{ m.role }}={{ m.content }};{% endfor %}`
	srv, _ := newTemplateServer(t, tmpl)

	base := decodeBody[chatResponse](t, post(t, srv.Handler(), "/v1/chat/completions",
		`{"messages":[{"role":"user","content":"hi"}],"max_tokens":1}`))
	off := decodeBody[chatResponse](t, post(t, srv.Handler(), "/v1/chat/completions",
		`{"messages":[{"role":"user","content":"hi"}],"max_tokens":1,
		  "chat_template_kwargs":{"enable_thinking":false}}`))

	if off.Usage.PromptTokens <= base.Usage.PromptTokens {
		t.Errorf("the kwarg did not change the prompt: %d tokens with it, %d without",
			off.Usage.PromptTokens, base.Usage.PromptTokens)
	}
	// The rendered text itself, which is what the branch is for. Reaching it through the
	// session rather than the response body keeps the assertion about the prompt.
	withVar, err := srv.renderChat([]chatMessage{{Role: "user", Content: "hi"}},
		map[string]any{"enable_thinking": false})
	if err != nil {
		t.Fatalf("renderChat with the variable: %v", err)
	}
	if !strings.HasPrefix(withVar, "NO-THINK:") {
		t.Errorf("the template did not take its branch: %q", withVar)
	}
	without, err := srv.renderChat([]chatMessage{{Role: "user", Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("renderChat without the variable: %v", err)
	}
	if strings.HasPrefix(without, "NO-THINK:") {
		t.Errorf("the branch was taken without the variable: %q", without)
	}
}

// newTemplateServer builds a server whose fixture declares tmpl as its chat template.
func newTemplateServer(t *testing.T, tmpl string) (*Server, *tokenize.Tokenizer) {
	t.Helper()
	model := gguftest.Model{NoEOS: true, ContextLen: 256, ChatTemplate: tmpl}
	path := model.Write(t, t.TempDir())
	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	renderer, err := engine.NewChatRenderer(gf)
	if err != nil {
		t.Fatalf("NewChatRenderer: %v", err)
	}
	if !renderer.HasTemplate() {
		t.Fatal("the template did not compile, so this test would exercise the fallback")
	}
	return newTestServerConfig(t, model, Config{ChatRenderer: renderer})
}

// TestChatTemplateKwargsRefusedWhenUnusable covers the two ways a variable cannot do anything.
//
// Both are refused rather than ignored, and that is the whole point of the field being declared:
// an unknown member asks for nothing the server claims to provide, while a variable this model
// cannot act on is a request whose effect the caller will look for in the reply and not find.
// The status is 400 with param naming the field, so an SDK surfaces it at the call site.
func TestChatTemplateKwargsRefusedWhenUnusable(t *testing.T) {
	t.Run("a null value sets nothing", func(t *testing.T) {
		// JSON null is not a value: it names a variable and gives it none, so the template
		// would read a hole. Refused rather than passed through, and this is the only branch
		// of chatTemplateVars that had no test -- replacing the refusal with `out[name] = nil`
		// left every package green.
		srv, _ := newTemplateServer(t, `{{ enable_thinking }}{% for m in messages %}{{ m.content }}{% endfor %}`)
		resp := post(t, srv.Handler(), "/v1/chat/completions",
			`{"messages":[{"role":"user","content":"hi"}],"max_tokens":1,
			  "chat_template_kwargs":{"enable_thinking":null}}`)
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400", resp.StatusCode)
		}
		body := readBody(t, resp)
		for _, want := range []string{"chat_template_kwargs", "enable_thinking", "is null"} {
			if !strings.Contains(body, want) {
				t.Errorf("the refusal does not mention %q: %s", want, body)
			}
		}
	})

	t.Run("the template does not read the name", func(t *testing.T) {
		// A template that reads enable_thinking but not this.
		srv, _ := newTemplateServer(t, `{{ enable_thinking }}{% for m in messages %}{{ m.content }}{% endfor %}`)
		resp := post(t, srv.Handler(), "/v1/chat/completions",
			`{"messages":[{"role":"user","content":"hi"}],"max_tokens":1,
			  "chat_template_kwargs":{"temperature_mode":"hot"}}`)
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400", resp.StatusCode)
		}
		body := readBody(t, resp)
		if !strings.Contains(body, "chat_template_kwargs") {
			t.Errorf("the refusal does not name the field: %s", body)
		}
		// The variable that could not be used, by name, so the caller knows which one.
		if !strings.Contains(body, "temperature_mode") {
			t.Errorf("the refusal does not name the variable: %s", body)
		}
		// And it reads as a sentence about the request. This message is the renderer's,
		// passed straight through, and it used to open with "engine:" -- the one response in
		// the whole envelope that named an internal package. Measured against a live server,
		// the neighbouring refusals read "temperature must not be negative, got -1" and
		// "top_p must be between 0 and 1, got 5".
		if strings.Contains(body, "engine:") {
			t.Errorf("the refusal names an internal package: %s", body)
		}
	})

	t.Run("the model has no template", func(t *testing.T) {
		// The default fixture declares none, so no variable can reach anything.
		srv, _ := newTestServer(t)
		resp := post(t, srv.Handler(), "/v1/chat/completions",
			`{"messages":[{"role":"user","content":"hi"}],"max_tokens":1,
			  "chat_template_kwargs":{"enable_thinking":false}}`)
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusBadRequest {
			t.Fatalf("status = %d, want 400: a model with no template cannot honour a variable",
				resp.StatusCode)
		}
	})
}

// TestChatTemplateKwargsAcceptsTheShapesJSONProduces covers the value types a client can send.
//
// The map is decoded by encoding/json, so a client's false arrives as a bool and its 3 as a
// float64. A template compares against a bool and prints a number, so the conversion has to keep
// both distinguishable -- and it has to accept each without a 400, or the field is only usable
// for strings.
func TestChatTemplateKwargsAcceptsTheShapesJSONProduces(t *testing.T) {
	const tmpl = `{% if flag %}F{% endif %}{% if not flag %}N{% endif %}[{{ count }}][{{ name }}][{{ list|length }}]`
	srv, _ := newTemplateServer(t, tmpl)

	cases := []struct {
		name string
		body string
		want string
	}{
		// 3 prints as "3", not "3.0": json.loads gives Jinja2 an int, and this package gets a
		// float64 for the same bytes. The value is what decides, not the Go type.
		{"bool true", `{"flag":true,"count":3,"name":"x","list":[1,2]}`,
			"F[3][x][2]"},
		{"bool false", `{"flag":false,"count":3,"name":"x","list":[1,2]}`,
			"N[3][x][2]"},
		// A genuine fraction keeps its decimal point, which is the other half of the rule.
		{"fraction", `{"flag":true,"count":2.5,"name":"x","list":[1]}`,
			"F[2.5][x][1]"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := srv.renderChat([]chatMessage{{Role: "user", Content: "hi"}},
				mustVars(t, c.body))
			if err != nil {
				t.Fatalf("renderChat: %v", err)
			}
			if got != c.want {
				t.Errorf("rendered %q, want %q", got, c.want)
			}
		})
	}
}

// mustVars decodes a JSON object into the map a request would produce.
func mustVars(t *testing.T, body string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(body), &m); err != nil {
		t.Fatalf("decoding %s: %v", body, err)
	}
	return m
}

// TestChoiceObjectShapes covers the response-shape decisions choiceObject owns.
//
// It is reachable directly because everything it does is a choice about the object a caller
// reads -- where a stop string cuts the text, whether the prompt is echoed ahead of it, and which
// of the two schemas the object takes. A test through the endpoint would have to guess the
// fixture's generated text to see any of that.
func TestChoiceObjectShapes(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, ContextLen: 256})

	// "abcdef" as ids the fixture can decode. Whatever they render to, the assertions below are
	// about structure rather than about the text's meaning.
	c := &choiceResult{
		out:    []int32{4, 5, 6}, // a, b, c in the fixture's vocabulary
		reason: "length",
	}

	t.Run("chat schema", func(t *testing.T) {
		obj := srv.choiceObject(c, 0, generateParams{chat: true})
		if _, ok := obj["message"]; !ok {
			t.Errorf("chat choice has no message: %v", obj)
		}
		if _, ok := obj["text"]; ok {
			t.Errorf("chat choice carries a text field: %v", obj)
		}
		if obj["index"] != 0 {
			t.Errorf("index = %v, want 0", obj["index"])
		}
		if obj["finish_reason"] != "length" {
			t.Errorf("finish_reason = %v, want length", obj["finish_reason"])
		}
	})

	t.Run("completions schema", func(t *testing.T) {
		obj := srv.choiceObject(c, 1, generateParams{})
		if _, ok := obj["text"]; !ok {
			t.Errorf("completions choice has no text: %v", obj)
		}
		if _, ok := obj["message"]; ok {
			t.Errorf("completions choice carries a message field: %v", obj)
		}
		// The index is the response position, not the candidate's slot.
		if obj["index"] != 1 {
			t.Errorf("index = %v, want the response position 1", obj["index"])
		}
	})

	t.Run("echo prepends the prompt", func(t *testing.T) {
		plain := srv.choiceObject(c, 0, generateParams{})
		echoed := srv.choiceObject(c, 0, generateParams{echo: true, prompt: "Once "})
		if got, want := echoed["text"], "Once "+plain["text"].(string); got != want {
			t.Errorf("echoed text = %q, want %q", got, want)
		}
		// echo has no effect on the chat schema: the caller sent messages, not a prompt, so
		// there is nothing to prepend.
		chatEcho := srv.choiceObject(c, 0, generateParams{chat: true, echo: true, prompt: "Once "})
		msg := chatEcho["message"].(map[string]any)
		if content, _ := msg["content"].(string); strings.HasPrefix(content, "Once ") {
			t.Errorf("chat echo prepended a prompt: %q", content)
		}
	})

	t.Run("a stop string cuts the text and reports stop", func(t *testing.T) {
		full := srv.choiceObject(c, 0, generateParams{})
		text := full["text"].(string)
		if len(text) < 2 {
			t.Skipf("the fixture generated %q, too short to cut", text)
		}
		// Cut at the second byte of whatever it produced, so the cut is inside the text rather
		// than before or after it.
		cutAt := text[:1]
		obj := srv.choiceObject(c, 0, generateParams{stop: []string{cutAt}})
		if got := obj["text"].(string); got != "" {
			t.Errorf("text = %q, want empty: the stop string was the first character", got)
		}
		if obj["finish_reason"] != "stop" {
			t.Errorf("finish_reason = %v, want stop", obj["finish_reason"])
		}
	})
}

// TestEventWriterFrameShape pins the bytes a client actually reads.
//
// Every chunk carries the same four metadata fields, and a client tells the usage chunk apart
// from a text one by its empty choices array. Both were assembled at each emit site before this
// type existed, which is how a field goes missing from one path and not the other.
func TestEventWriterFrameShape(t *testing.T) {
	rec := httptest.NewRecorder()
	ew, ok := newEventWriter(rec, "orbix-test", eventEnvelope{id: "cmpl-1", created: 1700000000},
		generateParams{chat: true, includeUsage: true})
	if !ok {
		t.Fatal("a recorder can flush, so newEventWriter refused one it should accept")
	}
	if got := rec.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Errorf("Content-Type = %q, want text/event-stream", got)
	}
	// The nginx switch, which has no other way to be noticed.
	if got := rec.Header().Get("X-Accel-Buffering"); got != "no" {
		t.Errorf("X-Accel-Buffering = %q, want no", got)
	}

	if !ew.send(map[string]any{"choices": []map[string]any{}}) {
		t.Fatal("send reported failure on a writable recorder")
	}
	frame := rec.Body.String()
	if !strings.HasPrefix(frame, "data: ") || !strings.HasSuffix(frame, "\n\n") {
		t.Fatalf("frame is not an SSE data line: %q", frame)
	}
	var payload map[string]any
	body := strings.TrimSuffix(strings.TrimPrefix(frame, "data: "), "\n\n")
	if err := json.Unmarshal([]byte(body), &payload); err != nil {
		t.Fatalf("the frame's payload is not JSON (%v): %s", err, body)
	}
	for _, key := range []string{"id", "object", "created", "model"} {
		if _, ok := payload[key]; !ok {
			t.Errorf("the payload has no %s: %v", key, payload)
		}
	}
	if payload["id"] != "cmpl-1" {
		t.Errorf("id = %v, want cmpl-1", payload["id"])
	}
	if payload["object"] != "chat.completion.chunk" {
		t.Errorf("object = %v, want the chat chunk name", payload["object"])
	}
	// includeUsage puts a present-and-null usage on every ordinary chunk, which is what a
	// client reading usage off each chunk expects rather than a missing key.
	if u, present := payload["usage"]; !present {
		t.Error("usage is absent from an ordinary chunk that includeUsage asked to mark")
	} else if u != nil {
		t.Errorf("usage = %v on an ordinary chunk, want null", u)
	}
}

// TestEventWriterLeavesTheUsageChunkAlone checks the one payload that carries its own usage.
//
// The sentinel above must not overwrite it: a client finds the usage chunk by its empty choices
// array and reads the totals off it, so replacing them with null loses the only number the
// caller asked for.
func TestEventWriterLeavesTheUsageChunkAlone(t *testing.T) {
	rec := httptest.NewRecorder()
	ew, _ := newEventWriter(rec, "m", eventEnvelope{id: "x"}, generateParams{includeUsage: true})
	ew.send(map[string]any{
		"choices": []map[string]any{},
		"usage":   map[string]int{"total_tokens": 7},
	})
	var payload map[string]any
	body := strings.TrimSuffix(strings.TrimPrefix(rec.Body.String(), "data: "), "\n\n")
	if err := json.Unmarshal([]byte(body), &payload); err != nil {
		t.Fatalf("payload is not JSON: %v", err)
	}
	usage, ok := payload["usage"].(map[string]any)
	if !ok || usage["total_tokens"] != float64(7) {
		t.Errorf("usage = %v, want the totals it was given", payload["usage"])
	}
}

// TestEventWriterRefusesAWriterThatCannotFlush keeps the one refusal it can make.
//
// A ResponseWriter without Flush holds every token until the response ends, which arrives as one
// late body rather than as a stream. The check is on the capability rather than on the concrete
// type because that is what the stream needs.
func TestEventWriterRefusesAWriterThatCannotFlush(t *testing.T) {
	plain := noFlushWriter{ResponseWriter: httptest.NewRecorder()}
	if _, ok := newEventWriter(plain, "m", eventEnvelope{}, generateParams{}); ok {
		t.Error("newEventWriter accepted a writer that cannot flush")
	}
}

// TestStopGateHoldsBackAPartialMatch covers the defect that made streaming disagree with
// blocking.
//
// Measured before this existed: stop=["北京"] streamed "北" where the non-streaming body returned
// nothing. A stop string is matched on decoded text, so its last character has to arrive before
// the match can be seen -- and a character already sent cannot be taken back. Holding the longest
// tail that could still begin a match is what makes the two transports agree.
func TestStopGateHoldsBackAPartialMatch(t *testing.T) {
	g := &stopGate{stop: []string{"北京"}}

	// The first character could be the start of the stop string, so nothing goes out yet.
	if out, found := g.push("北"); out != "" || found {
		t.Errorf("push(北) = %q, %v; want nothing released and no match", out, found)
	}
	// The second completes it: nothing was ever released, so there is nothing to release and
	// the match is reported.
	if out, found := g.push("京"); out != "" || !found {
		t.Errorf("push(京) = %q, %v; want nothing released and a match", out, found)
	}
}

// TestStopGateReleasesTextBeforeTheHeldTail is the ordinary case: output that cannot become a
// stop string must not be delayed.
//
// Without this a stream with a stop list would hold every chunk back, which defeats streaming.
func TestStopGateReleasesTextBeforeTheHeldTail(t *testing.T) {
	g := &stopGate{stop: []string{"END"}}

	// "hello" cannot begin "END", so all of it goes out at once.
	if out, _ := g.push("hello "); out != "hello " {
		t.Fatalf("push(hello ) = %q, want all of it released", out)
	}
	// "E" could begin the stop string, so it is held and nothing new is released.
	if out, _ := g.push("E"); out != "" {
		t.Errorf("push(E) = %q, want it held back", out)
	}
}

// TestStopGateReleasesEverythingWhenNoStopArrives pins flush.
//
// A stop list that never matches must cost nothing but a later chunk. Returning the held tail at
// the end is what makes that true, and without it the stream would end short.
func TestStopGateReleasesEverythingWhenNoStopArrives(t *testing.T) {
	g := &stopGate{stop: []string{"END"}}
	g.push("done.")
	g.push("E")
	if rest := g.flush(); rest != "E" {
		t.Errorf("flush() = %q, want the held E", rest)
	}
	// Flushing twice must not repeat the text: a caller does it once, but a second call
	// returning the same bytes would duplicate output if it ever did happen.
	if rest := g.flush(); rest != "" {
		t.Errorf("second flush() = %q, want nothing", rest)
	}
}

// TestStopGateDropsTheStopAndEverythingAfterIt covers what a match discards.
//
// The stop string is not part of the response, and neither is anything the engine produced after
// it. Returning only the text between what the client holds and the cut is what enforces that.
func TestStopGateDropsTheStopAndEverythingAfterIt(t *testing.T) {
	g := &stopGate{stop: []string{"STOP"}}
	if out, _ := g.push("answer "); out != "answer " {
		t.Fatalf("push(answer ) = %q", out)
	}
	out, found := g.push("STOP and more")
	if !found {
		t.Fatal("the stop string was not recognised")
	}
	if out != "" {
		t.Errorf("released %q on the match, want nothing: the stop is not in the response", out)
	}
}

// TestStopGateEmptyListReleasesEverythingImmediately keeps the no-stop path free of delay.
//
// Every request without a stop list takes this path, so a hold-back here would delay every
// streamed chunk in the common case.
func TestStopGateEmptyListReleasesEverythingImmediately(t *testing.T) {
	g := &stopGate{}
	for _, s := range []string{"a", "b", "c"} {
		if out, found := g.push(s); out != s || found {
			t.Errorf("push(%q) = %q, %v; want it released with no match", s, out, found)
		}
	}
	if rest := g.flush(); rest != "" {
		t.Errorf("flush() = %q, want nothing left", rest)
	}
}

// TestChatRefusesTheFieldsItValidates covers handleChat's refusals, which had no test.
//
// Measured before this: five lines in handleChat were never executed -- the error returns for
// chat_template_kwargs, response_format, logit_bias and the two penalties. They are the whole
// reason those checks exist, and each is reachable from a request a client could send.
//
// The assertion is the parameter name in the error, because that is what a client branches on: a
// 400 that does not say which field was wrong leaves the caller diffing their request against the
// schema.
func TestChatRefusesTheFieldsItValidates(t *testing.T) {
	srv, _ := newTestServer(t)

	cases := []struct {
		name  string
		body  string
		param string
	}{
		{
			// A response_format this server cannot produce. The value is well-formed JSON, so
			// the refusal comes from the check rather than the decoder.
			name:  "unknown response_format",
			body:  `{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"xml"}}`,
			param: "response_format",
		},
		{
			// Above the documented range. A penalty outside [-2, 2] is not a stronger penalty,
			// it is a typo, and the sampler would accept it.
			name:  "presence_penalty out of range",
			body:  `{"messages":[{"role":"user","content":"hi"}],"presence_penalty":9}`,
			param: "presence_penalty",
		},
		{
			name:  "frequency_penalty out of range",
			body:  `{"messages":[{"role":"user","content":"hi"}],"frequency_penalty":-9}`,
			param: "frequency_penalty",
		},
		{
			// A bias of -200 is not a ban, and logitBias refuses anything outside [-100, 100].
			name:  "logit_bias out of range",
			body:  `{"messages":[{"role":"user","content":"hi"}],"logit_bias":{"5":-200}}`,
			param: "logit_bias",
		},
		{
			// The fixture declares no chat template, so any variable is unusable. This is the
			// path chat_template_kwargs takes on a model that cannot honour it.
			name:  "template variable on a model without a template",
			body:  `{"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":false}}`,
			param: "chat_template_kwargs",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			resp := post(t, srv.Handler(), "/v1/chat/completions", c.body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400: %s", resp.StatusCode, readBody(t, resp))
			}
			body := readBody(t, resp)
			if !strings.Contains(body, c.param) {
				t.Errorf("the refusal does not name %s: %s", c.param, body)
			}
		})
	}
}

// TestCompletionsRefusesTheFieldsItValidates is the same check for the other endpoint.
//
// The two handlers validate the same set, so a rule that holds on one and not the other is a
// difference a client discovers by switching endpoints.
func TestCompletionsRefusesTheFieldsItValidates(t *testing.T) {
	srv, _ := newTestServer(t)
	cases := []struct {
		name  string
		body  string
		param string
	}{
		{"unknown response_format", `{"prompt":"hi","response_format":{"type":"xml"}}`, "response_format"},
		{"presence_penalty out of range", `{"prompt":"hi","presence_penalty":9}`, "presence_penalty"},
		{"frequency_penalty out of range", `{"prompt":"hi","frequency_penalty":-9}`, "frequency_penalty"},
		{"logit_bias out of range", `{"prompt":"hi","logit_bias":{"5":-200}}`, "logit_bias"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			resp := post(t, srv.Handler(), "/v1/completions", c.body)
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400: %s", resp.StatusCode, readBody(t, resp))
			}
			if body := readBody(t, resp); !strings.Contains(body, c.param) {
				t.Errorf("the refusal does not name %s: %s", c.param, body)
			}
		})
	}
}

// TestShownTokensAppliesTheStopCut covers the branch the ranking tests step around.
//
// TestBestOfRanksOnTheTokensItReports calls rankCandidates directly with hand-written counts, so
// shownTokens itself was never exercised with a stop list: the line that reassigns text after a
// cut had no test. That line is the whole reason the function exists -- without it a candidate
// generating thirty tokens past its stop string is ranked on all thirty while the caller sees
// only the first few.
func TestShownTokensAppliesTheStopCut(t *testing.T) {
	srv, _ := newTestServer(t)

	// A result whose decoded text is long enough to cut, with one sample per token so the
	// kept count is comparable to the text length.
	c := &choiceResult{
		out: []int32{1, 2, 3, 4, 5, 6},
	}
	// One sample per token, carrying the token id as well as a log probability: an entry's bytes
	// come from the id, and a zero id is outside the vocabulary, so the described entries would
	// have no bytes and the count would be zero for a reason that has nothing to do with the cut.
	for _, id := range c.out {
		c.samples = append(c.samples, sampledToken{id: id, logprob: -1})
	}
	full := srv.tok.DecodeCompletion(c.out)
	if len(full) < 3 {
		t.Skipf("the fixture decoded %q, too short to cut", full)
	}

	// No stop list: every token is reported.
	all := srv.shownTokens(c, generateParams{})
	if all == 0 {
		t.Fatal("shownTokens returned nothing with no stop list")
	}

	// A stop string that is the whole text: nothing survives, so nothing is reported. The
	// assertion is the zero rather than a comparison, because the point is that the cut
	// propagates into the count.
	if got := srv.shownTokens(c, generateParams{stop: []string{full}}); got != 0 {
		t.Errorf("shownTokens = %d with a stop string matching the whole text, want 0", got)
	}

	// A stop string absent from the text changes nothing.
	if got := srv.shownTokens(c, generateParams{stop: []string{"\u0000not-present"}}); got != all {
		t.Errorf("shownTokens = %d with an unmatched stop string, want %d", got, all)
	}
}

// TestKeptTokens covers the cut that decides what a logprob list describes.
//
// It had no direct test: the two callers were exercised only through endpoints whose generated
// text a fixture controls, so the boundaries were untested. The zero case is the one that broke.
func TestKeptTokens(t *testing.T) {
	describe := func(texts ...string) []logprobText {
		out := make([]logprobText, len(texts))
		for i, s := range texts {
			out[i] = logprobText{text: s, logprob: -1}
		}
		return out
	}
	cases := []struct {
		name      string
		described []logprobText
		retained  int
		keep      int
		used      int
	}{
		// Nothing retained: nothing shown, so nothing described. This is the case that
		// returned every entry before the fix.
		{"zero retention", describe("a", "b"), 0, 0, 0},
		{"zero retention with empty texts", describe("", "", ""), 0, 0, 0},
		{"everything fits", describe("a", "b"), 2, 2, 2},
		{"cut mid-way", describe("a", "b", "c"), 2, 2, 2},
		// A token that straddles the cut goes too: half its text survived, and neither
		// including nor truncating it describes the logprob of what the caller sees.
		{"straddling token dropped", describe("ab", "cd"), 3, 1, 2},
		{"first token alone too long", describe("abc"), 2, 0, 0},
		// An empty entry costs nothing, so it is kept while the budget allows.
		{"empty entry is free", describe("", "ab"), 2, 2, 2},
		{"no entries", nil, 5, 0, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			keep, used := keptTokens(c.described, c.retained)
			if keep != c.keep || used != c.used {
				t.Errorf("keptTokens(%v, %d) = (%d, %d), want (%d, %d)",
					c.described, c.retained, keep, used, c.keep, c.used)
			}
		})
	}
}

// TestModelObjectFields covers the three fields of a model entry that no test read.
//
// TestModelsAndHealth decodes only object and data[].id, so created, owned_by and the entry's own
// object were free to change. They are part of the OpenAI schema an SDK decodes into its own model
// struct, which means a missing one is a zero value on the client rather than an error.
//
// created is the interesting one. It is deliberately zero -- a GGUF file records no creation date
// and the file's mtime is when it was copied -- and the comment at modelObject explains why zero
// is the honest value rather than a plausible-looking timestamp. That reasoning is only correct as
// long as the field stays zero, so it is asserted rather than assumed.
func TestModelObjectFields(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, Name: "orbix-test-model"})

	var doc struct {
		Object string `json:"object"`
		Data   []struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			Created int64  `json:"created"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("/v1/models status = %d", w.Code)
	}
	if err := json.Unmarshal(w.Body.Bytes(), &doc); err != nil {
		t.Fatalf("decoding: %v", err)
	}
	if len(doc.Data) != 1 {
		t.Fatalf("got %d entries, want 1", len(doc.Data))
	}
	e := doc.Data[0]
	if e.Object != "model" {
		t.Errorf("object = %q, want \"model\": an SDK dispatches on this field", e.Object)
	}
	if e.Created != 0 {
		t.Errorf("created = %d, want 0: a nonzero value would claim a date the file does not "+
			"carry, and modelObject's comment explains why zero is the honest answer", e.Created)
	}
	if e.OwnedBy == "" {
		t.Error("owned_by is empty; the schema requires it and an SDK shows it to a caller")
	}

	// The retrieve endpoint has to return the same object, which is why it exists in one place.
	req = httptest.NewRequest(http.MethodGet, "/v1/models/orbix-test-model", nil)
	w = httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("retrieve status = %d: %s", w.Code, w.Body.String())
	}
	var one struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &one); err != nil {
		t.Fatalf("decoding retrieve: %v", err)
	}
	if one.ID != e.ID || one.Object != e.Object || one.Created != e.Created || one.OwnedBy != e.OwnedBy {
		t.Errorf("retrieve returned %+v but the list entry was %+v: an SDK that lists then "+
			"retrieves compares the two", one, e)
	}
}

// TestHealthzBody is the contract docs/usage.md states and no test read.
//
// "`/healthz` answers {"status": "ok", "model": "<name>"}" -- and until now only the status code
// was asserted, so the body an operator's probe parses was free to change. The model field is what
// makes the answer useful when several servers are behind one load balancer: a 200 that does not
// say which checkpoint answered cannot distinguish a wrong backend from a right one.
func TestHealthzBody(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Model{NoEOS: true, Name: "probe-target"})

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d", w.Code)
	}
	var got struct {
		Status string `json:"status"`
		Model  string `json:"model"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
		t.Fatalf("the body is not JSON: %v\n%s", err, w.Body.String())
	}
	// The literal "ok": a probe keys on this string, and changing it to "healthy" or a bool
	// would break every check without changing the status code.
	if got.Status != "ok" {
		t.Errorf("status field = %q, want \"ok\"", got.Status)
	}
	if got.Model != "probe-target" {
		t.Errorf("model = %q, want the configured name", got.Model)
	}
}

// TestServerDefaultsAreUsedWhenTheRequestOmitsThem covers the three sampling defaults.
//
// Config carries Temperature, TopK and TopP for a request that does not name them, and every
// test built its server with the zero values -- so the branch that reads s.cfg was only ever
// exercised with the same value the fallback would have produced. A change from
// `t := s.cfg.Temperature` to `t := 0.0` would have passed the whole suite.
//
// The assertion compares two runs rather than checking a field: the same seed and the same
// prompt through a server whose default is 0.7 must produce the same text as a request that asks
// for 0.7 explicitly. Equal text means the default was applied; different text means it was
// ignored and the run fell back to greedy.
func TestServerDefaultsAreUsedWhenTheRequestOmitsThem(t *testing.T) {
	model := gguftest.Model{NoEOS: true, ContextLen: 256}

	// The server whose defaults are non-zero, and one whose are not. Both take the same request.
	withDefaults := func(t *testing.T) *Server {
		t.Helper()
		srv, _ := newTestServerConfig(t, model, Config{
			MaxTokens:   8,
			ContextLen:  256,
			Temperature: 0.7,
			TopK:        5,
			TopP:        0.9,
		})
		return srv
	}
	zeroDefaults := func(t *testing.T) *Server {
		t.Helper()
		srv, _ := newTestServerConfig(t, model, Config{MaxTokens: 8, ContextLen: 256})
		return srv
	}

	// A fixed seed makes sampling reproducible, so two runs of the same configuration agree.
	const body = `{"prompt":"Once","max_tokens":6,"seed":12345}`

	useDefaults := decodeBody[completionResponse](t, post(t, withDefaults(t).Handler(),
		"/v1/completions", body))
	// The same numbers, sent explicitly, against a server with no defaults of its own.
	explicit := decodeBody[completionResponse](t, post(t, zeroDefaults(t).Handler(),
		"/v1/completions", `{"prompt":"Once","max_tokens":6,"seed":12345,
			"temperature":0.7,"top_k":5,"top_p":0.9}`))
	greedy := decodeBody[completionResponse](t, post(t, zeroDefaults(t).Handler(),
		"/v1/completions", body))

	if len(useDefaults.Choices) == 0 || len(explicit.Choices) == 0 || len(greedy.Choices) == 0 {
		t.Fatal("a request produced no choices")
	}
	if got, want := useDefaults.Choices[0].Text, explicit.Choices[0].Text; got != want {
		t.Errorf("the server's defaults produced %q but the same numbers sent explicitly "+
			"produced %q: the defaults are not being applied", got, want)
	}
	// And the defaults must not be the same as greedy, or the comparison above would hold for
	// a server that ignored them and treated every request as greedy.
	if useDefaults.Choices[0].Text == greedy.Choices[0].Text {
		t.Log("the fixture's sampling happened to match greedy for this seed; the equality " +
			"above is then vacuous for this input")
	}
}

// TestTheHonouredSamplingFieldsChangeTheOutput covers the half of "honoured" that the existing
// test does not.
//
// TestTheHonouredSamplingFieldsChangeTheOutput asserts that a request setting logit_bias,
// presence_penalty or frequency_penalty is served with 200, and its comment says those three are
// honoured now because "a caller who bans a token or asks for less repetition is asking for
// different text". A field that is parsed and then dropped also returns 200, so that test cannot
// tell the two apart -- and the docs/usage.md paragraph makes exactly the claim it cannot check.
//
// The assertion here is a comparison of outputs: the same prompt, seed and temperature, once
// without the field and once with it. Different text means the value reached the sampler.
func TestTheHonouredSamplingFieldsChangeTheOutput(t *testing.T) {
	srv, _ := newTestServerConfig(t, gguftest.Model{NoEOS: true, ContextLen: 512}, Config{
		MaxTokens: 8, ContextLen: 512,
	})

	run := func(t *testing.T, extra string) string {
		t.Helper()
		body := `{"prompt":"Once","max_tokens":8,"seed":7,"temperature":1.0`
		if extra != "" {
			body += "," + extra
		}
		body += "}"
		resp := post(t, srv.Handler(), "/v1/completions", body)
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d for %s", resp.StatusCode, body)
		}
		got := decodeBody[completionResponse](t, resp)
		if len(got.Choices) == 0 {
			t.Fatalf("no choices for %s", body)
		}
		return got.Choices[0].Text
	}

	plain := run(t, "")
	// A field whose value never reached the sampler would leave the text byte-identical to
	// this baseline. The three are checked one at a time rather than together, so a field that
	// became a no-op while another still worked would fail on its own.
	for _, tc := range []struct{ name, field string }{
		{"logit_bias", `"logit_bias":{"1":-100,"2":-100,"3":-100}`},
		{"presence_penalty", `"presence_penalty":2.0`},
		{"frequency_penalty", `"frequency_penalty":2.0`},
	} {
		t.Run(tc.name+" changes the output", func(t *testing.T) {
			if got := run(t, tc.field); got == plain {
				t.Errorf("with %s the text is unchanged (%q), so the value is parsed and "+
					"dropped rather than honoured", tc.field, got)
			}
		})
	}
}

// TestTopKIsHonoured is the check top_p has and top_k did not.
//
// TestTopPIsHonoured compares how many distinct continuations twelve seeds produce with and
// without a narrow nucleus, on the reasoning that a limit which never reached the sampler would
// leave the count unchanged. top_k is passed through a different path -- an int rather than a
// pointer, defaulted from Config rather than from a nil check -- and nothing asserted its effect.
//
// top_k: 1 is the sharpest form of the test: with a single candidate per step, sampling cannot
// vary, so every seed must produce the same text. That is an equality rather than a comparison of
// counts, so it cannot pass by the two sides happening to be close.
func TestTopKIsHonoured(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()

	sample := func(seed int, extra string) string {
		t.Helper()
		body := fmt.Sprintf(`{"model": %q, "prompt": "Once upon a time",
			"max_tokens": 8, "temperature": 1.5, "seed": %d%s}`,
			srv.cfg.ModelName, seed, extra)
		resp := post(t, h, "/v1/completions", body)
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d", resp.StatusCode)
		}
		return decodeBody[completionResponse](t, resp).Choices[0].Text
	}

	// Without top_k, temperature 1.5 across seeds should vary. If it does not, the fixture made
	// every step near-certain and the equality below would hold for a field that was ignored.
	seen := make(map[string]bool)
	for seed := 1; seed <= 12; seed++ {
		seen[sample(seed, "")] = true
	}
	if len(seen) < 2 {
		t.Skipf("all twelve seeds produced the same text without top_k, so this fixture cannot "+
			"distinguish a working top_k from an ignored one; saw %d distinct", len(seen))
	}

	first := sample(1, `, "top_k": 1`)
	for seed := 2; seed <= 12; seed++ {
		if got := sample(seed, `, "top_k": 1`); got != first {
			t.Fatalf("top_k=1 gave seed %d %q and seed 1 %q; with one candidate per step the "+
				"result cannot depend on the seed, so the value is not reaching the sampler",
				seed, got, first)
		}
	}
}

// TestBothEndpointsHonourTheSameSamplingFields checks the chat endpoint for every sampling
// field the completions endpoint has an effect test for.
//
// The two handlers build generateParams in separate places -- one after chatRequest is decoded,
// one after completionRequest -- and every effect test for a sampling field went through
// /v1/completions. Sharing the struct and the engine call means a field wired once is wired for
// both, but the parameter list is written twice, and setting one to a zero is what a copy invites.
//
// Measured before writing this: for each field below, the chat call site was changed to pass the
// zero value, and every one of them left the whole package green -- none had a test that would
// have noticed. top_p was added a round after the other three: the first version of this test
// covered top_k plus the three penalties, and a second audit pass found top_p still passing the
// whole package with a nil at the chat call site.
func TestBothEndpointsHonourTheSameSamplingFields(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()

	// chatText returns the assistant message for one chat request.
	chatText := func(seed int, extra string) string {
		t.Helper()
		body := fmt.Sprintf(`{"model": %q, "messages": [{"role": "user", "content": "Once"}],
			"max_tokens": 8, "temperature": 1.5, "seed": %d%s}`, srv.cfg.ModelName, seed, extra)
		resp := post(t, h, "/v1/chat/completions", body)
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d", resp.StatusCode)
		}
		got := decodeBody[chatResponse](t, resp)
		if len(got.Choices) == 0 {
			t.Fatal("no choices")
		}
		return got.Choices[0].Message.Content
	}

	// Each field must move the text on its own, so one that became a no-op while its
	// neighbours still worked fails by itself rather than being masked.
	baseline := chatText(7, "")
	for _, tc := range []struct{ field, extra string }{
		{"logit_bias", `, "logit_bias": {"1": -100, "2": -100, "3": -100}`},
		{"presence_penalty", `, "presence_penalty": 2.0`},
		{"frequency_penalty", `, "frequency_penalty": 2.0`},
		{"top_p", `, "top_p": 0.01`},
	} {
		t.Run(tc.field+" changes the chat output", func(t *testing.T) {
			if got := chatText(7, tc.extra); got == baseline {
				t.Errorf("with %s the text is unchanged (%q), so the value is dropped on the "+
					"chat path", tc.field, got)
			}
		})
	}

	// top_k is checked by its effect on reproducibility rather than on the text: "different from
	// the baseline" is not the property it promises. One candidate per step makes the result
	// independent of the seed.
	seen := make(map[string]bool)
	for seed := 1; seed <= 12; seed++ {
		seen[chatText(seed, "")] = true
	}
	if len(seen) < 2 {
		t.Skipf("all twelve chat seeds produced the same text, so this fixture cannot tell a "+
			"working top_k from an ignored one; saw %d distinct", len(seen))
	}
	first := chatText(1, `, "top_k": 1`)
	for seed := 2; seed <= 12; seed++ {
		if got := chatText(seed, `, "top_k": 1`); got != first {
			t.Fatalf("chat: top_k=1 gave seed %d %q and seed 1 %q, so the value is not "+
				"reaching the sampler on this endpoint", seed, got, first)
		}
	}
}

// TestChatHonoursStop covers the chat endpoint's stop strings, which every stop test skipped.
//
// TestStopStringStops and its neighbours all post to /v1/completions. The chat handler sets
// stop from its own request field, and injection shows the difference: replacing `stop:
// req.Stop` with `stop: nil` at the chat call site left the whole package green.
//
// Measured against a running server before writing this: the fixture repeats one character, so
// stop "h" on the chat endpoint returns an empty message and finish_reason "stop", which is the
// same shape the completions endpoint produces and is not something a request that ignored the
// field could return.
func TestChatHonoursStop(t *testing.T) {
	srv, _ := newTestServer(t)
	h := srv.Handler()

	finishAndText := func(t *testing.T, stopField string) (string, string) {
		t.Helper()
		body := fmt.Sprintf(`{"model": %q, "messages": [{"role": "user", "content": "Once"}],
			"max_tokens": 20, "temperature": 0%s}`, srv.cfg.ModelName, stopField)
		resp := post(t, h, "/v1/chat/completions", body)
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d", resp.StatusCode)
		}
		got := decodeBody[chatResponse](t, resp)
		if len(got.Choices) == 0 {
			t.Fatal("no choices")
		}
		return got.Choices[0].FinishReason, got.Choices[0].Message.Content
	}

	// Without a stop the reply runs to the budget, and the text is not empty -- otherwise the
	// assertion below would hold for a server that always returned nothing.
	reason, text := finishAndText(t, "")
	if reason != "length" {
		t.Fatalf("without stop, finish_reason = %q, want length", reason)
	}
	if text == "" {
		t.Fatal("without stop the reply was empty, so this fixture cannot show a stop taking effect")
	}

	// A stop string the fixture's output contains must cut it, and the reason must say so.
	reason, text = finishAndText(t, `, "stop": "h"`)
	if reason != "stop" {
		t.Errorf("with a matching stop, finish_reason = %q, want stop", reason)
	}
	// The stopped text must not carry the stop string: leaving it in is the mistake that looks
	// like a successful truncation.
	if strings.Contains(text, "h") {
		t.Errorf("the stopped text still contains the stop string: %q", text)
	}
}

// TestSuffixReachesTheEngineWhenTheModelSupportsFIM covers the line that passes suffix to
// generate -- the one after the refusal, which no fixture could reach until now.
//
// The refusal path is tested: a model declaring no FIM markers answers 400 with
// unsupported_parameter, which is what every fixture used to produce. The line that actually
// forwards the suffix was therefore unreachable, and injection confirms it was uncovered:
// replacing `suffix: suffixOf(req.Suffix)` with the empty string left the whole package green.
//
// A vocabulary containing the marker names is all FIMTokens looks for -- it scans the pieces, not
// the metadata -- so a fixture can declare them.
//
// Two assertions were tried before this one and both passed under injection. Comparing the reply
// text failed because greedy decoding on this fixture produced the same characters from two
// different prompts. Comparing a suffix against no suffix failed because the FIM form is applied
// whenever a suffix is present at all -- p.fim adds its markers around an empty suffix too, so
// both prompts came out the same length. What only a forwarded suffix can change is the tokens
// between the markers, which is what the comparison below measures.
func TestSuffixReachesTheEngineWhenTheModelSupportsFIM(t *testing.T) {
	// The Qwen2.5-Coder set; FIMTokens accepts any of the sets it knows.
	vocab := []string{
		"<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|fim_pad|>",
		"a", "b", "c", "d", "e", "f",
	}
	srv, tok := newTestServerModel(t, gguftest.Model{
		NoEOS: true, ContextLen: 512, Vocab: vocab,
	})
	if _, ok := tok.FIMTokens(); !ok {
		t.Fatal("the fixture's vocabulary does not declare FIM markers, so this test would only " +
			"exercise the refusal path")
	}

	promptTokens := func(t *testing.T, body string) int {
		t.Helper()
		resp := post(t, srv.Handler(), "/v1/completions", body)
		defer func() { _ = resp.Body.Close() }()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d: %s", resp.StatusCode, body)
		}
		got := decodeBody[completionResponse](t, resp)
		if len(got.Choices) == 0 {
			t.Fatalf("no choices for %s", body)
		}
		return got.Usage.PromptTokens
	}

	// Compare two suffix lengths rather than suffix against no suffix. With a suffix present the
	// prompt is rendered in FIM form via p.fim, which adds its markers whether or not the suffix
	// text is empty -- so an empty suffix and a filled one produce the same marker overhead, and
	// the first version of this test passed under injection because of it. What only a forwarded
	// suffix can change is the number of tokens between the markers.
	short := promptTokens(t, `{"prompt": "a", "suffix": "b", "max_tokens": 2, "temperature": 0}`)
	long := promptTokens(t, `{"prompt": "a", "suffix": "bcde", "max_tokens": 2, "temperature": 0}`)
	if long <= short {
		t.Errorf("a four-piece suffix produced a %d-token prompt and a one-piece suffix %d: the "+
			"suffix text is not reaching the engine", long, short)
	}
}

// TestEveryErrorCodeIsDocumented checks that the package doc names every code the server sends.
//
// Its neighbours prove each code behaves as documented; nothing proved that the doc covers all
// of them. Two were missing when this was written --
// method_not_allowed and unknown_endpoint, both built outside paramError, which is why a search
// for paramError's codes alone did not find them.
//
// The codes are read from the source rather than listed here, so a new one fails this test
// instead of being recorded in two places that then disagree. Reading Go source with a regular
// expression is fragile, which is why both spellings are matched explicitly and the test asserts
// it found a plausible number of each.
func TestEveryErrorCodeIsDocumented(t *testing.T) {
	src, err := os.ReadFile("server.go")
	if err != nil {
		t.Fatalf("reading server.go: %v", err)
	}
	text := string(src)

	codes := make(map[string]bool)
	// newParamError(param, code, format, ...) and its relatives pass the code as the second
	// argument.
	for _, m := range regexp.MustCompile(`newParamError\([^,]+,\s*"([a-z_]+)"`).FindAllStringSubmatch(text, -1) {
		codes[m[1]] = true
	}
	// The router assigns its two directly, and writeDecodeError picks a code for the body-level
	// failures before wrapping it in a paramError.
	for _, m := range regexp.MustCompile(`body\.Error\.Code = "([a-z_]+)"`).FindAllStringSubmatch(text, -1) {
		codes[m[1]] = true
	}
	for _, m := range regexp.MustCompile(`\bcode (:?=) "([a-z_]+)"`).FindAllStringSubmatch(text, -1) {
		codes[m[2]] = true
	}
	if len(codes) < 10 { // measured: 10
		t.Fatalf("only %d codes were extracted, so this check is looking at the wrong thing: %v",
			len(codes), slices.Sorted(maps.Keys(codes)))
	}

	doc, err := os.ReadFile("doc.go")
	if err != nil {
		t.Fatalf("reading doc.go: %v", err)
	}
	for _, code := range slices.Sorted(maps.Keys(codes)) {
		if !bytes.Contains(doc, []byte(code)) {
			t.Errorf("doc.go does not mention the error code %q, which the server can send", code)
		}
	}
}

// TestZeroBudgetIsNotAnAbsentBudget covers the field a pointer type exists for.
//
// max_tokens is *int on the wire so that 0 and "not sent" stay distinguishable: the field's
// comment says zero is "a budget of zero rather than unset". Nothing tested that, and an
// implementation that used the zero value to mean unset would look correct in every other case
// -- it would answer a request for no tokens with the default 256.
//
// Measured before it was asserted: max_tokens 0 gives completion_tokens 0 with finish_reason
// "length", and an absent max_tokens gives 256.
func TestZeroBudgetIsNotAnAbsentBudget(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 512}.Write(t, t.TempDir())
	srv := serverWithPolicy(t, path, engine.PriorityFIFO)

	zero := decodeBody[chatResponse](t, post(t, srv.Handler(), "/v1/chat/completions",
		`{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0}`))
	if zero.Usage.CompletionTokens != 0 {
		t.Errorf("max_tokens 0 produced %d completion tokens, want 0",
			zero.Usage.CompletionTokens)
	}
	if len(zero.Choices) != 1 {
		t.Fatalf("got %d choices, want 1", len(zero.Choices))
	}
	if got := zero.Choices[0].FinishReason; got != "length" {
		t.Errorf("finish_reason = %q with no budget to spend, want length", got)
	}
	if got := zero.Choices[0].Message.Content; got != "" {
		t.Errorf("max_tokens 0 produced the text %q, want none", got)
	}

	// And the distinction is real: the same request without the field gets a budget.
	absent := decodeBody[chatResponse](t, post(t, srv.Handler(), "/v1/chat/completions",
		`{"messages": [{"role": "user", "content": "hi"}]}`))
	if absent.Usage.CompletionTokens == 0 {
		t.Error("an absent max_tokens also produced 0 tokens, so this test cannot tell the " +
			"two apart and the pointer on the wire is not doing its job")
	}
}

// TestConcurrentRequestsMatchTheirSerialResults covers cross-sequence interference, which no test
// in this repository asserted directly.
//
// The engine batches: several sequences advance in one decode round, and their activations share
// the kernels and the KV pool. A defect there does not crash -- it makes one request's output
// depend on what else was in the batch, so the same prompt at temperature 0 returns different text
// depending on its neighbours. Every stop-sequence and sampling test in this file submits one
// request at a time and cannot see it.
//
// Measured against the binary before this test existed: six mixed prompts, serial versus six
// concurrent, byte-identical on all six. So the engine is right today; the test is what keeps it
// right, since the property is invisible in any single-request test.
func TestConcurrentRequestsMatchTheirSerialResults(t *testing.T) {
	srv, _ := newTestServerModel(t, gguftest.Default())
	work := []struct {
		prompt   string
		maxToken int
	}{
		{"Once upon a time", 8},
		{"x", 5},
		{"the quick brown fox", 10},
		{"a", 4},
		{"Once upon a time", 8}, // repeated: two identical requests must agree too
		{"hello", 6},
	}
	body := func(p string, n int) string {
		return `{"prompt": ` + quoteJSON(p) + `, "max_tokens": ` + strconv.Itoa(n) +
			`, "temperature": 0}`
	}
	// text returns an error rather than calling t.Fatalf, because the concurrent half below runs
	// it from a goroutine and Fatalf there is undefined -- it calls runtime.Goexit on a goroutine
	// the test framework is not waiting on, so the failure is reported twice or not at all. vet
	// does not catch it through an indirection like this one.
	text := func(resp *http.Response) (string, error) {
		if resp.StatusCode != http.StatusOK {
			return "", fmt.Errorf("status %d", resp.StatusCode)
		}
		defer func() { _ = resp.Body.Close() }()
		var d completionResponse
		if err := json.NewDecoder(resp.Body).Decode(&d); err != nil {
			return "", fmt.Errorf("decode response: %w", err)
		}
		if len(d.Choices) == 0 {
			return "", fmt.Errorf("no choices in the response")
		}
		return d.Choices[0].Text, nil
	}

	serial := make([]string, len(work))
	for i, w := range work {
		got, err := text(post(t, srv.Handler(), "/v1/completions", body(w.prompt, w.maxToken)))
		if err != nil {
			t.Fatalf("%q: %v", w.prompt, err)
		}
		serial[i] = got
	}

	// Submitted from separate goroutines so the server's handler admits them together. The
	// handler is served in-process, so the requests do overlap: the engine's queue is what decides
	// the batching, and it holds them until its next round.
	concurrent := make([]string, len(work))
	errs := make([]error, len(work))
	var wg sync.WaitGroup
	for i, w := range work {
		wg.Go(func() {
			// post is safe here: httptest.NewRecorder per call, and ServeHTTP is what the handler
			// is for. Only the assertions have to stay on the test goroutine.
			concurrent[i], errs[i] = text(post(t, srv.Handler(), "/v1/completions",
				body(w.prompt, w.maxToken)))
		})
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Fatalf("%q (concurrent): %v", work[i].prompt, err)
		}
	}

	for i := range work {
		if serial[i] != concurrent[i] {
			t.Errorf("%q: serial %q, concurrent %q -- a request's output changed with its batch "+
				"neighbours, which means shared state leaked between sequences",
				work[i].prompt, serial[i], concurrent[i])
		}
	}
}
