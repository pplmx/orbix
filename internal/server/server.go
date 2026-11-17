// Package server exposes the engine over HTTP with the OpenAI chat completions
// API.
//
// The package sits above engine and owns exactly the concerns that belong to a
// network boundary: reading JSON, writing SSE, mapping an engine failure onto a
// status code, and stopping generation when a client disconnects. It knows nothing
// about tokens, KV caches, or batching -- those are the engine's job -- which is
// what keeps the API surface of a serving layer from leaking into the scheduler.
//
// Why the OpenAI shape: it is the interface every client already speaks. A local
// engine that implements it can be pointed at by the OpenAI SDKs, by editor
// extensions, and by anything with a base-URL setting, which makes the layer
// testable against real clients instead of against a bespoke protocol.
//
// The chat format is the model's own. Models store it as Jinja2 in a metadata string
// under tokenizer.chat_template, and internal/chat evaluates the subset published
// templates actually use; a template outside that subset is rejected at parse time and
// the endpoint falls back to a neutral transcript, which the server reports at startup.
// A caller who wants exact control sends a pre-rendered prompt to the completions
// endpoint, which passes text through verbatim.
package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/pplmx/orbix/internal/engine"
	"github.com/pplmx/orbix/internal/llama"
	"github.com/pplmx/orbix/internal/otel"
	"github.com/pplmx/orbix/internal/tokenize"
)

// Config configures a Server.
type Config struct {
	// ModelName is the id clients must send, and the id echoed in responses.
	// Empty means "orbix".
	ModelName string

	// MaxTokens is the default generation budget when a request omits one. Zero
	// means 256.
	MaxTokens int

	// ContextLen caps prompt+output tokens for one request. It comes from the
	// model, and exists here so the server can reject an oversized request with a
	// clear message instead of letting the engine fail per token.
	ContextLen int

	// Temperature is the default sampling temperature. Zero means greedy, which is
	// the project's own default: reproducible output makes a wrong forward pass
	// obvious.
	Temperature float64

	// TopK is the default top-k cutoff. Zero disables it.
	TopK int

	// ChatRenderer renders conversations with the model's own template when one is
	// available. Nil means the neutral transcript, which is also what a model without
	// a usable template gets.
	//
	// It is passed in rather than built here because the server never opens the GGUF
	// file: the tokenizer and engine are already constructed by the time a Config
	// exists, and reopening the file for one metadata string would make the server
	// responsible for a file it otherwise does not touch.
	ChatRenderer *engine.ChatRenderer

	// Logger receives one record per request and any server-side failure. Nil discards
	// records, so a caller that does not configure logging pays almost nothing: the
	// request wrapper checks for nil before measuring anything.
	Logger *slog.Logger

	// Trace, when non-nil, records a span per request. It is passed in rather than
	// built here for the same reason the chat renderer is: the server does not own the
	// process lifetime that decides where traces go.
	Trace *otel.Exporter
}

// Server serves one loaded model over HTTP.
//
// An Engine is not safe for concurrent Pump calls, so the server owns a single
// pump goroutine. Every request is submitted to the engine and fed tokens from its
// channel; the pump advances whichever sequences are active. That is the whole
// concurrency model, and it is why Submit must not block on a slow client: each
// sequence's channel is buffered, and a client that falls too far behind is
// cancelled by its own context.
type Server struct {
	eng      *engine.Engine
	tok      *tokenize.Tokenizer
	cfg      Config
	mux      *http.ServeMux
	stop     chan struct{}
	stopOnce sync.Once
	wg       sync.WaitGroup

	// log is cfg.Logger, kept as the concrete field so the request wrapper can check
	// for nil cheaply.
	log *slog.Logger

	// requests and tokens are counters for /metrics and for tests.
	requests atomic.Int64
	tokens   atomic.Int64
}

// New builds a server around a loaded engine and tokenizer.
func New(eng *engine.Engine, tok *tokenize.Tokenizer, cfg Config) (*Server, error) {
	if eng == nil {
		return nil, errors.New("server: nil engine")
	}
	if tok == nil {
		return nil, errors.New("server: nil tokenizer")
	}
	if cfg.ModelName == "" {
		cfg.ModelName = "orbix"
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = 256
	}
	if cfg.ContextLen <= 0 {
		cfg.ContextLen = eng.Model().Config().ContextLen
	}

	s := &Server{eng: eng, tok: tok, cfg: cfg, log: cfg.Logger, stop: make(chan struct{})}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.handleChat)
	mux.HandleFunc("POST /v1/completions", s.handleCompletion)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("GET /healthz", s.handleHealth)
	mux.HandleFunc("GET /metrics", s.handleMetrics)
	s.mux = mux
	return s, nil
}

// Handler returns the server's HTTP handler, for serving or for tests.
// Handler returns the server's routes, wrapped so each request produces one record when
// a logger is configured.
func (s *Server) Handler() http.Handler { return s.logRequest(s.mux) }

// run pumps the engine until the context is cancelled or Shutdown is called. It is
// unexported because calling it without accounting for it first panics: the deferred
// Done has no matching Add. StartPump is the way in.
//
// The engine only advances inside Pump, so nothing generates until this runs. It
// is a separate call from ServeHTTP because the pump must be a single goroutine:
// ServeHTTP handlers run concurrently and each one only waits for its own tokens.
// idlePollShift bounds the idle backoff: the pump sleeps 1ms << idlePollShift (about 8ms)
// between polls once the engine has been empty for that many rounds.
//
// The backoff resets the moment a round has work, so a busy server polls as fast as before
// and only a genuinely idle one pays the longer interval.
const idlePollShift = 3

func (s *Server) run(ctx context.Context) {
	defer s.wg.Done()
	// idle counts consecutive rounds with no active sequences; it selects the backoff.
	idle := 0
	for {
		select {
		case <-ctx.Done():
			return
		case <-s.stop:
			return
		default:
		}
		// Pump reports how many sequences are still active; when there are none
		// there is nothing to do, and a sleep keeps an idle server from spinning a core.
		// The sleep is the cost of a poll loop, and it is the price of keeping the engine
		// free of a server-facing wakeup channel.
		if n := s.eng.Pump(); n > 0 {
			// Work is in flight: poll fast, so the next token is not delayed by the
			// idle backoff.
			idle = 0
			continue
		}
		// Nothing is running. A 1ms poll costs about 2.4% of a core continuously, which
		// is a lot to pay for a server that is usually idle, so the interval backs off
		// while there is nothing to do. The first token after a quiet period waits at
		// most this long, so the ceiling is the latency floor a new request can see.
		idle = min(idle+1, idlePollShift)
		select {
		case <-ctx.Done():
			return
		case <-s.stop:
			return
		case <-time.After(time.Millisecond << idle):
		}
	}
}

// StartPump runs the pump in a goroutine and returns once it is accounted for.
//
// Use this instead of `go srv.Run(ctx)`. The waitgroup has to be incremented before the
// goroutine exists, not inside it: an Add that happens in the goroutine leaves a window
// where Shutdown's Wait sees a zero counter, returns, and reports a clean stop while the
// pump is still starting. Run itself cannot close that window because by the time it runs
// the caller has already been given a goroutine handle.
func (s *Server) StartPump(ctx context.Context) {
	s.wg.Add(1)
	go s.run(ctx)
}

// Shutdown stops the pump goroutine and waits for it.
//
// Safe to call more than once, and safe to call before StartPump. Closing the channel
// twice panics, which a server handling two shutdown signals at once would hit: a signal
// handler and a deferred cleanup both running is the ordinary case, not an exotic one.
func (s *Server) Shutdown() {
	s.stopOnce.Do(func() { close(s.stop) })
	s.wg.Wait()
}

// Stats returns the request and token counters.
func (s *Server) Stats() (requests, tokens int64) {
	return s.requests.Load(), s.tokens.Load()
}

// chatRequest is the subset of the OpenAI chat request the server honours.
//
// Unknown fields are ignored rather than rejected: clients send fields like
// logit_bias and response_format that a local engine has no use for, and refusing
// them would make the server unusable from an SDK that always sends them.
type chatRequest struct {
	Model     string        `json:"model"`
	Messages  []chatMessage `json:"messages"`
	MaxTokens int           `json:"max_tokens"`

	// Priority ranks this request against others waiting for a slot; higher advances
	// first, and an omitted value means "with the rest". It is Orbix's own extension:
	// OpenAI's API has no equivalent, so a client that never sends it is unaffected.
	Priority int `json:"priority"`

	Temperature *float64  `json:"temperature"`
	TopK        int       `json:"top_k"`
	Stream      bool      `json:"stream"`
	Stop        stopField `json:"stop"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type completionRequest struct {
	Model     string `json:"model"`
	Prompt    string `json:"prompt"`
	MaxTokens int    `json:"max_tokens"`

	// Priority ranks this request against others waiting for a slot; see chatRequest.
	Priority int `json:"priority"`

	Temperature *float64  `json:"temperature"`
	TopK        int       `json:"top_k"`
	Stream      bool      `json:"stream"`
	Stop        stopField `json:"stop"`
}

// stopField accepts OpenAI's two spellings: a string or an array of strings.
type stopField []string

func (f *stopField) UnmarshalJSON(b []byte) error {
	if len(b) == 0 || string(b) == "null" {
		return nil
	}
	if b[0] == '[' {
		var list []string
		if err := json.Unmarshal(b, &list); err != nil {
			return err
		}
		*f = list
		return nil
	}
	var one string
	if err := json.Unmarshal(b, &one); err != nil {
		return err
	}
	*f = []string{one}
	return nil
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok", "model": s.cfg.ModelName})
}

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	// The OpenAI client libraries fetch this to validate a model name.
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data": []map[string]any{{
			"id":       s.cfg.ModelName,
			"object":   "model",
			"created":  0,
			"owned_by": "orbix",
		}},
	})
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages is required")
		return
	}
	prompt := s.renderChat(req.Messages)
	s.generate(w, r, generateParams{
		prompt:      prompt,
		maxTokens:   req.MaxTokens,
		priority:    req.Priority,
		temperature: req.Temperature,
		topK:        req.TopK,
		stream:      req.Stream,
		stop:        req.Stop,
		chat:        true,
	})
}

func (s *Server) handleCompletion(w http.ResponseWriter, r *http.Request) {
	var req completionRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if strings.TrimSpace(req.Prompt) == "" {
		writeError(w, http.StatusBadRequest, "prompt is required")
		return
	}
	s.generate(w, r, generateParams{
		prompt:      req.Prompt,
		maxTokens:   req.MaxTokens,
		priority:    req.Priority,
		temperature: req.Temperature,
		topK:        req.TopK,
		stream:      req.Stream,
		stop:        req.Stop,
	})
}

// renderChat applies the chat template.
//
// The rendering itself lives in engine.RenderChatPrompt so the HTTP server and the REPL
// cannot drift apart: the same conversation must produce the same tokens however it was
// asked. See that function for why the template is neutral rather than a model's own.
func (s *Server) renderChat(msgs []chatMessage) string {
	turns := make([]engine.Turn, len(msgs))
	for i, m := range msgs {
		turns[i] = engine.Turn{Role: m.Role, Text: m.Content}
	}
	// Render always returns a usable prompt: a model template that fails at render time
	// falls back to the neutral transcript inside the renderer, so this call cannot
	// leave the request without a prompt.
	prompt, _ := s.cfg.ChatRenderer.Render(turns)
	return prompt
}

// generateParams is what both endpoints share once their JSON is decoded.
type generateParams struct {
	prompt      string
	maxTokens   int
	priority    int
	temperature *float64
	topK        int
	stream      bool
	stop        []string
	chat        bool
}

func (s *Server) generate(w http.ResponseWriter, r *http.Request, p generateParams) {
	maxTokens := p.maxTokens
	if maxTokens <= 0 {
		maxTokens = s.cfg.MaxTokens
	}

	// Encode with BOS only when the model defines one: a model without a BOS id
	// encodes the prompt as-is.
	cfg := s.eng.Model().Config()
	//
	// A chat prompt is markup, so its control markers are matched whole; a completions
	// prompt is text the caller wrote, so a literal "<|im_start|>" there stays text.
	// The same distinction the CLI's chat command makes, for the same reason.
	var promptTokens []int32
	if p.chat {
		promptTokens = s.tok.EncodeWithSpecials(p.prompt, cfg.BOSID >= 0)
	} else {
		promptTokens = s.tok.Encode(p.prompt, cfg.BOSID >= 0)
	}
	if len(promptTokens) == 0 {
		writeError(w, http.StatusBadRequest, "prompt encoded to no tokens")
		return
	}
	if len(promptTokens) >= s.cfg.ContextLen {
		writeError(w, http.StatusBadRequest, fmt.Sprintf(
			"prompt is %d tokens, context is %d", len(promptTokens), s.cfg.ContextLen))
		return
	}
	// A request cannot exceed the window. Truncating the budget is the useful
	// behaviour: a client asking for 512 tokens with 100 left should get the 100.
	if room := s.cfg.ContextLen - len(promptTokens); maxTokens > room {
		maxTokens = room
	}

	sampler, err := s.sampler(p.temperature, p.topK)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	stopIDs := s.stopTokenIDs(p.stop)
	ch, err := s.eng.Submit(engine.Request{
		Prompt:    promptTokens,
		MaxTokens: maxTokens,
		Priority:  p.priority,
		Stop:      stopIDs,
		Sampler:   sampler,
		Cancel:    r.Context().Done(),
		Trace:     s.cfg.Trace,
	})
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	s.requests.Add(1)

	// OpenAI prefixes a completion id with "cmpl-" and a chat completion with "chatcmpl-".
	// The prefix is cosmetic to a parser but visible to anyone reading a log, and the wrong
	// one makes a completion look like a chat response.
	prefix := "cmpl-"
	if p.chat {
		prefix = "chatcmpl-"
	}
	id := fmt.Sprintf("%s%d", prefix, time.Now().UnixNano())
	if p.stream {
		s.streamResponse(w, r, ch, id, p.chat)
		return
	}
	s.blockingResponse(w, r, ch, id, p, promptTokens)
}

// blockingResponse collects the whole output and writes one JSON body.
func (s *Server) blockingResponse(w http.ResponseWriter, r *http.Request, ch <-chan engine.Event, id string, p generateParams, promptTokens []int32) {
	var out []int32
	reason := "stop"
	for ev := range ch {
		if ev.Err != nil {
			writeError(w, http.StatusInternalServerError, ev.Err.Error())
			return
		}
		if ev.HasToken {
			out = append(out, ev.Token)
			s.tokens.Add(1)
		}
		if ev.Done {
			reason = ev.FinishReason
		}
	}
	text := s.tok.DecodeCompletion(out)

	usage := map[string]int{
		"prompt_tokens":     len(promptTokens),
		"completion_tokens": len(out),
		"total_tokens":      len(promptTokens) + len(out),
	}
	stopReason := "stop"
	if reason == "length" || reason == "cancel" {
		stopReason = "length"
	}

	if p.chat {
		writeJSON(w, http.StatusOK, map[string]any{
			"id":      id,
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"model":   s.cfg.ModelName,
			"choices": []map[string]any{{
				"index":         0,
				"message":       map[string]string{"role": "assistant", "content": text},
				"finish_reason": stopReason,
			}},
			"usage": usage,
		})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id":      id,
		"object":  "text_completion",
		"created": time.Now().Unix(),
		"model":   s.cfg.ModelName,
		"choices": []map[string]any{{
			"index":         0,
			"text":          text,
			"finish_reason": stopReason,
		}},
		"usage": usage,
	})
}

// streamResponse writes server-sent events, one per token.
func (s *Server) streamResponse(w http.ResponseWriter, r *http.Request, ch <-chan engine.Event, id string, chat bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming is not supported by this connection")
		return
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	// OpenAI's spec is explicit that a completion is "always" text_completion, and its own
	// streaming example carries the same value as the non-streaming one -- the two endpoints
	// differ here, since chat streams object="chat.completion.chunk". A client that switches
	// on this field to tell a delta from a whole body looks for the chunk name it knows.
	object := "text_completion"
	if chat {
		object = "chat.completion.chunk"
	}
	send := func(payload map[string]any) bool {
		payload["id"] = id
		payload["object"] = object
		payload["created"] = time.Now().Unix()
		payload["model"] = s.cfg.ModelName
		b, err := json.Marshal(payload)
		if err != nil {
			return false
		}
		if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
			return false
		}
		flusher.Flush()
		return true
	}

	dec := s.tok.Seq()
	reason := "stop"
	for ev := range ch {
		if ev.Err != nil {
			// The response has already started, so a status code is no longer
			// available: report the failure in the stream and stop.
			send(map[string]any{"choices": []map[string]any{{
				"index": 0, "delta": map[string]string{}, "finish_reason": "error",
			}}, "error": ev.Err.Error()})
			return
		}
		if ev.HasToken {
			s.tokens.Add(1)
			text := dec.Token(ev.Token)
			if text == "" {
				continue
			}
			if !send(chunkPayload(chat, text)) {
				// The client went away. Its context is cancelled, so the engine
				// stops the sequence; draining is the caller's job and this loop
				// is already past that point.
				return
			}
		}
		if ev.Done {
			reason = ev.FinishReason
		}
	}
	stopReason := "stop"
	if reason == "length" || reason == "cancel" {
		stopReason = "length"
	}
	// One final chunk carries the finish reason and an empty delta, then the
	// sentinel every OpenAI stream ends with. Clients that accumulate deltas see no
	// text change from either.
	final := map[string]any{
		"id": id, "object": object, "created": time.Now().Unix(), "model": s.cfg.ModelName,
		"choices": []map[string]any{{"index": 0, "delta": map[string]string{}, "finish_reason": stopReason}},
	}
	if !chat {
		final["choices"] = []map[string]any{{"index": 0, "text": "", "finish_reason": stopReason}}
	}
	b, err := json.Marshal(final)
	if err != nil {
		return
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
		return
	}
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()
}

// chunkPayload builds one streaming chunk.
func chunkPayload(chat bool, text string) map[string]any {
	if chat {
		return map[string]any{"choices": []map[string]any{{
			"index": 0, "delta": map[string]string{"content": text}, "finish_reason": nil,
		}}}
	}
	return map[string]any{"choices": []map[string]any{{
		"index": 0, "text": text, "finish_reason": nil,
	}}}
}

// sampler builds the sampler a request asked for, or the server default.
func (s *Server) sampler(temp *float64, topK int) (llama.Sampler, error) {
	t := s.cfg.Temperature
	if temp != nil {
		t = *temp
	}
	if topK <= 0 {
		topK = s.cfg.TopK
	}
	if t < 0 {
		return nil, fmt.Errorf("temperature must not be negative")
	}
	if t == 0 {
		// Greedy is the default and the reproducible choice.
		return greedySampler{}, nil
	}
	seed := time.Now().UnixNano()
	r := newRand(seed)
	return temperatureSampler{temp: float32(t), topK: topK, src: r}, nil
}

// stopTokenIDs maps stop strings onto token ids.
//
// The engine stops on token ids, not strings, so a stop string is encoded and only
// its first token is used: a multi-token stop string would need prefix matching on
// the decoded text, which is the server's job and not the engine's. Encoding to one
// token covers the common cases ("\n", "###", a newline-terminated marker) and the
// limitation is documented rather than hidden.
func (s *Server) stopTokenIDs(stop []string) []int32 {
	var ids []int32
	for _, text := range stop {
		if text == "" {
			continue
		}
		enc := s.tok.Encode(text, false)
		if len(enc) == 0 {
			continue
		}
		ids = append(ids, enc[0])
	}
	return ids
}

func decodeJSON(r *http.Request, dst any) error {
	defer func() { _ = r.Body.Close() }()
	dec := json.NewDecoder(r.Body)
	// Unknown fields are ignored on purpose: see chatRequest.
	if err := dec.Decode(dst); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

// errorBody is OpenAI's error envelope, so client libraries surface the message.
type errorBody struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

func writeError(w http.ResponseWriter, status int, msg string) {
	var body errorBody
	body.Error.Message = msg
	body.Error.Type = "invalid_request_error"
	if status >= 500 {
		body.Error.Type = "server_error"
	}
	writeJSON(w, status, body)
}
