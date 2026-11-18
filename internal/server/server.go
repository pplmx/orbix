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

	// TopP is the default nucleus threshold. Zero disables it, which is the project's
	// default: top-k and top-p both narrow the candidate set, and turning on a second
	// limit by default would change what an existing flag combination does.
	TopP float64

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
	TopP        *float64  `json:"top_p"`
	Stream      bool      `json:"stream"`
	Stop        stopField `json:"stop"`

	// Seed makes a sampled reply reproducible: the same prompt, settings and seed produce
	// the same text. It is ignored when temperature is 0, because greedy is already
	// deterministic and spending a generator on it would only slow the request down.
	Seed int64 `json:"seed"`
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
	TopP        *float64  `json:"top_p"`
	Stream      bool      `json:"stream"`
	Stop        stopField `json:"stop"`

	// Seed makes a sampled reply reproducible: the same prompt, settings and seed produce
	// the same text. It is ignored when temperature is 0, because greedy is already
	// deterministic and spending a generator on it would only slow the request down.
	Seed int64 `json:"seed"`
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

// checkModel rejects a request naming a model this server does not serve.
//
// An empty name is accepted: the OpenAI clients omit it when built without one, and there is
// only one model here to mean. A name that does not match is refused rather than ignored,
// because ignoring it is the failure this check exists for -- a client pointed at the wrong
// server gets answers from a model it did not ask for, and every response echoes the server's
// own name back, so nothing in the output reveals the mix-up.
func (s *Server) checkModel(w http.ResponseWriter, name string) bool {
	if name == "" || name == s.cfg.ModelName {
		return true
	}
	var body errorBody
	body.Error.Message = fmt.Sprintf("The model %q does not exist; this server serves %q",
		name, s.cfg.ModelName)
	body.Error.Type = "invalid_request_error"
	body.Error.Param = "model"
	body.Error.Code = "model_not_found"
	writeJSON(w, http.StatusNotFound, body)
	return false
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
		// 415 for a body this server will not decode, 400 for one it tried to. The difference
		// matters to a client: the first says "change what you send", the second says "what you
		// sent was malformed".
		status := http.StatusBadRequest
		if errors.Is(err, errUnsupportedEncoding) {
			status = http.StatusUnsupportedMediaType
		}
		writeError(w, status, err.Error())
		return
	}
	if !s.checkModel(w, req.Model) {
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
		topP:        req.TopP,
		stream:      req.Stream,
		stop:        req.Stop,
		chat:        true,
		seed:        req.Seed,
	})
}

func (s *Server) handleCompletion(w http.ResponseWriter, r *http.Request) {
	var req completionRequest
	if err := decodeJSON(r, &req); err != nil {
		// 415 for a body this server will not decode, 400 for one it tried to. The difference
		// matters to a client: the first says "change what you send", the second says "what you
		// sent was malformed".
		status := http.StatusBadRequest
		if errors.Is(err, errUnsupportedEncoding) {
			status = http.StatusUnsupportedMediaType
		}
		writeError(w, status, err.Error())
		return
	}
	if !s.checkModel(w, req.Model) {
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
		topP:        req.TopP,
		stream:      req.Stream,
		stop:        req.Stop,
		seed:        req.Seed,
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
	topP        *float64
	stream      bool
	stop        []string
	chat        bool
	// seed makes sampling reproducible. Zero means "choose one", which is what a caller
	// that did not ask for reproducibility gets.
	seed int64
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

	sampler, err := s.sampler(p.temperature, p.topK, p.topP, p.seed)
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
		// A full queue is not the client's fault and not permanent: the request is well
		// formed and would succeed later. 503 with Retry-After is what a client or proxy
		// knows to back off on, where 400 would tell it to stop sending this at all.
		if errors.Is(err, engine.ErrQueueFull) {
			w.Header().Set("Retry-After", "1")
			writeError(w, http.StatusServiceUnavailable, err.Error())
			return
		}
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
		s.streamResponse(w, r, ch, id, p.chat, p.stop)
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
			// A pool that ran out is capacity, not a fault in the request or the server:
			// the same request succeeds once another sequence releases its blocks. 503
			// with Retry-After is what a client backs off on, and 500 would have it
			// treat a temporary condition as a permanent one.
			if errors.Is(ev.Err, llama.ErrPoolExhausted) {
				w.Header().Set("Retry-After", "1")
				writeError(w, http.StatusServiceUnavailable, ev.Err.Error())
				return
			}
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
	// A stop string is text, and generation halts where it would appear. OpenAI's API takes
	// strings, and a string need not be one token: " Paris" encodes as a space token
	// followed by the word's own. stopTokenIDs can only hand the engine a single token, so
	// the engine-side check cannot express this -- and taking the first token of a
	// multi-token string is actively wrong, since the space token then stops every sequence
	// at its first space and the response comes back empty.
	if cut, found := cutAtStop(text, p.stop); found {
		text = cut
		reason = "stop"
	}

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
func (s *Server) streamResponse(w http.ResponseWriter, r *http.Request, ch <-chan engine.Event, id string, chat bool, stop []string) {
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
	// seen is the text already handed to the client, kept only when a stop string has to
	// be matched against it.
	seen := ""
	// sent is how much of seen the client already has. With a stop list the tail is held
	// back until it cannot be the start of a match, so this trails seen.
	sent := 0
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
			// A stop string has to be applied to the stream as well as to the whole body.
			// The engine only knows about single tokens, so the check is on the text
			// accumulated so far: it can only be recognised once the rune that completes it
			// arrives, which is exactly when the client should stop seeing output.
			if len(stop) > 0 {
				seen += text
				if kept, found := cutAtStop(seen, stop); found {
					// Output up to the stop string is the client's; the stop and anything
					// after it are not. `sent` is what the client already has, so only the
					// part between them still needs sending.
					if tail := kept[min(len(kept), sent):]; tail != "" {
						if !send(chunkPayload(chat, tail)) {
							return
						}
					}
					// The sequence is still running; its context drives it to completion
					// unread, which is what the engine does for a client that stopped
					// reading. What changes here is only what this response reports.
					reason = "stop"
					break
				}
				// Nothing matched yet, but the tail may be the beginning of a stop string,
				// and a stop string only becomes recognisable once its last character
				// arrives -- by which time a character already sent cannot be taken back.
				// Hold back the longest tail that is a prefix of some stop string, and send
				// everything before it. Without this, stop=["北京"] streamed "北" where the
				// non-streaming body returned nothing.
				if safe := safePrefixLen(seen, stop); safe > sent {
					if !send(chunkPayload(chat, seen[sent:safe])) {
						return
					}
					sent = safe
				}
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
	// A character split across the last tokens is still held by the decoder, waiting for a
	// byte that never arrives. Without this the stream would end one character short of what
	// the same sequence produces without streaming, which is the contract README states.
	if tail := dec.Flush(); tail != "" {
		if !send(chunkPayload(chat, tail)) {
			return
		}
	}
	// Text held back in case it began a stop string is the client's after all: no match was
	// found, so the tail was ordinary output. Sending it is what makes a stop list that never
	// matches cost nothing but a later chunk.
	if reason != "stop" && sent < len(seen) {
		if !send(chunkPayload(chat, seen[sent:])) {
			return
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
func (s *Server) sampler(temp *float64, topK int, topP *float64, seed int64) (llama.Sampler, error) {
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
	// TopP defaults to the server's own value when the request omits it. A value outside
	// (0, 1] is refused rather than clamped: 0 and 1 both mean "no limit" to some callers
	// and "only the best token" to others, and guessing would change the output silently.
	p := s.cfg.TopP
	if topP != nil {
		p = *topP
	}
	if p < 0 || p > 1 {
		return nil, fmt.Errorf("top_p must be between 0 and 1, got %v", p)
	}
	if t == 0 {
		// Greedy is the default and the reproducible choice.
		return greedySampler{}, nil
	}
	// A caller-supplied seed makes the request reproducible, which is the whole point of
	// the parameter: the same prompt and settings give the same text. Leaving it to the
	// clock is what makes two otherwise identical requests differ.
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	return temperatureSampler{temp: float32(t), topK: topK, topP: float32(p), src: newRand(seed)}, nil
}

// stopTokenIDs maps stop strings onto token ids for the engine's early-out.
//
// Only a string that encodes to exactly one token is passed on. The engine stops on a
// token id, so a multi-token string cannot be expressed there, and using its first token
// is worse than not passing it at all: " Paris" encodes as a space token followed by the
// word, so the space token fires at the first space of the reply and the response comes
// back empty. Every stop string is handled correctly regardless, by cutting the decoded
// text in cutAtStop.
//
// The engine-side check is an optimisation: it ends generation a round earlier for the
// single-token cases ("\n", "###") and costs nothing when it does not apply.
// cutAtStop returns text up to the first stop string, and whether one was found.
//
// The earliest match wins: a client passing several stop strings expects generation to end
// at whichever appears first in the output, not at whichever was listed first.
// safePrefixLen reports how much of text can be sent without risking a stop string that is
// still only partly matched.
//
// A stop string becomes recognisable only once its final character has arrived, and by then
// anything already sent cannot be taken back. The caller therefore holds back the longest tail
// of text that could still turn into a match and sends the rest.
//
// Only the suffixes of text matter: a stop string starting earlier would already have been
// found by cutAtStop. So the tail to hold is the longest suffix that is a proper prefix of some
// stop string -- "北" when the stop list holds "北京" and text ends with "北".
func safePrefixLen(text string, stop []string) int {
	hold := 0
	for _, s := range stop {
		if s == "" {
			continue
		}
		// A suffix of text that is a prefix of s has to be shorter than s; equal length would
		// be a complete match, which cutAtStop already handled.
		for n := min(len(s)-1, len(text)); n > hold; n-- {
			if strings.HasPrefix(s, text[len(text)-n:]) {
				hold = n
				break
			}
		}
	}
	return len(text) - hold
}

func cutAtStop(text string, stop []string) (string, bool) {
	cut := -1
	for _, s := range stop {
		if s == "" {
			continue
		}
		i := strings.Index(text, s)
		if i < 0 {
			continue
		}
		if cut < 0 || i < cut {
			cut = i
		}
	}
	if cut < 0 {
		return text, false
	}
	return text[:cut], true
}

func (s *Server) stopTokenIDs(stop []string) []int32 {
	var ids []int32
	for _, text := range stop {
		if text == "" {
			continue
		}
		enc := s.tok.Encode(text, false)
		// Exactly one token, not at least one: the engine can only match that.
		if len(enc) != 1 {
			continue
		}
		ids = append(ids, enc[0])
	}
	return ids
}

// errUnsupportedEncoding marks a request whose body is compressed in a way this server does not
// decode, so the handler can answer 415 rather than a 400 that blames the JSON.
var errUnsupportedEncoding = errors.New("unsupported content encoding")

func decodeJSON(r *http.Request, dst any) error {
	defer func() { _ = r.Body.Close() }()
	// A client may compress its request body, and the only thing that must happen here is that
	// the result is not a JSON error: reading a gzip stream as text yields "invalid character
	// '\x1f'", which points at the bytes rather than at the encoding that produced them.
	//
	// The body is not decompressed. A local engine serves prompts measured in kilobytes, so the
	// bytes saved are worth less than the dependency and the decompression-bomb surface that
	// would come with handling them.
	//
	// Content-Type is deliberately not checked, and the asymmetry is the point: an encoding the
	// server will not apply changes the bytes, so proceeding yields a parse error about something
	// else, while a media type only says what the bytes are. A client that labels JSON as
	// text/plain has still sent JSON, and refusing it would reject a request this server can serve
	// perfectly well. curl -d sets application/x-www-form-urlencoded rather than leaving the header
	// off, which is exactly that case and is how most hand-written example requests here are sent.
	if enc := r.Header.Get("Content-Encoding"); enc != "" && !strings.EqualFold(enc, "identity") {
		return fmt.Errorf("%w: %s; send the body uncompressed", errUnsupportedEncoding, enc)
	}
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
		// Param names the request field at fault, when one is.
		Param string `json:"param,omitempty"`
		// Code is the machine-readable reason, which is what a client branches on. The
		// OpenAI libraries surface it as e.code and a 404 as a model-not-found error.
		Code string `json:"code,omitempty"`
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
