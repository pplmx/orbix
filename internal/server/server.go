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

	// MaxTokens is the generation budget for a request that does not name one. Zero
	// means 256.
	//
	// A request cannot ask for zero tokens: max_tokens is an int rather than a pointer, so
	// an explicit 0 is indistinguishable from the field being absent and both take this
	// default. Measured against stories15M (128 positions): with the prompt "Once upon a
	// time" -- five tokens -- both a body carrying {"max_tokens": 0} and one omitting the
	// field generated 123 tokens, the whole remaining window. The count follows the prompt,
	// so "hi" yields 126 from the same server; what the measurement establishes is that the
	// two spellings agree, not that 123 is a constant.
	//
	// The engine itself treats 0 as "no tokens" and finishes immediately, which is why the
	// CLI's --max-tokens 0 does produce nothing -- the two layers disagree, and the server
	// is the one that cannot express the request.
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
	// The SDKs' models.retrieve calls this path, so a client that validates a model by id
	// rather than by listing gets an answer instead of a 404.
	mux.HandleFunc("GET /v1/models/{model}", s.handleModel)
	mux.HandleFunc("GET /healthz", s.handleHealth)
	mux.HandleFunc("GET /metrics", s.handleMetrics)
	// Everything else under /v1 answers in the API's own error envelope. Without this, a Go
	// mux writes its plain-text "404 page not found", and the SDKs decode the body as JSON
	// before they raise: a request to an endpoint this server does not implement fails with a
	// JSON parse error inside the client rather than with the not-found error the API
	// documents. The plain-text body stays right for a probe at a path outside the API,
	// which is why this is scoped to /v1.
	//
	// It is method-less on purpose: a Go mux grants a method-less pattern the path, so a
	// request to a known route with the wrong method lands here rather than getting the
	// mux's own 405. That 405 is load-bearing -- README's CORS note rests on OPTIONS
	// answering it, since a cross-origin application/json request then never leaves the
	// preflight -- so handleUnimplemented restores it for a route that exists.
	mux.HandleFunc("/v1/", s.handleUnimplemented)
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
//
// The pump is what advances every active sequence, so shutting it down ends every response
// still in flight. A caller that wants those responses to finish must drain the HTTP server
// first and call this afterwards: cmd/orbix's serve does exactly that, and the order used to
// be reversed there, which left streaming responses with nothing to produce their next token
// until the drain's grace period cut them off. The sequencing is the caller's because this
// package does not know what is being served -- a caller using the engine directly may have
// no in-flight requests to wait for at all.
func (s *Server) Shutdown() {
	s.stopOnce.Do(func() { close(s.stop) })
	s.wg.Wait()
}

// chatRequest is the subset of the OpenAI chat request the server honours.
//
// Unknown fields are ignored rather than rejected: clients send fields like
// logit_bias and response_format that a local engine has no use for, and refusing
// them would make the server unusable from an SDK that always sends them.
//
// The fields below that change the *shape* of the response are declared so they can be
// refused instead. There are two kinds of unsupported field and they differ in what a
// client does when the server pretends not to see one:
//
//   - A field that only tunes sampling (logit_bias, presence_penalty) leaves the response a
//     client can still read: it is the wrong text, but choices[0].text is where it expected
//     it. Ignoring it costs quality and keeps working.
//   - A field that describes the response (n, logprobs, echo, suffix) is a contract about
//     the body. A client that sent n=3 indexes choices[2]; one that sent logprobs reads
//     choice.logprobs. Answering with one choice and no logprobs is not a smaller answer but
//     a different one, and the client fails with an IndexError or a KeyError somewhere far
//     from the request that caused it.
//
// So the second kind is refused with a 400 naming the field, which is a failure the caller
// can see and act on at the call site, and the first kind stays ignored.
type chatRequest struct {
	Model     string        `json:"model"`
	Messages  []chatMessage `json:"messages"`
	MaxTokens int           `json:"max_tokens"`

	// N, Logprobs, TopLogprobs and Echo change the response body, so they are refused
	// rather than ignored; see the type comment. Their types are loose on purpose: the
	// only thing read is whether the client asked for something, and a value this struct
	// would reject as malformed is still a value the caller meant to send.
	N           int   `json:"n"`
	Logprobs    *bool `json:"logprobs"`
	TopLogprobs *int  `json:"top_logprobs"`
	Echo        *bool `json:"echo"`

	// Priority ranks this request against others waiting for a slot; higher advances
	// first, and an omitted value means "with the rest". It is Orbix's own extension:
	// OpenAI's API has no equivalent, so a client that never sends it is unaffected.
	Priority int `json:"priority"`

	Temperature *float64  `json:"temperature"`
	TopK        int       `json:"top_k"`
	TopP        *float64  `json:"top_p"`
	Stream      bool      `json:"stream"`
	Stop        stopField `json:"stop"`

	// StreamOptions is OpenAI's stream_options. Only include_usage is honoured, because it
	// is the only member a client reads a value from; see generateParams.includeUsage.
	StreamOptions *streamOptions `json:"stream_options"`

	// Seed makes a sampled reply reproducible: the same prompt, settings and seed produce
	// the same text. It is ignored when temperature is 0, because greedy is already
	// deterministic and spending a generator on it would only slow the request down.
	//
	// Zero means "pick one", and it is what an absent field decodes to -- the field is a
	// plain int64 rather than a pointer, so an omitted seed and an explicit 0 are the same
	// request. That matches the API's own reading of an absent seed, and it is why the
	// sampler substitutes the clock rather than seeding a generator with zero.
	Seed int64 `json:"seed"`
}

// streamOptions is OpenAI's stream_options. Only the members this server acts on are
// declared; an unknown member is ignored like any other unknown field.
type streamOptions struct {
	// IncludeUsage asks for one extra chunk before [DONE] whose usage field is the whole
	// request's token counts. Without it a streaming response reports no usage at all,
	// which leaves a streaming client unable to account for what it spent.
	IncludeUsage bool `json:"include_usage"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type completionRequest struct {
	Model     string `json:"model"`
	Prompt    string `json:"prompt"`
	MaxTokens int    `json:"max_tokens"`

	// The response-shape fields, refused rather than ignored; see chatRequest.
	N        int     `json:"n"`
	Logprobs *int    `json:"logprobs"`
	Echo     *bool   `json:"echo"`
	Suffix   *string `json:"suffix"`
	BestOf   int     `json:"best_of"`

	// Priority ranks this request against others waiting for a slot; see chatRequest.
	Priority int `json:"priority"`

	Temperature *float64  `json:"temperature"`
	TopK        int       `json:"top_k"`
	TopP        *float64  `json:"top_p"`
	Stream      bool      `json:"stream"`
	Stop        stopField `json:"stop"`

	// StreamOptions is OpenAI's stream_options; see chatRequest.
	StreamOptions *streamOptions `json:"stream_options"`

	// Seed makes a sampled reply reproducible, and zero means "pick one"; see
	// chatRequest's Seed for why zero and absent are the same request.
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

// modelObject is one entry of the models list, and the body models.retrieve returns.
//
// It is built in one place because the list and the retrieve endpoint must not disagree: an
// SDK that lists a model and then retrieves it compares the two, and a field present in one
// and absent from the other reads as a server that changed underneath it.
//
// created is zero rather than a real timestamp because nothing here knows one: a GGUF file
// records no creation date, and the file's mtime is when it was copied rather than when the
// model was trained. Zero is the epoch, which is at least visibly "unknown" instead of
// plausible and wrong. A client that sorts models by created sees all of them equal, which
// is the truth for a server that serves exactly one.
func (s *Server) modelObject() map[string]any {
	return map[string]any{
		"id":       s.cfg.ModelName,
		"object":   "model",
		"created":  0,
		"owned_by": "orbix",
	}
}

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	// The OpenAI client libraries fetch this to validate a model name.
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data":   []map[string]any{s.modelObject()},
	})
}

// handleModel answers GET /v1/models/{model}, which the SDKs' models.retrieve calls.
//
// The id is matched the way checkModel matches a request: an unknown name is a 404 with the
// same `model_not_found` code the completion endpoints use, rather than a bare 404, so a
// client that retrieves before sending a request gets the answer it would have got from the
// request itself.
func (s *Server) handleModel(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("model")
	if !s.checkModel(w, name) {
		return
	}
	writeJSON(w, http.StatusOK, s.modelObject())
}

// handleUnimplemented answers any other path under /v1 in the API's error envelope.
//
// A missing endpoint is not a client bug to be silently ignored: the SDKs call endpoints this
// engine deliberately does not implement (embeddings, images, responses), and the honest
// answer is a JSON 404 naming the path rather than Go's plain-text "404 page not found",
// which the SDK cannot decode and reports as a parse failure.
// v1Routes is every path under /v1 this server serves, with the method each answers.
//
// It exists so the fallback can tell "this route does not take that method" from "this path
// is not a route": the first is a 405 with an Allow header, which is what a mux would have
// said had the method-less fallback not taken the path from it, and the second is a 404. The
// distinction is not cosmetic -- a cross-origin preflight is answered by the 405 and not by
// a 404, which is what keeps a browser from sending an unauthenticated inference request.
var v1Routes = map[string]string{
	"/v1/chat/completions": http.MethodPost,
	"/v1/completions":      http.MethodPost,
	"/v1/models":           http.MethodGet,
}

// handleUnimplemented answers a request under /v1 that no route above matched.
//
// Two cases arrive here, because the fallback pattern is method-less and therefore wins the
// path from the mux:
//
//   - The path is a route but the method is not the one it takes. That is a 405 with Allow,
//     the same answer the mux gives when the fallback is absent. OPTIONS lands here, which is
//     the CORS preflight: answering 404 instead would tell a browser the route is missing,
//     while 405 tells it the method is not allowed, and the preflight fails either way -- but
//     only the 405 is true, and README's account of the CORS gap depends on it.
//   - The path is not a route at all. That is a 404 in the API's envelope, so an SDK that
//     called an endpoint this engine does not implement raises a not-found rather than a
//     JSON parse error.
func (s *Server) handleUnimplemented(w http.ResponseWriter, r *http.Request) {
	// A models sub-path is a dynamic route, so it is matched by prefix rather than by the
	// exact entries above.
	allowed, known := v1Routes[r.URL.Path]
	if !known && strings.HasPrefix(r.URL.Path, "/v1/models/") {
		allowed, known = http.MethodGet, true
	}
	if known && allowed != r.Method {
		w.Header().Set("Allow", allowed)
		var body errorBody
		body.Error.Message = fmt.Sprintf("The method %s is not allowed on %s; it takes %s",
			r.Method, r.URL.Path, allowed)
		body.Error.Type = "invalid_request_error"
		body.Error.Code = "method_not_allowed"
		writeJSON(w, http.StatusMethodNotAllowed, body)
		return
	}
	var body errorBody
	body.Error.Message = fmt.Sprintf("The endpoint %s %s is not implemented by this server",
		r.Method, r.URL.Path)
	body.Error.Type = "invalid_request_error"
	body.Error.Code = "unknown_endpoint"
	writeJSON(w, http.StatusNotFound, body)
}

// shapeField is one request field that describes the response body rather than tuning the
// sampling, with the reason this server cannot honour it.
type shapeField struct {
	name string
	// asked reports whether the decoded request set this field. It takes the request by
	// pointer through any because the two endpoints have different structs and the field
	// names do not line up exactly (chat has top_logprobs, completions has best_of and
	// suffix), so a shared name list would have to say which endpoint each belongs to.
	asked func(any) bool
	why   string
}

// chatShapeFields are the response-shape fields the chat endpoint refuses.
var chatShapeFields = []shapeField{
	{"n", func(v any) bool { return v.(*chatRequest).N != 0 },
		"n>1 needs several sequences per request, and this server generates one"},
	{"logprobs", func(v any) bool { return v.(*chatRequest).Logprobs != nil },
		"token log-probabilities are not computed or returned"},
	{"top_logprobs", func(v any) bool { return v.(*chatRequest).TopLogprobs != nil },
		"alternative-token log-probabilities are not computed or returned"},
	{"echo", func(v any) bool { return v.(*chatRequest).Echo != nil },
		"the response never includes the prompt, and a silent omission is worse than a refusal"},
}

// completionShapeFields are the response-shape fields the completions endpoint refuses.
var completionShapeFields = []shapeField{
	{"n", func(v any) bool { return v.(*completionRequest).N != 0 },
		"n>1 needs several sequences per request, and this server generates one"},
	{"logprobs", func(v any) bool { return v.(*completionRequest).Logprobs != nil },
		"token log-probabilities are not computed or returned"},
	{"echo", func(v any) bool { return v.(*completionRequest).Echo != nil },
		"the response never includes the prompt, and a silent omission is worse than a refusal"},
	{"suffix", func(v any) bool { return v.(*completionRequest).Suffix != nil },
		"suffix generation is not implemented, so the reply would silently lack it"},
	{"best_of", func(v any) bool { return v.(*completionRequest).BestOf != 0 },
		"best_of needs several candidates per request, and this server samples one"},
}

// rejectShapeFields writes a 400 naming every response-shape field the request asked for,
// and reports whether it wrote one.
//
// These are refused rather than ignored, which is the opposite of how this server treats an
// unknown field like logit_bias, and the difference is what the client does with the answer.
// A sampling knob that is ignored produces text that is not what was asked for but is still
// the shape the client reads. A response-shape field that is ignored produces a body the
// client cannot parse: one that sent n=3 indexes choices[1] and gets an IndexError, one that
// sent logprobs reads choice.logprobs and gets a KeyError. In both cases the failure surfaces
// in the caller's code, far from the request that caused it, and the 400 here is the same
// fact said at the call site.
//
// Every field is named rather than the first: a caller fixing these should not need one
// round trip per field. That is the rule cmd/orbix's bench follows for the flags --suite
// measures over, and the reason is the same -- one request, one complete list of what to fix.
func (s *Server) rejectShapeFields(w http.ResponseWriter, fields []shapeField, req any) bool {
	var names, whys []string
	for _, f := range fields {
		if f.asked(req) {
			names = append(names, f.name)
			whys = append(whys, f.why)
		}
	}
	if len(names) == 0 {
		return false
	}
	writeRequestError(w, http.StatusBadRequest, newParamError(
		names[0], "unsupported_parameter",
		"this server does not implement %s; %s",
		strings.Join(names, ", "), strings.Join(whys, "; ")))
	return true
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := decodeJSON(w, r, &req, s.maxBodyBytes()); err != nil {
		writeDecodeError(w, err)
		return
	}
	if !s.checkModel(w, req.Model) {
		return
	}
	if s.rejectShapeFields(w, chatShapeFields, &req) {
		return
	}
	if len(req.Messages) == 0 {
		writeErrorParam(w, http.StatusBadRequest, "messages is required", "messages", "missing_required_parameter")
		return
	}
	prompt := s.renderChat(req.Messages)
	s.generate(w, r, generateParams{
		prompt:       prompt,
		maxTokens:    req.MaxTokens,
		priority:     req.Priority,
		temperature:  req.Temperature,
		topK:         req.TopK,
		topP:         req.TopP,
		stream:       req.Stream,
		stop:         req.Stop,
		chat:         true,
		seed:         req.Seed,
		includeUsage: req.StreamOptions != nil && req.StreamOptions.IncludeUsage,
	})
}

func (s *Server) handleCompletion(w http.ResponseWriter, r *http.Request) {
	var req completionRequest
	if err := decodeJSON(w, r, &req, s.maxBodyBytes()); err != nil {
		writeDecodeError(w, err)
		return
	}
	if !s.checkModel(w, req.Model) {
		return
	}
	if s.rejectShapeFields(w, completionShapeFields, &req) {
		return
	}
	if strings.TrimSpace(req.Prompt) == "" {
		writeErrorParam(w, http.StatusBadRequest, "prompt is required", "prompt", "missing_required_parameter")
		return
	}
	s.generate(w, r, generateParams{
		prompt:       req.Prompt,
		maxTokens:    req.MaxTokens,
		priority:     req.Priority,
		temperature:  req.Temperature,
		topK:         req.TopK,
		topP:         req.TopP,
		stream:       req.Stream,
		stop:         req.Stop,
		seed:         req.Seed,
		includeUsage: req.StreamOptions != nil && req.StreamOptions.IncludeUsage,
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
	return s.cfg.ChatRenderer.Render(turns)
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
	// includeUsage asks for the extra chunk that carries token usage at the end of a
	// stream; see stream_options. It has no effect on a non-streaming response, which
	// always carries usage.
	includeUsage bool
	// seed makes sampling reproducible. Zero means "choose one", which is what a caller
	// that did not ask for reproducibility gets.
	seed int64
}

func (s *Server) generate(w http.ResponseWriter, r *http.Request, p generateParams) {
	if p.maxTokens < 0 {
		// Refused rather than treated as "unset". The field is a plain int, so zero is the
		// only distinguishable "not sent", and the code below reaches it with "<= 0" -- which
		// meant a negative budget silently became the server's default. A client asking for
		// -5 tokens asked for nothing this server can do, and answering with 256 tokens of
		// text is the one reply it cannot have meant. It is also the rule every command in
		// cmd/orbix follows for the same flag value.
		writeErrorParam(w, http.StatusBadRequest,
			fmt.Sprintf("max_tokens must not be negative, got %d", p.maxTokens),
			"max_tokens", "invalid_value")
		return
	}
	maxTokens := p.maxTokens
	if maxTokens <= 0 {
		maxTokens = s.cfg.MaxTokens
	}

	// Encode with BOS only when the model defines one: a model without a BOS id
	// encodes the prompt as-is.
	cfg := s.eng.Model().Config()

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
		// A coded error, not a bare message: OpenAI's guidance to clients is to branch on
		// `code`, and its own context overflow carries `context_length_exceeded` with a
		// `param` naming the side that overflowed. A client that follows that guidance
		// otherwise sees an uncoded 400 and cannot tell "shorten and retry" from "your
		// request was malformed".
		writeRequestError(w, http.StatusBadRequest, newParamError(
			promptParam(p.chat), "context_length_exceeded",
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
		// The sampler's errors are all about one request field, so they carry it rather
		// than making the client infer which of temperature and top_p was wrong.
		writeRequestError(w, http.StatusBadRequest, err)
		return
	}

	stopIDs := s.stopTokenIDs(p.stop)
	// OpenAI prefixes a completion id with "cmpl-" and a chat completion with "chatcmpl-".
	// The prefix is cosmetic to a parser but visible to anyone reading a log, and the wrong
	// one makes a completion look like a chat response.
	prefix := "cmpl-"
	if p.chat {
		prefix = "chatcmpl-"
	}
	id := fmt.Sprintf("%s%d", prefix, time.Now().UnixNano())

	// The two endpoints differ only in how the prompt was rendered, so the operation name
	// is what tells them apart in a trace.
	op := "text_completion"
	if p.chat {
		op = "chat"
	}
	ch, err := s.eng.Submit(engine.Request{
		Prompt:        promptTokens,
		MaxTokens:     maxTokens,
		Priority:      p.priority,
		Stop:          stopIDs,
		Sampler:       sampler,
		Cancel:        r.Context().Done(),
		Trace:         s.cfg.Trace,
		Operation:     op,
		ResponseID:    id,
		Stream:        p.stream,
		ResponseModel: s.cfg.ModelName,
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

	if p.stream {
		s.streamResponse(w, r, ch, id, p.chat, p.stop, p.includeUsage, len(promptTokens))
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
	stopReason := finishReasonFor(reason)

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
//
// includeUsage asks for the closing usage chunk OpenAI sends when
// stream_options.include_usage is set. Its contract has two halves and both matter: the
// extra chunk has an empty choices array and carries the totals, and every ordinary chunk
// in the same stream carries "usage": null. A client written against the documented shape
// distinguishes the two by choices being empty, and one that reads usage off every chunk
// without a null there would see the field missing rather than absent.
func (s *Server) streamResponse(w http.ResponseWriter, r *http.Request, ch <-chan engine.Event, id string, chat bool, stop []string, includeUsage bool, promptTokens int) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming is not supported by this connection")
		return
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	// nginx buffers proxied responses by default (proxy_buffering on), which holds an SSE
	// stream until the response ends -- the stream still arrives intact, but all at once, and
	// a token stream that arrives at the end is not a token stream. This header is nginx's own
	// per-response switch for that, and nginx will not infer it: its maintainers declined to
	// special-case text/event-stream on the grounds that a media type describes how a payload
	// is formatted, not how it is delivered (nginx/nginx#1505).
	//
	// It costs nothing where it is not understood. A proxy that does not know it ignores it and
	// a direct client has no use for it, so the only reader is nginx, and only to turn
	// buffering off for this one response. The rest of the API keeps the proxy's default,
	// which is what that default is for.
	h.Set("X-Accel-Buffering", "no")
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
		if includeUsage {
			// Set on every ordinary chunk, not only the usage one: OpenAI documents the
			// field as present-and-null here, and a client that inspects it on each chunk
			// then needs no branch for "the field is missing".
			if _, ok := payload["usage"]; !ok {
				payload["usage"] = nil
			}
		}
		b, err := json.Marshal(payload)
		if err != nil {
			// Unreachable with the payloads built here -- maps of strings, ints, nils and
			// nested maps of the same, none of which can fail to encode. Kept because a
			// future field that does fail should end the stream rather than write a chunk
			// with a hole in it, and returning false already means "stop streaming".
			//
			// It shares a return value with the write failure below, and the two mean
			// different things: a write failure means the client is gone, while this one
			// would mean the server built something it could not describe. Nothing above
			// distinguishes them today because nothing can reach this branch.
			return false
		}
		if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
			return false
		}
		// A flush error is not checked, and that is deliberate: the request context is what
		// notices a client that left, and it is already passed to the engine as Request.Cancel.
		// Measured over a raw TCP connection that is reset mid-stream, the engine stops within
		// two tokens of the reset -- the cancel fires first, so a flush check would add a second
		// route to the same place.
		//
		// A test that measures this over http.Post and closes the body sees something else: a
		// graceful close leaves the connection usable, no reset is sent, and generation runs to
		// its full budget. That is a property of the client, not of this code, and it is worth
		// knowing before reading a large number as waste.
		flusher.Flush()
		return true
	}

	dec := s.tok.Seq()
	reason := "stop"
	// generated counts engine tokens for the usage chunk; see the HasToken branch.
	generated := 0
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
			// Counted here rather than from the emitted text: the usage figures describe what
			// the model produced, and a token that decodes to nothing is still a token the
			// engine generated. The non-streaming body counts the same way (len(out)), so the
			// two agree for one request.
			generated++
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
					// The engine keeps producing after this point, and completion_tokens has
					// to include that output: OpenAI defines it as every token the model
					// generated, not the text that reached the caller, and the blocking body
					// counts them all (len(out)). Counting only the events this loop happened
					// to read reported 2 where the same request reported 17 through the other
					// endpoint -- the two disagreeing about one request is the defect.
					//
					// So the terminal event is read for its Generated figure. It is small and
					// comes one round after the stop, so the wait is a round, not the rest of
					// the budget. Nothing is written while draining.
					for ev2 := range ch {
						if ev2.Done || ev2.Err != nil {
							if ev2.Generated > generated {
								generated = ev2.Generated
							}
							break
						}
					}
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
	stopReason := finishReasonFor(reason)
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
	if includeUsage {
		// The finish chunk is built here rather than by send, so it needs the null too.
		// Missing it here is easy to overlook and visible to a client: the docs say every
		// chunk in the stream carries the field, so a client reading usage off each chunk
		// sees one absence in the middle of the nulls.
		final["usage"] = nil
	}
	b, err := json.Marshal(final)
	if err != nil {
		return
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
		return
	}
	// The usage chunk, when asked for, goes after the finish chunk and before [DONE].
	// OpenAI's shape is specific and a client depends on both halves: choices is an empty
	// array (which is how a client tells this chunk apart from a text one) and usage carries
	// the whole request's counts. Sending it with a choice in it, or without the empty array,
	// breaks the check a client uses to find it.
	if includeUsage {
		usage := map[string]any{
			"id": id, "object": object, "created": time.Now().Unix(), "model": s.cfg.ModelName,
			"choices": []map[string]any{},
			"usage": map[string]int{
				"prompt_tokens":     promptTokens,
				"completion_tokens": generated,
				"total_tokens":      promptTokens + generated,
			},
		}
		ub, err := json.Marshal(usage)
		if err != nil {
			return
		}
		if _, err := fmt.Fprintf(w, "data: %s\n\n", ub); err != nil {
			return
		}
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
		return nil, newParamError("temperature", "invalid_value",
			"temperature must not be negative, got %v", t)
	}
	// TopP defaults to the server's own value when the request omits it. A value outside
	// (0, 1] is refused rather than clamped: 0 and 1 both mean "no limit" to some callers
	// and "only the best token" to others, and guessing would change the output silently.
	p := s.cfg.TopP
	if topP != nil {
		p = *topP
	}
	if p < 0 || p > 1 {
		return nil, newParamError("top_p", "invalid_value",
			"top_p must be between 0 and 1, got %v", p)
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

// maxBodyBytes is the largest request body this server will read.
//
// Derived from the context rather than fixed, because the two are the same constraint seen
// from different ends: a request cannot usefully carry more text than the model can attend
// over. bytesPerToken is deliberately generous -- a token's surface text is a few bytes of
// UTF-8, and JSON escapes some of them -- so the limit never rejects a servable request
// while still bounding the work to something proportional to the context.
//
// The floor matters as much as the multiple: a tiny context would otherwise compute a
// limit small enough to reject ordinary conversation, and the point of the bound is to
// stop a huge body, not to make a small model unusable.
func (s *Server) maxBodyBytes() int64 {
	const (
		bytesPerToken = 48
		minLimit      = 1 << 20 // 1 MiB
	)
	limit := int64(s.cfg.ContextLen) * bytesPerToken
	if limit < minLimit {
		return minLimit
	}
	return limit
}

// errUnsupportedEncoding marks a request whose body is compressed in a way this server does not
// decode, so the handler can answer 415 rather than a 400 that blames the JSON.
var errUnsupportedEncoding = errors.New("unsupported content encoding")

func decodeJSON(w http.ResponseWriter, r *http.Request, dst any, limit int64) error {
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
	//
	// The cost of that tolerance is real and is worth stating where the decision lives, because it
	// is not visible from either endpoint: a cross-origin page can make the server generate.
	// POST with Content-Type: text/plain is a "simple request" under the Fetch standard, so the
	// browser sends it with no CORS preflight, and this server parses the body whatever the header
	// says. The page cannot read the reply -- no Access-Control-Allow-Origin is sent -- but the
	// request runs. Measured with --max-active 2: six such requests from a page's origin were
	// accepted in a row and generated 240 tokens between them. The preflighted alternative is
	// already refused: OPTIONS on these routes answers 405, so a cross-origin application/json
	// request never leaves the preflight.
	//
	// Keeping the tolerance is the decision, and the cost is an unauthenticated API that performs
	// work rather than a Content-Type problem. The tolerance is what makes the text/plain path
	// usable at all, so checking the header would trade a documented convenience for less exposure
	// than the word "bypass" suggests: the page still cannot read anything, and the request path
	// it needs is the one already documented for curl -d. What closes the exposure is binding
	// loopback or authenticating in front, which is what the README's "No CORS and no
	// authentication" section tells a deployer to do.
	if enc := r.Header.Get("Content-Encoding"); enc != "" && !strings.EqualFold(enc, "identity") {
		return fmt.Errorf("%w: %s; send the body uncompressed", errUnsupportedEncoding, enc)
	}
	// Bound the read before decoding it. Without this the decoder consumes whatever the
	// client sends: measured against this server, a 64 MiB body was read in full and turned
	// into a context_length_exceeded error that reported the prompt as 524290 tokens, after
	// 82 seconds of CPU. No status code can be sent until the body is parsed, so the work is
	// spent before the request is known to be impossible -- which makes an unbounded body a
	// cheaper attack than a long prompt, since the tokens do not have to be valid.
	//
	// The limit is generous by construction rather than by guess: a request may carry at
	// most ContextLen tokens, and a single token's text is at most a few UTF-8 bytes, so a
	// body above this many bytes cannot describe a servable request. MaxBytesReader also
	// stops reading at the limit, so the cost is bounded even when the client keeps sending.
	//
	// w is passed rather than nil so the reader can mark the connection for closing once
	// the limit is hit. With nil the server answers 413 and then tears the connection down
	// mid-request, which a client sees as a reset rather than as the response it was just
	// sent -- measured: a 64 MiB body produced ConnectionResetError instead of 413.
	r.Body = http.MaxBytesReader(w, r.Body, limit)
	dec := json.NewDecoder(r.Body)
	// Unknown fields are ignored on purpose: see chatRequest.
	if err := dec.Decode(dst); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			return fmt.Errorf("%w: body exceeds %d bytes", errBodyTooLarge, tooBig.Limit)
		}
		return fmt.Errorf("invalid JSON: %w", err)
	}
	return nil
}

// writeDecodeError answers a body this server could not accept.
//
// Three failures with three different meanings, and the status is what a client acts on:
// 400 says the JSON was malformed and retrying the same bytes is pointless, 415 says the
// encoding is unsupported and the client should change that, 413 says the body was too
// large and the request must be made smaller. Reporting all three as 400 would send a
// client with a compressed or oversized body looking for a syntax error.
func writeDecodeError(w http.ResponseWriter, err error) {
	status := http.StatusBadRequest
	switch {
	case errors.Is(err, errUnsupportedEncoding):
		status = http.StatusUnsupportedMediaType
	case errors.Is(err, errBodyTooLarge):
		status = http.StatusRequestEntityTooLarge
	}
	writeError(w, status, err.Error())
}

// errBodyTooLarge marks a request body past the server's size limit, so the handler can
// answer 413 rather than a 400 that blames the JSON.
var errBodyTooLarge = errors.New("request body too large")

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

// promptParam names the field a context-window failure is about.
//
// The two endpoints name their input differently: a chat request carries `messages`, a
// completion carries `prompt`. OpenAI reports `messages` for a chat overflow, so echoing
// the field the caller actually sent is what lets a client highlight the right part of its
// request instead of a field it never used.
func promptParam(chat bool) string {
	if chat {
		return "messages"
	}
	return "prompt"
}

// writeErrorParam is writeError for a request-validation failure that can name the field at
// fault. OpenAI's error object documents `param` for exactly this, and its clients surface it as
// e.param, so a caller writing "--prompt" instead of "prompt" gets the field name rather than
// having to parse it out of the message.
func writeErrorParam(w http.ResponseWriter, status int, msg, param, code string) {
	var body errorBody
	body.Error.Message = msg
	body.Error.Type = "invalid_request_error"
	body.Error.Param = param
	body.Error.Code = code
	writeJSON(w, status, body)
}

// paramError is a request-validation failure that knows which field it is about.
//
// It exists because OpenAI's own guidance is to branch on `code` rather than on message
// text, and a handler that writes its message with writeError hands a client no code at
// all. Two paths did exactly that -- the context-window check and the sampler checks -- so
// a client that follows the guidance could not tell "your prompt is too long" (trim it and
// retry) from "your temperature is negative" (fix the value and retry), or either from a
// malformed body. Carrying param and code on the error keeps the pairing with the message
// that produced them, rather than leaving it to whichever writer the call site happens to
// reach for.
//
// error's own Error() is the message, so a caller that only wants the text can ignore the
// rest, and errors.As recovers the fields.
type paramError struct {
	message string
	param   string
	code    string
}

func (e *paramError) Error() string { return e.message }

// newParamError builds a validation failure naming the field at fault.
func newParamError(param, code, format string, args ...any) error {
	return &paramError{message: fmt.Sprintf(format, args...), param: param, code: code}
}

// writeRequestError writes a validation failure, adding the field and code when the error
// carries them.
//
// This is the writer the request paths should use: it degrades to a plain message for an
// error that is not a paramError, so a call site cannot lose the code by accident, and one
// that does carry fields gets them without a second argument list to keep in step with the
// message.
func writeRequestError(w http.ResponseWriter, status int, err error) {
	var pe *paramError
	if errors.As(err, &pe) {
		writeErrorParam(w, status, pe.message, pe.param, pe.code)
		return
	}
	writeError(w, status, err.Error())
}

func writeError(w http.ResponseWriter, status int, msg string) {
	var body errorBody
	body.Error.Message = msg
	body.Error.Type = "invalid_request_error"
	switch {
	case status == http.StatusServiceUnavailable:
		// OpenAI gives 503 its own type. A client that branches on `type` -- which the
		// official SDKs do, mapping it to a retryable error rather than a failure -- would
		// otherwise read "the server broke" where this engine means "come back shortly".
		body.Error.Type = "service_unavailable_error"
	case status >= 500:
		body.Error.Type = "server_error"
	}
	writeJSON(w, status, body)
}

// finishReasonFor maps an engine finish reason to the value the OpenAI schema carries.
//
// The two vocabularies are not the same size. The engine distinguishes stop, length, cancel,
// abandoned and error, because a caller reading /metrics needs to tell a client that walked away
// from one that ran out of budget. The response schema has three values, and only two of them are
// meaningful here:
//
//   - "stop" is the model emitting its end token, or the request being cut by a stop string.
//   - "length" is the token budget or the context window running out.
//   - "tool_calls" does not apply: this server does not parse tool calls.
//
// cancel maps to "length" rather than "stop" because the generation did not complete on its own:
// the sequence was cut short by the client leaving. The schema has no value for that, and the
// client that caused it has already gone, so the response is read by a log or a proxy.
//
// abandoned follows the same reasoning but keeps "stop", which is the mapping that was in place
// before this function existed. Changing it is a behavioural change rather than a consolidation and
// is not made here.
//
// error maps to "stop" because the schema has no failure value; the stream carries an error event
// separately for the case that matters, and a non-streaming error is a 500 rather than a completion.
//
// One function rather than two copies, which is what the duplicated expression had become: the
// streaming and non-streaming paths must agree, and a mapping written twice is a mapping that can
// disagree.
func finishReasonFor(reason string) string {
	if reason == "length" || reason == "cancel" {
		return "length"
	}
	return "stop"
}
