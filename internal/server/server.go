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
	"io"
	"log/slog"
	"math"
	"net/http"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/pplmx/orbix/internal/engine"
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

	// ChatFallbackAnnounced says the caller has already told the operator that the renderer
	// falls back -- typically a startup banner that probed the template. It suppresses the
	// request-time warning for a fallback that was known before the first request, so the
	// same fact is not reported twice. A fallback the caller could not have seen still
	// warns, which is why this is a flag the caller sets rather than a condition the server
	// infers: only the caller knows what it printed.
	ChatFallbackAnnounced bool

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
	// pumpOnce makes StartPump idempotent, the way stopOnce makes Shutdown idempotent.
	// A second StartPump would put two goroutines inside Pump, which is the one thing the
	// engine's contract forbids: measured, it is a data race on the sequence fields rather
	// than a panic, so the symptom appears in the engine while the cause is here.
	pumpOnce sync.Once
	wg       sync.WaitGroup

	// idSeq makes response ids unique. The timestamp alone is not enough: concurrent callers
	// can read the clock in the same nanosecond -- measured on this machine, 32 goroutines
	// over 200 trials collided 12 times, while 8 never did -- and the collision is silent
	// when it happens. A counter costs one atomic add and removes the question.
	idSeq atomic.Uint64

	// log is cfg.Logger, kept as the concrete field so the request wrapper can check
	// for nil cheaply.
	log *slog.Logger

	// chatFallbackOnce makes the render-fallback warning appear once rather than per request.
	// A template that compiles but cannot render every conversation falls back on every one of
	// them, so an unguarded warning would fill the log with the same line.
	chatFallbackOnce sync.Once

	// refused counts requests turned away for lack of capacity, by status. It lives here
	// rather than in the engine because that is where the decision is made: Submit returns
	// ErrQueueFull and the pool error arrives on an event, and neither reaches the engine's
	// own counters -- by design, since a request that never entered the queue must not make
	// the admission rate look better than it is.
	//
	// The gap that left: those requests were invisible everywhere. Measured with
	// --max-active 1 --max-queued 1 and 24 requests in bursts of 8, 16 were refused with
	// 503 and orbix_requests_submitted_total still read 8, so a deployment shedding most of
	// its load reported a healthy admission rate and a queue gauge pinned at 0. Refusal is
	// the capacity signal an operator acts on, and it needs its own number.
	refusedQueueFull atomic.Uint64
	refusedPool      atomic.Uint64
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
	// mux's own 405. That 405 is load-bearing -- docs/usage.md's CORS note rests on OPTIONS
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
// StartPump starts the goroutine that advances the engine. Calling it twice is a no-op
// rather than a second pump: the engine may have only one goroutine inside Pump, and two
// would race on the sequence fields it writes there.
//
// The once also means StartPump after Shutdown does nothing. That is not a restart API and
// cannot become one without recreating the stop channel: the pump exits when stop closes,
// so a goroutine started afterwards would return on its first iteration. A server is
// started once and shut down once; rebuilding one means a new Server.
//
// ctx must not be nil. The pump selects on ctx.Done(), and a nil interface has no such
// method, so the goroutine panics on its first iteration -- which is worse than it sounds,
// because a panic in a goroutine the caller never sees cannot be recovered: the process
// exits. The context package says not to pass nil, so this is a contract note rather than a
// check; rejecting nil here would imply the function could otherwise do something sensible
// with it, and there is nothing to do. The one production caller passes the signal context.
func (s *Server) StartPump(ctx context.Context) {
	s.pumpOnce.Do(func() { s.wg.Go(func() { s.run(ctx) }) })
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
//   - A field that only tunes sampling leaves the response a client can still read: it is the
//     wrong text, but choices[0].text is where it expected it. Ignoring it costs quality and keeps
//     working. `presence_penalty` and `frequency_penalty` were in this list and are now honoured:
//     they still do not change the response's shape, but a caller who sets one is asking for
//     different text rather than expressing a tolerance, and the two fields measured identical
//     output with and without them.
//
// `response_format` is the one field that moved the other way. It is not a sampling preference and
// not a shape the server can produce either: json_object promises valid JSON, which needs
// constrained decoding this engine does not have, and ignoring it returned a markdown fence where a
// client expected parseable JSON. It is refused by name now, with the forms that ask for what the
// server already does -- absent, or "text" -- still served.
//
// `logit_bias` used to be in that second list and does not belong there. OpenAI documents -100
// and 100 as "a ban or exclusive selection", so a client that bans a token is relying on it not
// appearing rather than expressing a preference -- and a silent ban is the closed-mouth failure
// this classification exists to avoid. It is honoured on both endpoints, and applied by wrapping
// the sampler so that temperature 0, the mode a caller bans in, sees it too.
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
	MaxTokens *int          `json:"max_tokens"`
	// LogitBias shifts named tokens' logits before sampling, which is a ban or an exclusive
	// choice at the extremes OpenAI documents (-100 and 100). It is honoured rather than
	// ignored for that reason: a client that bans a token is relying on it not appearing, and
	// silence would break that reliance without saying so.
	LogitBias map[string]float64 `json:"logit_bias"`
	// PresencePenalty and FrequencyPenalty discourage repetition over the tokens already seen.
	// Honoured for the same reason: they change what the text says, and ignoring them produces
	// a plausible answer to a request the caller did not make.
	PresencePenalty  *float64 `json:"presence_penalty"`
	FrequencyPenalty *float64 `json:"frequency_penalty"`
	// ResponseFormat is checked rather than ignored: json_object promises valid JSON, and this
	// server cannot deliver that without constrained decoding. See checkResponseFormat.
	ResponseFormat json.RawMessage `json:"response_format"`

	// MaxCompletionTokens is OpenAI's newer spelling of MaxTokens, and it is honoured rather
	// than refused: max_tokens is deprecated in favour of it, so a migrated client sends only
	// the newer name. A pointer so an omitted field is distinguishable from a zero, which is
	// what lets generate tell "not sent" from "sent as 0".
	//
	// Sending both is refused, because the two are alternatives rather than a pair: OpenAI
	// documents max_completion_tokens as the replacement, so a request carrying both is a
	// caller mid-migration whose two numbers may disagree, and picking one silently would
	// answer with a budget the caller might not have meant.
	MaxCompletionTokens *int `json:"max_completion_tokens"`

	// N, Logprobs, TopLogprobs and Echo all change the response body, and all four are
	// implemented now: they left the shape-field list when they started working. The comment
	// that stood here said they were refused, which had stopped being true and would have sent a
	// reader looking for a refusal that no longer happens -- n: 2 returns two choices, measured.
	//
	// Two refusals remain, and neither is about the shape-field rule. Chat still refuses echo
	// because its response never includes the prompt. Completions refuses suffix only when the
	// checkpoint declares no fill-in-the-middle markers, which is a statement about the model
	// rather than about this server; see the handler. docs/usage.md has the reader-facing form.
	N int `json:"n"`
	// Function calling. Tools and ToolChoice are honoured -- the signatures are rendered into
	// the prompt by the model's own template and the reply is parsed for the calls it asked for
	// -- while the legacy functions/function_call pair is still refused, because it is a
	// different schema (a flat name+parameters list, no tool_choice modes) and translating it
	// would be this server inventing a mapping the caller did not ask for. See tools.go.
	Tools        json.RawMessage `json:"tools"`
	ToolChoice   json.RawMessage `json:"tool_choice"`
	Functions    json.RawMessage `json:"functions"`
	FunctionCall json.RawMessage `json:"function_call"`
	Logprobs     *bool           `json:"logprobs"`
	TopLogprobs  *int            `json:"top_logprobs"`
	Echo         *bool           `json:"echo"`

	// ChatTemplateKwargs sets variables in the model's own chat template, which is where a
	// checkpoint's switches live: Qwen3's template branches on enable_thinking, and without a
	// way to set it every request gets the default mode.
	//
	// The name is the one vLLM introduced and the OpenAI-compatible ecosystem uses. It is
	// refused rather than ignored when this model cannot act on it, which is the opposite of
	// how an unknown member is treated: an unknown member asks for nothing the server claims
	// to provide, while a variable this model's template does not read is a request whose
	// effect the caller will otherwise look for in the reply and not find.
	//
	// Decoded as a map of JSON values. Anything other than an object is refused by
	// decodeJSON naming this field.
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs"`

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

// chatMessage is one turn of a chat request.
//
// Content is a string or an array of content parts, which are OpenAI's two spellings for
// the same field and both appear in real clients: the Python SDK sends the array form
// whenever a message is built from parts, and a plain string otherwise. Only the text is
// turned into a prompt, so a text part contributes its text and every other part type is
// refused -- a request carrying an image would otherwise be answered from its text alone,
// which is a reply to a question the caller did not ask.
type chatMessage struct {
	Role    string      `json:"role"`
	Content chatContent `json:"content"`
	// ToolCalls is what an assistant turn asked to call, in the wire shape the response also
	// writes: id, type and a function carrying name and arguments-as-JSON-string. It is decoded
	// rather than kept raw because the conversation it belongs to has to be rendered back into
	// the prompt, and a marshalled blob would reach the template as text rather than as calls.
	//
	// A malformed entry is refused by name in resolveChatTools, not by the decoder: the
	// decoder's message would name this Go type rather than messages[i].tool_calls.
	ToolCalls json.RawMessage `json:"tool_calls"`
	// ToolCallID names the call a tool result answers. It is checked against the ids the
	// conversation actually declared: a result for a call the model cannot see is a turn it
	// cannot place.
	ToolCallID string `json:"tool_call_id"`
	Name       string `json:"name"`
	// parsedToolCalls is ToolCalls after validation, filled by resolveChatTools. The raw field
	// is kept because the error messages name it and because re-marshalling would be a second
	// spelling of the same data.
	parsedToolCalls []engine.ToolCall
}

// chatContent is a message body: a JSON string, or an array of {"type","text"} parts.
//
// It decodes both into the text a prompt can be rendered from, and it answers the failures
// itself, in the envelope's terms, because the decoder's own message would name this Go
// type rather than the field the caller sent.
type chatContent string

// chatContentParam is the JSON path every refusal about this field names. The message index
// is deliberately not part of it: the type is decoded without knowing which message it
// belongs to, and a path with a made-up index is worse than one the whole field shares.
const chatContentParam = "messages.content"

// chatContentPart is one entry of the array form. Both members are strings, so a part that
// arrives with the wrong types is reported as such rather than silently becoming "".
type chatContentPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// UnmarshalJSON accepts either spelling and keeps the text.
//
// null is an empty body rather than an error: an assistant turn in a transcript a client is
// replaying can carry it, and an empty string is what this server would render for it
// anyway.
//
// Every value here arrives already parsed by the outer decoder, so the bytes are valid JSON
// and the only failure either branch can actually see is a type error. The bare `return err`
// arms are the honest fallback for a decoder that grew another failure mode, not a case this
// server can reach; they are left uncovered rather than forced.
func (c *chatContent) UnmarshalJSON(b []byte) error {
	if len(b) == 0 || string(b) == "null" {
		return nil
	}
	if b[0] != '[' {
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			var te *json.UnmarshalTypeError
			if errors.As(err, &te) {
				return unmarshalTypeFailure(chatContentParam,
					"a string or an array of text parts", te)
			}
			return err
		}
		*c = chatContent(s)
		return nil
	}
	var parts []chatContentPart
	if err := json.Unmarshal(b, &parts); err != nil {
		// The entries are typed, so the decoder knows which member went wrong but not how
		// to say it without naming a Go struct. want is the shape the caller has to write.
		var te *json.UnmarshalTypeError
		if errors.As(err, &te) {
			return unmarshalTypeFailure(chatContentParam,
				"an array of objects with a string \"type\" and a string \"text\"", te)
		}
		return newParamError(chatContentParam, "invalid_value",
			"%s must be a string or an array of text parts: %s", chatContentParam, err)
	}
	var text strings.Builder
	for i, p := range parts {
		switch p.Type {
		case "text":
			text.WriteString(p.Text)
		case "":
			// A part with no type is not a text part this server can recognise, and
			// reading its "text" anyway would be guessing at the caller's intent.
			return newParamError(chatContentParam, "missing_required_parameter",
				"content part %d of %s has no \"type\"; this server renders parts of "+
					"type \"text\"", i, chatContentParam)
		default:
			return newParamError(chatContentParam, "unsupported_parameter",
				"this server renders text content only: content part %d of %s has type "+
					"%q, and dropping it would answer a question the caller did not ask",
				i, chatContentParam, p.Type)
		}
	}
	*c = chatContent(text.String())
	return nil
}

type completionRequest struct {
	Model     string `json:"model"`
	Prompt    string `json:"prompt"`
	MaxTokens *int   `json:"max_tokens"`
	// LogitBias is honoured: it is a constraint rather than a preference. See the note on
	// chatRequest for why that distinction decides between honouring and ignoring a field.
	LogitBias map[string]float64 `json:"logit_bias"`
	// PresencePenalty and FrequencyPenalty are honoured for the same reason logit_bias is: a
	// caller who sets them is asking for different text, and the request is the only place the
	// setting can come from. See the class comment.
	PresencePenalty  *float64 `json:"presence_penalty"`
	FrequencyPenalty *float64 `json:"frequency_penalty"`
	// ResponseFormat is checked rather than ignored: json_object promises valid JSON, and this
	// server cannot deliver that without constrained decoding. See checkResponseFormat.
	ResponseFormat json.RawMessage `json:"response_format"`

	// MaxCompletionTokens is honoured here even though OpenAI defines it only for chat: the
	// official SDK's completions parameters do not include it, so no SDK client can send it,
	// and a raw request carrying it is reaching for a limit. Honouring it is a deliberate
	// superset. The alternative -- treating it as an unknown member and ignoring it, which is
	// what this server does with unknown fields -- would generate the default length for a
	// caller who asked for less, and nothing in the response would say so. Ignoring a limit is
	// the one silent omission this file keeps refusing to make. Both at once is refused; see
	// chatRequest.MaxCompletionTokens.
	MaxCompletionTokens *int `json:"max_completion_tokens"`

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

// UnmarshalJSON decodes either spelling, and answers a wrong type itself.
//
// The errors json.Unmarshal returns here carry no field path -- a custom unmarshaller is
// handed the value alone, so the decoder has no struct field to name -- and the generic
// answer for that is "the request body must be a JSON object", which is wrong for a body that
// is an object and has one bad member. Measured before this: {"prompt":"x","stop":5} answered
// with that sentence instead of naming stop.
//
// As in chatContent's decoder, the bytes here are already valid JSON, so the type error is
// the only failure these arms can see and the bare `return err` arms are a fallback.
func (f *stopField) UnmarshalJSON(b []byte) error {
	if len(b) == 0 || string(b) == "null" {
		return nil
	}
	if b[0] == '[' {
		var list []string
		if err := json.Unmarshal(b, &list); err != nil {
			var te *json.UnmarshalTypeError
			if errors.As(err, &te) {
				return unmarshalTypeFailure("stop", "a string or an array of strings", te)
			}
			return err
		}
		*f = list
		return nil
	}
	var one string
	if err := json.Unmarshal(b, &one); err != nil {
		var te *json.UnmarshalTypeError
		if errors.As(err, &te) {
			return unmarshalTypeFailure("stop", "a string or an array of strings", te)
		}
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
//     only the 405 is true, and docs/usage.md's account of the CORS gap depends on it.
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
	// asked reports whether the request asks for a response shape this server does not
	// produce. It is deliberately not "was the field sent": a field that was sent with the
	// value the response already carries is a request this server can serve.
	//
	// Sending such a value is not hypothetical. The official SDKs serialise a keyword
	// argument the caller passed even when it equals the API's default, so
	// completions.create(best_of=1) puts "best_of": 1 on the wire. Refusing that told a
	// caller "best_of needs several candidates" about best_of=1, and the request was one
	// this server answers correctly.
	//
	// It takes the request by pointer through any because the two endpoints have different
	// structs and the field names do not line up exactly (chat has top_logprobs,
	// completions has best_of and suffix), so a shared name list would have to say which
	// endpoint each belongs to.
	asked func(any) bool
	// why is what a refusal says instead of the value. It has to be true of every value
	// asked accepts, which is why the ranges are narrowed in asked rather than here.
	why string
	// invalid reports a value the API does not define at all, and the message to answer
	// with. It runs before asked so a malformed count is reported as the malformed number it
	// is rather than as an unimplemented feature. best_of carries both: -1 is a number the
	// API does not define, while 2 is a ranking this server cannot compute, and answering -1
	// with the reason for 2 sends the caller to change the wrong thing.
	invalid func(any) (string, bool)
}

// chatShapeFields are the response-shape fields the chat endpoint refuses.
var chatShapeFields = []shapeField{
	// n is validated here rather than refused: the server generates several choices now, so
	// the only thing left to reject is a count outside the range it will honour. It stays in
	// the table so that bound is checked before any refusal below it, which is the order a
	// client relies on when a request has more than one problem.
	{name: "n", invalid: func(v any) (string, bool) { return checkChoices(v.(*chatRequest).N) }},
	// logprobs and top_logprobs are not here: they are computed now. Their ranges are
	// checked by checkLogprobs, which runs beside the other invalid checks so a bad count is
	// still reported before any refusal below it.
	{name: "top_logprobs", invalid: func(v any) (string, bool) {
		return checkLogprobs(v.(*chatRequest).TopLogprobs, maxChatTopLogprobs, "top_logprobs")
	}},
	{name: "echo",
		asked: func(v any) bool {
			p := v.(*chatRequest).Echo
			return p != nil && *p
		},
		why: "the response never includes the prompt, and a silent omission is worse than a refusal"},
}

// completionShapeFields are the response-shape fields the completions endpoint refuses.
var completionShapeFields = []shapeField{
	{name: "n", invalid: func(v any) (string, bool) { return checkChoices(v.(*completionRequest).N) }},
	{name: "logprobs", invalid: func(v any) (string, bool) {
		return checkLogprobs(v.(*completionRequest).Logprobs, maxCompletionLogprobs, "logprobs")
	}},
	// best_of is honoured now. Its range, including the rule that it must exceed n, is checked
	// here because this is the one place both numbers are available; see checkBestOf.
	{name: "best_of", invalid: func(v any) (string, bool) {
		r := v.(*completionRequest)
		return checkBestOf(r.BestOf, r.N)
	}},
}

// The two endpoints cap log probabilities differently in OpenAI's schema: chat documents
// top_logprobs at a maximum of 20, completions documents logprobs at 5. Both are enforced
// rather than clamped, because a caller that asked for more alternatives and got fewer has
// been silently given a different answer -- the count is in the response's own arrays, so the
// discrepancy is visible but unexplained.
const (
	maxChatTopLogprobs    = 20
	maxCompletionLogprobs = 5
)

// checkLogprobs validates an optional count of alternatives.
//
// The field is a pointer because both endpoints distinguish "not sent" from a zero, and a
// zero is a real request: it asks for the chosen token's log probability and no alternatives.
func checkLogprobs(count *int, max int, name string) (string, bool) {
	if count == nil {
		return "", false
	}
	if *count < 0 {
		return fmt.Sprintf("%s must not be negative, got %d", name, *count), true
	}
	if *count > max {
		return fmt.Sprintf("%s must be at most %d, got %d", name, max, *count), true
	}
	return "", false
}

// maxChoices is the largest n this server honours.
//
// Every choice is a separate engine sequence with its own KV cache, and they are all in
// flight at once, so n multiplies what one request costs. The engine's default queue holds
// 8*MaxActive entries -- 64 by default -- and a cap of 8 keeps a single request from taking a
// meaningful share of it at any default setting. OpenAI's own ceiling is 128, which this
// engine could only reach with a pool sized for it, so the bound is reported rather than
// guessed at: a caller that needs more can send the request several times.
const maxChoices = 8

// checkChoices validates n, which used to be refused outright and is now generated.
//
// A negative count and a count past maxChoices are both malformed values rather than
// unimplemented features, so both are reported with invalid_value and the number that arrived.
func checkChoices(n int) (string, bool) {
	if n < 0 {
		return fmt.Sprintf("n must not be negative, got %d", n), true
	}
	if n > maxChoices {
		return fmt.Sprintf("n must be at most %d, got %d", maxChoices, n), true
	}
	return "", false
}

// derefInt reads an optional count as a number, treating absent as zero.
func derefInt(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}

// rejectShapeFields writes a 400 for a response-shape request field this server cannot
// honour, and reports whether it wrote one.
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
// The line runs through the value, not the field name. A field sent with the value the
// response already carries asks for nothing, so it is served: measured through the OpenAI
// Python SDK, n=1, best_of=1, echo=false and logprobs=false were all refused with "this
// server does not implement ..." before, and each is a keyword the SDK passes through
// verbatim when the caller sets it.
//
// n is not refused at all any more -- the server generates the choices it asks for -- so its
// table entry carries only an invalid check, which is why asked may be nil here. An entry
// with no asked participates in validation and never in refusal.
//
// Every field is named rather than the first: a caller fixing these should not need one
// round trip per field. That is the rule cmd/orbix's bench follows for the flags --suite
// measures over, and the reason is the same -- one request, one complete list of what to fix.
func (s *Server) rejectShapeFields(w http.ResponseWriter, fields []shapeField, req any) bool {
	// A malformed value is reported before an unimplemented one. The two ask different
	// things of the caller: a negative count has to be changed whatever the server supports,
	// while an unimplemented field has to be dropped.
	for _, f := range fields {
		if f.invalid == nil {
			continue
		}
		if msg, bad := f.invalid(req); bad {
			writeRequestError(w, http.StatusBadRequest,
				newParamError(f.name, "invalid_value", "%s", msg))
			return true
		}
	}
	var names, whys []string
	for _, f := range fields {
		// A nil asked means the field is validated but no longer refused; see the comment
		// above about n.
		if f.asked != nil && f.asked(req) {
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
	if s.legacyToolFields(w, &req) {
		return
	}
	if len(req.Messages) == 0 {
		writeErrorParam(w, http.StatusBadRequest, "messages is required", "messages", "missing_required_parameter")
		return
	}
	// Function calling is resolved before rendering, because the two halves depend on each
	// other: the tool list goes into the prompt by way of the model's own template, and the
	// calls in a conversation have to be in the turns the template renders. A model with no
	// usable template cannot take tools at all, which checkTools and the renderer report.
	declared, tools, err := checkTools(req.Tools)
	if err != nil {
		writeRequestError(w, http.StatusBadRequest, err)
		return
	}
	choice, err := checkToolChoice(req.ToolChoice, declared)
	if err != nil {
		writeRequestError(w, http.StatusBadRequest, err)
		return
	}
	if err := resolveToolCallMessages(req.Messages); err != nil {
		writeRequestError(w, http.StatusBadRequest, err)
		return
	}
	turns, err := checkToolMessages(req.Messages)
	if err != nil {
		writeRequestError(w, http.StatusBadRequest, err)
		return
	}
	// A tool_choice that cannot take effect is refused rather than dropped, for the reason the
	// rest of this file refuses fields: `tool_choice: "required"` promises a call, and a template
	// here has no way to print that constraint, so honouring it would mean answering with
	// whatever the model chose while the caller believes the choice was forced. "auto" is served
	// because it asks for what the server does anyway; "none" is served by rendering the
	// conversation with no tools, which is the one direction that needs no prompt rewrite.
	if choice.Mode == modeRequired || choice.Mode == modeNamed {
		writeRequestError(w, http.StatusBadRequest, newParamError(
			"tool_choice", "unsupported_parameter",
			"tool_choice %q cannot be honoured: no chat template in this project's subset has a "+
				"field that forces a call, so the constraint would reach no prompt and the reply "+
				"would be whatever the model chose; send \"auto\" (the default), or \"none\" to "+
				"withhold the tools", choice.Mode))
		return
	}
	if choice.Mode == modeNone {
		// The tools are still validated above -- a malformed list is an error whatever the
		// choice says -- but they do not reach the prompt, which is what "none" means.
		tools = nil
	}
	vars, err := chatTemplateVars(req.ChatTemplateKwargs)
	if err != nil {
		writeRequestError(w, http.StatusBadRequest, err)
		return
	}
	prompt, err := s.renderChatTurns(turns, tools, vars)
	if err != nil {
		// The model cannot honour what the caller sent, so answering would produce a prompt that
		// ignored it. Naming the field and the reason is what lets a client fix the request
		// rather than wonder why the reply did not change.
		//
		// The param is "chat_template_kwargs" because that is the only field whose value the
		// renderer rejects on its own; a tools refusal is what this server decides before
		// rendering, so it never arrives here.
		writeRequestError(w, http.StatusBadRequest, &paramError{
			message: err.Error(), param: "chat_template_kwargs", code: "unsupported_parameter"})
		return
	}
	if err := checkResponseFormat(req.ResponseFormat); err != nil {
		writeRequestError(w, http.StatusBadRequest, err)
		return
	}
	bias, err := logitBias(req.LogitBias, s.eng.Model().Config().VocabSize)
	if err != nil {
		writeRequestError(w, http.StatusBadRequest, err)
		return
	}
	presence, err := checkPenalty(req.PresencePenalty, "presence_penalty")
	if err != nil {
		writeRequestError(w, http.StatusBadRequest, err)
		return
	}
	frequency, err := checkPenalty(req.FrequencyPenalty, "frequency_penalty")
	if err != nil {
		writeRequestError(w, http.StatusBadRequest, err)
		return
	}
	s.generate(w, r, generateParams{
		prompt:              prompt,
		logitBias:           bias,
		presencePenalty:     presence,
		frequencyPenalty:    frequency,
		maxTokens:           derefInt(req.MaxTokens),
		maxTokensGiven:      req.MaxTokens != nil,
		priority:            req.Priority,
		temperature:         req.Temperature,
		topK:                req.TopK,
		topP:                req.TopP,
		stream:              req.Stream,
		stop:                req.Stop,
		chat:                true,
		seed:                req.Seed,
		choices:             req.N,
		maxCompletionTokens: req.MaxCompletionTokens,
		logprobs:            req.Logprobs != nil && *req.Logprobs,
		topLogprobs:         derefInt(req.TopLogprobs),
		includeUsage:        req.StreamOptions != nil && req.StreamOptions.IncludeUsage,
		tools:               tools,
	})
}

// jsonAsks reports whether a raw JSON value asks for something, given that "no" is spelled four
// ways in practice: an absent field, an explicit null, an empty array, and the string "none"
// that tool_choice and function_call use.
//
// It is a string test rather than a decode because the fields carry structures this server does
// not model, and the only question here is whether the caller asked for them.
func jsonAsks(raw json.RawMessage) bool {
	switch strings.TrimSpace(string(raw)) {
	case "", "null", "[]", `"none"`:
		return false
	}
	return true
}

// legacyToolFields refuses the deprecated functions/function_call pair.
//
// Function calling itself is implemented now (see tools.go), so this covers only the older schema
// OpenAI replaced. Translating it would mean this server inventing a mapping -- a flat
// {name, parameters} list has no descriptions, no tool_choice modes and no call ids, so a
// conversion would drop fields the caller sent while looking like it worked. The message names
// the replacement, which is what a caller sending the old spelling needs.
func (s *Server) legacyToolFields(w http.ResponseWriter, req *chatRequest) bool {
	for _, f := range []struct {
		name string
		raw  json.RawMessage
	}{
		{"functions", req.Functions},
		{"function_call", req.FunctionCall},
	} {
		if !jsonAsks(f.raw) {
			continue
		}
		writeRequestError(w, http.StatusBadRequest, newParamError(f.name, "unsupported_parameter",
			"%s is the deprecated spelling of OpenAI's function calling; send \"tools\" and "+
				"\"tool_choice\" instead -- this server renders the tool signatures into the "+
				"prompt and returns the calls it parses back as tool_calls", f.name))
		return true
	}
	return false
}

// handleCompletion and handleChat check a request in the same order, and the order is the
// contract: decode, model, response-shape fields, required fields, then the values inside
// generate. A client that sends several problems at once sees the first one in that list, so
// changing the order changes what a client is told to fix.
//
// The reasoning, cheapest-and-most-fundamental first: a body the decoder cannot parse has no
// fields to look at; a model this server does not serve makes every later answer a reply from
// the wrong model, so 404 precedes any 400 about the request; a shape field the server cannot
// honour means the response would be the wrong shape whatever the prompt says; a missing
// prompt means there is nothing to generate from. Value ranges come last because they can only
// be judged against a request that is otherwise well-formed.
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
	// suffix turns the request from a continuation into an insertion: the prompt is the text
	// before the insertion point and suffix is the text after it. That needs a checkpoint
	// trained for it, recognised by its marker names, so the refusal here is about the model
	// rather than about this server -- the same request works against a code-tuned checkpoint.
	var fim *tokenize.FIM
	if req.Suffix != nil && *req.Suffix != "" {
		markers, ok := s.tok.FIMTokens()
		if !ok {
			writeRequestError(w, http.StatusBadRequest, newParamError(
				"suffix", "unsupported_parameter",
				"suffix needs a fill-in-the-middle checkpoint, and this model declares none of "+
					"the marker sets this server knows (%s); generating anyway would answer from the "+
					"prefix alone and silently ignore the suffix. Send the request without suffix, "+
					"or serve a code-tuned checkpoint",
				fimMarkerHint()))
			return
		}
		// echo returns the prompt, and with an insertion the "prompt" the caller wrote is only
		// the part before the insertion point. Echoing that would return half of what they
		// sent under a name that promises the prompt, so the pair is refused rather than
		// guessed at.
		if req.Echo != nil && *req.Echo {
			writeRequestError(w, http.StatusBadRequest, newParamError(
				"suffix", "unsupported_parameter",
				"suffix and echo cannot be combined: with an insertion the prompt is only the "+
					"text before the insertion point, so echo would return part of what was sent"))
			return
		}
		fim = &markers
	}
	if err := checkResponseFormat(req.ResponseFormat); err != nil {
		writeRequestError(w, http.StatusBadRequest, err)
		return
	}
	bias, err := logitBias(req.LogitBias, s.eng.Model().Config().VocabSize)
	if err != nil {
		writeRequestError(w, http.StatusBadRequest, err)
		return
	}
	presence, err := checkPenalty(req.PresencePenalty, "presence_penalty")
	if err != nil {
		writeRequestError(w, http.StatusBadRequest, err)
		return
	}
	frequency, err := checkPenalty(req.FrequencyPenalty, "frequency_penalty")
	if err != nil {
		writeRequestError(w, http.StatusBadRequest, err)
		return
	}
	s.generate(w, r, generateParams{
		prompt:              req.Prompt,
		logitBias:           bias,
		presencePenalty:     presence,
		frequencyPenalty:    frequency,
		maxTokens:           derefInt(req.MaxTokens),
		maxTokensGiven:      req.MaxTokens != nil,
		priority:            req.Priority,
		temperature:         req.Temperature,
		topK:                req.TopK,
		topP:                req.TopP,
		stream:              req.Stream,
		stop:                req.Stop,
		seed:                req.Seed,
		choices:             req.N,
		maxCompletionTokens: req.MaxCompletionTokens,
		echo:                req.Echo != nil && *req.Echo,
		suffix:              suffixOf(req.Suffix),
		fim:                 fim,
		// On this endpoint logprobs is a count rather than a switch: its presence turns the
		// feature on and its value is how many alternatives to describe.
		logprobs:     req.Logprobs != nil,
		topLogprobs:  derefInt(req.Logprobs),
		bestOf:       req.BestOf,
		includeUsage: req.StreamOptions != nil && req.StreamOptions.IncludeUsage,
	})
}

// renderChat renders a conversation with no tools declared, which is the shape the tests and the
// REPL share. It is a wrapper so there is one rendering path: everything below goes through the
// same ChatRenderer call the tool-carrying endpoint uses.
func (s *Server) renderChat(msgs []chatMessage, vars map[string]any) (string, error) {
	turns := make([]engine.Turn, len(msgs))
	for i, m := range msgs {
		turns[i] = engine.Turn{Role: m.Role, Text: string(m.Content)}
	}
	return s.renderChatTurns(turns, nil, vars)
}

// renderChatTurns applies the chat template to a conversation that may declare tools.
//
// The rendering itself lives in engine.RenderChatPrompt so the HTTP server and the REPL
// cannot drift apart: the same conversation must produce the same tokens however it was
// asked. See that function for why the template is neutral rather than a model's own.
//
// A model with no usable template cannot take tools, and that comes back as a *paramError naming
// "tools" rather than as a template-variable failure: the caller's mistake is declaring tools
// against a model that cannot read them, not setting a variable wrong.
func (s *Server) renderChatTurns(turns []engine.Turn, tools []engine.ToolSpec, vars map[string]any) (string, error) {
	// RenderWith reports a variable the model cannot honour and falls back for a template that
	// fails at render time. The two are different: a render failure still produces a usable
	// prompt, while a variable this model does not read would produce one that silently ignored
	// what the caller asked for. vars is nil for every request that sends none, which is the
	// ordinary case.
	prompt, err := s.cfg.ChatRenderer.RenderConversation(turns, tools, vars)
	// The fallback is silent otherwise, and the startup banner cannot catch this case: it
	// reads UsesModelTemplate before any request has rendered, and renderReason is only
	// written when a render fails. Measured with a template of "{{ messages[99].content }}":
	// the banner said "chat: model template", every request answered 200 from the neutral
	// transcript, and nothing in the log said so.
	//
	// Warned once because such a template fails on every conversation, and a line per request
	// would bury every other message.
	//
	// ChatFallbackAnnounced is the guard, not "a fallback reason is set": a caller whose
	// banner already named the fallback has told the operator once, and repeating it per
	// process would be noise. What still warns is a fallback the caller could not have
	// announced -- a template that renders the startup probe and then fails on a real
	// conversation, which only the server sees.
	if s.cfg.ChatRenderer != nil && !s.cfg.ChatFallbackAnnounced {
		s.chatFallbackOnce.Do(func() {
			if reason := s.cfg.ChatRenderer.FallbackReason(); reason != "" {
				s.Logger().Warn("the model's chat template is not in use; prompts are using "+
					"the neutral transcript instead", slog.String("reason", reason))
			}
		})
	}
	return prompt, err
}

// chatTemplateVars normalises the request's chat_template_kwargs into the map the renderer
// takes, refusing the forms that cannot mean anything.
//
// The two refusals here are about shape, not about the model: a null value, or an empty object,
// says nothing to set. Whether the model's template reads a name is decided by the renderer,
// which is the only place that knows -- see engine.ChatRenderer.RenderWith.
//
// Nothing is silently dropped. That is the difference between this field and an unknown member:
// a client that sends {"enable_thinking": false} expects the prompt to change, and a server that
// accepted it and rendered the other mode would answer a different request than the one made.
func chatTemplateVars(raw map[string]any) (map[string]any, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	out := make(map[string]any, len(raw))
	for name, v := range raw {
		if v == nil {
			return nil, newParamError("chat_template_kwargs", "invalid_value",
				"chat_template_kwargs.%s is null, which sets nothing; send a value or omit the "+
					"variable", name)
		}
		out[name] = v
	}
	return out, nil
}

// suffixOf reads an optional suffix as a string, treating absent as empty.
func suffixOf(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}

// fimMarkerHint names the marker sets a checkpoint could declare, so a refusal can tell an
// operator what the served model is missing instead of only that something is missing.
func fimMarkerHint() string {
	sets := tokenize.FIMMarkerNames()
	out := make([]string, len(sets))
	for i, set := range sets {
		out[i] = set[0] + " and " + set[2]
	}
	return strings.Join(out, "; ")
}

// fimPrompt assembles a fill-in-the-middle prompt from its parts.
//
// The parts are encoded separately and joined around the marker ids rather than concatenated
// into one string and encoded, for two reasons. The markers are vocabulary entries, so they
// have to be their own ids instead of pieces the encoder might split; and the caller's text has
// to stay text, so a literal "<|fim_middle|>" typed into the prefix is encoded as characters
// rather than becoming the marker that opens the insertion.
//
// No BOS is added. The markers are the structure here, and a checkpoint trained for this sees
// the prefix marker at the start of the sequence.
func (s *Server) fimPrompt(prefix, suffix string, fim tokenize.FIM) []int32 {
	out := make([]int32, 0, len(prefix)/4+len(suffix)/4+3)
	out = append(out, fim.Prefix)
	out = append(out, s.tok.Encode(prefix, false)...)
	out = append(out, fim.Suffix)
	out = append(out, s.tok.Encode(suffix, false)...)
	out = append(out, fim.Middle)
	return out
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
	// choices is the request's n: how many completions to generate. Zero means one, which is
	// what an omitted field decodes to. Every choice is its own engine sequence with its own
	// sampler, and the range has already been checked by checkChoices.
	choices int
	// logitBias shifts named tokens' logits before sampling. It reaches the greedy path too,
	// which is the difference between a ban and a suggestion.
	logitBias []engine.TokenBias
	// presencePenalty and frequencyPenalty discourage tokens already seen. They reach the
	// engine rather than a sampler because they need the generated run, which a sampler does not
	// receive.
	presencePenalty  float32
	frequencyPenalty float32
	// maxTokensGiven reports that max_tokens was sent, which matters because zero is a budget
	// of zero rather than "unset". The field is a pointer on the wire so the two are
	// distinguishable; this carries that distinction into generate, where the budget is
	// reconciled with max_completion_tokens.
	maxTokensGiven bool
	// maxCompletionTokens is OpenAI's newer name for the budget. Nil means the field was not
	// sent; the two names are reconciled in generate, which is where the precedence is
	// visible next to the negative check it has to stay ordered against.
	maxCompletionTokens *int
	// echo asks the completions endpoint to return the prompt ahead of the generated text.
	// It has no meaning on the chat endpoint, whose schema has no such field.
	echo bool
	// logprobs asks for each generated token's log probability. It is one flag rather than a
	// count because the two endpoints spell "the chosen token and no alternatives"
	// differently: chat sends logprobs: true, completions sends logprobs: 0.
	logprobs bool
	// topLogprobs is how many alternatives to describe at each position. Zero means the
	// chosen token only.
	topLogprobs int
	// bestOf is the completions endpoint's candidate count: generate this many and return the
	// choiceCount best. Zero or one means no selection, which is the default.
	bestOf int
	// suffix, with fim, asks for an insertion rather than a continuation: the prompt is the
	// text before the insertion point and suffix is the text after it. fim is non-nil only
	// when the checkpoint declares markers for it, which the handler checks.
	suffix string
	fim    *tokenize.FIM
	// tools is the tool list the request declared, empty when it declared none. It is here so
	// the response can decide whether to parse the reply for calls: a model writing the marker
	// unasked must not have its text cut out of a reply that never asked to be a call.
	tools []engine.ToolSpec
	// responseID is the id this request's response carries. Tool calls are given ids derived
	// from it, so a call id is unique across responses, which is what a client relies on when
	// it sends the id back in tool_call_id.
	responseID string
}

// choiceCount is how many completions the request asks for, with the omitted field reading
// as one.
func (p generateParams) choiceCount() int {
	if p.choices <= 0 {
		return 1
	}
	return p.choices
}

func (s *Server) generate(w http.ResponseWriter, r *http.Request, p generateParams) {
	promptTokens, maxTokens, fimStopIDs, ok := s.resolvePrompt(w, &p)
	if !ok {
		return
	}
	choices := p.choiceCount()
	// best_of asks for more sequences than it returns, and ranking them needs each token's log
	// probability, so the engine is asked for them whenever selection is on even if the caller
	// did not request them. They are only put in the response when the caller did.
	//
	// Selection needs the candidates to differ, so each one gets its own sampler exactly as it
	// would if the caller had asked for n of them.
	if p.bestOf > 1 {
		if p.stream {
			// OpenAI documents best_of as "results cannot be streamed", and that is structural
			// rather than a limitation of this server: the candidates are only comparable once
			// they have all finished, so there is nothing to stream until the choice is made.
			writeRequestError(w, http.StatusBadRequest, newParamError(
				"best_of", "unsupported_parameter",
				"best_of cannot be combined with stream=true: the candidates are ranked after "+
					"they finish, so there is no result to stream until the choice is made. "+
					"Drop best_of to receive one of them streamed, or set stream=false"))
			return
		}
	}
	candidates := choices
	if p.bestOf > 1 {
		candidates = p.bestOf
	}
	wantLogprobs := p.logprobs || p.bestOf > 1
	samplers, effectiveTemp, err := s.samplers(p.temperature, p.topK, p.topP, p.seed, candidates)
	if err != nil {
		// The sampler's errors are all about one request field, so they carry it rather
		// than making the client infer which of temperature and top_p was wrong.
		writeRequestError(w, http.StatusBadRequest, err)
		return
	}
	if len(p.logitBias) > 0 {
		// Wrapped after the sampler is built, so the bias reaches every path including greedy.
		// A bias applied inside Temperature would be invisible to a temperature-0 request, and
		// temperature 0 is exactly when a caller bans a token: it is the reproducible mode.
		for i := range samplers {
			samplers[i] = engine.NewBiased(p.logitBias, samplers[i])
		}
	}

	// A request that cannot fit the block pool is refused now, as a 400, rather than being
	// submitted and failed later as a 503 with Retry-After.
	//
	// Those are two different situations that used to share one answer. A pool that is busy
	// is temporary: the same request succeeds once another sequence releases its blocks, and
	// Retry-After is exactly what a client or proxy should act on. A request whose own
	// prompt plus budget needs more blocks than the pool holds is permanent: nothing another
	// request does makes room for it, so every retry gets the same 503 and the client backs
	// off forever. Measured on the 256-position fixture with --blocks 2 --block-size 16 --
	// two blocks, 32 positions -- an 11-token prompt asking for 4 tokens needs 3 blocks: it
	// returned 503 with Retry-After 1 on four attempts two seconds apart while
	// orbix_kv_pool_free stayed at 2, and a shorter prompt succeeded against the same idle
	// server.
	//
	// Refusing costs one multiplication and saves the client a retry loop that cannot end.
	// The wording names the two numbers a caller can act on, the pool's positions and what
	// their request needs, because "block pool exhausted" reads as transient.
	//
	// The order against the checks above and below is deliberate: this one runs after the
	// prompt-length and sampler checks, so a request with a bad field is reported for the
	// field it got wrong. When it was placed before sampler, a request carrying
	// temperature -2 and no max_tokens against a small pool was refused as
	// context_length_exceeded -- the default 256-token budget made it look oversized, and
	// the caller was told to shorten a prompt whose real problem was the temperature. A
	// refusal that names the wrong field costs more than the retry loop it was added to stop.
	if st := s.eng.Stats(); st.PoolBlocks > 0 {
		if need, have := blocksNeeded(len(promptTokens)+maxTokens, st.PoolBlockSize)*candidates,
			st.PoolBlocks; need > have {
			// param names whichever side accounts for more of the request, because the check
			// compares prompt+budget and either can be what makes it unservable. Naming the
			// prompt field unconditionally was wrong for the shape a caller is most likely to
			// hit by accident: "Hi" with max_tokens 2000 is 2002 positions, and a client that
			// follows `param` shortens a two-token prompt while the budget stays the reason.
			// The reverse -- a long prompt with a small budget -- is what this test's fixture
			// builds, and there the prompt field is the right answer. Splitting on which is
			// larger gets both right; it is also the split the message itself already makes
			// by printing both numbers.
			blamed := promptParam(p.chat)
			if maxTokens > len(promptTokens) {
				blamed = "max_tokens"
			}
			// With several choices the need is n times one choice's, and a message that
			// printed only the prompt-plus-budget figure would read as if one choice needed
			// those blocks. The detail clause names the multiplication when there is one, so
			// the arithmetic in the message accounts for the total it is quoting.
			positions := len(promptTokens) + maxTokens
			detail := fmt.Sprintf("%d positions for the prompt and budget", positions)
			if candidates > 1 {
				detail = fmt.Sprintf("%d positions for the prompt and budget, times %d "+
					"sequences at %d blocks each", positions, candidates,
					blocksNeeded(positions, st.PoolBlockSize))
			}
			writeRequestError(w, http.StatusBadRequest, newParamError(
				blamed, "context_length_exceeded",
				"this request needs %d KV cache blocks of %d positions (%s) but the pool holds "+
					"%d (%d positions), so no other request finishing can make room for it; "+
					"shorten the prompt or lower max_tokens or n, or start the server with a "+
					"larger --blocks",
				need, st.PoolBlockSize, detail, have, have*st.PoolBlockSize))
			return
		}
	}

	stopIDs := append(s.stopTokenIDs(p.stop), fimStopIDs...)
	// OpenAI prefixes a completion id with "cmpl-" and a chat completion with "chatcmpl-".
	// The prefix is cosmetic to a parser but visible to anyone reading a log, and the wrong
	// one makes a completion look like a chat response.
	prefix := "cmpl-"
	if p.chat {
		prefix = "chatcmpl-"
	}
	id := s.responseID(prefix, time.Now().UnixNano())
	// Recorded on the params so the response builder can derive per-call ids from it. The id is
	// created here and nowhere else, so a builder that needs it cannot be handed a different one.
	p.responseID = id
	// One timestamp for the whole response. Every chunk of a stream and the final usage chunk
	// carry this same value, because a client reads it as when the response was created, not
	// as when each chunk happened to be written. Computing it per chunk made a long stream
	// report two different creation times -- measured, a fixture reply is short enough to fit
	// in one second, so it took a deliberate slow stream to see, but nothing prevented it.
	created := time.Now().Unix()

	// The two endpoints differ only in how the prompt was rendered, so the operation name
	// is what tells them apart in a trace.
	op := "text_completion"
	if p.chat {
		op = "chat"
	}

	// One sequence per choice, each with its own sampler. Submitting them together is what
	// makes n choices cost a batch rather than n requests: the engine advances every active
	// sequence in the same rounds.
	chans := make([]<-chan engine.Event, 0, choices)
	for i, sm := range samplers {
		ch, err := s.eng.Submit(engine.Request{
			Prompt:           promptTokens,
			ChoiceIndex:      i,
			ChoiceCount:      len(samplers),
			MaxTokens:        maxTokens,
			Priority:         p.priority,
			Stop:             stopIDs,
			Sampler:          sm,
			PresencePenalty:  p.presencePenalty,
			FrequencyPenalty: p.frequencyPenalty,
			Logprobs:         wantLogprobs,
			TopLogprobs:      p.topLogprobs,
			Temperature:      effectiveTemp,
			Cancel:           r.Context().Done(),
			Trace:            s.cfg.Trace,
			Operation:        op,
			ResponseID:       id,
			Stream:           p.stream,
			ResponseModel:    s.cfg.ModelName,
		})
		if err != nil {
			// A later choice failing leaves the earlier ones running, and the engine abandons
			// a sequence whose events nobody reads only after abandonGrace -- holding its KV
			// blocks until then. Draining them here releases the blocks as soon as each
			// finishes instead of leaving the pool short for two seconds.
			//
			// No test covers this loop, and deleting it leaves the whole package green. What it
			// changes is when blocks come back, not whether they do -- abandonGrace is two
			// seconds, so a test would have to assert "the pool was free well before that",
			// and a threshold loose enough not to flake is loose enough to pass without the
			// loop. Reaching the branch at all also needs a request where choice 0 is accepted
			// and a later one is refused, which against a fixture this fast means racing the
			// two submissions. Both properties are why AGENTS.md puts this shape in comments
			// rather than in tests.
			for _, done := range chans {
				go func() {
					for range done {
					}
				}()
			}
			// A full queue is not the client's fault and not permanent: the request is well
			// formed and would succeed later. 503 with Retry-After is what a client or proxy
			// knows to back off on, where 400 would tell it to stop sending this at all.
			if errors.Is(err, engine.ErrQueueFull) {
				s.refusedQueueFull.Add(1)
				w.Header().Set("Retry-After", "1")
				// Same reasoning as the exhausted-pool refusal below: the engine's sentence
				// gives the counts an operator needs ("1 queued for 1 active slots") and stops
				// there. The appended half names the two flags that change it, because a client
				// logging this body otherwise learns only that something was full.
				writeError(w, http.StatusServiceUnavailable, err.Error()+
					"; retrying after the Retry-After delay succeeds once the queue drains, "+
					"but a backlog that fills repeatedly needs a larger --max-queued or a "+
					"higher --max-active")
				return
			}
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		chans = append(chans, ch)
	}

	if p.stream {
		s.streamResponse(w, r, chans, id, created, p, len(promptTokens))
		return
	}
	s.blockingResponse(w, r, chans, id, created, p, promptTokens)
}

// choiceResult is one sequence's collected output, shared between the drain goroutines and
// the code that assembles the response.
//
// Each element is written by exactly one goroutine -- the one draining that sequence's channel
// -- and read by this goroutine only after all of them have finished, which is why no lock is
// needed for a slice shared this way.
type choiceResult struct {
	out     []int32
	samples []sampledToken
	reason  string
	closed  bool
	failure error
}

// score is the candidate's mean log probability over the tokens it generated.
//
// The mean rather than the sum, because a sum always favours the shortest completion: every
// term is negative, so a five-token answer scores higher than a hundred-token one that was
// equally unsurprising. OpenAI documents best_of as comparing "log probability per token",
// which is the same correction.
//
// A candidate that generated nothing scores -Inf, so it never wins: it has no per-token figure
// to compare, and ranking it first on an empty sum would return an empty completion as "best".
//
// Nothing in production calls this: ranking uses scoreShown, which applies the same mean to the
// run the response actually reports. It is kept because two tests need the whole-run figure as
// the thing scoreShown differs from -- TestBestOfRanksOnTheTokensItReports builds a pair of
// candidates whose whole-run and visible orderings disagree, and asserts both, so that the
// ranking test would fail if the two scorings ever became the same. Removing this as dead code
// would leave those cases passing without showing the distinction they exist for.
func (c *choiceResult) score() float64 {
	if len(c.samples) == 0 || len(c.samples) != len(c.out) {
		// The length check matters: samples is only filled when log probabilities were
		// requested, and a caller reaching here without them has no basis to rank on.
		return math.Inf(-1)
	}
	var sum float64
	for _, s := range c.samples {
		sum += float64(s.logprob)
	}
	return sum / float64(len(c.samples))
}

// blockingResponse collects the whole output and writes one JSON body.
//
// chans holds one event channel per requested choice, in choice order, so the response's
// choices[i] is the i-th channel's output. The channels are drained concurrently rather than
// one after another: the engine advances every sequence in the same rounds, and its emit path
// gives a full event buffer two seconds before deciding the caller is gone. Draining them in
// turn would let the choice that is not being read hit that deadline while the first choice
// is still generating, which abandons it and truncates the answer.
func (s *Server) blockingResponse(w http.ResponseWriter, r *http.Request, chans []<-chan engine.Event, id string, created int64, p generateParams, promptTokens []int32) {
	// One result per channel, each written by exactly one goroutine, so the slice is shared
	// between them without a lock: different elements are different memory.
	results := make([]choiceResult, len(chans))
	var wg sync.WaitGroup
	for i, ch := range chans {
		c := &results[i]
		c.reason = "stop"
		wg.Go(func() {
			// An error stops the drain and is reported after it, rather than from inside the
			// callback: a callback cannot return from this function, and writing a status from
			// inside one would leave the drain still reading a channel nobody wants.
			c.closed = drainEvents(r.Context(), ch, func(ev engine.Event) {
				if ev.Err != nil {
					if c.failure == nil {
						c.failure = ev.Err
					}
					return
				}
				if ev.HasToken {
					c.out = append(c.out, ev.Token)
					// Positioned by construction: the engine sets HasLogprob on every token
					// event when the request asked for them, so samples ends up the same
					// length as out and index i describes token i.
					if ev.HasLogprob {
						c.samples = append(c.samples, sampledToken{
							id: ev.Token, logprob: ev.Logprob, top: ev.Top,
						})
					}
				}
				if ev.Done {
					c.reason = ev.FinishReason
				}
			})
		})
	}
	wg.Wait()

	// The first failure in choice order decides the status, so the answer does not depend on
	// which goroutine happened to be scheduled first.
	for i := range results {
		if f := results[i].failure; f != nil {
			// A pool that ran out is capacity, not a fault in the request or the server: the
			// same request succeeds once another sequence releases its blocks. 503 with
			// Retry-After is what a client backs off on, and 500 would have it treat a
			// temporary condition as a permanent one.
			//
			// The sentinel is the engine's, not the cache's: which errors are transient is a
			// scheduling judgement, and this layer should not have to import internal/llama to
			// learn it. The message is unchanged either way -- engine.poolAwareErr keeps the
			// storage layer's sentence, block count included.
			if errors.Is(f, engine.ErrPoolExhausted) {
				s.refusedPool.Add(1)
				w.Header().Set("Retry-After", "1")
				// The storage layer's sentence carries the geometry an operator needs -- "16
				// blocks of 16 positions" -- but none of it says what to do, and the prefixes
				// name internal packages a caller cannot see. Both halves are kept: the
				// original text, because it is the measurement, and an appended sentence with
				// the two levers and the fact that retrying alone will not help a request that
				// is simply too large for the pool.
				writeError(w, http.StatusServiceUnavailable, f.Error()+
					"; retrying after the Retry-After delay succeeds once another sequence "+
					"releases its blocks, but a pool that is too small for the traffic it is "+
					"given needs a larger --blocks, a smaller --block-size, fewer concurrent "+
					"requests, or a shorter prompt and max_tokens")
				return
			}
			writeError(w, http.StatusInternalServerError, f.Error())
			return
		}
	}
	for i := range results {
		if !results[i].closed {
			// The request was abandoned before the sequence finished, so what arrived is a
			// prefix and there is no honest 200 to build from it. Saying so gives the client
			// a status it can retry on; the alternative was no response at all.
			//
			// One abandoned choice is enough to fail the whole request. A body with n
			// choices where some are prefixes and some are complete would be a 200 whose
			// content is partly truncated, and nothing in the schema says which choice that
			// was.
			writeError(w, http.StatusServiceUnavailable,
				"the request was abandoned before it finished")
			return
		}
	}

	// Which of the collected sequences the response returns. Normally all of them, in order;
	// with best_of, the top n by mean log probability.
	selected := make([]int, len(results))
	for i := range selected {
		selected[i] = i
	}
	if p.bestOf > 1 {
		// Rank on the tokens the response will actually show. A stop string cuts the
		// reported log probabilities to the surviving text, and ranking on the untrimmed run
		// meant a candidate could be selected for a mean the caller never sees: measured on
		// Gemma 3 270M with n=7 best_of=8 and stop ".", the returned choices were not in
		// descending order of the log probabilities the response carried -- -10.772 came
		// before -10.626 -- because the first of those two had generated further tokens after
		// its stop string than the second.
		trimmed := make([]int, len(results))
		for i := range results {
			trimmed[i] = s.shownTokens(&results[i], p)
		}
		selected = rankCandidates(results, trimmed, p.choiceCount())
	}

	// completion_tokens counts every sequence generated, discarded candidates included.
	// OpenAI's own description of best_of warns that it "can quickly consume your token
	// quota", which says the candidates are generated and charged whether or not they are
	// returned; reporting only the returned ones would understate the work by best_of/n and
	// flatter exactly the number a caller budgets from.
	generated := 0
	for i := range results {
		generated += len(results[i].out)
	}

	choices := make([]map[string]any, len(selected))
	for out, idx := range selected {
		choices[out] = s.choiceObject(&results[idx], out, p)
	}

	usage := map[string]int{
		"prompt_tokens":     len(promptTokens),
		"completion_tokens": generated,
		"total_tokens":      len(promptTokens) + generated,
	}
	object := "text_completion"
	if p.chat {
		object = "chat.completion"
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id":      id,
		"object":  object,
		"created": created,
		"model":   s.cfg.ModelName,
		"choices": choices,
		"usage":   usage,
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
// drainEvents feeds every event on ch to fn until the channel closes or ctx is done.
//
// It returns true when the channel closed -- the sequence finished, one way or another -- and
// false when ctx was cancelled first, which means the request is being abandoned: the client
// hung up, or the server is shutting down and will not advance this sequence again.
//
// A `for ev := range ch` cannot express the second case, and that is a real difference rather
// than a stylistic one. The channel is closed by the engine when it finishes a sequence, and a
// sequence the pump never advances again is never finished, so the range blocks until the
// process exits and the client gets nothing. Measured against a real server: four non-streaming
// requests in flight at a 3-second shutdown grace all came back as curl exit 52, "empty reply
// from server", while the same four at a 60-second grace answered 200.
func drainEvents(ctx context.Context, ch <-chan engine.Event, fn func(engine.Event)) bool {
	for {
		select {
		case ev, ok := <-ch:
			if !ok {
				return true
			}
			fn(ev)
		case <-ctx.Done():
			return false
		}
	}
}

// streamItem is one thing a choice's reader wants written to the client.
//
// The reader does the deciding -- decoding, stop-string matching, holding back a partial
// match -- and hands the result here, so the code that writes to the socket has nothing to
// know about tokens and the code that reads tokens has nothing to know about the socket.
type streamItem struct {
	// index is the choice this belongs to.
	index int
	// delta is text to send. It is empty on a terminal or error item.
	delta string
	// reason, when set, makes this the choice's final chunk. It is the engine's own
	// vocabulary ("stop", "length", "cancel"), not the schema's.
	reason string
	// err, when set, is a mid-stream failure to report in band.
	err error
	// logprobs, when non-nil, is this token's entry in the chunk's logprobs field. The reader
	// builds it because the object needs the token's byte offset in the response, which only
	// the reader is tracking; the writer would have to duplicate the count.
	logprobs any
	// calls, when non-empty, is the tool calls this choice's reply asked for. It is carried on
	// the terminal item and travels in the same chunk as the finish reason, which is where the
	// schema puts it: a client accumulates deltas and reads tool_calls once the turn is over.
	calls []toolCall
}

// streamResponse writes server-sent events, one per token.
//
// chans holds one event channel per choice, in choice order. n=1 -- the common case -- runs
// its single choice inline, so its output is byte-for-byte what it was before choices existed.
// n>1 runs one reader goroutine per choice and writes what they produce from this goroutine,
// because http.ResponseWriter is not safe for concurrent use and the chunks have to be
// interleaved by whatever order the choices actually advance in.
//
// includeUsage asks for the closing usage chunk OpenAI sends when
// stream_options.include_usage is set. Its contract has two halves and both matter: the
// extra chunk has an empty choices array and carries the totals, and every ordinary chunk
// in the same stream carries "usage": null. A client written against the documented shape
// distinguishes the two by choices being empty, and one that reads usage off every chunk
// without a null there would see the field missing rather than absent.
func (s *Server) streamResponse(w http.ResponseWriter, r *http.Request, chans []<-chan engine.Event, id string, created int64, p generateParams, promptTokens int) {
	ew, ok := newEventWriter(w, s.cfg.ModelName, eventEnvelope{id: id, created: created}, p)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming is not supported by this connection")
		return
	}
	send := ew.send

	// writeItem is the one place a streamItem becomes bytes, so the inline and the fan-in
	// paths cannot disagree about the shape of a chunk.
	writeItem := func(it streamItem) bool {
		if it.err != nil {
			// The response has already started, so a status code is no longer available:
			// report the failure in the stream and stop.
			//
			// The refusal counters still move, even though the client cannot be told the cause
			// through a status code. They are what an operator watches, and a streaming
			// deployment whose requests all lose the race for blocks would otherwise report
			// zero pool refusals while refusing every one of them -- measured, four streaming
			// requests against an exhausted pool left refused_pool_total at 0 against 4 for the
			// same four sent blocking. The count is the only signal that survives the
			// protocol's inability to say it in the response.
			if errors.Is(it.err, engine.ErrPoolExhausted) {
				s.refusedPool.Add(1)
			}
			return send(map[string]any{
				"choices": []map[string]any{{
					"index": it.index, "delta": map[string]string{}, "finish_reason": "error",
				}},
				"error": it.err.Error(),
			})
		}
		if it.reason != "" {
			return send(finalChoice(p.chat, it.index, finishReasonFor(it.reason), it.calls))
		}
		return send(chunkPayload(p.chat, it.index, it.delta, it.logprobs))
	}

	// echo returns the prompt ahead of the generated text, so it has to arrive before any
	// generated delta. One chunk per choice, because with n>1 every choice's text includes
	// the prompt and a single shared chunk would land in whichever index the client happened
	// to be accumulating. Nothing is read from the engine until this has been written.
	if p.echo && !p.chat {
		for i := range chans {
			if !send(chunkPayload(p.chat, i, p.prompt, nil)) {
				return
			}
		}
	}

	generated, cancelled, broken := s.streamChoices(chans, p, writeItem)
	if broken {
		// The client is gone, or a choice reported a failure that was already written. Either
		// way there is nobody left to read a usage chunk or a sentinel, and writing one after a
		// broken pipe is the case the write error already covered.
		return
	}

	// The usage chunk, when asked for, goes after the last choice's finish chunk and before
	// [DONE]. OpenAI's shape is specific and a client depends on both halves: choices is an
	// empty array (which is how a client tells this chunk apart from a text one) and usage
	// carries the whole request's counts. Sending it with a choice in it, or without the empty
	// array, breaks the check a client uses to find it.
	//
	// The totals span every choice, matching the non-streaming body: prompt_tokens counted
	// once because the prompt is one text, completion_tokens summed because each choice
	// generated its own.
	if p.includeUsage {
		// Through the same writer as every other chunk: the envelope -- id, object, created,
		// model -- is what a client uses to tell this chunk apart from a text one, and building
		// it here as well is how a field gets left off one of the two paths.
		ew.send(map[string]any{
			"choices": []map[string]any{},
			"usage": map[string]int{
				"prompt_tokens":     promptTokens,
				"completion_tokens": generated,
				"total_tokens":      promptTokens + generated,
			},
		})
	}
	if cancelled {
		// No [DONE]: the stream did not complete, and saying it did is the difference between a
		// client that retries and one that keeps a truncated reply. Flushing what is already
		// buffered still happens so the deltas that did arrive are delivered.
		ew.flusher.Flush()
		return
	}
	_, _ = io.WriteString(ew.w, "data: [DONE]\n\n")
	ew.flusher.Flush()
}

// streamChoices runs every choice and reports the stream's totals.
//
// One choice runs here, inline. More than one runs one goroutine per channel behind a fan-in
// channel, because the deltas have to reach the socket in the order the choices produce them
// and http.ResponseWriter cannot be written from several goroutines. writeItem is only ever
// called from this goroutine, which is what makes that safe.
//
// broken means the stream must stop without a sentinel: the client stopped reading, or a choice
// failed and its error chunk has already been written.
//
// Once broken, the other readers are told to stop through done rather than being allowed to
// run on. That is deliberate: letting them finish would make the error response wait for the
// remaining budget of every choice, and a client that has already been told the request failed
// has nothing to gain from the wait. The sequences they leave behind are abandoned by the
// engine within abandonGrace, which releases their KV blocks -- the same trade the n=1 path
// makes when a client hangs up, and the reason its pump is bounded.
func (s *Server) streamChoices(chans []<-chan engine.Event, p generateParams, writeItem func(streamItem) bool) (generated int, cancelled, broken bool) {
	type totals struct {
		generated int
		cancelled bool
	}
	if len(chans) == 1 {
		g, c, ok := s.streamOneChoice(chans[0], 0, p, writeItem)
		return g, c, !ok
	}

	results := make([]totals, len(chans))
	// done lets the writers give up promptly once this function has stopped reading.
	done := make(chan struct{})
	items := make(chan streamItem, streamFanInBuffer)
	var wg sync.WaitGroup
	for i, ch := range chans {
		t := &results[i]
		wg.Go(func() {
			// ok is deliberately ignored here: a choice whose sink refused has already told
			// this function through an item, or the stream is stopping for another reason.
			g, c, _ := s.streamOneChoice(ch, i, p, func(it streamItem) bool {
				select {
				case items <- it:
					return true
				case <-done:
					return false
				}
			})
			t.generated, t.cancelled = g, c
		})
	}
	go func() {
		wg.Wait()
		close(items)
	}()

	for it := range items {
		if broken {
			// Still draining: the readers are blocked on this channel and would never return
			// if it stopped being read once the stream was declared broken.
			continue
		}
		if !writeItem(it) || it.err != nil {
			broken = true
			close(done)
		}
	}
	// close(done) above may not have run; the drain has finished either way by now.
	select {
	case <-done:
	default:
		close(done)
	}
	for i := range results {
		generated += results[i].generated
		cancelled = cancelled || results[i].cancelled
	}
	return generated, cancelled, broken
}

// streamFanInBuffer is how many items the choice readers can have queued before one blocks.
//
// One choice can produce at most a token per round, and the readers only queue work that is
// already decoded, so a small buffer is enough to keep them from being the bottleneck without
// letting a stalled reader hold an unbounded amount of text.
const streamFanInBuffer = 64

// streamOneChoice consumes one choice's events and hands every write to emit.
//
// It returns the choice's own token total (for usage) and whether the engine cancelled it
// (for withholding the sentinel). ok is false when emit refused a write, which means the
// stream is over and the caller should stop.
func (s *Server) streamOneChoice(ch <-chan engine.Event, index int, p generateParams, emit func(streamItem) bool) (generated int, cancelled, ok bool) {
	dec := s.tok.Seq()
	reason := "stop"
	// stopMatched is true only when a stop string was found in the text. It is a separate
	// variable because reason cannot carry it: the engine reports "stop" for an ordinary
	// completion, and the two would be indistinguishable.
	stopMatched := false
	gate := &stopGate{stop: p.stop}
	// tcg is non-nil only when the request declared tools on the chat endpoint, which is the one
	// combination whose output has to be withheld until a call can be recognised. With no tools
	// declared the stream is byte-for-byte what it was, which is why the plain-text tests still
	// describe it.
	var tcg *toolCallGate
	if p.chat && len(p.tools) > 0 {
		tcg = &toolCallGate{}
	}
	// tokenOffset is how many bytes of this choice's text the tokens so far have accounted
	// for. It is the offset a logprob entry reports, and it is tracked separately from sent
	// because held-back text has an offset before it has been written.
	tokenOffset := 0
	// held collects the described tokens whose text the tool gate is still withholding. It is
	// empty whenever the gate is not in use, and emptied as soon as the gate releases anything.
	var held []heldToken
	// write releases one piece of a choice's text to the client, routing it through the tool
	// gate when the request declared tools. Every path below goes through it rather than
	// straight to emit, because the tool gate is not an alternative to the other two
	// behaviours -- a stop list holds bytes back and a call has to be withheld -- but a third
	// stage both of them feed. Measured before this was the only route: a request carrying both
	// `tools` and `stop` streamed the whole `<tool_call>{...}</tool_call>` to the client as
	// content and reported no tool_calls at all, while the same request non-streamed returned
	// the call. The stop branch simply continued before the gate was consulted.
	// chunkFor renders one chunk's logprobs object in the shape the endpoint uses. A chat chunk
	// carries a list of per-token objects and a completions chunk parallel arrays, so a held run
	// released in one chunk is one object built from the whole run.
	chunkFor := func(described []logprobText, offset int) any {
		if described == nil {
			return nil
		}
		if p.chat {
			return chatLogprobs(described)
		}
		return completionLogprobs(described, offset)
	}
	write := func(part string, described []logprobText, offset int) bool {
		if tcg == nil {
			if part == "" && described == nil {
				return true
			}
			return emit(streamItem{index: index, delta: part, logprobs: chunkFor(described, offset)})
		}
		before := tcg.dropped
		out := tcg.push(part)
		if tcg.dropped > before {
			// Some of these bytes turned out to be part of a call and will never reach the
			// client. That settles every description the gate was still holding as well: they
			// were waiting on exactly this decision, and it went against them.
			held = nil
			return emit(streamItem{index: index, delta: out})
		}
		if out == "" {
			// The gate is holding this piece: it may still turn out to be a call, and a call's
			// bytes must not reach the client as content. The description is held with it
			// rather than dropped, because whether the bytes are prose is decided later -- a
			// '<' in prose is released by this same gate on a later push. Measured before the
			// description was held: "a < b tail" streamed 8 logprob entries for 9 tokens, the
			// missing one being the '<' the client did receive.
			if described != nil {
				held = append(held, heldToken{described, offset})
			}
			return true
		}
		// Whatever was held comes out with the first piece the gate releases, which is the same
		// chunk the text itself arrives in. The entries are contiguous in the reply, so the
		// first one's offset is where the run starts.
		if len(held) > 0 {
			described, offset = appendHeld(held, described, offset)
			held = nil
		}
		return emit(streamItem{index: index, delta: out, logprobs: chunkFor(described, offset)})
	}
	for ev := range ch {
		if ev.Err != nil {
			emit(streamItem{index: index, err: ev.Err})
			return generated, cancelled, false
		}
		if ev.HasToken {
			// Counted here rather than from the emitted text: the usage figures describe what
			// the model produced, and a token that decodes to nothing is still a token the
			// engine generated. The non-streaming body counts the same way (len(out)), so the
			// two agree for one request.
			generated++
			text := dec.Token(ev.Token)
			// The token's byte offset in this choice's text, counted before it is written:
			// a stop list can hold text back, and the offset describes where the text lands
			// rather than when it was released.
			var describedToken []logprobText
			// advance is how much of the choice's text this token accounts for. Without
			// logprobs it is the text the stream decoder released; with them it is the length
			// of the token's own rendering, because that is what the blocking path counts and
			// the two offsets have to agree. They differ exactly when a character spans
			// several tokens, where the decoder holds text back and attributes it all to one.
			advance := len(text)
			if p.logprobs && ev.HasLogprob {
				// first says whether this is the token the leading-space rule applies to.
				described := s.describeStreamedToken(ev.Token, text, ev.Logprob, ev.Top, tokenOffset == 0)
				advance = len(described[0].text)
				describedToken = described
			}
			tokenStart := tokenOffset
			tokenOffset += advance
			// A token that decodes to nothing still has a log probability, and a caller who
			// asked for them expects one entry per generated token. Without this the streamed
			// list would be shorter than the blocking one for the same request.
			if text == "" && describedToken == nil {
				continue
			}
			// A stop string has to be applied to the stream as well as to the whole body.
			// The engine only knows about single tokens, so the check is on the text
			// accumulated so far: it can only be recognised once the rune that completes it
			// arrives, which is exactly when the client should stop seeing output.
			if len(p.stop) > 0 {
				release, found := gate.push(text)
				if found {
					// Output up to the stop string is the client's; the stop and anything
					// after it are not. `sent` is what the client already has, so only the
					// part between them still needs sending.
					// No log probability travels with this tail, and none is sent for the
					// token that completed the match. That token's text is the stop string,
					// which is not in the response, so a logprob for it would describe
					// something the caller cannot see -- and when its text straddles the cut,
					// part of it survives and neither "include" nor "truncate" is right.
					// Dropping the straddling token is what the blocking path does, so the two
					// transports report the same list for the same request.
					if release != "" {
						if !write(release, nil, 0) {
							return generated, cancelled, false
						}
					}
					// The sequence is still running; its context drives it to completion
					// unread, which is what the engine does for a client that stopped
					// reading. What changes here is only what this response reports.
					reason = "stop"
					stopMatched = true
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
				if release != "" {
					if !write(release, describedToken, tokenStart) {
						return generated, cancelled, false
					}
				} else if describedToken != nil {
					// Everything visible is already out and this token's text is being held,
					// so the entry travels alone. The text follows when the tail is released.
					if !emit(streamItem{index: index, logprobs: chunkFor(describedToken, tokenStart)}) {
						return generated, cancelled, false
					}
				}
				continue
			}
			if !write(text, describedToken, tokenStart) {
				// The client went away. Its context is cancelled, so the engine
				// stops the sequence; draining is the caller's job and this loop
				// is already past that point.
				return generated, cancelled, false
			}
		}
		if ev.Done {
			reason = ev.FinishReason
			// A cancel is the engine being told to stop, which on a shutdown is this server
			// going away rather than the model finishing. The stream cannot say so with
			// finish_reason -- the schema has no value for it -- but it can decline to send
			// the completion sentinel, and that is the one signal a client already knows
			// how to read: no [DONE] means the stream was cut.
			if ev.FinishReason == "cancel" {
				cancelled = true
			}
		}
	}
	// A character split across the last tokens is still held by the decoder, waiting for a
	// byte that never arrives. Without this the stream would end one character short of what
	// the same sequence produces without streaming, which breaks the contract
	// TestStreamingReassemblesToSameText states.
	if tail := dec.Flush(); tail != "" {
		if !write(tail, nil, 0) {
			return generated, cancelled, false
		}
	}
	// Text held back in case it began a stop string is the client's after all: no match was
	// found, so the tail was ordinary output. Sending it is what makes a stop list that never
	// matches cost nothing but a later chunk.
	// stopMatched, not reason, is the condition: reason is the engine's own finish reason, and
	// an ordinary completion ends with "stop" too. Gating on it meant a reply that finished
	// normally lost whatever the gate was holding -- measured, `x y z` with stop=["zzz"]
	// streamed "x y " while the blocking body returned all of it, because the trailing "z"
	// could still have begun the stop string. The two ideas share a word and nothing else: one
	// says the model stopped because it had nothing more to write, the other says this server
	// cut the reply because it found a stop string.
	if !stopMatched {
		if rest := gate.flush(); rest != "" {
			if !write(rest, nil, 0) {
				return generated, cancelled, false
			}
		}
	}
	// Whatever the tool gate still holds is decided now that generation has ended: text that was
	// only ever prose goes out, and a complete call is left to the final chunk's tool_calls.
	var calls []toolCall
	if tcg != nil {
		// Whatever the gate was still holding, and the descriptions that went with it, is
		// decided now: flush releases it when no complete call was in the reply, so the text is
		// prose and its log probabilities belong to the chunk that carries it.
		rest := tcg.flush()
		// flush releases the held region only when the whole reply held no complete call, and
		// in that case the region is prose: its log probabilities belong to the chunk carrying
		// it. When it stays empty the region was a call, and held == nil has just said so.
		if run, offset := appendHeld(held, nil, 0); rest != "" && run != nil {
			if !emit(streamItem{index: index, delta: rest, logprobs: chunkFor(run, offset)}) {
				return generated, cancelled, false
			}
			held, rest = nil, ""
		}
		held = nil
		if rest != "" {
			if !emit(streamItem{index: index, delta: rest}) {
				return generated, cancelled, false
			}
		}
		if _, parsed := parseToolCalls(tcg.text); len(parsed) > 0 {
			reason = "tool_calls"
			calls = assignToolCallIDs(p.responseID, parsed)
		}
	}
	// One final chunk carries the finish reason, the calls when there are any, and an empty
	// delta. Clients that accumulate deltas see no text change from it, and it is what tells a
	// client that this index stopped.
	//
	// The calls ride here rather than on an item of their own because writeItem classifies an
	// item as an error, a terminal, or a delta -- a fourth shape would fall through to
	// chunkPayload and be dropped, which is exactly what an earlier version did: the stream
	// reported finish_reason "tool_calls" with no tool_calls anywhere in it.
	if !emit(streamItem{index: index, reason: reason, calls: calls}) {
		return generated, cancelled, false
	}
	return generated, cancelled, true
}

// chunkPayload builds one streaming delta chunk for a choice.
//
// index is the choice's position, which for n=1 is always 0 and is why this took no index
// before. A client accumulates deltas per index, so a chunk carrying the wrong index appends
// one choice's text to another's.
func chunkPayload(chat bool, index int, text string, logprobs any) map[string]any {
	if chat {
		return map[string]any{"choices": []map[string]any{{
			"index": index, "delta": map[string]string{"content": text},
			"finish_reason": nil, "logprobs": logprobs,
		}}}
	}
	return map[string]any{"choices": []map[string]any{{
		"index": index, "text": text, "finish_reason": nil, "logprobs": logprobs,
	}}}
}

// finalChoice builds the chunk that ends one choice: its finish reason, and the tool calls the
// reply asked for when there are any.
//
// Every choice gets one, including under n=1: a client uses it to know that particular index
// stopped rather than to infer it from the sentinel, which arrives once for the whole stream.
//
// The calls ride on the terminal chunk rather than a chunk of their own because a client reads
// them once, after the deltas: the schema's delta object carries either content or tool_calls, and
// splitting them across two chunks makes the turn end twice.
//
// The delta then carries tool_calls and an empty content, which is the shape the SDKs parse. An
// empty content string is sent rather than omitted because the same field appears on every other
// delta of the turn, and a client that indexes it would otherwise have to special-case the last
// chunk.
func finalChoice(chat bool, index int, reason string, calls []toolCall) map[string]any {
	if chat {
		if len(calls) > 0 {
			return map[string]any{"choices": []map[string]any{{
				"index":         index,
				"delta":         map[string]any{"content": "", "tool_calls": toolCallsJSON(calls)},
				"finish_reason": reason,
			}}}
		}
		return map[string]any{"choices": []map[string]any{{
			"index": index, "delta": map[string]string{}, "finish_reason": reason,
		}}}
	}
	return map[string]any{"choices": []map[string]any{{
		"index": index, "text": "", "finish_reason": reason,
	}}}
}

// samplers builds one sampler per choice.
//
// Each choice gets its own sampler so that sampled choices differ from one another; the seed is
// anchored on the request and offset by the choice index, so choice i gets seed+i and the same
// request reproduces the same set of completions in the same order, which is what a caller
// comparing samples needs.
//
// Distinct seeds do not by themselves make the choices differ: at temperature 0 every sampler
// here is greedy, and a greedy sampler never reads its seed, so n>1 returns n copies of one
// completion. Measured, n:4 at the default temperature returned the same six tokens four times,
// identically across three requests. This comment used to claim the choices differ whenever
// they are seeded, which is true only once sampling is on.
//
// Every seed is resolved here rather than in sampler(), including the clock read for a request
// that supplied none. That is what makes the choices differ in that case too: reading the clock
// once and offsetting it gives n distinct seeds, while letting each choice read the clock for
// itself is a race that two calls in the same nanosecond can lose -- the sampler would then
// return the same text for both, which is exactly the case n>1 exists to avoid.
func (s *Server) samplers(temp *float64, topK int, topP *float64, seed int64, n int) ([]engine.Sampler, float32, error) {
	reproducible := seed != 0
	base := seed
	if !reproducible {
		base = time.Now().UnixNano()
	}
	out := make([]engine.Sampler, n)
	// The temperature the samplers actually use is reported back because the engine needs it
	// to describe the distribution when log probabilities are requested, and re-deriving it
	// there would be a second copy of the default-and-validate rule.
	var effective float32
	for i := range n {
		// seeded is true even for the clock case: the seed is resolved at this point, so
		// sampler() must use the value rather than substitute one. A derived seed of zero is
		// legitimate here -- seed -1 with two choices derives -1 and 0 -- and passes through
		// as the real stream seed.
		sm, t, err := s.sampler(temp, topK, topP, base+int64(i), true)
		if err != nil {
			return nil, 0, err
		}
		out[i] = sm
		effective = t
	}
	return out, effective, nil
}

// sampler builds the sampler a request asked for, or the server default.
func (s *Server) sampler(temp *float64, topK int, topP *float64, seed int64, seeded bool) (engine.Sampler, float32, error) {
	t := s.cfg.Temperature
	if temp != nil {
		t = *temp
	}
	if t < 0 {
		return nil, 0, newParamError("temperature", "invalid_value",
			"temperature must not be negative, got %v", t)
	}
	// A negative top_k is refused rather than treated as "unset". The field is a plain int,
	// so zero is the only distinguishable "not sent", and the code below reaches it with
	// "<= 0" -- which meant top_k: -5 silently became the server's default and the request
	// ran with a cutoff the caller did not ask for and was not told about. The CLI's --top-k
	// already refuses it for the same reason (see cmd/orbix's run and chat), and max_tokens
	// and top_p above are refused here, so this was the one sampling knob that quietly
	// substituted a value.
	if topK < 0 {
		return nil, 0, newParamError("top_k", "invalid_value",
			"top_k must not be negative, got %d", topK)
	}
	if topK == 0 {
		topK = s.cfg.TopK
	}
	// TopP defaults to the server's own value when the request omits it. A value outside
	// [0, 1] is refused rather than clamped. Zero is accepted and means "no nucleus": it is
	// the documented "disabled" value on the CLI's --top-p, and engine.NewTemperature treats a
	// value at or below zero the same way as one at or above 1 -- measured, top_p: 0 and
	// top_p: 1 returned identical text for the same seed. What is refused is a value with no
	// reading at all, since clamping -0.1 to 0 would let a typo through as "off".
	//
	// This comment said "(0, 1]" and called 0 one of the ambiguous values, which stopped
	// being true when the lower bound was widened: the check below is "< 0 || > 1", and a
	// comment naming a different range than the code is how a reader learns the wrong rule.
	p := s.cfg.TopP
	if topP != nil {
		p = *topP
	}
	if p < 0 || p > 1 {
		return nil, 0, newParamError("top_p", "invalid_value",
			"top_p must be between 0 and 1, got %v", p)
	}
	if t == 0 {
		// Greedy is the default and the reproducible choice.
		return greedySampler{}, 0, nil
	}
	// A caller-supplied seed makes the request reproducible, which is the whole point of
	// the parameter: the same prompt and settings give the same text. Leaving it to the
	// clock is what makes two otherwise identical requests differ.
	//
	// seeded rather than "seed != 0": the caller has already decided, and a derived seed for a
	// later choice can legitimately be 0 (seed -1 with two choices derives -1 and 0). Reading
	// 0 here as "pick one" would make that one choice irreproducible while its siblings were
	// not, which is worse than either behaviour on its own. Zero is a usable stream seed; it is
	// only the /request/ field's zero that means "unset", because a JSON number has no way to
	// say absent otherwise.
	if !seeded {
		seed = time.Now().UnixNano()
	}
	return temperatureSampler{temp: float32(t), topK: topK, topP: float32(p), src: newRand(seed)}, float32(t), nil
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

// blocksNeeded is how many KV cache blocks positions positions occupy.
//
// The arithmetic is PagedKVCache.ensureBlocks': position p lives in block p/blockSize, so n
// positions need ceil(n/blockSize) blocks, and a request that generates up to position n-1
// holds exactly that many. Stated as a function and tested against a literal, because
// getting it wrong in either direction turns a permanent failure into a retry loop or
// refuses a request the pool could have served.
//
// blockSize <= 0 returns 0: it means the engine has no pool, and a caller that reached here
// with one has nothing to compare.
func blocksNeeded(positions, blockSize int) int {
	if positions <= 0 || blockSize <= 0 {
		return 0
	}
	return (positions + blockSize - 1) / blockSize
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
	// loopback or authenticating in front, which is what the "No CORS and no authentication"
	// section of docs/usage.md tells a deployer to do.
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
		// A value of the wrong type is not a malformed body: the client wrote valid JSON
		// and named a field, so it can be told which one. Reporting it as invalid_json
		// sends a caller looking for a syntax error in a body that parses, and it is the
		// one failure this path still shared with a genuinely broken document. Measured
		// before this branch existed: {"max_tokens":"8"} answered 400 invalid_json with no
		// param, and an OpenAI SDK surfaces that as a malformed request rather than as
		// "max_tokens must be a number".
		var typeErr *json.UnmarshalTypeError
		if errors.As(err, &typeErr) {
			return &typeMismatchError{typeErr: typeErr}
		}
		// A field's own UnmarshalJSON can already know what to say -- chat content is the
		// case that does -- and it says it in the envelope's terms rather than as a decode
		// failure. Passing it through keeps that answer instead of replacing it with
		// "invalid JSON" and a Go type name.
		var pe *paramError
		if errors.As(err, &pe) {
			return pe
		}
		return fmt.Errorf("invalid JSON: %w", err)
	}
	return nil
}

// typeMismatchError is a request field that arrived as the wrong JSON type.
//
// It exists to keep two facts the decoder knows and the envelope needs: which field, and
// what type it actually got. json.UnmarshalTypeError carries both, but its message is Go's
// -- "cannot unmarshal string into Go struct field chatRequest.max_tokens of type int" --
// which names a struct the client never sent and a type no client thinks in. The message
// here says the same thing in the caller's terms, because the caller is who has to edit
// their request.
type typeMismatchError struct {
	typeErr *json.UnmarshalTypeError
	// field overrides the decoder's field path. It is set by a field with its own
	// UnmarshalJSON, which the decoder hands the value alone: there is no struct field in
	// scope to name, so the type that refused the value supplies its own JSON key.
	field string
	// want overrides the expected type when the field accepts more than one. The decoder
	// reports the Go type it was trying to build at the point it failed, which for a union
	// (chat content, stop strings) is one arm of several rather than the whole rule.
	want string
}

// Error renders the failure the way the caller has to read it: the field, the type it sent
// and the type it should have sent.
//
// One case reads badly and is left that way: a number outside the range of the Go type it was
// being decoded into -- {"max_tokens": 1e20} -- reaches this through the same UnmarshalTypeError
// as a string in a number field, so the message is "max_tokens must be an integer, but it is
// number", which is true and unhelpful. Telling the two apart means parsing the raw token again
// to see whether it is a number at all; the caller still gets a 400 naming the field, and a
// request with a 10^20 token budget is not one with a plausible intended answer.
func (e *typeMismatchError) Error() string {
	if e.param() == "" {
		return fmt.Sprintf("the request body must be a JSON object, but it is %s",
			jsonTypeName(e.typeErr.Value))
	}
	want := e.want
	if want == "" {
		want = goTypeName(e.typeErr.Type)
	}
	return fmt.Sprintf("%s must be %s, but it is %s", e.param(),
		want, jsonTypeName(e.typeErr.Value))
}

// param is the request field at fault, as the path the caller can find in their body.
//
// The decoder builds it from the JSON keys it walked, so it is already the caller's
// vocabulary: "max_tokens" for a top-level field and "messages.0.role" for one inside an
// array element. It is empty only when the body was not an object at all, where there is no
// field to name.
func (e *typeMismatchError) param() string {
	if e.field != "" {
		return e.field
	}
	return e.typeErr.Field
}

// unmarshalTypeFailure builds the refusal for a field that decodes itself.
//
// The decoder's field path is empty on this path, so key and want have to come from the
// field: key is the JSON name the caller wrote, and want is the rule the value broke.
func unmarshalTypeFailure(key, want string, te *json.UnmarshalTypeError) error {
	return &typeMismatchError{typeErr: te, field: key, want: want}
}

// writeTypeMismatch answers a request whose JSON parsed but whose fields did not fit.
//
// The code is OpenAI's own invalid_type, and that is the point of the whole branch: a
// client that branches on `code` could not tell a wrong-typed field from a truncated body,
// and the two have nothing in common to act on -- one is an edit, the other is a resend.
func writeTypeMismatch(w http.ResponseWriter, err *typeMismatchError) {
	writeRequestError(w, http.StatusBadRequest, newParamError(
		err.param(), "invalid_type", "%s", err.Error()))
}

// jsonTypeName says what a JSON value is, in the words the spec uses.
//
// The decoder reports the value as "string", "number 1.5" or "object", so the leading word
// is the type and anything after it is the offending value. Naming the type alone is what
// a caller needs; quoting a long object back at them is not.
func jsonTypeName(value string) string {
	if v, _, ok := strings.Cut(value, " "); ok {
		return v
	}
	return value
}

// goTypeName says what a Go field is declared as, in JSON's words.
//
// The decoder hands back a reflect.Type whose String() is Go syntax: int64, []string,
// *float64. A caller wrote JSON, so the names here are JSON's; a type this server has no
// word for falls back to the Go name rather than inventing one.
func goTypeName(t reflect.Type) string {
	switch t.Kind() {
	case reflect.String:
		return "a string"
	case reflect.Bool:
		return "a boolean"
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return "an integer"
	case reflect.Float32, reflect.Float64:
		return "a number"
	case reflect.Slice, reflect.Array:
		if t.Elem().Kind() == reflect.String {
			return "an array of strings"
		}
		return "an array"
	case reflect.Map:
		return "an object"
	case reflect.Pointer:
		return goTypeName(t.Elem())
	}
	return t.String()
}

// writeDecodeError answers a body this server could not accept.
//
// Three failures with three different meanings, and the status is what a client acts on:
// 400 says the JSON was malformed and retrying the same bytes is pointless, 415 says the
// encoding is unsupported and the client should change that, 413 says the body was too
// large and the request must be made smaller. Reporting all three as 400 would send a
// client with a compressed or oversized body looking for a syntax error.
//
// Two more arrive through here and are answered before those three: a field of the wrong
// JSON type (invalid_type, with the field in param) and a field whose own decoder refused it
// (whatever code that decoder chose). Both name a field the three below cannot, and the
// first used to be flattened into invalid_json, which sent a caller looking for a syntax
// error in a body that parses.
//
// Each case also carries a `code`. This is the writer the paramError comment calls out by
// name -- a client that branches on code rather than message text had nothing to match here,
// so a malformed body, an unsupported encoding and an oversized one all arrived as an uncoded
// error alongside the coded ones the handlers write. The three codes are the ones OpenAI's
// own API uses for the same conditions where it has them.
func writeDecodeError(w http.ResponseWriter, err error) {
	// A body that parsed but carried the wrong type for a field is answered before the
	// window checks below: it has its own code, invalid_type, and the field to name.
	var mismatch *typeMismatchError
	if errors.As(err, &mismatch) {
		writeTypeMismatch(w, mismatch)
		return
	}
	// A field that refused itself wrote its own param and code. Its failure is about that
	// field -- a chat content part this server cannot render, say -- so the answer names it
	// rather than saying the document did not parse.
	var coded *paramError
	if errors.As(err, &coded) {
		writeRequestError(w, http.StatusBadRequest, coded)
		return
	}
	status := http.StatusBadRequest
	code := "invalid_json"
	switch {
	case errors.Is(err, errUnsupportedEncoding):
		status = http.StatusUnsupportedMediaType
		code = "unsupported_encoding"
	case errors.Is(err, errBodyTooLarge):
		status = http.StatusRequestEntityTooLarge
		code = "body_too_large"
	}
	// No `param`: these failures are about the body as a whole, and there is no field to name
	// -- the decoder never reached one.
	writeRequestError(w, status, newParamError("", code, "%s", err.Error()))
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
// the sequence was cut short by something outside the model. The schema has no value for that, so
// any choice is wrong about the cause; "length" at least says the reply is not what the request
// asked for.
//
// It carries no claim about the token budget, and a caller must not read it as one. Two things
// can cancel a sequence: a client that hung up, and a shutdown that gave up draining. The first
// leaves nobody to read the answer; the second leaves a client that is still listening and would
// reasonably conclude it got the whole reply. That is why the streaming path withholds its [DONE]
// sentinel on a cancel -- the sentinel, not this value, is how a cut stream is recognised.
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
	switch reason {
	case "length", "cancel":
		return "length"
	case "tool_calls":
		// Passed through rather than folded into "stop": a client branches on this to decide
		// whether to run a tool, and "stop" tells it the assistant finished speaking.
		return "tool_calls"
	}
	return "stop"
}

// responseID builds the id for one response from a timestamp the caller supplies.
//
// Timestamp first so ids sort by arrival in a log, then the sequence number so two requests
// in the same nanosecond still differ. Neither alone is enough: a bare counter says nothing
// about when, and the bare clock can repeat -- see the idSeq field for the measurement.
//
// The timestamp is a parameter rather than a time.Now call inside so a test can hold it fixed
// and watch the counter do its work. Without that the uniqueness it adds is not observable:
// the HTTP path spaces requests by routing and JSON decoding, far more than a nanosecond, so
// reverting to a bare timestamp still passes a request-level test twenty times over.
func (s *Server) responseID(prefix string, unixNano int64) string {
	return fmt.Sprintf("%s%d-%d", prefix, unixNano, s.idSeq.Add(1))
}

// logitBias converts the request's map into the sampler's list.
//
// The keys are token ids as strings, which is the JSON shape the schema uses because a token id
// is an object key. Two things are refused rather than skipped: a key that is not a non-negative
// integer, and a bias outside OpenAI's documented [-100, 100]. A skipped key would be a ban the
// caller believed in and the server never applied, which is the failure mode this whole path
// exists to avoid.
//
// A bias whose id is past the vocabulary is refused too, for the same reason: the caller has the
// wrong id -- probably a tokenizer's id from another model -- and applying it to nothing is
// indistinguishable from ignoring the request.
func logitBias(raw map[string]float64, vocab int) ([]engine.TokenBias, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	out := make([]engine.TokenBias, 0, len(raw))
	for key, bias := range raw {
		id, err := strconv.ParseInt(key, 10, 32)
		if err != nil || id < 0 {
			return nil, newParamError("logit_bias", "invalid_value",
				"logit_bias keys are token ids written as strings, so %q is not one; send "+
					"{\"50256\": -100} rather than a token's text", key)
		}
		if int(id) >= vocab {
			return nil, newParamError("logit_bias", "invalid_value",
				"logit_bias names token %d but this model's vocabulary ends at %d", id, vocab-1)
		}
		if bias < -100 || bias > 100 {
			return nil, newParamError("logit_bias", "invalid_value",
				"logit_bias values are between -100 and 100, so token %d's %v is outside it",
				id, bias)
		}
		out = append(out, engine.TokenBias{ID: int32(id), Bias: float32(bias)})
	}
	// Sorted so the sampler's work is deterministic: a map's range order is randomised, and
	// two biases on the same id would then be applied in either order -- which matters once
	// float addition is involved, and matters more for reproducing a report.
	slices.SortFunc(out, func(a, b engine.TokenBias) int { return int(a.ID) - int(b.ID) })
	return out, nil
}

// checkPenalty validates one of OpenAI's repetition penalties.
//
// The documented range is [-2, 2] and both ends are legal: a positive value discourages a token
// that has already appeared, a negative one encourages it. A value outside the range is refused
// rather than clamped, because clamping -3 to -2 would answer a request the caller did not make
// with text that looks like it honoured the one they did.
func checkPenalty(v *float64, name string) (float32, error) {
	if v == nil {
		return 0, nil
	}
	if math.IsNaN(*v) || *v < -2 || *v > 2 {
		return 0, newParamError(name, "invalid_value",
			"%s is between -2 and 2, so %v is outside it", name, *v)
	}
	return float32(*v), nil
}

// checkResponseFormat refuses the formats this server cannot honour.
//
// OpenAI's json_object is a promise, not a hint: "ensures the message the model generates is valid
// JSON". Delivering that needs constrained decoding -- every sampling step restricted to tokens that
// keep the output a prefix of a valid JSON document -- and this engine has no grammar layer, only a
// sampler over raw logits. Measured on Gemma 3 270M without one, the reply to "Name two colors as
// JSON" was a markdown fence:
//
//	```json
//	{ "color1": "red", "color2": "blue", ...
//
// which is not JSON, and a client that called json.loads on it fails somewhere far from the request.
// Ignoring the field is what produced that; refusing it names the field and the way out.
//
// The values that describe batching rather than output -- an absent field, {"type": "text"} -- are
// served, because they ask for what the server would do anyway. Same rule the tool fields follow.
func checkResponseFormat(raw json.RawMessage) error {
	if len(raw) == 0 {
		return nil
	}
	var f struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		return newParamError("response_format", "invalid_value",
			"response_format must be an object with a \"type\", so %s is not one", raw)
	}
	switch f.Type {
	case "text":
		// What the server does already.
		return nil
	case "json_object":
		return newParamError("response_format", "unsupported_parameter",
			"response_format json_object promises valid JSON, which needs constrained decoding "+
				"this engine does not implement; asking for JSON in the prompt works without the "+
				"promise, or set response_format {\"type\": \"text\"} explicitly")
	case "json_schema":
		return newParamError("response_format", "unsupported_parameter",
			"response_format json_schema needs constrained decoding this engine does not "+
				"implement; asking for the shape in the prompt works without the promise")
	default:
		return newParamError("response_format", "invalid_value",
			"response_format type %q is not one this server knows; the types are \"text\", "+
				"\"json_object\" and \"json_schema\"", f.Type)
	}
}

// scoreShown is the candidate's mean log probability over the tokens the response reports.
//
// The same correction score makes -- mean rather than sum -- applied to the run after a stop string
// has cut it. A candidate that generated a hundred tokens past its stop is not a better answer than
// one that stopped on the same string after two, and the caller can only see the two.
func (c *choiceResult) scoreShown(shown int) float64 {
	if shown <= 0 || shown > len(c.samples) {
		return math.Inf(-1)
	}
	var sum float64
	for _, s := range c.samples[:shown] {
		sum += float64(s.logprob)
	}
	return sum / float64(shown)
}

// resolvePrompt validates the budget, encodes the prompt and fits both against the window.
//
// ok=false means a response has already been written: every refusal here is about the request,
// and the writer is where the status and the coded error belong. Splitting this out leaves
// generate with the sampling, admission and streaming, which is what its name suggests.
//
// The budget comes back truncated to the window, because that is the same decision: a client
// asking for 512 tokens with 100 left gets the 100 rather than a refusal.
func (s *Server) resolvePrompt(w http.ResponseWriter, p *generateParams) (
	promptTokens []int32, maxTokens int, fimStopIDs []int32, ok bool) {
	cfg := s.eng.Model().Config()
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
		return nil, 0, nil, false
	}
	// The two names for the budget are alternatives, not a pair. OpenAI documents
	// max_tokens as deprecated in favour of max_completion_tokens, so a request carrying
	// both is a caller mid-migration, and the two numbers can disagree -- 3 and 300 are a
	// typo either way round, and choosing one silently answers with a budget the caller may
	// not have meant. Naming both is the one answer that cannot be wrong about which field
	// to fix.
	//
	// The check comes after the negative one so a request with max_tokens: -5 and nothing
	// else is reported for the value it got wrong rather than for a conflict that is not
	// there.
	if p.maxCompletionTokens != nil && *p.maxCompletionTokens < 0 {
		writeRequestError(w, http.StatusBadRequest, newParamError(
			"max_completion_tokens", "invalid_value",
			"max_completion_tokens must not be negative, got %d", *p.maxCompletionTokens))
		return nil, 0, nil, false
	}
	if p.maxCompletionTokens != nil && p.maxTokensGiven {
		writeRequestError(w, http.StatusBadRequest, newParamError(
			"max_completion_tokens", "invalid_value",
			"max_tokens and max_completion_tokens are two spellings of one limit and cannot "+
				"both be set (got %d and %d); send max_completion_tokens alone",
			p.maxTokens, *p.maxCompletionTokens))
		return nil, 0, nil, false
	}
	// Absent, not zero: an omitted budget takes the server's default, while an explicit zero is
	// a budget of zero and is passed through. The engine finishes a sequence whose budget is
	// already spent before sampling anything, so the caller gets an empty completion with
	// finish_reason "length" -- the request it made, rather than the server's default length.
	//
	// This used to be one int, so zero meant both and an explicit zero silently became the
	// default: measured, max_tokens: 0 came back with a full-length paragraph.
	maxTokens = s.cfg.MaxTokens
	switch {
	case p.maxCompletionTokens != nil:
		maxTokens = *p.maxCompletionTokens
	case p.maxTokensGiven:
		maxTokens = p.maxTokens
	}

	// A chat prompt is markup, so its control markers are matched whole; a completions
	// prompt is text the caller wrote, so a literal "<|im_start|>" there stays text.
	// The same distinction the CLI's chat command makes, for the same reason.
	//
	// fimStopIDs are the marker ids the engine stops on, kept separate from p.stop because the
	// markers are control tokens: their text encodes to a run of characters rather than to the
	// one token, so stopTokenIDs cannot find them.
	switch {
	case p.fim != nil:
		promptTokens = s.fimPrompt(p.prompt, p.suffix, *p.fim)
		// The generated middle ends at a marker the checkpoint emits, and which one is the
		// model's business rather than ours. Measured on Qwen2.5-Coder without the closing
		// markers: the correct expression came first and the reply then carried on writing the
		// caller's own following lines, because nothing told it to stop.
		p.stop = append(slices.Clone(p.stop), p.fim.StopStrings()...)
		fimStopIDs = p.fim.StopIDs()
	case p.chat:
		promptTokens = s.tok.EncodeWithSpecials(p.prompt, cfg.BOSID >= 0)
	default:
		promptTokens = s.tok.Encode(p.prompt, cfg.BOSID >= 0)
	}
	if len(promptTokens) == 0 {
		writeError(w, http.StatusBadRequest, "prompt encoded to no tokens")
		return nil, 0, nil, false
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
		return nil, 0, nil, false
	}
	// A request cannot exceed the window. Truncating the budget is the useful behaviour: a
	// client asking for 512 tokens with 100 left should get the 100 rather than a refusal.
	//
	// This is an optimisation rather than the guarantee, and the difference is worth knowing
	// before changing it. The engine stops a sequence at the window too (engine.decode checks
	// s.pos against ContextLen), so a server that skipped this cut would still not generate past
	// the window -- measured, with the cut removed the same request returned 253 tokens and
	// finish_reason "length" either way. What the cut buys is the response's own numbers: usage
	// reports the budget that was actually used, so a caller sees 253 rather than a request for
	// 100000 reported as though it had been honoured.
	//
	// A test cannot distinguish the two implementations for that reason, which is why there is
	// none here: a case asserting completion_tokens == 253 passes with and without the cut.
	if room := s.cfg.ContextLen - len(promptTokens); maxTokens > room {
		maxTokens = room
	}
	return promptTokens, maxTokens, fimStopIDs, true
}

// choiceObject builds one entry of the response's choices array.
//
// It is separated from blockingResponse because everything it does is a decision about the
// response's shape rather than about generation: where a stop string cuts the text, which
// logprob entries survive that cut, whether the prompt is echoed ahead of the answer, and which
// of the two schemas -- chat or completions -- the object takes.
//
// out is the position in the response, not the candidate's original slot: a caller reads
// choices[0] as the best one, and leaving a discarded candidate's number there would name an
// index that no longer exists.
func (s *Server) choiceObject(c *choiceResult, out int, p generateParams) map[string]any {
	text := s.tok.DecodeCompletion(c.out)
	// A stop string is text, and generation halts where it would appear. OpenAI's API
	// takes strings, and a string need not be one token: " Paris" encodes as a space
	// token followed by the word's own. stopTokenIDs can only hand the engine a single
	// token, so the engine-side check cannot express this -- and taking the first token
	// of a multi-token string is actively wrong, since the space token then stops every
	// sequence at its first space and the response comes back empty.
	reason := c.reason
	if cut, found := cutAtStop(text, p.stop); found {
		text = cut
		reason = "stop"
	}
	// A chat reply is parsed here rather than at the end, because the parse decides what the
	// response shows and the logprobs describe that. It runs only when the request declared
	// tools: a model that writes <tool_call> unasked -- a conversation pasted in from
	// elsewhere, or a base model imitating the marker -- would otherwise have its text cut out
	// of a reply nobody asked to be a call.
	var (
		calls []toolCall
		// spans is the byte ranges of text the response shows once calls are cut out, and nil
		// when no call was found -- at which point the whole of text is what is shown and a
		// length budget over it is the right cut.
		spans []proseSpan
	)
	if p.chat && len(p.tools) > 0 {
		if _, parsed, ranges := parseToolCallsSpans(text); len(parsed) > 0 {
			calls, spans = parsed, ranges
		}
	}
	// The logprob lists are cut to the text the response shows, which is the stop-cut text, or
	// the prose once a call has been parsed out of it. The echoed prompt is not part of either:
	// it was not generated, so it has no logprob here, and with echo the offsets move past it
	// because they index into the text the caller receives.
	//
	// A token whose text straddles the cut is dropped rather than half-reported: half of
	// its text survived, and neither keeping it nor truncating it would describe the
	// probability of the thing the caller can see.
	var logprobs any
	if p.logprobs {
		described := s.describeTokens(c.samples)
		described = shownLogprobTokens(described, text, calls, spans)
		if p.chat {
			logprobs = chatLogprobs(described)
		} else {
			offset := 0
			if p.echo {
				offset = len(p.prompt)
			}
			logprobs = completionLogprobs(described, offset)
		}
	}

	// echo prepends the prompt after the stop cut, not before: the cut applies to what
	// the model generated, and a stop string that happens to appear in the prompt must
	// not truncate the part of the answer the caller already typed. The prompt is echoed
	// as the caller sent it rather than re-decoded from its token ids, because that text
	// is exactly what they are asking to see back.
	//
	// The echoed prompt is not counted again in usage: prompt_tokens already covers it,
	// and adding it to completion_tokens as well would bill the same text twice. This is
	// a choice, not a documented OpenAI rule -- the schema does not say -- and it is the
	// one that keeps prompt_tokens + completion_tokens equal to the tokens actually
	// processed.
	if p.echo && !p.chat {
		text = p.prompt + text
	}
	// The index is the position in the response, not the candidate's original slot: a caller
	// reads choices[0] as the best one, and leaving a discarded candidate's number there would
	// name an index that no longer exists.
	if p.chat {
		message := map[string]any{"role": "assistant", "content": text}
		if calls != nil {
			prose, _ := parseToolCalls(text)
			calls = assignToolCallIDs(p.responseID, calls)
			message["content"] = prose
			message["tool_calls"] = toolCallsJSON(calls)
			// finish_reason is tool_calls, which is what the schema reserves for a turn
			// that asks the caller to run something. "stop" would tell a client the
			// assistant finished speaking and nothing is expected of it.
			reason = "tool_calls"
		}
		return map[string]any{
			"index":         out,
			"message":       message,
			"finish_reason": finishReasonFor(reason),
			"logprobs":      logprobs,
		}
	}
	return map[string]any{
		"index":         out,
		"text":          text,
		"finish_reason": finishReasonFor(reason),
		"logprobs":      logprobs,
	}
}

// eventEnvelope is the metadata every server-sent event carries.
//
// It is separate because the same four fields go on every chunk and on the usage chunk, and the
// two were assembled at each site -- which is how a chunk ends up missing its model name.
type eventEnvelope struct {
	id      string
	created int64
}

// eventWriter writes one response's server-sent events.
//
// Headers, the object name and the flush are protocol concerns that have nothing to do with
// which tokens the engine produced, and keeping them here means the two paths that emit events --
// blocking and streaming -- cannot disagree about the shape of a frame.
//
// send returning false means the stream must stop: the client is gone, or a payload could not be
// encoded. The caller cannot tell those apart, and nothing above needs to: both end the response.
type eventWriter struct {
	w            http.ResponseWriter
	flusher      http.Flusher
	model        string
	env          eventEnvelope
	object       string
	includeUsage bool
}

// newEventWriter checks that the connection can stream and writes the response headers.
//
// ok=false means the response has already been written with an error and the caller must return.
// The check is on http.Flusher rather than on the response type because that is the capability a
// stream needs: a writer that cannot flush holds every token until the response ends, which
// arrives as one late body rather than as a stream.
func newEventWriter(w http.ResponseWriter, model string, env eventEnvelope, p generateParams) (*eventWriter, bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, false
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	// nginx buffers proxied responses by default (proxy_buffering on), which holds an SSE
	// stream until the response ends -- the stream still arrives intact, but all at once, and a
	// token stream that arrives at the end is not a token stream. This header is nginx's own
	// per-response switch for that, and nginx will not infer it: its maintainers declined to
	// special-case text/event-stream on the grounds that a media type describes how a payload is
	// formatted, not how it is delivered (nginx/nginx#1505).
	//
	// It costs nothing where it is not understood. A proxy that does not know it ignores it and a
	// direct client has no use for it, so the only reader is nginx, and only to turn buffering
	// off for this one response. The rest of the API keeps the proxy's default, which is what
	// that default is for.
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	// OpenAI's spec is explicit that a completion is "always" text_completion, and its own
	// streaming example carries the same value as the non-streaming one -- the two endpoints
	// differ here, since chat streams object="chat.completion.chunk". A client that switches on
	// this field to tell a delta from a whole body looks for the chunk name it knows.
	object := "text_completion"
	if p.chat {
		object = "chat.completion.chunk"
	}
	return &eventWriter{
		w: w, flusher: flusher, model: model, env: env, object: object,
		includeUsage: p.includeUsage,
	}, true
}

// send writes one event and flushes it.
func (ew *eventWriter) send(payload map[string]any) bool {
	payload["id"] = ew.env.id
	payload["object"] = ew.object
	payload["created"] = ew.env.created
	payload["model"] = ew.model
	if ew.includeUsage {
		// Set on every ordinary chunk, not only the usage one: OpenAI documents the field as
		// present-and-null here, and a client that inspects it on each chunk then needs no
		// branch for "the field is missing".
		if _, ok := payload["usage"]; !ok {
			payload["usage"] = nil
		}
	}
	b, err := json.Marshal(payload)
	if err != nil {
		// Unreachable with the payloads built here -- maps of strings, ints, nils and nested
		// maps of the same, none of which can fail to encode. Kept because a future field that
		// does fail should end the stream rather than write a chunk with a hole in it, and
		// returning false already means "stop streaming".
		return false
	}
	if _, err := fmt.Fprintf(ew.w, "data: %s\n\n", b); err != nil {
		return false
	}
	// A flush error is not checked, and that is deliberate: the request context is what notices
	// a client that left, and it is already passed to the engine as Request.Cancel. Measured
	// over a raw TCP connection that is reset mid-stream, the engine stops within two tokens of
	// the reset -- the cancel fires first, so a flush check would add a second route to the same
	// place.
	//
	// A test that measures this over http.Post and closes the body sees something else: a
	// graceful close leaves the connection usable, no reset is sent, and generation runs to its
	// full budget. That is a property of the client, not of this code, and it is worth knowing
	// before reading a large number as waste.
	ew.flusher.Flush()
	return true
}

// stopGate decides which streamed text may be sent while a stop list is in force.
//
// A stop string is matched against decoded text, so a match can only be recognised once its
// last character arrives -- by which time an earlier character already sent cannot be taken
// back. Measured before this existed: stop=["北京"] streamed "北" where the non-streaming body
// returned nothing, so the two transports disagreed about one request.
//
// The fix is to hold back the longest tail that could still begin a stop string, and to send
// everything before it. This type is that bookkeeping: seen is the text this choice has produced,
// sent is how much of it the client already holds, and push returns what became safe to send.
//
// The zero value is not usable; build one with the stop list, which may be empty.
type stopGate struct {
	stop []string
	// seen is every byte of this choice's text the gate has been given.
	seen string
	// sent is how many bytes of seen the client has. It trails seen by the length of the
	// held-back tail.
	sent int
}

// push adds text and returns what may now be sent, and whether a stop string matched.
//
// On a match the returned text is the part between what the client holds and the cut, and the
// gate is left with nothing more to send: everything after the stop string belongs to nobody.
func (g *stopGate) push(text string) (release string, found bool) {
	g.seen += text
	if kept, hit := cutAtStop(g.seen, g.stop); hit {
		// Only the part between what the client already has and the stop string still needs
		// sending. Reading it after the cut rather than from the increment is what makes the
		// held-back tail from earlier pushes come out here.
		return kept[min(len(kept), g.sent):], true
	}
	// Nothing matched, so the tail that could still become a stop string stays back and
	// everything before it goes out.
	if safe := safePrefixLen(g.seen, g.stop); safe > g.sent {
		out := g.seen[g.sent:safe]
		g.sent = safe
		return out, false
	}
	return "", false
}

// flush returns text still held back when no stop string matched.
//
// It is what makes a stop list that never matches cost nothing but a later chunk: the tail was
// ordinary output all along.
func (g *stopGate) flush() string {
	// No guard against a second call: advancing sent is what makes this idempotent, and a
	// sent >= len(seen) check above it would be dead weight -- measured, removing the guard
	// changes nothing because the slice is empty by then anyway, while removing the
	// assignment below makes a second call repeat the tail.
	out := g.seen[g.sent:]
	g.sent = len(g.seen)
	return out
}
