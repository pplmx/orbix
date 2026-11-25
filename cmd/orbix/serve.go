package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/pplmx/orbix/internal/engine"
	"github.com/pplmx/orbix/internal/llama"
	"github.com/pplmx/orbix/internal/otel"
	"github.com/pplmx/orbix/internal/server"
)

// cmdServe implements "orbix serve": the OpenAI-compatible HTTP server.
// otelResource is the set of attributes describing the process, as opposed to the work it does.
//
// service.name is Required by the resource conventions. Nothing from gen_ai.* belongs here: those
// describe one operation and are set on the span. The model is still recorded, under this
// repository's own prefix, because a collector separating two deployments by checkpoint needs it
// at the process level and no semantic convention offers a resource attribute for it.
func otelResource(version, model string) []otel.Attribute {
	return []otel.Attribute{
		otel.StringAttr("service.name", "orbix"),
		otel.StringAttr("service.version", version),
		otel.StringAttr("orbix.model", model),
	}
}

// newHTTPServer builds the server with the timeouts this service needs.
//
// ReadHeaderTimeout is small because headers are small. IdleTimeout needs an explicit value:
// net/http falls back to ReadTimeout, and ReadTimeout cannot be set here because it bounds the
// whole request including the handler's run, which for a generation that streams for minutes
// would cut the response off mid-flight. Zero for both means an idle keep-alive socket is held
// for the life of the process -- measured, 30 idle connections held 30 descriptors.
//
// WriteTimeout is left unset for the same reason as ReadTimeout: a slow generation is a
// legitimate response, not a slow client.
func newHTTPServer(h http.Handler, idle, readHeader time.Duration) *http.Server {
	return &http.Server{
		Handler:           h,
		ReadHeaderTimeout: readHeader,
		IdleTimeout:       idle,
	}
}

// newHTTPServerWithRequests gives every request a base context the caller can cancel.
//
// Shutdown needs it. http.Server.Shutdown stops accepting and waits for handlers to return, but
// nothing cancels their contexts, so a handler waiting for an engine that will not advance again
// has no reason to give up: it blocks until the process exits and the client receives nothing at
// all. Measured against this server, four non-streaming requests in flight at a 3-second grace
// all came back as curl exit 52, "empty reply from server"; the same four at a 60-second grace
// answered 200.
//
// With a cancellable base context, the handlers see Done, take their cancellation branch, and
// write a 503 the client can act on.
func newHTTPServerWithRequests(h http.Handler, idle, readHeader time.Duration,
	base context.Context) *http.Server {
	srv := newHTTPServer(h, idle, readHeader)
	srv.BaseContext = func(net.Listener) context.Context { return base }
	return srv
}

// accessLogger builds the request log for a --log-format value.
//
// Extracted from cmdServe so the three accepted spellings and the refusal are reachable without
// starting a server: cmdServe's body is a twenty-step setup that blocks on a listener, and none of
// it ran under any test, so a --log-format typo would have been caught by the operator rather than
// by the suite. Returning nil for "off" is the contract the server relies on -- it treats a nil
// Logger as silent rather than installing a discard handler.
func accessLogger(format string) (*slog.Logger, error) {
	switch format {
	case "off":
		return nil, nil
	case "text":
		return slog.New(slog.NewTextHandler(os.Stderr, nil)), nil
	case "json":
		return slog.New(slog.NewJSONHandler(os.Stderr, nil)), nil
	default:
		return nil, fmt.Errorf("serve: --log-format must be %s, not %q", logFormatChoices, format)
	}
}

// The log format is a switch whose default arm is an error, so the flag's initial value has to
// be one of the accepted spellings. Both are named here rather than at the two sites, so the
// help text, the default, and the error cannot drift apart.
const (
	defaultLogFormat = "text"
	logFormatChoices = "text, json, or off"
)

func cmdServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	model := modelFlag(fs)
	addr := fs.String("addr", ":8080", "address to listen on")
	maxTokens := maxTokensFlag(fs, 256)
	maxActive := fs.Int("max-active", 8, "requests advanced per scheduling round")
	maxQueued := fs.Int("max-queued", 0,
		"requests accepted but not yet active before new ones are refused with 503 (0 = 8x --max-active)")
	cache := registerCacheFlags(fs, true)
	prefixCache := fs.Bool("prefix-cache", false, "reuse KV blocks across requests that share a prompt prefix (needs --blocks)")
	otlpEndpoint := fs.String("otlp-endpoint", "", "OTLP/HTTP collector for traces, e.g. http://localhost:4318; empty disables tracing")
	logFormat := fs.String("log-format", defaultLogFormat, "access log format: "+logFormatChoices)
	shutdownGraceFlag := fs.Duration("shutdown-grace", shutdownGrace,
		"how long a SIGTERM or SIGINT waits for in-flight responses before cutting them; "+
			"a cut stream ends without [DONE]")
	policy := fs.String("policy", "fifo", "admission policy when more requests are ready than --max-active: fifo, shortest, or fair")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if err := resolveModel(fs, model); err != nil {
		return err
	}
	// The policy is parsed here rather than after the model is opened: a typo used to cost a
	// full checkpoint load first, and the error then arrived from engine.New instead of naming
	// the flag. ParsePriority does not depend on the model, so there is nothing to gain by
	// checking it late.
	schedPolicy, err := engine.ParsePriority(*policy)
	if err != nil {
		return err
	}
	// Resolved once and used by both the banner and the drain. Reading *shutdownGraceFlag at
	// each site let a mutation of one pass every test: the banner would report the configured
	// value while the drain used the default, and nothing would notice.
	grace := *shutdownGraceFlag
	if err := validateServeFlags(serveFlags{
		model:         *model,
		addr:          *addr,
		maxTokens:     *maxTokens,
		maxActive:     *maxActive,
		maxQueued:     *maxQueued,
		cache:         cache,
		prefixCache:   *prefixCache,
		otlpEndpoint:  *otlpEndpoint,
		logFormat:     *logFormat,
		shutdownGrace: grace,
		policy:        schedPolicy,
	}); err != nil {
		return err
	}
	// One open for the metadata, the tokenizer, the template and the config. This used to open
	// the file twice -- once here to check --block-size against the context, once below for the
	// tokenizer -- and engine.New reached into the loader for a third parse.
	//
	// The context length is needed before the pool is built, which is why the model is opened
	// before the engine: a --block-size wider than the window is refused rather than warned
	// about, like the --blocks check above. Both configurations start a server that serves text,
	// so the only signal would be the line claiming a feature the operator asked for, and a
	// warning under it reads as advice rather than as "this did not happen".
	m, err := openModel(*model)
	if err != nil {
		return err
	}
	plan, err := planCache("serve", m.cfg, cache, false, *prefixCache)
	if err != nil {
		return err
	}
	tok, renderer := m.tok, m.renderer

	eng, err := engine.New(*model, engine.Options{
		MaxActive:   *maxActive,
		MaxQueued:   *maxQueued,
		Policy:      schedPolicy,
		Workers:     *cache.workers,
		BlockSize:   plan.blockSize,
		Blocks:      plan.blocks,
		PrefixCache: *prefixCache,
	})
	if err != nil {
		return err
	}
	cfg := eng.Model().Config()

	// Tracing is opt-in and off by default: a collector is one more thing that can be
	// down, and a server whose traces have nowhere to go should not behave differently.
	var traceExporter *otel.Exporter
	if *otlpEndpoint != "" {
		traceExporter, err = otel.NewExporter(otel.ExporterOptions{
			Endpoint: *otlpEndpoint,
			Resource: otelResource(version, cfg.Name),
		})
		if err != nil {
			return err
		}
		defer func() { _ = traceExporter.Close() }()
	}

	// The access log is opt-in at the format level but on by default in text: a server
	// that says nothing about the requests it served makes a failure indistinguishable
	// from a request that never arrived.
	accessLog, err := accessLogger(*logFormat)
	if err != nil {
		return err
	}

	srv, err := server.New(eng, tok, server.Config{
		ModelName:    cfg.Name,
		MaxTokens:    *maxTokens,
		ContextLen:   cfg.ContextLen,
		Temperature:  0,
		ChatRenderer: renderer,
		// The banner below names the fallback whenever the probe found one, so the server does
		// not repeat it per process. A template that renders the probe and then fails a real
		// conversation is still unannounced, and the server warns for that.
		ChatFallbackAnnounced: !renderer.UsesModelTemplate(),
		Trace:                 traceExporter,
		Logger:                accessLog,
	})
	if err != nil {
		return err
	}

	ln, err := net.Listen("tcp", *addr)
	if err != nil {
		return err
	}
	// The listener's real address is what to print: :0 means the kernel picked a
	// port, and a caller (or a test) needs the one it got.
	url := "http://" + ln.Addr().String()

	// Requests get a context of their own so a drain that runs out of grace can cancel them.
	// It is separate from the pump's context, which has to keep the engine advancing until the
	// in-flight responses are done with it.
	reqCtx, cancelRequests := context.WithCancel(context.Background())
	defer cancelRequests()
	httpSrv := newHTTPServerWithRequests(srv.Handler(), 120*time.Second, 10*time.Second, reqCtx)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	srv.StartPump(ctx)

	fmt.Printf("orbix serving %s on %s\n", cfg, url)
	queued := defaultQueued(*maxQueued, *maxActive)
	fmt.Printf("scheduling: %d active, %d queued max, %s policy\n", *maxActive, queued, schedPolicy)
	fmt.Printf("%s", shutdownNote(grace))
	// The weights, before any cache exists. LoadWeights widens every tensor to float32 once,
	// so this figure follows the parameter count and not the file size -- and it is usually
	// the largest number an operator has to plan around. Nothing else prints it: the model
	// summary above gives no count, and `run --stats` reports params only to a caller who asked
	// for stats on a one-shot generation. On Llama 3.2 1B it is 4.6 GiB, which a 553 MiB file
	// gives no hint of.
	fmt.Printf("weights:    %s resident (4 bytes per parameter, widened from the file at load)\n",
		humanBytes(weightsResidentBytes(eng.Model().W)))
	// Say what the contiguous cache costs, which nothing else reports. A paged cache prints
	// its pool size below from the pool itself; a contiguous one reserves a whole context per
	// accepted request, so the backlog multiplies whatever one request costs and the model
	// file size says nothing about it. For a small model that is a few MB; for a 4k-context
	// model with a wide KV it is hundreds, and the operator reading this at startup is the one
	// who can still change the flags.
	if *cache.blocks == 0 {
		fmt.Printf("%s\n", contiguousMemoryNote(cfg, *maxActive, queued))
		// The number above is virtual, not resident: the arenas are one make per layer and
		// Linux only backs a page when it is first written, so a server can look healthy
		// while holding a reservation it could never fault in. Saying so is the difference
		// between an operator who changes the flags now and one who finds out when a burst
		// of long requests drives the process into the OOM killer.
		if note := reservationWarning(cfg, *maxActive, queued); note != "" {
			fmt.Printf("%s\n", note)
		}
	}
	// Report which chat format is in use. A fallback is not an error -- the server
	// still answers -- but an operator who sees oddly shaped replies should be able to
	// tell whether the model's own format was applied.
	if traceExporter != nil {
		fmt.Printf("traces:     %s (OTLP/HTTP)\n", *otlpEndpoint)
	}
	if accessLog != nil {
		// stderr is named because it is not where this banner goes. A deployment that captures
		// only stdout -- which several container platforms do by default -- gets the banner and
		// none of the access log, and this line is the only place the operator can learn that
		// from the server itself. Measured: redirecting only stdout produced six banner lines
		// and zero request records.
		fmt.Printf("access log: %s (on stderr, not stdout)\n", *logFormat)
	}
	if renderer.UsesModelTemplate() {
		fmt.Println("chat:       model template")
	} else {
		fmt.Printf("chat:       neutral transcript (%s)\n", renderer.FallbackReason())
	}
	// The paged cache's lines belong with the contiguous one's, above the address: everything
	// before "POST" describes what was configured, and that line announces the server is
	// ready. Printing the pool after it put two configuration figures -- and, worse, the
	// capacity warning -- below the line a reader takes as "nothing more follows", which is
	// how a warning that only appears at startup gets missed. The contiguous branch above
	// already had this order; only the paged one was split.
	if pool := eng.Pool(); pool != nil {
		fmt.Printf("kv cache:   paged, %d blocks x %d positions, %s shared\n",
			pool.Blocks(), pool.BlockSize(), humanBytes(pool.Bytes()))
		if pool.PrefixIndex() != nil {
			fmt.Println("prefix cache: on (shared prompt prefixes reuse their KV blocks)")
		}
		if note := poolReservationWarning(pool.Bytes()); note != "" {
			fmt.Printf("%s\n", note)
		}
		if note := poolCapacityWarning(pool.Blocks()*pool.BlockSize(), eng.Model().Config().ContextLen); note != "" {
			fmt.Printf("%s\n", note)
		}
	}
	fmt.Printf("POST %s/v1/chat/completions\n", url)

	// A signal ends the server the way a caller would: stop accepting new requests, let the
	// ones in flight finish, then stop the pump and return.
	//
	// The order matters and used to be wrong. cancel() stops the pump, and the pump is what
	// advances every active sequence -- including the ones behind a streaming response. So
	// cancelling first and asking http.Server to drain afterwards left in-flight streams with
	// nothing to produce their next token: each one blocked until the grace period ran out,
	// the client got a truncated response, and the process exited with "context deadline
	// exceeded" for what was an ordinary Ctrl-C. Measured before the fix: a stream that had
	// produced 92 chunks was held for the full 5s and then cut.
	//
	// Now the drain comes first, which is what the grace period is for, and the pump is
	// stopped only once no response is waiting on it.
	//
	// The select below is the one part of this command without a test, and it stays that way
	// rather than being restructured for one.
	//
	// The two cases look like two ways in, but only the signal is reachable on its own. The error
	// case needs Serve to return before the signal arrives, and the ways Serve returns early are
	// shutting the server down -- which only the signal case does before this point -- or the
	// listener breaking under it. The listener is made earlier in this function from *addr and never
	// leaves this function, so nothing can close it from outside. Measured with a standalone
	// httptest-shaped program: closing the listener under Serve returns "use of closed network
	// connection", not ErrServerClosed, so that path would reach the return err below rather than
	// the nil return -- which is why it is worth saying the listener is unreachable rather than
	// only that Shutdown comes later. So the entry point is a process signal, and signal.Notify is
	// process-wide: a test that sent SIGTERM would end the test binary. Making it testable means
	// taking the signal channel as a parameter, which is a change to the command for the sake of
	// its test. What the signal case does -- drain first, stop the pump after -- is covered
	// through drainAndStop, which has its own tests and full statement coverage.
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	errCh := make(chan error, 1)
	go func() { errCh <- httpSrv.Serve(ln) }()

	select {
	case err := <-errCh:
		cancel()
		srv.Shutdown()
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			return err
		}
		return nil
	case <-sig:
		fmt.Println("\nshutting down")
		return drainAndStop(httpSrv, cancelRequests, func() {
			cancel()
			srv.Shutdown()
		}, grace)
	}
}

// shutdownGrace is the default bound on how long an in-flight response may delay a shutdown.
//
// It is a default rather than a constant because five seconds is not long enough for a real
// generation. Measured on this engine: Llama 3.2 1B produces 64 tokens in 5.04 s and 128 in
// 10.4 s; TinyLlama 1.1B needs 5.17 s for 64; Qwen2.5-0.5B needs 3.41 s. So an ordinary reply
// from a modest model already sits at or past the grace, and a rolling restart will cut it.
// The comment here used to say the value was "long enough that ordinary replies finish", which
// those numbers contradict.
//
// `--shutdown-grace` lets a deployment choose: a few seconds for a service whose clients retry,
// minutes for one that would rather finish than restart fast.
const shutdownGrace = 5 * time.Second

// shutdownSettle bounds how long a cancelled handler gets to write its response after a drain
// has expired. Writing a status is not work that takes time; the bound is here so a handler that
// ignores its cancellation cannot hold the process open.
const shutdownSettle = 2 * time.Second

// shutdownNote reports the effective drain grace as a banner line.
//
// The line exists so an operator can see that --shutdown-grace took effect without waiting for a
// signal, and it is the only observable form of the wiring: the flag reaches drainAndStop
// through one expression that no test can reach, and drainAndStop prints the grace only when it
// expires. Pulling the text into a function gives that value something to assert on.
//
// It names the consequence rather than only the number. "5s" alone does not tell a reader what
// happens at five seconds; "then they are cut" does, and it is the fact a streaming client sees.
func shutdownNote(grace time.Duration) string {
	if grace == 0 {
		// Zero is a special case worth spelling out: "0s grace, then they are cut" reads like a
		// mistake rather than a choice.
		return "shutdown:  configured not to wait; in-flight responses are cut immediately\n"
	}
	return fmt.Sprintf("shutdown:  %s grace for in-flight responses, then they are cut\n", grace)
}

// drainAndStop stops an HTTP server and the engine pump behind it, in that order.
//
// It is a function rather than four inline calls because the order is the whole point and
// was wrong: the pump advances every active sequence, so stopping it before the drain leaves
// each in-flight streaming response with nothing to produce its next token. The response
// then blocks until the grace period expires, the client gets a truncated stream, and the
// process reports "context deadline exceeded" for an ordinary Ctrl-C. Measured before the
// fix: a stream that had emitted 92 chunks was held for the full 5s and then cut; after it,
// the same stream finished intact in 1.9s and the client waited 0.4s.
//
// A response that outlives the grace period is the documented behaviour of a bounded drain,
// not a failure: the signal asked for a shutdown within the grace and got one. Saying so
// beats an "error:" line that reads like a crash, and the exit code stays success because
// nothing failed.
//
// The drain's own return value is otherwise passed through: a listener failure is worth
// reporting.
func drainAndStop(httpSrv *http.Server, cancelRequests, stopPump func(), grace time.Duration) error {
	ctx, done := context.WithTimeout(context.Background(), grace)
	defer done()
	err := httpSrv.Shutdown(ctx)
	// Whatever is still running now has no reason to finish on its own, so it is told to stop
	// and given a moment to write its answer. Shutdown has already given up on waiting, but the
	// connections are still open: a handler that returns within this window still reaches the
	// client, which is the difference between a 503 and a connection that dies silently.
	cancelRequests()
	settleHandlers(httpSrv)
	// The pump stops after the drain, whichever way the drain ended: a stream that outlived
	// the grace period was cut by Shutdown, and leaving the pump running would keep advancing
	// a sequence nobody can read.
	stopPump()
	if err != nil && errors.Is(err, context.DeadlineExceeded) {
		fmt.Printf("shutdown: a response was still in flight when the %s grace expired; it was cut\n", grace)
		return nil
	}
	return err
}

// settleHandlers waits for handlers that were cancelled to write their responses.
//
// A second Shutdown is the wait: the first one returned when its context expired, but it closed
// the listener and the connections are still open, so a handler that returns now still reaches
// its client. Calling Shutdown again blocks until every handler has returned, and the bounded
// context keeps a handler that ignores its cancellation from holding the process open.
//
// The alternative was to let cmdServe return, which exits the process and takes the pending
// writes with it -- measured as curl exit 52, "empty reply from server", for four requests in
// flight at a 3-second grace.
func settleHandlers(httpSrv *http.Server) {
	ctx, done := context.WithTimeout(context.Background(), shutdownSettle)
	defer done()
	// The error is the deadline expiring, which is this function's own bound rather than a
	// failure: a handler that has still not returned by then is one nothing can help.
	_ = httpSrv.Shutdown(ctx)
}

// contiguousMemoryNote describes the KV footprint of the contiguous cache, which is the mode
// where the reservation is not obvious from a flag.
func contiguousMemoryNote(cfg llama.Config, maxActive, queued int) string {
	// A sequence reserves its whole context the moment it is accepted, so the worst case is
	// every slot plus every queued request holding one.
	perRequest := int64(cfg.ContextLen) * int64(cfg.KVHeads) * int64(cfg.HeadDim) * 2 * 4 * int64(cfg.Layers)
	total := perRequest * int64(maxActive+queued)
	return fmt.Sprintf("kv cache:   contiguous, about %s per request, %s for %d active + %d queued",
		humanBytes(perRequest), humanBytes(total), maxActive, queued)
}

// poolReservationWarning says when a block pool reserves more than a machine should hand out
// casually, and returns "" when it does not.
//
// The pool allocates its whole backing store up front -- one slice per layer covering every
// block -- so the reservation is the block count times the per-block cost, whether or not the
// blocks are ever used. A block count that reads as a round number can therefore ask the
// kernel for tens of gigabytes, and the same virtual-not-resident caveat applies as for the
// contiguous cache: resident memory stays low until the blocks are touched.
func poolReservationWarning(reserved int64) string {
	if reserved <= 1<<30 {
		return ""
	}
	return fmt.Sprintf("warning:    the pool reserves all %s up front; a smaller --blocks lowers "+
		"it, and Linux commits pages only as blocks are used, so resident memory will be lower "+
		"until the pool fills.", humanBytes(reserved))
}

// poolCapacityWarning reports a pool whose total capacity is smaller than one request's
// context, which means no request can run to completion.
//
// The pool report above already prints the block count and the positions per block, so the
// shortage is derivable -- but it is not visible, and the failure arrives per request as a
// refusal. Measured with --blocks 1 --block-size 1 against a 128-position model: the server
// started, printed "paged, 1 blocks x 1 positions", accepted the connection, and refused
// every prompt longer than one token. A config that cannot serve anything should say so at
// startup, which is where the operator is still looking.
//
// The status is 400, not 503. It used to say 503, and when that refusal became a 400 for
// requests whose size can never be met, this line kept advertising the old one -- the
// warning was the only place a reader saw a status, and TestPoolCapacityWarning asserted
// "503" because that is what the code produced. Both moved together in that change; the
// wording now names the status the server actually sends.
//
// This mirrors poolReservationWarning in the other direction: that one warns about a pool too
// large to be free, this one about a pool too small to be useful.
func poolCapacityWarning(capacity, contextLen int) string {
	if capacity >= contextLen || contextLen <= 0 {
		return ""
	}
	return fmt.Sprintf("warning:    the pool holds %d positions but one request needs up to %d, "+
		"so requests longer than %d are refused with 400 (they can never fit, so retrying "+
		"does not help); raise --blocks or --block-size",
		capacity, contextLen, capacity)
}

// weightsResidentBytes is what the loaded weights occupy, and it is exactly 4 bytes per
// parameter.
//
// The name says "resident" rather than "size" because those differ here: a 553 MiB Llama 3.2
// file holds 4.6 GiB once every tensor has been widened to float32, and the file size predicts
// neither this figure nor the peak the process reaches. Params counts every float32 the model
// holds -- including all of a mixture-of-experts model's experts, not just the ones a token
// routes to -- so multiplying by the width of a float32 is the whole calculation.
//
// A function rather than an inline `Params() * 4` so the arithmetic has a name a test can call.
// It was inline for one round, and the banner line it fed had no test at all.
func weightsResidentBytes(w *llama.Weights) int64 {
	return w.Params() * 4
}

// humanBytes renders a byte count the way an operator reads it: whole kibibytes below a
// mebibyte, one decimal place above that. The one-decimal rule applies at every scale above a
// mebibyte including gibibytes -- an earlier version of this comment said "whole megabytes above a
// gigabyte", which the code has never done.
func humanBytes(n int64) string {
	// The divisors are powers of 1024 and the labels say so: MiB, not MB. bench.go already
	// labels the same arithmetic with MiB, and calling a binary megabyte a megabyte makes two
	// reports of the same pool disagree by the 4.9% between them.
	const (
		mib = 1 << 20
		gib = 1 << 30
	)
	switch {
	case n >= gib:
		return fmt.Sprintf("%.1f GiB", float64(n)/gib)
	case n >= mib:
		return fmt.Sprintf("%.1f MiB", float64(n)/mib)
	default:
		return fmt.Sprintf("%d KiB", n/(1<<10))
	}
}

// reservationWarning says when the contiguous reservation is large enough that the operator
// should choose a paged cache instead, and returns "" when it is not worth mentioning.
//
// The threshold is deliberately a round number rather than a fraction of the machine: the
// process cannot know how much memory it may use, and the point is to start the
// conversation, not to make the decision. A gibibyte is where the reservation stops being
// noise on a shared host.
func reservationWarning(cfg llama.Config, maxActive, queued int) string {
	const warnAbove = 1 << 30
	perRequest := int64(cfg.ContextLen) * int64(cfg.KVHeads) * int64(cfg.HeadDim) * 2 * 4 * int64(cfg.Layers)
	if perRequest*int64(maxActive+queued) < warnAbove {
		return ""
	}
	return fmt.Sprintf(
		"warning:    this reservation is virtual, and Linux only commits a page when it is " +
			"written, so resident memory will look smaller than the figure above until long " +
			"requests touch the whole context. Use --blocks and --block-size to share one " +
			"pool instead of reserving a context per request.")
}

// defaultQueued is how many requests may wait when --max-queued is unset.
//
// Eight times the active slots rather than a fixed number because the useful backlog scales with the
// service: a server admitting 4 at a time wants a different depth from one admitting 64, and a
// constant would either starve the small one's retries or let the large one accept work it will not
// reach for minutes. The multiplication is the whole rule, so it is stated once and tested rather
// than living in the middle of the startup banner where nothing could reach it.
func defaultQueued(maxQueued, maxActive int) int {
	if maxQueued > 0 {
		return maxQueued
	}
	return 8 * maxActive
}

// serveFlags is what cmdServe assembled from its flags, before anything is loaded.
//
// It exists so the checks and the assembly are separate: a reader following what serve does with
// a request no longer steps over a hundred lines of flag validation, and the checks have one exit
// point instead of eight.
type serveFlags struct {
	model         string
	addr          string
	maxTokens     int
	maxActive     int
	maxQueued     int
	cache         cacheFlags
	prefixCache   bool
	otlpEndpoint  string
	logFormat     string
	shutdownGrace time.Duration
	policy        engine.Priority
}

// validateServeFlags checks every flag that can be wrong before anything is loaded.
//
// Every check is deliberately ahead of openModel. A typo in --addr or --otlp-endpoint used to
// cost a full checkpoint load first -- measured at 3.2 seconds on a 2 GB Qwen2 file -- and the
// error then arrived from net.Listen rather than naming the flag. Each comment below records
// what the check was measured doing before it existed.
func validateServeFlags(v serveFlags) error {
	if err := v.cache.validate("serve", false); err != nil {
		return err
	}
	if v.maxActive < 1 {
		return fmt.Errorf("serve: --max-active must be at least 1")
	}
	// Prefix reuse lives in the paged cache: a contiguous arena has no block index to
	// attach a previous request's KV to. Accepting the flag without --blocks silently ran a
	// contiguous server with no reuse, and the startup log said "kv cache: contiguous" with
	// no mention of the flag at all -- so an operator who passed it believed reuse was on.
	// A malformed endpoint does not fail the exporter: it is built, every send fails, and the
	// spans are dropped into the counter. That is the right behaviour for a collector that is
	// down -- the server must keep serving -- but not for a typo, which would silently disable
	// the tracing the operator just asked for. Checked here with the other flags, before the
	// model is opened, so a bad value is reported without loading 8 GB first.
	if v.otlpEndpoint != "" {
		if u, uerr := url.Parse(v.otlpEndpoint); uerr != nil || u.Host == "" ||
			(u.Scheme != "http" && u.Scheme != "https") {
			return fmt.Errorf("serve: --otlp-endpoint %q is not an http(s) URL with a host",
				v.otlpEndpoint)
		}
	}
	// A negative grace cannot be a bounded drain: context.WithTimeout with a negative duration
	// is already expired, so every shutdown would cut every response while claiming to drain.
	// Zero is allowed and means "do not wait", which is a real choice for a process whose
	// requests are already being retried elsewhere.
	if v.shutdownGrace < 0 {
		return fmt.Errorf("serve: --shutdown-grace must not be negative (0 means do not wait)")
	}
	// The listen address is checked here, with the other flags, for the reason --otlp-endpoint
	// is: net.Listen runs after engine.New, so a typo in --addr used to cost a full model load
	// before it was reported. Measured on the 2 GB Qwen2 file, `--addr not-an-addr` spent 3.2
	// seconds loading before failing.
	//
	// Only the syntax is checked, not availability: whether the port is free is a fact that can
	// change between this check and the listen, and binding early to find out would hold the
	// socket through the load. A malformed value is a different thing -- it cannot become valid
	// by waiting.
	//
	// The same ordering has a second consequence worth stating, because docs/usage.md promises
	// it: because nothing listens until the engine exists, a successful connection to /healthz
	// means the weights are resident. A readiness probe can rely on that. Measured on the
	// 553 MiB Llama 3.2 checkpoint: 6.3 s from process start to the first 200. It also means a
	// model that fails to load exits before any socket exists, rather than serving an unhealthy
	// instance -- measured, a missing file and a bad magic both exit 1 with the port still
	// closed.
	// ResolveTCPAddr alone is not enough: it accepts a service name, so "127.0.0.1:http"
	// resolves to port 80 and passes, then fails at net.Listen after the load. A port this
	// server binds is a number, and requiring one is what makes the check cover the whole
	// flag rather than the part Go's parser happens to reject.
	if _, err := net.ResolveTCPAddr("tcp", v.addr); err != nil {
		return fmt.Errorf("serve: --addr %q is not a host:port to listen on: %w", v.addr, err)
	}
	if _, port, err := net.SplitHostPort(v.addr); err == nil {
		if _, perr := strconv.Atoi(port); perr != nil {
			return fmt.Errorf("serve: --addr %q has a port that is not a number (%q); "+
				"a service name would be resolved before the listen, which is not what this "+
				"flag says", v.addr, port)
		}
	}
	// A negative backlog is not a smaller backlog: the reporting below treats anything <= 0
	// as "unset" and falls back to 8x --max-active, so -1 silently became 64 here. A typo
	// that produces a valid-looking server is worse than one that stops at the flag.
	if v.maxQueued < 0 {
		return fmt.Errorf("serve: --max-queued must not be negative (0 means 8x --max-active)")
	}
	// The budget reaches server.Config, which replaces anything <= 0 with 256. So a negative
	// value here is not "generate nothing" -- it is a server that hands every request which
	// omits max_tokens the 256-token default, and reports the reply as ordinary. Measured
	// against the 256-position fixture: `serve --max-tokens -5` started, answered a request
	// with no max_tokens field, and returned 253 tokens with finish_reason "length".
	//
	// Zero is refused for the same reason, and this is where serve's meaning of zero differs
	// from run's. run accepts 0 because it generates once into its own output, where "prefill
	// only" is observable. This flag is not a per-request budget: it is the default for
	// requests that omit max_tokens, and server.Config reads anything below 1 as 256. Accepting
	// 0 would therefore configure a 256-token default under a flag that says 0, which is the
	// silent substitution this check exists to prevent.
	//
	// A request that sends max_tokens: 0 is unaffected and is served: the field is a pointer on
	// the wire, so an explicit zero is distinguishable from an absent one, and that request
	// comes back empty with finish_reason "length". The message below said the server could not
	// express a zero-token request, which was wrong about the protocol -- measured, an explicit
	// zero has always been served -- and the reason for refusing the flag is the default, not
	// expressibility.
	if v.maxTokens < 1 {
		return fmt.Errorf("serve: --max-tokens must be at least 1; it sets the default for " +
			"requests that omit max_tokens, and 0 there would quietly become the 256-token " +
			"default rather than a zero-token reply")
	}
	// The policy name is parsed here rather than after the model is opened, where it used to
	// be: a typo cost a full checkpoint load first, and the error then arrived from
	// engine.New rather than naming the flag. ParsePriority has no dependency on the model, so
	// there is nothing to gain by checking it late; schedPolicy below is reused as-is.
	if v.prefixCache && (*v.cache.blocks) == 0 {
		return fmt.Errorf("serve: --prefix-cache needs a paged cache; add --blocks (and " +
			"optionally --block-size), or drop --prefix-cache")
	}
	return nil
}
