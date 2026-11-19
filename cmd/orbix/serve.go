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
	"syscall"
	"time"

	"github.com/pplmx/orbix/internal/engine"
	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/llama"
	"github.com/pplmx/orbix/internal/otel"
	"github.com/pplmx/orbix/internal/server"
	"github.com/pplmx/orbix/internal/tokenize"
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
	logFormat := fs.String("log-format", "text", "access log format: text, json, or off")
	policy := fs.String("policy", "fifo", "admission policy when more requests are ready than --max-active: fifo, shortest, or fair")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if err := resolveModel(fs, model); err != nil {
		return err
	}
	if err := cache.validate("serve"); err != nil {
		return err
	}
	if *maxActive < 1 {
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
	if *otlpEndpoint != "" {
		if u, uerr := url.Parse(*otlpEndpoint); uerr != nil || u.Host == "" ||
			(u.Scheme != "http" && u.Scheme != "https") {
			return fmt.Errorf("serve: --otlp-endpoint %q is not an http(s) URL with a host",
				*otlpEndpoint)
		}
	}
	// A negative backlog is not a smaller backlog: the reporting below treats anything <= 0
	// as "unset" and falls back to 8x --max-active, so -1 silently became 64 here. A typo
	// that produces a valid-looking server is worse than one that stops at the flag.
	if *maxQueued < 0 {
		return fmt.Errorf("serve: --max-queued must not be negative (0 means 8x --max-active)")
	}
	if *prefixCache && *cache.blocks == 0 {
		return fmt.Errorf("serve: --prefix-cache needs a paged cache; add --blocks (and " +
			"optionally --block-size), or drop --prefix-cache")
	}

	gf, err := gguf.Open(*model)
	if err != nil {
		return err
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		return err
	}
	schedPolicy, err := engine.ParsePriority(*policy)
	if err != nil {
		return err
	}
	eng, err := engine.New(*model, engine.Options{
		MaxActive:   *maxActive,
		MaxQueued:   *maxQueued,
		Policy:      schedPolicy,
		Workers:     *cache.workers,
		BlockSize:   *cache.blockSize,
		Blocks:      *cache.blocks,
		PrefixCache: *prefixCache,
	})
	if err != nil {
		return err
	}
	cfg := eng.Model().Config()

	renderer, err := engine.NewChatRenderer(gf)
	if err != nil {
		return err
	}

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
	var accessLog *slog.Logger
	switch *logFormat {
	case "off":
	case "text":
		accessLog = slog.New(slog.NewTextHandler(os.Stderr, nil))
	case "json":
		accessLog = slog.New(slog.NewJSONHandler(os.Stderr, nil))
	default:
		return fmt.Errorf("serve: --log-format must be text, json, or off, not %q", *logFormat)
	}

	srv, err := server.New(eng, tok, server.Config{
		ModelName:    cfg.Name,
		MaxTokens:    *maxTokens,
		ContextLen:   cfg.ContextLen,
		Temperature:  0,
		ChatRenderer: renderer,
		Trace:        traceExporter,
		Logger:       accessLog,
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

	httpSrv := newHTTPServer(srv.Handler(), 120*time.Second, 10*time.Second)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	srv.StartPump(ctx)

	fmt.Printf("orbix serving %s on %s\n", cfg, url)
	queued := *maxQueued
	if queued <= 0 {
		queued = 8 * *maxActive
	}
	fmt.Printf("scheduling: %d active, %d queued max, %s policy\n", *maxActive, queued, schedPolicy)
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
		fmt.Printf("access log: %s\n", *logFormat)
	}
	if renderer.UsesModelTemplate() {
		fmt.Println("chat:       model template")
	} else {
		fmt.Printf("chat:       neutral transcript (%s)\n", renderer.FallbackReason())
	}
	fmt.Printf("POST %s/v1/chat/completions\n", url)
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
		return drainAndStop(httpSrv, func() {
			cancel()
			srv.Shutdown()
		}, shutdownGrace)
	}
}

// shutdownGrace bounds how long an in-flight response may delay a shutdown.
const shutdownGrace = 5 * time.Second

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
func drainAndStop(httpSrv *http.Server, stopPump func(), grace time.Duration) error {
	ctx, done := context.WithTimeout(context.Background(), grace)
	defer done()
	err := httpSrv.Shutdown(ctx)
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
// 503 naming "block pool exhausted". Measured with --blocks 1 --block-size 1 against a
// 128-position model: the server started, printed "paged, 1 blocks x 1 positions", accepted
// the connection, and refused every prompt longer than one token. A config that cannot serve
// anything should say so at startup, which is where the operator is still looking.
//
// This mirrors poolReservationWarning in the other direction: that one warns about a pool too
// large to be free, this one about a pool too small to be useful.
func poolCapacityWarning(capacity, contextLen int) string {
	if capacity >= contextLen || contextLen <= 0 {
		return ""
	}
	return fmt.Sprintf("warning:    the pool holds %d positions but one request needs up to %d, "+
		"so requests longer than %d will be refused with 503; raise --blocks or --block-size",
		capacity, contextLen, capacity)
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
