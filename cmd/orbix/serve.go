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
	// --block-size only sets how the pool's blocks are divided; without --blocks there is no
	// pool to divide. Accepting it alone left the contiguous cache in place, which on a
	// 128k-context model reserves 8 GB per accepted request -- and the operator who passed a
	// block size read the resulting "kv cache: contiguous" line as the size they asked for.
	if *cache.blockSize > 0 && *cache.blocks == 0 {
		return fmt.Errorf("serve: --block-size needs --blocks; without a pool there are no " +
			"blocks to size, and each request still reserves a whole context")
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
			Resource: []otel.Attribute{
				otel.StringAttr("service.name", "orbix"),
				otel.StringAttr("service.version", version),
				otel.StringAttr("gen_ai.request.model", cfg.Name),
			},
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

	httpSrv := &http.Server{
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
	}

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
	}

	// A signal ends the server the way a caller would: stop accepting, stop
	// pumping, return. The 5s grace is enough for an in-flight stream to finish
	// its current token, which is all a local engine needs.
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
		cancel()
		shutdownCtx, done := context.WithTimeout(context.Background(), 5*time.Second)
		defer done()
		err := httpSrv.Shutdown(shutdownCtx)
		srv.Shutdown()
		return err
	}
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
