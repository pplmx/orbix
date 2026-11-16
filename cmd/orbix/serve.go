package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/pplmx/orbix/internal/engine"
	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/server"
	"github.com/pplmx/orbix/internal/tokenize"
)

// cmdServe implements "orbix serve": the OpenAI-compatible HTTP server.
func cmdServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	model := fs.String("model", "", "path to a GGUF model file (required)")
	addr := fs.String("addr", ":8080", "address to listen on")
	maxTokens := fs.Int("max-tokens", 256, "default generation budget per request")
	maxActive := fs.Int("max-active", 8, "requests advanced per scheduling round")
	workers := fs.Int("workers", 0, "kernel worker goroutines; 0 means one per CPU")
	blockSize := fs.Int("block-size", 0, "KV cache block size in positions; 0 keeps the contiguous cache")
	blocks := fs.Int("blocks", 0, "KV cache blocks shared by all requests; implies a paged cache")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *model == "" {
		if v := os.Getenv("ORBIX_MODEL"); v != "" {
			*model = v
		} else {
			return fmt.Errorf("serve: --model is required (or set ORBIX_MODEL)")
		}
	}
	if *maxActive < 1 {
		return fmt.Errorf("serve: --max-active must be at least 1")
	}

	gf, err := gguf.Open(*model)
	if err != nil {
		return err
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		return err
	}
	eng, err := engine.New(*model, engine.Options{
		MaxActive: *maxActive,
		Workers:   *workers,
		BlockSize: *blockSize,
		Blocks:    *blocks,
	})
	if err != nil {
		return err
	}
	cfg := eng.Model().Config()

	srv, err := server.New(eng, tok, server.Config{
		ModelName:   cfg.Name,
		MaxTokens:   *maxTokens,
		ContextLen:  cfg.ContextLen,
		Temperature: 0,
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
	go srv.Run(ctx)

	fmt.Printf("orbix serving %s on %s\n", cfg, url)
	fmt.Printf("POST %s/v1/chat/completions\n", url)
	if pool := eng.Pool(); pool != nil {
		fmt.Printf("kv cache: paged, %d blocks x %d positions = %.2f MiB shared\n",
			pool.Blocks(), pool.BlockSize(), float64(pool.Bytes())/(1<<20))
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
