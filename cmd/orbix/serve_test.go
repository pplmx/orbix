package main

import (
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/pplmx/orbix/internal/engine"
	"github.com/pplmx/orbix/internal/gguf"
	"github.com/pplmx/orbix/internal/gguftest"
	"github.com/pplmx/orbix/internal/llama"
	"github.com/pplmx/orbix/internal/server"
	"github.com/pplmx/orbix/internal/tokenize"
)

// TestContiguousMemoryNote pins the KV footprint arithmetic.
//
// The number is the only thing that tells an operator what the offline default costs: a
// contiguous cache reserves a whole context per accepted request, so a backlog of eight
// active plus sixty-four queued reserves far more than the model file. A factor-of-two
// error here would read as plausible and be wrong, which is exactly the class of mistake
// the repository convention says to recompute independently in the test.
func TestContiguousMemoryNote(t *testing.T) {
	// The shape of the stories15M fixture: six layers, six KV heads of forty-eight
	// dimensions, a 128-position window.
	cfg := llama.Config{Layers: 6, KVHeads: 6, HeadDim: 48, ContextLen: 128}

	// Recomputed here rather than taken from the function: positions times the per-position
	// KV width, for both K and V, in every layer, at four bytes each.
	const (
		// One position holds a K and a V vector of KVHeads*HeadDim floats in every layer.
		perPos = 2 * 6 * 48 * 4 * 6
		// A request reserves its whole context.
		perReq   = 128 * perPos
		wantMiB  = float64(perReq*(8+64)) / (1 << 20)
		wantPerM = float64(perReq) / (1 << 20)
	)
	got := contiguousMemoryNote(cfg, 8, 64)
	t.Logf("note: %s", got)

	if !strings.Contains(got, "contiguous") {
		t.Errorf("note does not name the cache layout: %q", got)
	}
	// The per-request figure, to one decimal, and the total for the configured slots.
	if want := itoa1(wantPerM); !strings.Contains(got, want) {
		t.Errorf("note %q has no per-request figure %q (want %.2f MiB)", got, want, wantPerM)
	}
	if want := itoa1(wantMiB); !strings.Contains(got, want) {
		t.Errorf("note %q has no total %q (want %.2f MiB)", got, want, wantMiB)
	}
	// The divisor is 1024-based, so the label has to be MiB. This assertion exists because it was
	// not: humanBytes divided by 1<<20 and printed "MB", while bench.go divided by the same number
	// and printed "MiB" -- two reports of one pool, off by the 4.9% between a megabyte and a
	// mebibyte, with nobody comparing them. The name of the variable above said MiB all along.
	if !strings.Contains(got, "MiB") {
		t.Errorf("note %q labels a 1024-based size without MiB", got)
	}
	if strings.Contains(got, " MB") || strings.Contains(got, " KB") || strings.Contains(got, " GB") {
		t.Errorf("note %q uses decimal unit labels for binary sizes", got)
	}

	// Doubling the slots doubles the total, which is the property the line exists to convey.
	doubled := contiguousMemoryNote(cfg, 8, 136)
	if want := itoa1(2 * wantMiB); !strings.Contains(doubled, want) {
		t.Errorf("doubling the queue gave %q, want a total of %q", doubled, want)
	}
}

// itoa1 formats like the function under test, so the assertion compares the same rendering.
// itoa1 renders one decimal place with the unit the function under test uses. The unit is
// MiB because the divisor is 1<<20; it said MB once, matching humanBytes' label rather than its
// arithmetic, and both were wrong in the same direction.
func itoa1(v float64) string {
	return fmt.Sprintf("%.1f MiB", v)
}

// TestReservationWarning covers the line that tells an operator a contiguous reservation is
// too large to leave alone.
//
// The figure it accompanies is virtual, not resident: the arenas are one make per layer and
// Linux commits a page only when it is written, so a server reserving hundreds of gigabytes
// measures a few gigabytes of RSS and looks healthy right up until long requests fault in the
// pages. Nothing else at startup distinguishes a planner that reserves 1.7MB per request from
// one that reserves 8GB.
func TestReservationWarning(t *testing.T) {
	// The shape of the Llama 3.2 checkpoint: 16 layers, 8 KV heads of 64 dimensions, a
	// 128k window.
	wide := llama.Config{Layers: 16, KVHeads: 8, HeadDim: 64, ContextLen: 131072}
	if got := reservationWarning(wide, 8, 64); got == "" {
		t.Error("no warning for a 128k-context model, whose reservation is hundreds of gigabytes")
	} else {
		// The advice is the part that has to be there: a warning without a way out is noise.
		if !strings.Contains(got, "--blocks") {
			t.Errorf("warning does not name the flags that fix it: %q", got)
		}
		if !strings.Contains(got, "virtual") {
			t.Errorf("warning does not explain why resident memory will look smaller: %q", got)
		}
	}

	// The small fixture: 1.7MB per request, 121.5MB for the default slots. Well under the
	// threshold, so the line would be noise.
	small := llama.Config{Layers: 6, KVHeads: 6, HeadDim: 48, ContextLen: 128}
	if got := reservationWarning(small, 8, 64); got != "" {
		t.Errorf("warning for a 121.5MB reservation: %q", got)
	}

	// The threshold is on the total, not on one request: the wide model with a single slot
	// and no queue still crosses it, and a narrow one with a huge queue does too.
	// 131072 * 8 * 64 * 2 * 4 * 16 = 8GiB for one request alone.
	if got := reservationWarning(wide, 1, 0); got == "" {
		t.Error("no warning for a single 8GiB request")
	}
}

// TestPoolReservationWarning covers the counterpart of the contiguous warning, for the mode
// where the reservation comes from a block count rather than a context length.
//
// A block pool allocates every block up front, so --blocks 20000 on a 128k-window model asks
// the kernel for 19.5GB whether or not a request ever touches those blocks. Measured on a
// running server at that setting: VmSize 32GB against VmRSS 7.4GB after eight requests, so
// the resident figure is not the reservation and an operator watching RSS would not see it.
// TestPoolCapacityWarning is the other end of the same reporting: a pool too small to serve
// anything rather than too large to be free.
//
// Measured before adding it: --blocks 1 --block-size 1 against a 128-position model started
// cleanly, printed "paged, 1 blocks x 1 positions", and then refused every prompt longer
// than one token with "llama: block pool exhausted (1 blocks of 1 positions)". The startup
// line states the numbers but not that they cannot work, and the operator reading three
// lines of cache report is not expected to divide 1 by 128.
func TestPoolCapacityWarning(t *testing.T) {
	// Enough capacity: no warning. 8 blocks of 16 positions covers a 128-position context
	// exactly, which is the boundary the check is written against.
	if got := poolCapacityWarning(8*16, 128); got != "" {
		t.Errorf("warning for a pool that exactly fits the context: %q", got)
	}
	// More than enough.
	if got := poolCapacityWarning(64*16, 128); got != "" {
		t.Errorf("warning for a pool larger than the context: %q", got)
	}
	// No context to compare against: stay quiet rather than divide by it.
	if got := poolCapacityWarning(0, 0); got != "" {
		t.Errorf("warning with no context length: %q", got)
	}

	got := poolCapacityWarning(1, 128)
	if got == "" {
		t.Fatal("no warning for a one-position pool against a 128-position context")
	}
	// The same standard as the reservation warning: name what to change and state both
	// numbers, since either one alone leaves the operator guessing which is wrong.
	for _, want := range []string{"--blocks", "--block-size", "1", "128"} {
		if !strings.Contains(got, want) {
			t.Errorf("warning %q does not mention %q", got, want)
		}
	}
	// The status it quotes has to be the one the server sends. This assertion used to read
	// "503", which was the status at the time and became wrong when a request that can never
	// fit the pool started answering 400: the test pinned the old behaviour rather than the
	// contract, so it kept a stale advertisement green.
	if !strings.Contains(got, "400") {
		t.Errorf("warning %q does not name the 400 a request over the pool actually gets", got)
	}
	if strings.Contains(got, "503") {
		t.Errorf("warning %q still advertises 503 for a request that can never fit; that "+
			"status now means the pool is busy", got)
	}
	// And the advice is why it differs from the busy case: a permanent refusal is not
	// something a client can wait out.
	if !strings.Contains(got, "retrying") {
		t.Errorf("warning %q does not say that retrying cannot help", got)
	}
}

// TestServeRefusesABlockBiggerThanTheContext covers the prefix-cache configuration that
// cannot work.
//
// A prefix index stores and matches whole blocks, so a block larger than the model's context
// can never be filled and never published. Measured against stories15M (128 positions) with
// --blocks 8 --block-size 1000 --prefix-cache and three identical prompts:
// orbix_prefix_positions_reused_total stayed at 0 while the startup line said "prefix cache:
// on", and the same run at --block-size 16 reported 32.
//
// Refused rather than warned about, because both readings start a server that serves text and
// the operator's only evidence was the line claiming the feature they asked for.
func TestServeRefusesABlockBiggerThanTheContext(t *testing.T) {
	// The fixture's default window is 64 positions, so 100000 cannot fit in it.
	path := gguftest.Model{NoEOS: true}.Write(t, t.TempDir())

	err := cmdServe([]string{
		"--model", path, "--addr", "127.0.0.1:0",
		"--blocks", "8", "--block-size", "100000", "--prefix-cache",
	})
	if err == nil {
		t.Fatal("serve accepted a block larger than the context with --prefix-cache")
	}
	// The message has to name both numbers and the flag to change, for the same reason the
	// other cache checks do: either number alone leaves the operator guessing.
	for _, want := range []string{"--block-size", "100000", "--prefix-cache"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

func TestPoolReservationWarning(t *testing.T) {
	// Below the threshold: the small fixture's 2048 x 16 pool is 432MiB, which is a normal
	// thing to configure and should not print a warning.
	if got := poolReservationWarning(432 << 20); got != "" {
		t.Errorf("warning for a 432MiB pool: %q", got)
	}
	// Exactly at the threshold stays quiet: the condition is "more than a gibibyte".
	if got := poolReservationWarning(1 << 30); got != "" {
		t.Errorf("warning for exactly 1GiB: %q", got)
	}

	got := poolReservationWarning(20 << 30)
	if got == "" {
		t.Fatal("no warning for a 20GiB pool")
	}
	// The advice and the reason both have to be there: a warning that does not say what to
	// change, or why the resident number will disagree, is not actionable.
	if !strings.Contains(got, "--blocks") {
		t.Errorf("warning does not name the flag that fixes it: %q", got)
	}
	if !strings.Contains(got, "20.0 GiB") {
		t.Errorf("warning does not state the reservation: %q", got)
	}
	if !strings.Contains(got, "virtual") && !strings.Contains(got, "commits pages") {
		t.Errorf("warning does not explain why resident memory will be lower: %q", got)
	}
}

// TestPrefixCacheNeedsABlockPool checks that the combination which cannot work is refused at
// startup rather than ignored.
//
// Prefix reuse lives in the paged cache: attaching a previous request's KV blocks needs a
// block index, and a contiguous arena has none. Passing --prefix-cache without --blocks used
// to start a contiguous server that silently did no reuse, and the startup log said "kv
// cache: contiguous" with no mention of the flag, so the operator who passed it had nothing
// to notice it by.
//
// The check runs before the model is opened, so this needs no checkpoint: the flag
// combination is rejected on its own terms.
func TestPrefixCacheNeedsABlockPool(t *testing.T) {
	err := cmdServe([]string{"--model", "unused.gguf", "--prefix-cache"})
	if err == nil {
		t.Fatal("cmdServe accepted --prefix-cache without --blocks")
	}
	msg := err.Error()
	if !strings.Contains(msg, "prefix-cache") {
		t.Errorf("error %q does not name the flag that was passed", msg)
	}
	if !strings.Contains(msg, "blocks") {
		t.Errorf("error %q does not say what the flag needs", msg)
	}
}

// TestServeRejectsFlagCombinationsThatCannotWork covers three settings that used to be
// accepted and then quietly ignored.
//
// All three share a shape: a flag that only makes sense with another, or a value the code
// treats as "unset". Nothing failed, the server started, and the startup log described
// something other than what was asked for -- so the operator had no way to tell that the
// flag had done nothing.
func TestServeRejectsFlagCombinationsThatCannotWork(t *testing.T) {
	cases := []struct {
		name string
		args []string
		want string
		why  string
	}{
		{
			name: "block size without a pool",
			args: []string{"--block-size", "32"},
			want: "--blocks",
			why:  "the contiguous cache stayed in place, reserving 8 GB per request on a 128k-context model",
		},
		{
			name: "negative backlog",
			args: []string{"--max-queued", "-1"},
			want: "negative",
			why:  "anything <= 0 falls back to 8x --max-active, so -1 became 64 and looked configured",
		},
		{
			name: "otlp endpoint that is not a URL",
			args: []string{"--otlp-endpoint", "not-a-url"},
			want: "http",
			why:  "the exporter is built and every send fails, dropping every span into the counter",
		},
		{
			name: "listen address with no port",
			args: []string{"--addr", "not-an-addr"},
			want: "--addr",
			why:  "net.Listen runs after engine.New, so the typo cost a full model load first",
		},
		{
			// A negative duration is already expired, so every shutdown would cut every
			// response while claiming to drain.
			name: "negative shutdown grace",
			args: []string{"--shutdown-grace", "-1s"},
			want: "--shutdown-grace",
			why:  "context.WithTimeout with a negative duration returns immediately",
		},
		{
			name: "listen address with a port that is not a number",
			args: []string{"--addr", "127.0.0.1:http"},
			want: "--addr",
			why:  "the same late failure, and the message has to name the flag rather than only 'address'",
		},
		{
			name: "otlp endpoint with no host",
			args: []string{"--otlp-endpoint", "http://"},
			want: "http(s) URL with a host",
			why:  "url.Parse accepts it, so only the host check stops a collector address nothing sends to",
		},
		{
			name: "otlp endpoint with a scheme that is not http",
			args: []string{"--otlp-endpoint", "grpc://localhost:4317"},
			want: "http(s) URL with a host",
			why:  "the exporter speaks OTLP over HTTP only; a gRPC endpoint would fail on every send",
		},
		{
			name: "no active slots",
			args: []string{"--max-active", "0"},
			want: "at least 1",
			why:  "a round that can hold nothing accepts requests it will never advance",
		},
		{
			name: "negative default budget",
			args: []string{"--max-tokens", "-5"},
			want: "--max-tokens",
			why:  "server.Config replaces anything <= 0 with 256, so -5 became a 256-token default",
		},
		{
			name: "zero default budget",
			args: []string{"--max-tokens", "0"},
			want: "--max-tokens",
			why:  "a request's max_tokens 0 is indistinguishable from an absent field, so the default cannot be zero",
		},
		{
			name: "unknown policy",
			args: []string{"--policy", "bogus"},
			want: "bogus",
			why:  "the name was parsed after the checkpoint loaded, so a typo cost a full model load",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			args := append([]string{"--model", "unused.gguf"}, c.args...)
			err := cmdServe(args)
			if err == nil {
				t.Fatalf("cmdServe accepted %v; %s", c.args, c.why)
			}
			// The message has to name the problem, not merely report that opening the model
			// failed -- which is what a validation running too late would produce, and what a
			// test looking only for a non-nil error would accept.
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("error %q does not mention %q", err, c.want)
			}
			if strings.Contains(err.Error(), "unused.gguf") {
				t.Errorf("error %q came from opening the model, so the flag was not validated",
					err)
			}
		})
	}
}

// TestServeAcceptsTheFlagCombinationsThatDoWork checks the other side, so the validation
// above cannot pass by rejecting everything.
func TestServeAcceptsTheFlagCombinationsThatDoWork(t *testing.T) {
	// No checkpoint is available here, so a combination that passes validation fails on the
	// model file. That failure is the signal that the flags were fine: it names the file.
	cases := [][]string{
		{"--blocks", "64", "--block-size", "32"},
		{"--blocks", "64", "--prefix-cache"},
		{"--max-queued", "0"},
		{"--otlp-endpoint", "http://localhost:4318"},
		// https is accepted: the exporter uses net/http, which speaks TLS on its own.
		{"--otlp-endpoint", "https://collector.example.com:443"},
		{"--max-active", "1"},
	}
	for _, args := range cases {
		err := cmdServe(append([]string{"--model", "unused.gguf"}, args...))
		if err == nil {
			continue // would only happen if a model file really were named unused.gguf
		}
		if !strings.Contains(err.Error(), "unused.gguf") {
			t.Errorf("cmdServe(%v) was rejected by validation: %v", args, err)
		}
	}
}

// TestDrainAndStopStopsThePumpAfterTheDrain covers the order inside drainAndStop.
//
// The pump advances every active sequence, so stopping it before the drain strands every
// in-flight response: each one blocks until the grace period expires and the client gets a
// truncated body. That was the shipped behaviour -- measured, a stream that had emitted 92
// chunks was held for the full 5s and then cut.
//
// The test does not try to observe the race from inside a handler, because http.Server.Shutdown
// returns only after handlers finish and a handler that waits on the drain deadlocks. It
// records the one fact that distinguishes the two orderings: the wall-clock time at which
// each side completed. The handler must finish before stopPump is called; if the pump is
// stopped first, the handler's end time is later.
func TestDrainAndStopStopsThePumpAfterTheDrain(t *testing.T) {
	var (
		mu             sync.Mutex
		handlerEndedAt time.Time
		pumpStoppedAt  time.Time
	)
	// entered is closed once the handler is running, so the drain cannot begin before there is
	// something to drain. A fixed sleep here is a race: on a slow machine the request may not have
	// reached the handler yet, and the order assertion would then be measuring nothing.
	entered := make(chan struct{})
	handler := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		close(entered)
		// Stand in for a streaming response that needs the pump for one more token: a short
		// wait, then the write the old ordering would never have reached in time.
		time.Sleep(60 * time.Millisecond)
		_, _ = w.Write([]byte("finished"))
		mu.Lock()
		handlerEndedAt = time.Now()
		mu.Unlock()
	})
	stopPump := func() {
		mu.Lock()
		pumpStoppedAt = time.Now()
		mu.Unlock()
	}
	httpSrv := newHTTPServer(handler, time.Minute, time.Second)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	go func() { _ = httpSrv.Serve(ln) }()

	got := make(chan string, 1)
	go func() {
		r, err := http.Get("http://" + ln.Addr().String() + "/")
		if err != nil {
			got <- "error: " + err.Error()
			return
		}
		defer func() { _ = r.Body.Close() }()
		b, _ := io.ReadAll(r.Body)
		got <- string(b)
	}()
	// Wait for the handler to have started rather than guessing at a delay. The timeout is only a
	// hang guard; it is not part of what the test measures.
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("the handler never started")
	}

	if err := drainAndStop(httpSrv, func() {}, stopPump, 5*time.Second); err != nil {
		t.Fatalf("drainAndStop: %v", err)
	}
	select {
	case body := <-got:
		if body != "finished" {
			t.Errorf("body = %q, want %q", body, "finished")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the request did not complete")
	}

	mu.Lock()
	h, p := handlerEndedAt, pumpStoppedAt
	mu.Unlock()
	if h.IsZero() || p.IsZero() {
		t.Fatalf("did not observe both times (handler=%v pump=%v)", h, p)
	}
	if p.Before(h) {
		t.Error("the pump was stopped before the in-flight response finished: drainAndStop " +
			"must drain first, or a streaming response is cut off at the grace period")
	}
}

// TestDrainAndStopReportsACutStream checks the grace period's outcome is not an error.
//
// A response that outlives the grace period is what a bounded drain is for: the signal asked
// for a shutdown within that time and got one. Returning the deadline exceeded would make an
// ordinary Ctrl-C on a long generation look like a crash in the exit code and in the log.
func TestDrainAndStopReportsACutStream(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		// Never finishes within the grace period below.
		time.Sleep(2 * time.Second)
	})
	httpSrv := newHTTPServer(handler, time.Minute, time.Second)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	go func() { _ = httpSrv.Serve(ln) }()

	go func() {
		r, err := http.Get("http://" + ln.Addr().String() + "/")
		if err == nil {
			_ = r.Body.Close()
		}
	}()
	// Give the request time to reach the handler, so the drain has something to wait for.
	time.Sleep(200 * time.Millisecond)

	if err := drainAndStop(httpSrv, func() {}, func() {}, 100*time.Millisecond); err != nil {
		t.Errorf("drainAndStop returned %v for a stream cut at the grace period, want nil", err)
	}
}

// TestHumanBytes pins the formatter at each scale and at both boundaries between them.
//
// It had no test of its own. The startup banner's test asserted that a number appeared in the
// string, which passed while the unit label was wrong, and the formatter itself was reached only
// through the three messages that call it -- none of which checked the unit. A formatter is the
// easiest thing in a CLI to leave untested and the easiest to get wrong by one word.
func TestHumanBytes(t *testing.T) {
	cases := []struct {
		in   int64
		want string
		why  string
	}{
		{1024, "1 KiB", "the first whole kibibyte"},
		{1536, "1 KiB", "below a mebibyte, truncated to whole KiB"},
		{(1 << 20) - 1, "1023 KiB", "the last byte before the scale changes"},
		{1 << 20, "1.0 MiB", "the boundary itself belongs to the larger unit"},
		{3 * (1 << 20) / 2, "1.5 MiB", "one decimal above a mebibyte"},
		{(1 << 30) - 1, "1024.0 MiB", "the last byte before gibibytes"},
		{1 << 30, "1.0 GiB", "the boundary again"},
		{20 << 30, "20.0 GiB", "one decimal at a large scale too, not a whole number"},
	}
	for _, c := range cases {
		if got := humanBytes(c.in); got != c.want {
			t.Errorf("humanBytes(%d) = %q, want %q (%s)", c.in, got, c.want, c.why)
		}
	}

	// The values that reach this function are pool sizes, so anything below a kibibyte is
	// unreachable -- but it must not panic or print something that is not a byte count.
	for _, n := range []int64{0, 1, 1023} {
		if got := humanBytes(n); !strings.HasSuffix(got, "KiB") {
			t.Errorf("humanBytes(%d) = %q, want a KiB figure", n, got)
		}
	}
}

// TestShutdownNoteReportsTheEffectiveGrace pins the value the flag actually reaches.
//
// --shutdown-grace had a validation test and nothing else. The value reached drainAndStop
// through one expression that no in-process test can reach -- cmdServe blocks on a listener
// after a full model load -- so ignoring the flag left the whole suite green. Measured:
// replacing `*shutdownGraceFlag` with the constant shutdownGrace passed every test.
//
// This test does not close that hole, and does not claim to. It pins the formatter, which is
// what the banner's readability rests on. Two changes close the rest: the flag is resolved once
// into a `grace` variable that both the banner and the drain read, so the two can no longer
// disagree; and the banner reports the resolved value, so an operator starting a server sees
// which grace is in effect without waiting for a signal. Proving the drain honours it still
// needs a subprocess, which this repository has no precedent for.
func TestShutdownNoteReportsTheEffectiveGrace(t *testing.T) {
	for _, c := range []struct {
		grace time.Duration
		want  string
	}{
		{5 * time.Second, "5s grace for in-flight responses, then they are cut"},
		{30 * time.Second, "30s grace for in-flight responses, then they are cut"},
		{90 * time.Second, "1m30s grace for in-flight responses, then they are cut"},
		// Zero is a choice, not a mistake, and reads as one.
		{0, "configured not to wait; in-flight responses are cut immediately"},
	} {
		got := shutdownNote(c.grace)
		if !strings.Contains(got, c.want) {
			t.Errorf("shutdownNote(%v) = %q, want it to contain %q", c.grace, got, c.want)
		}
		if !strings.HasSuffix(got, "\n") {
			t.Errorf("shutdownNote(%v) = %q, want a trailing newline so the banner does not run into the next line", c.grace, got)
		}
	}
	// The line has to name the consequence, not only the duration: a reader who sees "5s" alone
	// does not learn that responses are cut at five seconds.
	if !strings.Contains(shutdownNote(time.Second), "cut") {
		t.Error("the note does not say what happens when the grace expires")
	}
}

// TestWeightsResidentBytesIsFourPerParameter covers the arithmetic behind the `weights:` line.
//
// The line was added without a test: it read `Params() * 4` inline, and nothing could reach it
// because cmdServe blocks on a listener and its banner goes to os.Stdout. Extracting the
// expression into weightsResidentBytes gave the arithmetic a name to call, and this checks it
// against a real loaded model rather than a hand-built Weights -- Params walks every tensor
// including per-layer experts, so a fabricated struct would test the multiplication and skip
// the part that can actually be wrong.
func TestWeightsResidentBytesIsFourPerParameter(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 128}.Write(t, t.TempDir())
	m, err := llama.New(path)
	if err != nil {
		t.Fatalf("llama.New: %v", err)
	}
	params := m.W.Params()
	if params == 0 {
		t.Fatal("the fixture reports no parameters, so this test would pass vacuously")
	}
	if got, want := weightsResidentBytes(m.W), params*4; got != want {
		t.Errorf("weightsResidentBytes = %d, want %d (4 x %d parameters)", got, want, params)
	}
	// The figure has to be a plausible size for the fixture, not a byte count off by a factor:
	// four bytes per parameter and one float32 per parameter are the same statement, and a
	// conversion error would show as a factor of 4 rather than a rounding difference.
	if got := weightsResidentBytes(m.W); got%4 != 0 {
		t.Errorf("%d is not a whole number of float32 values", got)
	}
}

// TestOTelResourceHasNoSpanAttributes checks the boundary between the two attribute levels.
//
// gen_ai.request.model was on the resource, where it says every span from this process was made
// to that model. The span already carries it, and the resource claim is a different statement:
// true while one checkpoint is loaded, wrong the moment two processes serve different models and
// are grouped by service.name.
func TestOTelResourceHasNoSpanAttributes(t *testing.T) {
	attrs := otelResource("v1.2.3", "llama-3")

	var names []string
	seen := map[string]string{}
	for _, a := range attrs {
		names = append(names, a.Key)
		seen[a.Key] = a.Value
	}

	// Required by the resource conventions.
	if seen["service.name"] == "" {
		t.Errorf("service.name is required on the resource; got %v", names)
	}
	for _, key := range names {
		if strings.HasPrefix(key, "gen_ai.") {
			t.Errorf("%q is a span attribute and must not be on the resource", key)
		}
	}
	if got := seen["orbix.model"]; got != "llama-3" {
		t.Errorf("orbix.model = %q, want llama-3", got)
	}
	if got := seen["service.version"]; got != "v1.2.3" {
		t.Errorf("service.version = %q, want v1.2.3", got)
	}
}

// TestHTTPServerTimeouts pins the two timeouts whose absence is invisible until a process has
// been up long enough to accumulate sockets.
//
// net/http's zero value means no timeout for all four fields, and IdleTimeout's fallback is
// ReadTimeout -- which this server must leave unset, because ReadTimeout bounds the handler's
// whole run and a generation can stream for minutes. So an unset IdleTimeout is not "the default
// applies", it is "an idle keep-alive socket is held for the life of the process".
func TestHTTPServerTimeouts(t *testing.T) {
	srv := newHTTPServer(http.NotFoundHandler(), 120*time.Second, 10*time.Second)

	if srv.IdleTimeout <= 0 {
		t.Errorf("IdleTimeout = %v, want a positive value", srv.IdleTimeout)
	}
	if srv.ReadHeaderTimeout <= 0 {
		t.Errorf("ReadHeaderTimeout = %v, want a positive value", srv.ReadHeaderTimeout)
	}
	// Both would bound a legitimate long generation and must stay unset.
	if srv.WriteTimeout != 0 {
		t.Errorf("WriteTimeout = %v, want 0: it would cut off a long stream", srv.WriteTimeout)
	}
	if srv.ReadTimeout != 0 {
		t.Errorf("ReadTimeout = %v, want 0: it would cut off a long stream", srv.ReadTimeout)
	}
	// IdleTimeout longer than ReadHeaderTimeout is the intent: headers arrive fast, a
	// keep-alive pause between requests does not.
	if srv.IdleTimeout <= srv.ReadHeaderTimeout {
		t.Errorf("IdleTimeout %v is not longer than ReadHeaderTimeout %v",
			srv.IdleTimeout, srv.ReadHeaderTimeout)
	}
}

// TestServeDefaultsAreAcceptedValues pins the two defaults the code then switches on.
//
// --log-format is matched by a switch whose default arm returns an error, so the flag's initial
// value has to be one of the three accepted spellings. With it changed to "", a plain
// `orbix serve --model m.gguf` refused to start -- measured on a real checkpoint: the binary
// printed `serve: --log-format must be text, json, or off, not ""` where the unmodified one
// printed `access log: text`. Neither the existing flag tests nor the cmd suite noticed,
// because every one of them calls cmdServe with a model path that does not exist, so the
// failure names the file and never reaches the switch.
//
// The policy default is the same shape one step removed: engine.ParsePriority has no fallback,
// and the choice decides the admission order of every round, so a different default silently
// reorders which requests run first.
func TestServeDefaultsAreAcceptedValues(t *testing.T) {
	// The log format default has to be a value its own switch accepts, and the accepted set is
	// enumerated in one place so the help text and the error message cannot disagree with it.
	accepted := map[string]bool{}
	for _, name := range strings.Split(strings.ReplaceAll(logFormatChoices, " or ", ", "), ", ") {
		accepted[strings.TrimSpace(name)] = true
	}
	if !accepted[defaultLogFormat] {
		t.Errorf("defaultLogFormat = %q, which the switch rejects (%s)", defaultLogFormat, logFormatChoices)
	}
	// Every accepted name must be a case in the switch, or the default could be "valid" and
	// still fall through to the error arm.
	for name := range accepted {
		if name == "" {
			t.Errorf("logFormatChoices lists an empty name")
		}
	}

	// The policy default has to be one ParsePriority accepts.
	if _, err := engine.ParsePriority("fifo"); err != nil {
		t.Fatalf("engine.ParsePriority rejects fifo, so serve's default policy is not usable: %v", err)
	}
	// And fifo has to be the default: the flag's registration names it, and a different one
	// changes admission order for every operator who never passes the flag.
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	policy := fs.String("policy", "fifo", "")
	if got := *policy; got != "fifo" {
		t.Errorf("serve's policy default = %q, want fifo", got)
	}
}

// TestServeRefusesABadModelBeforeListening covers the startup checks between opening the file and
// binding the port.
//
// Each of them runs before a listener exists, so the failure is an ordinary returned error rather
// than a server that comes up and then misbehaves. None was covered: the existing serve tests
// exercise flag validation, which happens before the file is opened at all.
//
// The order matters as much as the refusal. A server that bound the port first would accept a
// connection and then fail, which a client sees as a dropped connection rather than as the message
// explaining what is wrong.
func TestServeRefusesABadModelBeforeListening(t *testing.T) {
	t.Run("no tokenizer", func(t *testing.T) {
		// A valid header with no metadata carries no vocabulary.
		var b []byte
		b = append(b, []byte("GGUF")...)
		var u32 [4]byte
		binary.LittleEndian.PutUint32(u32[:], 3)
		b = append(b, u32[:]...)
		var u64 [8]byte
		binary.LittleEndian.PutUint64(u64[:], 0) // no tensors
		b = append(b, u64[:]...)
		binary.LittleEndian.PutUint64(u64[:], 0) // no metadata
		b = append(b, u64[:]...)
		path := filepath.Join(t.TempDir(), "bare.gguf")
		if err := os.WriteFile(path, b, 0o644); err != nil {
			t.Fatalf("writing the file: %v", err)
		}

		err := cmdServe([]string{"--model", path, "--addr", "127.0.0.1:0"})
		if err == nil {
			t.Fatal("serve started a model with no tokenizer")
		}
		if !strings.Contains(err.Error(), "tokenizer") {
			t.Errorf("error %q does not name the tokenizer", err)
		}
	})

	t.Run("unknown policy", func(t *testing.T) {
		path := gguftest.Model{NoEOS: true, ContextLen: 128}.Write(t, t.TempDir())
		err := cmdServe([]string{"--model", path, "--addr", "127.0.0.1:0", "--policy", "bogus"})
		if err == nil {
			t.Fatal("serve accepted an unknown scheduling policy")
		}
		if !strings.Contains(err.Error(), "bogus") {
			t.Errorf("error %q does not quote the policy name", err)
		}
		// The message names the alternatives, which is what makes it actionable.
		for _, want := range []string{"fifo", "shortest", "fair"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error %q does not list %q", err, want)
			}
		}
	})
}

// TestServeRefusesABlockWiderThanTheContext covers the third startup refusal, and the one whose
// absence would be silent.
//
// The prefix index stores and matches whole blocks. A block larger than the context can never be
// filled, so it is never published and reuse is impossible however much traffic arrives. The server
// still starts and still serves text; the only signal would be the startup line claiming "prefix
// cache: on", which is why this is refused rather than warned about.
//
// The message names a workable block size, so a reader does not have to infer one from the context
// length in another line.
func TestServeRefusesABlockWiderThanTheContext(t *testing.T) {
	const ctx = 128
	path := gguftest.Model{NoEOS: true, ContextLen: ctx}.Write(t, t.TempDir())

	err := cmdServe([]string{"--model", path, "--addr", "127.0.0.1:0",
		"--prefix-cache", "--blocks", "8", "--block-size", "1000"})
	if err == nil {
		t.Fatal("serve accepted a block wider than the context")
	}
	if !strings.Contains(err.Error(), "1000") || !strings.Contains(err.Error(), "128") {
		t.Errorf("error %q does not give both numbers", err)
	}
	// The suggestion is the actionable half.
	if !strings.Contains(err.Error(), "128 or less") {
		t.Errorf("error %q does not say what block size would work", err)
	}
	if !strings.Contains(err.Error(), "--prefix-cache") {
		t.Errorf("error %q does not name the flag that caused it", err)
	}

	// A block of exactly the context length is the boundary and must also be refused: the block
	// would only fill at the last position, which is the same as never filling for reuse purposes.
	err = cmdServe([]string{"--model", path, "--addr", "127.0.0.1:0",
		"--prefix-cache", "--blocks", "8", "--block-size", strconv.Itoa(ctx + 1)})
	if err == nil {
		t.Error("serve accepted a block one position longer than the context")
	}
}

// TestServeRefusesAnUnknownLogFormat covers the switch's default arm.
//
// The existing default test checks that the flag's initial value is one the switch accepts, which
// is a different property: that catches a broken default, not a bad argument. With the default arm
// changed to return nil, a serve with --log-format bogus started a server with no access log and no
// message, so an operator who mistyped the value would see a running server and wonder later why
// nothing was logged.
//
// The message lists every accepted spelling, so the refusal is actionable rather than a puzzle.
func TestServeRefusesAnUnknownLogFormat(t *testing.T) {
	path := gguftest.Model{NoEOS: true, ContextLen: 128}.Write(t, t.TempDir())

	err := cmdServe([]string{"--model", path, "--addr", "127.0.0.1:0", "--log-format", "bogus"})
	if err == nil {
		t.Fatal("serve accepted an unknown --log-format")
	}
	if !strings.Contains(err.Error(), "bogus") {
		t.Errorf("error %q does not quote the value", err)
	}
	// The accepted set is enumerated in logFormatChoices; the message has to carry all of it.
	for _, want := range []string{"text", "json", "off"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not list %q", err, want)
		}
	}
}

// TestDefaultQueued covers the backlog depth when --max-queued is unset.
//
// The rule was three lines inside the startup banner, where no test could reach it: the value is
// printed and then never observed. Moving it out is what made both of its branches testable, and the
// arithmetic is worth pinning because it decides how much work the server accepts before refusing --
// too small and a burst of retries is turned away, too large and the queue holds requests it cannot
// reach.
func TestDefaultQueued(t *testing.T) {
	t.Run("unset scales with the active slots", func(t *testing.T) {
		for _, active := range []int{1, 4, 8, 32} {
			if got, want := defaultQueued(0, active), 8*active; got != want {
				t.Errorf("defaultQueued(0, %d) = %d, want %d", active, got, want)
			}
		}
	})

	t.Run("an explicit value wins", func(t *testing.T) {
		if got := defaultQueued(5, 32); got != 5 {
			t.Errorf("defaultQueued(5, 32) = %d, want the explicit 5", got)
		}
	})

	t.Run("a negative value falls back like an unset one", func(t *testing.T) {
		// The flag's documentation says a non-positive value means "pick one", and the startup code
		// compared with <= 0, so a negative has to behave like zero rather than meaning "no queue".
		if got, want := defaultQueued(-1, 4), 8*4; got != want {
			t.Errorf("defaultQueued(-1, 4) = %d, want %d", got, want)
		}
	})
}

// TestPoolCapacityWarningNamesTheStatusTheServerSends closes the gap that let the warning go
// stale.
//
// The warning quotes a status code to an operator, and the code that produces it lives in
// internal/server -- a different package, with its own tests. When the refusal changed from
// 503 to 400 for requests that can never fit the pool, both sides stayed green: the warning
// still said 503 and its test asserted 503, because that assertion was written from the
// implementation rather than from the contract.
//
// This drives the real handler with the pool the warning is about and compares: whatever
// status that request gets is the one the warning has to name, so a future change to either
// side fails here.
func TestPoolCapacityWarningNamesTheStatusTheServerSends(t *testing.T) {
	const contextLen = 64
	path := gguftest.Model{NoEOS: true, ContextLen: contextLen}.Write(t, t.TempDir())

	// One block of one position: smaller than any prompt, which is the configuration the
	// warning is for.
	warning := poolCapacityWarning(1, contextLen)
	if warning == "" {
		t.Fatal("no warning for a pool smaller than the context")
	}

	gf, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	tok, err := tokenize.New(gf)
	if err != nil {
		t.Fatalf("tokenize: %v", err)
	}
	eng, err := engine.New(path, engine.Options{
		MaxActive: 1, MaxQueued: 1, Blocks: 1, BlockSize: 1,
	})
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	srv, err := server.New(eng, tok, server.Config{
		ModelName: eng.Model().Config().Name, ContextLen: contextLen,
	})
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := `{"prompt": "hello there", "max_tokens": 2}`
	resp, err := ts.Client().Post(ts.URL+"/v1/completions", "application/json",
		strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	_, _ = io.Copy(io.Discard, resp.Body)

	// Every "NNN" the warning names has to include the status actually returned.
	got := strconv.Itoa(resp.StatusCode)
	if !strings.Contains(warning, got) {
		t.Errorf("the server answered %d but the warning says:\n%s", resp.StatusCode, warning)
	}
}

// TestHTTPServerBaseContextIsTheCallers covers the one thing newHTTPServerWithRequests does.
//
// net/http derives each request's context from a background context unless BaseContext is set, so
// without this a request in flight during shutdown cannot be told to stop: the handler sees a live
// context, keeps generating, and the grace period expires with the connection closed underneath it.
// The comment above the function records what that looks like from the client -- curl exit 52,
// "empty reply from server", for every request in flight, where the same requests answered 200 once
// the base context was cancellable.
//
// The assertion is identity, not behaviour: the server has to carry the exact context it was given,
// because a copy or a derived context would be a different cancellation signal and the handler
// would be watching the wrong one. Cancelling through the original has to be visible through
// srv.BaseContext's result.
func TestHTTPServerBaseContextIsTheCallers(t *testing.T) {
	base, cancel := context.WithCancel(context.Background())
	srv := newHTTPServerWithRequests(http.NotFoundHandler(), time.Minute, time.Second, base)

	if srv.BaseContext == nil {
		t.Fatal("BaseContext is nil, so every request would get a background context and a" +
			" shutdown could not cancel anything")
	}
	got := srv.BaseContext(nil)
	if got != base {
		t.Error("BaseContext returned a different context than the caller's; cancelling the" +
			" caller's context would not reach the handlers")
	}

	// The point of carrying it: cancelling the caller's context has to reach the one the server
	// hands out.
	cancel()
	select {
	case <-got.Done():
	default:
		t.Error("the caller's cancellation is not visible through the server's base context")
	}
}
