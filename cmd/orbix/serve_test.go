package main

import (
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/pplmx/orbix/internal/llama"
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
	handler := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
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
	// Let the request be accepted and reach the handler before the drain begins.
	time.Sleep(20 * time.Millisecond)

	if err := drainAndStop(httpSrv, stopPump, 5*time.Second); err != nil {
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

	if err := drainAndStop(httpSrv, func() {}, 100*time.Millisecond); err != nil {
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
