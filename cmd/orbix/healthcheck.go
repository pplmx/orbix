package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

// cmdHealthcheck implements "orbix healthcheck": probe a running server's /healthz.
//
// It exists for the container case. The runtime image is `scratch` -- the binary is statically
// linked, so nothing else is needed -- and that means there is no curl, no wget and no shell to
// write a probe in. A `HEALTHCHECK` line and a Kubernetes httpGet probe both need an http client
// that is already inside the image, and this command is that client.
//
// The check is deliberately the same request a load balancer would make, rather than a "is the
// process alive" test: /healthz is answered from the HTTP layer after the engine exists, so a
// 200 means the weights are resident. A probe that only proved the process had not exited would
// call a still-loading server healthy, which is the failure mode a start-period is meant to
// cover and a process-existence probe cannot.
func cmdHealthcheck(args []string) error {
	fs := flag.NewFlagSet("healthcheck", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	addr := fs.String("addr", "http://127.0.0.1:8080", "base URL of the server to probe")
	timeout := fs.Duration("timeout", 3*time.Second, "how long the whole probe may take")
	quiet := fs.Bool("quiet", false, "print nothing on success; the exit status is the answer")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("healthcheck: unexpected argument %q", fs.Arg(0))
	}
	if *timeout <= 0 {
		return fmt.Errorf("healthcheck: --timeout must be positive, got %v", *timeout)
	}

	url := trimTrailingSlash(*addr) + "/healthz"
	// A base URL needs a scheme, and the flag is named --addr like serve's, whose value is a bare
	// host:port. Passing that form here reaches client.Get as "127.0.0.1:8080/healthz" and fails
	// with `parse "127.0.0.1:8080/healthz": first path segment in URL cannot contain colon`, which
	// says what the parser saw and not what to type. The check below turns it into the fix.
	if !strings.Contains(*addr, "://") {
		return fmt.Errorf("healthcheck: --addr is a base URL such as http://127.0.0.1:8080, got %q; "+
			"the bare host:port that serve accepts is not a URL", *addr)
	}
	client := &http.Client{Timeout: *timeout}

	resp, err := client.Get(url)
	if err != nil {
		return fmt.Errorf("healthcheck: %s: %w", url, err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("healthcheck: %s answered %d, want 200", url, resp.StatusCode)
	}
	// The body names the model. Decoding it is what separates "something is listening on this
	// port" from "this server is serving a model" -- a proxy or a stale process on the same
	// port would pass the status check alone.
	var body struct {
		Status string `json:"status"`
		Model  string `json:"model"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return fmt.Errorf("healthcheck: %s: decode body: %w", url, err)
	}
	if body.Status != "ok" {
		return fmt.Errorf("healthcheck: %s reported status %q, want \"ok\"", url, body.Status)
	}
	if !*quiet {
		fmt.Printf("ok: %s serving %s\n", url, body.Model)
	}
	return nil
}

// trimTrailingSlash removes trailing slashes from a base URL so "/healthz" is appended to a
// path that ends once, not twice. `http://host:8080/` and `http://host:8080` are the same server
// and are meant to work the same way.
func trimTrailingSlash(s string) string {
	for len(s) > 0 && s[len(s)-1] == '/' {
		s = s[:len(s)-1]
	}
	return s
}
