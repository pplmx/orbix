package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestHealthcheckReportsSuccessAndFailure covers the probe command.
//
// It exists because the container runtime is `scratch`: a statically linked binary and a CA
// bundle, with no curl, no wget and no shell. Both a Docker HEALTHCHECK and a Kubernetes
// exec probe need an http client inside the image, and this command is it. A probe that was
// wrong in the permissive direction -- reporting success against a server that is not serving --
// would let a container be marked ready before its weights were resident.
func TestHealthcheckReportsSuccessAndFailure(t *testing.T) {
	ok := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/healthz" {
			t.Errorf("probed %q, want /healthz", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok","model":"tiny"}`))
	}))
	defer ok.Close()

	out := captureStdout(t, func() {
		if err := cmdHealthcheck([]string{"--addr", ok.URL}); err != nil {
			t.Errorf("healthcheck against a healthy server: %v", err)
		}
	})
	if !strings.Contains(out, "tiny") {
		t.Errorf("the report does not name the model: %q", out)
	}

	// --quiet prints nothing: the exit status is the answer, which is what a probe reads.
	quiet := captureStdout(t, func() {
		if err := cmdHealthcheck([]string{"--addr", ok.URL, "--quiet"}); err != nil {
			t.Errorf("healthcheck --quiet: %v", err)
		}
	})
	if quiet != "" {
		t.Errorf("--quiet printed %q", quiet)
	}

	// A trailing slash is the same server and must not become "//healthz".
	for _, suffix := range []string{"/", "//", "///"} {
		slashed := captureStdout(t, func() {
			if err := cmdHealthcheck([]string{"--addr", ok.URL + suffix, "--quiet"}); err != nil {
				t.Errorf("healthcheck with %q appended to the address: %v", suffix, err)
			}
		})
		if slashed != "" {
			t.Errorf("--quiet with %q appended printed %q", suffix, slashed)
		}
	}
}

// TestHealthcheckQuietStillFails covers the one pairing the container's health probe depends on.
//
// --quiet means "print nothing on success", not "print nothing and report success". The Dockerfile
// runs `healthcheck --quiet` as its HEALTHCHECK, so a --quiet that swallowed the error would make
// every container report healthy while its server was down -- the one signal an orchestrator acts
// on, inverted. The success cases above check the silence; nothing checked that the failure
// survives it.
//
// Measured against the built binary before writing this: a dead port exits 1 with and without
// --quiet, a live one exits 0 with and without it. Don't read $? after a pipe to head -- it reports
// head's status, and that mistake made this gap look like a defect for a few minutes.
func TestHealthcheckQuietStillFails(t *testing.T) {
	dead := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	addr := dead.URL
	dead.Close() // nothing is listening now, which is the case that must fail

	for _, args := range [][]string{
		{"--addr", addr},
		{"--addr", addr, "--quiet"},
	} {
		var err error
		_ = captureStdout(t, func() { err = cmdHealthcheck(args) })
		if err == nil {
			t.Errorf("healthcheck %v against a dead server returned no error; --quiet must not "+
				"turn a failure into a success", args)
			continue
		}
		if !strings.Contains(err.Error(), "healthcheck") {
			t.Errorf("healthcheck %v: error %q does not name the command", args, err)
		}
	}
}

// TestTrimTrailingSlash removes every trailing slash, not just one.
//
// The loop it replaced this shape of was untested: trimming a single slash instead left every
// test green, because the healthy-server case above appends exactly one and the handler checks
// only the path. A caller passing "http://host:8080//" -- from a template that appends a slash
// to a value that already ended with one -- would then probe "//healthz", which a mux serving
// only "/healthz" answers with 404.
func TestTrimTrailingSlash(t *testing.T) {
	cases := []struct{ in, want string }{
		{"http://host:8080", "http://host:8080"},
		{"http://host:8080/", "http://host:8080"},
		{"http://host:8080///", "http://host:8080"},
		{"http://host/prefix//", "http://host/prefix"},
		// Everything is a slash: the answer is the empty string, and the caller builds
		// "/healthz", which http.Get rejects as a missing scheme rather than probing.
		{"///", ""},
		{"", ""},
	}
	for _, c := range cases {
		if got := trimTrailingSlash(c.in); got != c.want {
			t.Errorf("trimTrailingSlash(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestHealthcheckFailsOnADegradedServer covers the cases a status-code-only check would pass.
//
// A 200 from a proxy, a stale process, or a server answering an empty body are all "something
// is listening" and none of them is "this model is ready". Each is asserted separately so a
// future rewrite cannot keep the status check and quietly drop the body check.
func TestHealthcheckFailsOnADegradedServer(t *testing.T) {
	cases := []struct {
		name    string
		handler http.HandlerFunc
	}{
		{"wrong status", func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
		}},
		{"status not ok", func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`{"status":"loading","model":"tiny"}`))
		}},
		{"body is not json", func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`<html>hello</html>`))
		}},
		{"empty body", func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		}},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			srv := httptest.NewServer(c.handler)
			defer srv.Close()
			_ = captureStdout(t, func() {
				if err := cmdHealthcheck([]string{"--addr", srv.URL, "--quiet"}); err == nil {
					t.Error("healthcheck reported success against a server that is not healthy")
				}
			})
		})
	}
}

// TestHealthcheckRefusesBadArguments covers the two ways the command itself can be wrong.
func TestHealthcheckRefusesBadArguments(t *testing.T) {
	_ = captureStdout(t, func() {
		if err := cmdHealthcheck([]string{"--timeout", "0s"}); err == nil {
			t.Error("a zero timeout was accepted; every probe would fail without saying why")
		}
	})
	_ = captureStdout(t, func() {
		if err := cmdHealthcheck([]string{"stray"}); err == nil {
			t.Error("a stray positional argument was accepted")
		}
	})
}
