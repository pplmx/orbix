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
	slashed := captureStdout(t, func() {
		if err := cmdHealthcheck([]string{"--addr", ok.URL + "/", "--quiet"}); err != nil {
			t.Errorf("healthcheck with a trailing slash: %v", err)
		}
	})
	if slashed != "" {
		t.Errorf("--quiet with a trailing slash printed %q", slashed)
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
