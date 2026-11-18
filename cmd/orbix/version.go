package main

import (
	"encoding/json"
	"fmt"
	"runtime"
	"runtime/debug"
)

// Build metadata. These are overridden at build time with -ldflags, e.g.
//
//	go build -ldflags "-X main.version=v0.1.0 -X main.commit=$(git rev-parse --short HEAD)"
//
// Keeping them as variables rather than constants means a plain `go build`
// still produces a working binary that reports "dev".
var (
	version = "dev"
	commit  = "none"
	date    = "unknown"
)

// cmdVersion implements "orbix version".
//
// -json exists because this is the command a build script reads: checking that an artifact
// reports cgo:false is what the cross-build job does by hand today, and parsing a two-column
// text report for that is worse than parsing a document. The other commands refuse an unknown
// flag through the flag package; this one reads its arguments itself and used to ignore them,
// so `version --json` printed the human report and exited zero -- a caller piping it into a
// JSON parser got a parse error where the flag said success.
func cmdVersion(args []string) error {
	asJSON := false
	for _, a := range args {
		switch a {
		case "-h", "--help":
			fmt.Println("Usage: orbix version [-json]")
			return nil
		case "-json", "--json":
			asJSON = true
		default:
			return fmt.Errorf("version: unknown argument %q, want no arguments or -json", a)
		}
	}
	if asJSON {
		return writeVersionJSON()
	}
	fmt.Printf("orbix %s\n", version)
	// One column width for every label, so the values line up and a reader can scan them. The
	// widest label is CGO_ENABLED, which only appears for a binary built by the go tool; the
	// rest are padded to match rather than to the longest one present, because a report whose
	// alignment changed depending on how the binary was built would be harder to compare.
	const label = "  %-13s%s\n"
	fmt.Printf(label, "commit:", commit)
	fmt.Printf(label, "built:", date)
	fmt.Printf(label, "go:", runtime.Version())
	fmt.Printf(label, "platform:", runtime.GOOS+"/"+runtime.GOARCH)
	fmt.Printf(label, "cgo:", fmt.Sprint(cgoEnabled()))
	if bi, ok := debug.ReadBuildInfo(); ok {
		for _, s := range bi.Settings {
			if s.Key == "CGO_ENABLED" {
				fmt.Printf(label, "CGO_ENABLED:", s.Value)
			}
		}
	}
	return nil
}

// writeVersionJSON prints the same facts as the text report, as one JSON object.
//
// The fields are named after the labels a reader already sees, so the two outputs describe the
// same thing rather than two overlapping ones. cgo_enabled is omitted when the binary carries no
// build info to read it from, which is the same condition the text report uses to leave the line
// out.
func writeVersionJSON() error {
	out := struct {
		Version    string  `json:"version"`
		Commit     string  `json:"commit"`
		Built      string  `json:"built"`
		Go         string  `json:"go"`
		Platform   string  `json:"platform"`
		Cgo        bool    `json:"cgo"`
		CgoEnabled *string `json:"cgo_enabled,omitempty"`
	}{
		Version:  version,
		Commit:   commit,
		Built:    date,
		Go:       runtime.Version(),
		Platform: runtime.GOOS + "/" + runtime.GOARCH,
		Cgo:      cgoEnabled(),
	}
	if bi, ok := debug.ReadBuildInfo(); ok {
		for _, s := range bi.Settings {
			if s.Key == "CGO_ENABLED" {
				v := s.Value
				out.CgoEnabled = &v
			}
		}
	}
	b, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return fmt.Errorf("version: encode: %w", err)
	}
	_, err = fmt.Println(string(b))
	return err
}

// cgoEnabled reports whether this binary was built with CGo.
//
// The build tag and the runtime setting can disagree when CGO_ENABLED was set
// in the environment at build time, so report the build-time fact from debug
// info when it is available.
func cgoEnabled() bool {
	bi, ok := debug.ReadBuildInfo()
	if ok {
		for _, s := range bi.Settings {
			if s.Key == "CGO_ENABLED" {
				return s.Value == "1"
			}
		}
	}
	return cgoCompiled
}
