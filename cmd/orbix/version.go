package main

import (
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
func cmdVersion(args []string) error {
	if len(args) > 0 && (args[0] == "-h" || args[0] == "--help") {
		fmt.Println("Usage: orbix version")
		return nil
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
