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
	fmt.Printf("  commit:  %s\n", commit)
	fmt.Printf("  built:   %s\n", date)
	fmt.Printf("  go:      %s\n", runtime.Version())
	fmt.Printf("  platform:%s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("  cgo:     %v\n", cgoEnabled())
	if bi, ok := debug.ReadBuildInfo(); ok {
		for _, s := range bi.Settings {
			if s.Key == "CGO_ENABLED" {
				fmt.Printf("  CGO_ENABLED: %s\n", s.Value)
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
