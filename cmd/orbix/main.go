// Command orbix is the Orbix CLI.
//
// The CLI is a thin shell around the engine packages: it parses flags, opens
// files, and prints results. Inference itself lives in the packages under
// internal/ and the public orbix package, so the same code paths are used by
// the server, the REPL, and embedded callers.
package main

import (
	"fmt"
	"os"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "orbix:", err)
		os.Exit(1)
	}
}

// run dispatches a command. It returns an error rather than exiting so tests
// can call it directly.
func run(args []string) error {
	if len(args) == 0 {
		usage()
		return nil
	}
	switch args[0] {
	case "run":
		return cmdRun(args[1:])
	case "bench":
		return cmdBench(args[1:])
	case "model":
		return cmdModel(args[1:])
	case "version":
		return cmdVersion(args[1:])
	case "help", "-h", "--help":
		usage()
		return nil
	default:
		return fmt.Errorf("unknown command %q, run \"orbix help\"", args[0])
	}
}

func usage() {
	fmt.Print(`Orbix - a pure-Go LLM inference engine.

Usage:
  orbix <command> [arguments]

Commands:
  run <prompt>           Generate text from a prompt
  bench                  Measure generation throughput
  model inspect <path>   Print GGUF metadata, tensor types, quantization
  version                Print version and build info
  help                   Print this message

Run "orbix <command> -h" for command-specific flags.
`)
}
