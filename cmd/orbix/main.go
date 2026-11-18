// Command orbix is the Orbix CLI.
//
// The CLI is a thin shell around the engine packages: it parses flags, opens
// files, and prints results. Inference itself lives in the packages under
// internal/ and the public orbix package, so the same code paths are used by
// the server, the REPL, and embedded callers.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
)

func main() {
	os.Exit(report(run(os.Args[1:]), os.Stderr))
}

// report writes the message for err and returns the status the process should exit with.
//
// It is separate from main so the mapping from an error to an exit status is testable: main
// calls os.Exit, which ends the test binary.
func report(err error, w io.Writer) int {
	if err == nil {
		return 0
	}
	// -h and -help are a request for the usage text, which the flag package has already
	// printed, not a failure. The stdlib's own ExitOnError exits 0 for exactly this case;
	// returning it as an error made every command's help exit 1 with an extra
	// "orbix: flag: help requested" line on stderr, while `orbix help` exited 0 -- so the two
	// documented ways to see help disagreed, and a script probing `orbix serve -h` to check
	// the binary read a healthy binary as a broken command.
	if errors.Is(err, flag.ErrHelp) {
		return 0
	}
	// The write is best effort for the same reason it is in chat.go: the only way it fails is
	// a closed or broken stderr, and a CLI reporting a failure has nothing left to report on
	// if the channel for reporting is gone. Exiting non-zero is what a caller can act on.
	_, _ = fmt.Fprintln(w, "orbix:", err)
	return 1
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
	case "chat":
		return cmdChat(args[1:])
	case "serve":
		return cmdServe(args[1:])
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
  chat                   Interactive REPL with conversation history
  serve                  Serve an OpenAI-compatible HTTP API
  bench                  Measure generation throughput; --suite runs every scenario
  model inspect <path>   Print GGUF metadata, tensor types, quantization
  version                Print version and build info
  help                   Print this message

Run "orbix <command> -h" for command-specific flags.
`)
}
