package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/pplmx/orbix/internal/gguf"
)

// cmdModel implements "orbix model".
func cmdModel(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("model: missing subcommand, want \"inspect\"")
	}
	switch args[0] {
	case "inspect":
		return cmdModelInspect(args[1:])
	case "-h", "--help", "help":
		fmt.Print(`Usage:
  orbix model inspect [flags] <path.gguf>

Flags:
  --json    Print metadata and the tensor directory as JSON
`)
		return nil
	default:
		return fmt.Errorf("model: unknown subcommand %q, want \"inspect\"", args[0])
	}
}

// cmdModelInspect prints everything worth knowing about a GGUF file.
//
// This doubles as a teaching tool: the output presents the file in the order
// the loader meets it — header, metadata, then tensors.
func cmdModelInspect(args []string) error {
	fs := flag.NewFlagSet("model inspect", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	asJSON := fs.Bool("json", false, "print metadata and the tensor directory as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("model inspect: want exactly one file path, got %d", fs.NArg())
	}
	path := fs.Arg(0)

	f, err := gguf.Open(path)
	if err != nil {
		return err
	}
	return inspect(os.Stdout, f, path, inspectOptions{JSON: *asJSON})
}
