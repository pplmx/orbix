package main

import (
	"flag"
	"strings"
	"testing"
)

// TestPromptFromStdin covers the --stdin path.
//
// `echo "..." | orbix run --stdin` is the shape a shell script reaches for. Before it existed
// the only way in was an argument, which needs quoting for anything with spaces and hits the
// argument-length limit for a large prompt.
//
// The trimming rule is asserted rather than assumed: a prompt assembled by `cat file` ends with
// a newline that the caller did not mean to send, so trailing newlines go -- but leading
// whitespace and interior newlines are part of the prompt and are kept.
func TestPromptFromStdin(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{"plain", "Once upon a time", "Once upon a time"},
		{"trailing newline", "Once upon a time\n", "Once upon a time"},
		{"trailing crlf", "Once upon a time\r\n", "Once upon a time"},
		{"several trailing newlines", "hello\n\n\n", "hello"},
		{"interior newline kept", "line one\nline two", "line one\nline two"},
		{"trailing spaces kept", "hello  ", "hello  "},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fs := flag.NewFlagSet("run", flag.ContinueOnError)
			got, err := promptFromStdin(strings.NewReader(c.input), fs)
			if err != nil {
				t.Fatalf("promptFromStdin(%q): %v", c.input, err)
			}
			if got != c.want {
				t.Errorf("promptFromStdin(%q) = %q, want %q", c.input, got, c.want)
			}
		})
	}
}

// TestPromptFromStdinRefusesEmpty covers the case a script hits when the thing it meant to pipe
// produced nothing.
//
// Passing an empty prompt on would reach the tokenizer and come back as "prompt encoded to no
// tokens", which names the wrong problem: the input was empty, and that is what the caller can
// fix.
func TestPromptFromStdinRefusesEmpty(t *testing.T) {
	for _, input := range []string{"", "\n", "\r\n\r\n"} {
		fs := flag.NewFlagSet("run", flag.ContinueOnError)
		if _, err := promptFromStdin(strings.NewReader(input), fs); err == nil {
			t.Errorf("an input of %q was accepted as a prompt", input)
		}
	}
}

// TestPromptFromStdinRefusesAPositionalArgument pins that the two ways of supplying a prompt
// cannot be mixed.
//
// Silently preferring one would hide half of what the caller wrote; refusing names both.
func TestPromptFromStdinRefusesAPositionalArgument(t *testing.T) {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	if err := fs.Parse([]string{"a stray prompt"}); err != nil {
		t.Fatal(err)
	}
	_, err := promptFromStdin(strings.NewReader("from stdin"), fs)
	if err == nil {
		t.Fatal("--stdin with a positional argument was accepted")
	}
	if !strings.Contains(err.Error(), "a stray prompt") {
		t.Errorf("error %q does not name the argument that is ambiguous", err)
	}
}
