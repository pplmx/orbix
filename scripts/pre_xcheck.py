#!/usr/bin/env python3
"""Cross-check the GPT-2 pre-tokenizer against the pattern OpenAI published.

    ORBIX_PREDUMP=/tmp/go_pre.json go test ./internal/tokenize/ -run TestDumpPreSplit
    uv run --offline --with regex python3 scripts/pre_xcheck.py /tmp/go_pre.json

The expectations here come from `openai/gpt-2`'s `src/encoder.py`, written out from that source
rather than from the Go implementation. That is the point: pre_reference_cases.go's expectations
were transcribed from llama.cpp's expressions, so a mistake in reading either side would not show
up against itself.

Only preGPT2 is covered. The other families' patterns live in the same llama.cpp file, but each
would need its own transcription and its own corpus, and the two done this way (this one and the
reference table) already disagree whenever a rule drifts.
"""

import json
import sys

import regex

# openai/gpt-2 src/encoder.py:
#   re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
PATTERN = regex.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def split(text: str) -> list[str]:
    """split returns the chunks the published pattern produces.

    An empty input is one empty chunk rather than none: the Go side's preSplit returns
    []string{""} for it, and a tokenizer that produced nothing for the empty string would make
    the count disagree for a reason that has nothing to do with the rule.
    """
    if text == "":
        return [""]
    return [m.group(0) for m in PATTERN.finditer(text)]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    with open(sys.argv[1]) as f:
        go = json.load(f)

    differ = 0
    for text, want in go.items():
        got = split(text)
        if got != want:
            differ += 1
            print(f"DIFFER {text!r}\n  python {got}\n  go     {want}")
    print(f"{len(go)} inputs, {len(go) - differ} agree, {differ} differ")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
