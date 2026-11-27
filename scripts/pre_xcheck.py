#!/usr/bin/env python3
"""Cross-check the GPT-2 pre-tokenizer against the pattern OpenAI published.

    ORBIX_PREDUMP=/tmp/go_pre.json go test ./internal/tokenize/ -run TestDumpPreSplit
    uv run --offline --with regex python3 scripts/pre_xcheck.py /tmp/go_pre.json

The expectations here come from `openai/gpt-2`'s `src/encoder.py`, written out from that source
rather than from the Go implementation. That is the point: pre_reference_cases.go's expectations
were transcribed from llama.cpp's expressions, so a mistake in reading either side would not show
up against itself.

Four families are covered: gpt2, llama3, qwen2 and tekken. The other families' patterns live in
the same llama.cpp file, but each would need its own transcription and its own corpus, and the ones
done this way already disagree whenever a rule drifts.

Adding qwen2 and tekken was what found a real defect: this repository folded case on the clitic
alternative for every family, while llama.cpp writes llama3's as "(?i:'s|'t|...)" and gpt2's and
starcoder's without the flag. "DON'T" therefore split as ["DON", "'T"] for all four where gpt2 and
starcoder want ["DON", "'", "T"]. A family-by-family comparison is what makes that visible; against
one pattern it cannot be.
"""

import json
import sys

import regex

# Each pattern is written from the source named above it, not from the Go implementation.
PATTERNS = {
    # openai/gpt-2 src/encoder.py:
    #   re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    "gpt2": regex.compile(
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    ),
    # Meta's Llama 3 tokenizer, as llama.cpp spells it in llm_tokenizer_bpe:
    #   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
    # Three differences from GPT-2's, and each is a rule the Go side encodes separately: digits
    # group in threes, a letter run may take any non-letter as its prefix (so "a.b" is
    # ["a", ".b"]), and newlines get their own alternatives.
    "llama3": regex.compile(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
        r"|\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
    # Qwen2's, from llama.cpp's llama-vocab.cpp under LLAMA_VOCAB_PRE_TYPE_QWEN2. The contraction
    # branch is spelled out rather than written with (?i:) as Llama 3's is, and the digit branch is
    # `\p{N}` -- one digit per chunk, where GPT-2 takes a run and Llama 3 takes up to three. Those
    # two are the differences worth checking, so the corpus below puts digits and contractions side
    # by side.
    "qwen2": regex.compile(
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
        r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
        r"|\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
    # BailingMoE's, same file under LLAMA_VOCAB_PRE_TYPE_BAILINGMOE. Llama 3's shape with a bare
    # `\p{N}` instead of `\p{N}{1,3}`, and the newline alternative without Llama 3's `+` -- which
    # changes nothing, since both consume the whole run. The contraction alternative spells every
    # letter both ways rather than using (?i:), so its accepted set equals Llama 3's.
    "bailingmoe": regex.compile(
        r"'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])"
        r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
        r"|\s*[\r\n]|\s+(?!\S)|\s+"
    ),
    # Jais2's, same file under LLAMA_VOCAB_PRE_TYPE_JAIS2. Llama 3's expression with the whitespace
    # tail replaced by power-of-two tiers tried largest first, so a run of spaces is cut into
    # 512, 256, ... and the final space leads whatever follows. Writing out the tiers is what makes
    # this worth having: the interaction between "\s{4}(?!\S)", "\s{1,2}(?!\S)" and "\s{1}" decides
    # where a run breaks, and reading that off by hand is exactly the kind of mistake this file
    # exists to catch.
    "jais2": regex.compile(
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
        r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
        r"|\s*[\r\n]+|\s{512}(?!\S)|\s{256}(?!\S)|\s{128}(?!\S)|\s{64}(?!\S)"
        r"|\s{32}(?!\S)|\s{16}(?!\S)|\s{8}(?!\S)|\s{4}(?!\S)|\s{1,2}(?!\S)|\s{1}"
    ),
    # Tekken's, same file under LLAMA_VOCAB_PRE_TYPE_TEKKEN. The llama.cpp entry carries both the
    # readable original from the model's tokenizer.json and a rewritten form with lookaheads; the
    # original is transcribed here, since a comparison against the rewrite could only confirm the
    # rewrite. Its letter alternatives cut at a lower-to-upper boundary, which is the property the
    # corpus is built to exercise.
    "tekken": regex.compile(
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
        r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
        r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
}


# Families whose llama.cpp entry lists more than one expression. llama.cpp runs them in order,
# each over the previous expression's output, so comparing against a single joined alternation
# would test something the engine does not do. superBPE's second expression is a zero-width
# lookahead -- `(?=(\d{3})+(?!\d))` -- which splits a digit run without consuming anything, and
# only means "group from the right" because it runs after `\p{N}+` has already isolated the run.
TWO_PASS = {
    # tiny_aya's, same file under LLAMA_VOCAB_PRE_TYPE_TINY_AYA. Two expressions whose digit
    # classes differ: the first cuts a \d run into threes from the right where a word boundary
    # follows, the second handles what is left with the wider \p{N}{1,3}. Folding them into one
    # pass gets "1\u00b2" wrong -- \u00b2 is \p{N} but not \d, and a boundary follows the 1, so
    # the first pass cuts there and the second leaves the superscript alone.
    "tinyaya": [
        r"\d{1,3}(?=(?:\d{3})*\b)",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?"
        r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?"
        r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    ],
    "superbpe": [r"\p{N}+", r"(?=(\d{3})+(?!\d))"],
    "laguna": [
        r"[^\n]+|[\n]+",
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
        r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
        r"|\s*[\r\n]+|\s+(?!\S)|\s+",
    ],
}


def split(pattern: regex.Pattern, text: str) -> list[str]:
    """split returns the chunks the published pattern produces.

    An empty input is one empty chunk rather than none: the Go side's preSplit returns
    []string{""} for it, and a tokenizer that produced nothing for the empty string would make
    the count disagree for a reason that has nothing to do with the rule.
    """
    if text == "":
        return [""]
    return [m.group(0) for m in pattern.finditer(text)]


def split_by(text: str, expr: str) -> list[str]:
    """split_by applies one expression to text, keeping the matches and what lies between."""
    out, idx = [], 0
    for m in regex.finditer(expr, text):
        if m.start() > idx:
            out.append(text[idx:m.start()])
        if m.end() > m.start():
            out.append(m.group(0))
        idx = m.end()
    if idx < len(text):
        out.append(text[idx:])
    return out


def split_two_pass(exprs: list[str], text: str) -> list[str]:
    """split_two_pass runs each expression over the previous one's output, as llama.cpp does."""
    if text == "":
        return [""]
    pieces = [text]
    for expr in exprs:
        nxt = []
        for piece in pieces:
            nxt.extend(split_by(piece, expr))
        pieces = nxt
    return pieces


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    with open(sys.argv[1]) as f:
        go = json.load(f)

    total = differ = 0
    for family, cases in go.items():
        pattern = PATTERNS.get(family)
        exprs = TWO_PASS.get(family)
        if pattern is None and exprs is None:
            print(f"unknown family {family!r}: add its published pattern above")
            return 1
        for text, want in cases.items():
            total += 1
            got = split_two_pass(exprs, text) if exprs else split(pattern, text)
            if got != want:
                differ += 1
                print(f"DIFFER {family} {text!r}\n  python {got}\n  go     {want}")
    print(f"{len(go)} families, {total} inputs, {total - differ} agree, {differ} differ")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
