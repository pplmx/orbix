#!/usr/bin/env python3
"""Parse gguftest's fixture with the Python gguf package and compare against orbix's reading.

    just build
    uv run --offline --with gguf python3 scripts/fixture_xcheck.py

The fixture is what most tests in this repository run on: a few-KB model built by gguftest and
parsed by the same reader the tests exercise. That makes a wrong field in it invisible -- every
test downstream would be asserting against the same mistake.

The Python package is an independent implementation of the container format, so parsing the same
bytes with both and comparing the numbers, the metadata and the tensor directory is a check the
fixture's own tests cannot perform.

Writes the fixture to a temporary file first, since gguftest only produces one through a Go test.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import gguf

# A file with the same options the rest of the suite uses by default.
GO_TEST = """
package gguftest

import (
	"os"
	"testing"
)

func TestWriteFixtureForScript(t *testing.T) {
	out := os.Getenv("ORBIX_FIXTURE_OUT")
	if out == "" {
		t.Skip("set ORBIX_FIXTURE_OUT")
	}
	path := Model{NoEOS: true, ContextLen: 128}.Write(t, t.TempDir())
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	if err := os.WriteFile(out, b, 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
}
"""


def build_fixture(dest: Path) -> None:
    src = Path("internal/gguftest/fixture_script_test.go")
    src.write_text(GO_TEST)
    try:
        subprocess.run(
            ["go", "test", "./internal/gguftest/", "-run", "TestWriteFixtureForScript", "-count=1"],
            check=True,
            env={**os.environ, "ORBIX_FIXTURE_OUT": str(dest)},
            capture_output=True,
        )
    finally:
        src.unlink()


def main() -> int:
    if len(sys.argv) > 1:
        fixture = Path(sys.argv[1])
    else:
        # NamedTemporaryFile(delete=False) because gguftest writes the file through a Go test,
        # which needs a path that outlives this call. That means the file is the caller's to
        # remove, and for a while it was nobody's: measured, one 185 KB fixture was left in
        # /tmp per run. Harmless while the script was only run by hand, and not harmless once
        # it ran on every push.
        tmp = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        tmp.close()
        fixture = Path(tmp.name)
        try:
            build_fixture(fixture)
        except subprocess.CalledProcessError as e:
            print(f"building the fixture failed: {e.stderr.decode()}", file=sys.stderr)
            return 2
        try:
            return compare(fixture)
        finally:
            fixture.unlink(missing_ok=True)

    return compare(fixture)


def compare(fixture: Path) -> int:
    reader = gguf.GGUFReader(str(fixture))

    out = subprocess.run(
        ["bin/orbix", "model", "inspect", "--json", str(fixture)],
        check=True, capture_output=True,
    )
    go = json.loads(out.stdout)

    differ = 0

    def check(what: str, a, b) -> None:
        nonlocal differ
        if a != b:
            differ += 1
            print(f"DIFFER {what}: python {a!r} go {b!r}")

    check("version", int(reader.fields["GGUF.version"].parts[-1][0]), go["version"])
    check("tensor_count", len(reader.tensors), len(go["tensors"]))

    # The tensor directory, by name: dimensions and type have to agree for every entry.
    py_tensors = {t.name: (list(t.shape), t.tensor_type.name) for t in reader.tensors}
    go_tensors = {
        t["name"]: (t["dims"], t["type"]) for t in go["tensors"]
    }
    if set(py_tensors) != set(go_tensors):
        differ += 1
        print(f"DIFFER tensor names: only python {sorted(set(py_tensors) - set(go_tensors))}, "
              f"only go {sorted(set(go_tensors) - set(py_tensors))}")
    for name in sorted(set(py_tensors) & set(go_tensors)):
        # gguf reports dimensions in file order; orbix reverses to row-major, so compare the
        # multiset and the product rather than the sequence.
        pd, pt = py_tensors[name]
        gd, gt = go_tensors[name]
        if sorted(pd) != sorted(gd):
            differ += 1
            print(f"DIFFER {name} dims: python {pd} go {gd}")
        if pt != gt:
            differ += 1
            print(f"DIFFER {name} type: python {pt} go {gt}")

    # The metadata, for the keys both sides report as scalars.
    py_meta = {}
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        try:
            py_meta[key] = field.contents()
        except Exception:
            continue
    for key, want in sorted(py_meta.items()):
        if key not in go["metadata"]:
            differ += 1
            print(f"DIFFER metadata {key}: python has {want!r}, go does not")
            continue
        got = go["metadata"][key]
        # orbix summarises an array as "[363 string]" rather than printing it: the token lists
        # are hundreds of entries and would swamp the output. Compare the length it reports and
        # leave the contents to TestFixtureTokenizes, which has them in Go form already.
        if isinstance(want, list):
            n = len(want)
            # The names orbix uses for array element types, mapped from Python's.
            kind = {float: "float32", int: "int32", str: "string", bool: "bool"}.get(
                type(want[0]), type(want[0]).__name__)
            if got != f"[{n} {kind}]":
                differ += 1
                print(f"DIFFER metadata {key} length: python {n} entries, go says {got!r}")
            continue
        if isinstance(want, float) and isinstance(got, float):
            if abs(want - got) > 1e-9 * max(1.0, abs(want)):
                differ += 1
                print(f"DIFFER metadata {key}: python {want!r} go {got!r}")
        elif want != got:
            differ += 1
            print(f"DIFFER metadata {key}: python {want!r} go {got!r}")

    print(f"{len(reader.fields) - 3} metadata keys, {len(reader.tensors)} tensors, "
          f"{differ} differences")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
