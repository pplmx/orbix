"""Cross-check a GGUF file against an independent parser, tensor by tensor.

This is the check AGENTS.md records as the way to validate a real checkpoint: parse the
same file with a reader nobody on this project wrote, and compare what the two see. A
reader can be self-consistently wrong -- it reads the header it wrote, or it agrees with
its own fixture builder -- and only a second implementation catches that.

For every tensor it compares three things:

  - the type name (Q4_K, F32, ...);
  - the dimensions, in the file's own order;
  - a SHA-256 over the raw stored bytes, read at the offset each side computed.

The hash is what makes this stronger than a metadata diff: two readers can agree on a
tensor's shape and disagree about where it starts, and a wrong offset produces a
plausible-looking matrix from the neighbouring tensor's data.

Run it with uv, which provides the `gguf` package -- an implementation of the format
maintained outside this repository:

    uv run --with gguf python3 scripts/gguf_xcheck.py /tmp/tinyllama-q2k.gguf

Needs the orbix binary for the Go side; pass its path as the second argument or let it
default to ./bin/orbix. Exit status is 1 if anything disagrees, so it can gate by hand.

A note on the shape of the comparison, because getting it wrong is easy: gguf's
`Tensor.shape` is already in GGUF's stored order, the same order orbix prints as `dims`.
Reversing one side to "match" the other turns every non-square tensor into a reported
disagreement -- measured, 112 of TinyLlama's 201 tensors, all of them a bare reversal.
"""

import hashlib
import json
import os
import subprocess
import sys

import gguf


def sha256_tensor(reader, tensor):
    """Hash the bytes the Python reader would hand a dequantizer."""
    data = tensor.data
    # gguf exposes a memoryview or an array depending on the type; bytes() covers both.
    return hashlib.sha256(bytes(data)).hexdigest()


def sha256_orbix(path, offset, nbytes):
    """Hash the bytes at the offset orbix reported, read straight from the file.

    The offset is relative to the tensor-data section, not the file: GGUF stores a single
    `tensor_data_offset` in the header and every tensor's offset is measured from there.
    Hash from the file start and every tensor disagrees -- measured, 33 of 33 on the 7 MB
    MoE file, because the 13696-byte header was included in each read.
    """
    with open(path, "rb") as fh:
        fh.seek(offset)
        remaining = nbytes
        h = hashlib.sha256()
        while remaining > 0:
            chunk = fh.read(min(remaining, 1 << 20))
            if not chunk:
                raise SystemExit(f"short read at offset {offset}: {nbytes - remaining} of {nbytes}")
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: gguf_xcheck.py <file.gguf> [orbix-binary]")
    path = sys.argv[1]
    orbix = sys.argv[2] if len(sys.argv) > 2 else "./bin/orbix"
    if not os.path.exists(orbix):
        raise SystemExit(f"{orbix} not found; build it with `just build` or pass the path")

    reader = gguf.GGUFReader(path)
    raw = json.loads(
        subprocess.run([orbix, "model", "inspect", "--json", path],
                       capture_output=True, text=True, check=True).stdout
    )
    go_tensors = {t["name"]: t for t in raw["tensors"]}

    py_names = {t.name for t in reader.tensors}
    go_names = set(go_tensors)
    if py_names != go_names:
        print("tensor name sets differ", file=sys.stderr)
        for name in sorted(py_names - go_names)[:8]:
            print(f"  only in the reference parser: {name}", file=sys.stderr)
        for name in sorted(go_names - py_names)[:8]:
            print(f"  only in orbix: {name}", file=sys.stderr)
        return 1

    bad = 0
    for tensor in reader.tensors:
        go = go_tensors[tensor.name]
        py_type = str(tensor.tensor_type.name)
        py_dims = tuple(int(d) for d in tensor.shape)
        go_dims = tuple(go["dims"])
        if (py_type, py_dims) != (go["type"], go_dims):
            print(f"{tensor.name}: type/shape differ: "
                  f"reference {py_type}{py_dims} vs orbix {go['type']}{go_dims}", file=sys.stderr)
            bad += 1
            continue
        # bytes is omitted when the count overflows an int64; a comparison needs the count,
        # so a tensor without it is a disagreement rather than something to skip silently.
        if go.get("bytes") is None:
            print(f"{tensor.name}: orbix reported no byte count", file=sys.stderr)
            bad += 1
            continue
        want = sha256_tensor(reader, tensor)
        got = sha256_orbix(path, raw["tensor_data_offset"] + go["offset"], go["bytes"])
        if want != got:
            print(f"{tensor.name}: data differs at offset {go['offset']} "
                  f"({go['bytes']} bytes): reference {want[:16]} vs orbix {got[:16]}",
                  file=sys.stderr)
            bad += 1

    compared = len(reader.tensors)
    if bad:
        print(f"{compared} tensors compared; {bad} disagree", file=sys.stderr)
        return 1
    # Say what was compared rather than only that it passed: a run over zero tensors would
    # otherwise report success, which is the failure mode the stress script's self-check
    # exists to prevent.
    print(f"{compared} tensors compared; all agree on type, shape and SHA-256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
