#!/usr/bin/env python3
"""Rotate heads with numpy and compare against internal/llama's rope.

    ORBIX_ROPEDUMP=/tmp/rope.json go test ./internal/llama/ -run TestDumpRope
    uv run --offline --with numpy python3 scripts/rope_xcheck.py /tmp/rope.json

The rotation is the one from RoFormer: dimension pair k turns by pos * base^(-2k/d), with the
normal layout pairing (2k, 2k+1) and NEOX pairing (k, half+k). Writing it here from that
definition rather than from rope.go is what makes a wrong pairing or a wrong frequency visible --
both produce plausible-looking floats rather than an error.

The theta is built as a running product, not as a power per index, because that is what
ggml_rope_cache_init does and the two differ in the last float bits. Comparing against a power
would report differences that are not mistakes.
"""

import json
import struct
import sys

import numpy as np


def f32(x) -> np.float32:
    return np.float32(x)


def rotate(x: np.ndarray, pos: int, rope_dim: int, head_dim: int, base: float,
           scale: float, layout: str, freqs: list | None = None) -> np.ndarray:
    half = rope_dim // 2
    theta_scale = base ** (-2.0 / rope_dim)
    # Build cos/sin for every pair the same way the Go table does: one multiply per pair,
    # starting from the position itself.
    # The angle is computed in float64 and truncated once, matching the Go side's
    # float64 -> float32 conversion. Rounding the angle to float32 before the cosine, which is
    # what an obvious numpy version does, shifts the result by one or two units in the last
    # place and looks like a disagreement.
    cos = np.empty(half, dtype=np.float32)
    sin = np.empty(half, dtype=np.float32)
    theta = float(pos)
    for k in range(half):
        # The frequency factor divides this dimension's angle and nothing else. Dividing the
        # running theta instead would slow every later dimension too, and the first few pairs
        # would still agree, so the mistake only shows past the last factored pair -- which is
        # why the corpus carries a list covering half the pairs as well as all of them. freqs is
        # indexed by pair, so a shorter list leaves the remaining pairs on the plain rate.
        dim = theta
        if freqs is not None and k < len(freqs):
            dim /= float(freqs[k])
        t = scale * dim
        cos[k] = f32(np.cos(t))
        sin[k] = f32(np.sin(t))
        theta *= theta_scale

    out = x.copy()
    for k in range(half):
        if layout == "neox":
            i0, i1 = k, half + k
        else:
            i0, i1 = 2 * k, 2 * k + 1
        # The products are float32 -- both operands are -- but the sum is formed in float64 and
        # truncated, which is how Go evaluates `x0*cos - x1*sin` on float32 operands.
        x0, x1 = float(out[i0]), float(out[i1])
        c, s_ = float(cos[k]), float(sin[k])
        out[i0] = f32(np.float32(x0 * c) - np.float32(x1 * s_))
        out[i1] = f32(np.float32(x0 * s_) + np.float32(x1 * c))
    return out


def bits(v: np.float32) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    with open(sys.argv[1]) as f:
        doc = json.load(f)

    differ = 0
    for c in doc["cases"]:
        freqs = c.get("freqs")
        name = f'{c["layout"]}-{c["head_dim"]}-{c["rope_dim"]}-{c["base"]:g}-{c["scale"]:g}'
        if freqs:
            name += "-freqs" + "".join(f"{f:g}" for f in freqs)
        x = np.array(c["x"], dtype=np.float32)
        want = c["y"]
        positions = c["pos"]
        head = c["head_dim"]
        bad = 0
        first = None
        for i, pos in enumerate(positions):
            got = rotate(x, pos, c["rope_dim"], head, c["base"], c["scale"], c["layout"], freqs)
            for j in range(head):
                g, w = bits(got[j]), bits(np.float32(want[i * head + j]))
                if g != w:
                    bad += 1
                    if first is None:
                        first = (pos, j, got[j], float(np.float32(want[i * head + j])), g, w)
        if bad:
            differ += 1
            pos, j, gv, wv, gb, wb = first
            print(f"DIFFER {name} at pos {pos} dim {j}: python {gv!r} (0x{gb:08x}) "
                  f"go {wv!r} (0x{wb:08x}); {bad} of {head * len(positions)} differ")
        else:
            print(f"{name}: {head * len(positions)} values agree")
    print(f'{len(doc["cases"])} cases, {len(doc["cases"]) - differ} agree, {differ} differ')
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
