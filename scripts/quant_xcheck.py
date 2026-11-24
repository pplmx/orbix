#!/usr/bin/env python3
"""Decode ggml blocks from their published layout and compare with the Go implementation.

    ORBIX_QUANTDUMP=/tmp/q.json go test ./internal/quant/ -run TestDumpDequant
    uv run --offline --with numpy python3 scripts/quant_xcheck.py /tmp/q.json

Each layout below is written from ggml's `quantize.c`, which is the reference every GGUF reader
ports. Writing them from that source rather than from internal/quant is the point: a misreading
of a block layout, repeated in both the code and its tests, is exactly what a same-source
comparison cannot see.

The layouts, for the record:

  Q4_0   2-byte f16 scale, then 16 bytes. Byte j holds weight j in its low nibble and weight
         j+16 in its high nibble -- interleaved, not packed as 2j and 2j+1. Codes are offset
         by 8, giving -8..7. w = d * (code - 8).
  Q4_1   2-byte f16 scale, 2-byte f16 minimum, then 16 bytes of the same interleaving.
         w = d * code + m, with no offset.
  Q5_0   2-byte f16 scale, 4 bytes of high bits, then 16 bytes of low nibbles.
         w = d * ((code | (hi_bit << 4)) - 16), giving -16..15.
  Q5_1   2-byte f16 scale, 2-byte f16 minimum, 4 bytes of high bits, 16 bytes of low nibbles.
         w = d * (code | (hi_bit << 4)) + m.
  Q8_0   2-byte f16 scale, then 32 signed bytes. w = d * int8.
  BF16   the high half of a float32; widening is a shift with no rounding.
  F16    IEEE 754 binary16.
"""

import base64
import json
import struct
import sys


def f16(b: bytes) -> float:
    return struct.unpack("<e", b)[0]


def dequant_q4_0(src: bytes) -> list[float]:
    d = f16(src[0:2])
    qs = src[2:18]
    out = []
    for j in range(16):
        out.append(d * ((qs[j] & 0x0F) - 8))
    for j in range(16):
        out.append(d * ((qs[j] >> 4) - 8))
    return out


def dequant_q4_1(src: bytes) -> list[float]:
    d = f16(src[0:2])
    m = f16(src[2:4])
    qs = src[4:20]
    out = []
    for j in range(16):
        out.append(d * (qs[j] & 0x0F) + m)
    for j in range(16):
        out.append(d * (qs[j] >> 4) + m)
    return out


def dequant_q5_0(src: bytes) -> list[float]:
    d = f16(src[0:2])
    qh = int.from_bytes(src[2:6], "little")
    qs = src[6:22]
    out = []
    for j in range(16):
        h = (qh >> j) & 1
        out.append(d * (((qs[j] & 0x0F) | (h << 4)) - 16))
    for j in range(16):
        h = (qh >> (j + 16)) & 1
        out.append(d * (((qs[j] >> 4) | (h << 4)) - 16))
    return out


def dequant_q5_1(src: bytes) -> list[float]:
    d = f16(src[0:2])
    m = f16(src[2:4])
    qh = int.from_bytes(src[4:8], "little")
    qs = src[8:24]
    out = []
    for j in range(16):
        h = (qh >> j) & 1
        out.append(d * ((qs[j] & 0x0F) | (h << 4)) + m)
    for j in range(16):
        h = (qh >> (j + 16)) & 1
        out.append(d * ((qs[j] >> 4) | (h << 4)) + m)
    return out


def dequant_q8_0(src: bytes) -> list[float]:
    d = f16(src[0:2])
    return [d * struct.unpack("<b", bytes([b]))[0] for b in src[2:34]]


def dequant_bf16(src: bytes) -> list[float]:
    out = []
    for i in range(0, len(src), 2):
        bits = int.from_bytes(src[i : i + 2], "little") << 16
        out.append(struct.unpack("<f", struct.pack("<I", bits))[0])
    return out


def dequant_f16(src: bytes) -> list[float]:
    return [f16(src[i : i + 2]) for i in range(0, len(src), 2)]


DECODERS = {
    "Q4_0": dequant_q4_0,
    "Q4_1": dequant_q4_1,
    "Q5_0": dequant_q5_0,
    "Q5_1": dequant_q5_1,
    "Q8_0": dequant_q8_0,
    "BF16": dequant_bf16,
    "F16": dequant_f16,
}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    with open(sys.argv[1]) as f:
        go = json.load(f)

    differ = 0
    for name, case in go.items():
        decode = DECODERS.get(name)
        if decode is None:
            print(f"unknown format {name!r}: add its layout above")
            return 1
        # Go encodes []byte as base64 in JSON.
        src = base64.b64decode(case["src"])
        # A block is 32 weights for the quantized formats and one for the wide ones; decode
        # whatever the encoded length covers rather than assuming a count.
        block = 32 if name.startswith("Q") else 1
        want = case["bits"]
        size = len(src) // (len(want) // block)
        got = []
        for i in range(0, len(src), size):
            got.extend(decode(src[i : i + size]))
        if len(got) != len(want):
            print(f"{name}: decoded {len(got)} values, Go produced {len(want)}")
            differ += 1
            continue
        # Compare bit patterns rather than float equality: a NaN differs from itself under ==,
        # and -0.0 compares equal to 0.0, so a sign the decoder dropped would go unreported.
        bad = [
            (k, g, w)
            for k, (g, w) in enumerate(zip(got, want))
            if struct.unpack("<I", struct.pack("<f", g))[0] != w
        ]
        if bad:
            differ += 1
            k, g, w = bad[0]
            print(f"DIFFER {name} at index {k} of {len(want)}: python {g!r} "
                  f"(0x{struct.unpack('<I', struct.pack('<f', g))[0]:08x}) "
                  f"go 0x{w:08x} ({len(bad)} of {len(want)} differ)")
        else:
            print(f"{name}: {len(want)} values agree")
    print(f"{len(go)} formats, {len(go) - differ} agree, {differ} differ")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
