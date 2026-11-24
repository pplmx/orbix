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


def get_scale_min_k4(j: int, q: bytes) -> tuple[int, int]:
    """get_scale_min_k4 unpacks sub-block j's 6-bit scale and minimum.

    ggml's own helper of the same name. The low four sub-blocks take their scale from the low
    six bits of bytes 0-3 and their min from bytes 4-7; the high four take the low nibble of
    bytes 8-11 as the low four bits and two bits from an earlier byte as the top two.
    """
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def dequant_q4_k(src: bytes) -> list[float]:
    """Q4_K: 2-byte d, 2-byte dmin, 12 scale bytes, then 128 code bytes for 256 weights.

    The codes are four iterations of 64 weights. Within one iteration the first 32 come from
    the low nibbles of 32 bytes and the next 32 from the high nibbles of those same bytes --
    not interleaved per byte the way Q4_0 is.
    """
    d = f16(src[0:2])
    dmin = f16(src[2:4])
    scales = src[4:16]
    qs = src[16:144]
    out = [0.0] * 256
    for it in range(4):
        sc0, m0 = get_scale_min_k4(2 * it, scales)
        sc1, m1 = get_scale_min_k4(2 * it + 1, scales)
        d1, m1f = d * sc0, dmin * m0
        d2, m2f = d * sc1, dmin * m1
        for l in range(32):
            q = qs[it * 32 + l]
            out[it * 64 + l] = d1 * (q & 0x0F) - m1f
            out[it * 64 + 32 + l] = d2 * (q >> 4) - m2f
    return out


def dequant_q6_k(src: bytes) -> list[float]:
    """Q6_K: 128 low-nibble bytes, 64 high-2-bit bytes, 16 int8 sub-scales, then d.

    d comes last here, unlike every other format. Two halves of 128 weights; within a half,
    each of 32 byte positions yields four weights from ql[l] and ql[l+32] with the top two bits
    from successive bit pairs of qh[l].
    """
    ql = src[0:128]
    qh = src[128:192]
    sc = src[192:208]
    d = f16(src[208:210])
    out = [0.0] * 256
    for half in range(2):
        ql_off = half * 64
        qh_off = half * 32
        out_off = half * 128
        # The scale pointer advances by 8 per half in ggml, so the second half uses sc[8..15].
        sc_off = half * 8
        for l in range(32):
            is_ = l // 16
            q1 = (ql[ql_off + l] & 0x0F) | (((qh[qh_off + l] >> 0) & 3) << 4)
            q2 = (ql[ql_off + l + 32] & 0x0F) | (((qh[qh_off + l] >> 2) & 3) << 4)
            q3 = (ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)
            q4 = (ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)
            s0 = d * struct.unpack("<b", bytes([sc[sc_off + is_]]))[0]
            s2 = d * struct.unpack("<b", bytes([sc[sc_off + is_ + 2]]))[0]
            s4 = d * struct.unpack("<b", bytes([sc[sc_off + is_ + 4]]))[0]
            s6 = d * struct.unpack("<b", bytes([sc[sc_off + is_ + 6]]))[0]
            out[out_off + l] = s0 * (q1 - 32)
            out[out_off + l + 32] = s2 * (q2 - 32)
            out[out_off + l + 64] = s4 * (q3 - 32)
            out[out_off + l + 96] = s6 * (q4 - 32)
    return out


# ggml's kvalues_iq4nl. A nibble indexes this table rather than being read as a signed offset,
# which is what separates IQ4_NL from Q4_0 -- and reading it as an offset produces output of the
# right shape and entirely the wrong values.
KVALUES_IQ4NL = [
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113,
]


def dequant_iq4_nl(src: bytes) -> list[float]:
    """IQ4_NL: 2-byte f16 scale, then 16 packed nibbles in Q4_0's split-half order."""
    d = f16(src[0:2])
    qs = src[2:18]
    out = [0.0] * 32
    for j in range(16):
        out[j] = d * KVALUES_IQ4NL[qs[j] & 0x0F]
        out[j + 16] = d * KVALUES_IQ4NL[qs[j] >> 4]
    return out


DECODERS = {
    "Q4_0": dequant_q4_0,
    "Q4_1": dequant_q4_1,
    "Q5_0": dequant_q5_0,
    "Q5_1": dequant_q5_1,
    "Q8_0": dequant_q8_0,
    "BF16": dequant_bf16,
    "F16": dequant_f16,
    "Q4_K": dequant_q4_k,
    "Q6_K": dequant_q6_k,
    "IQ4_NL": dequant_iq4_nl,
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
        # Sub-block count per encoded block. Named rather than inferred from the format name:
        # IQ4_NL is 32 weights but does not start with "Q".
        BLOCKS = {"Q4_0": 32, "Q4_1": 32, "Q5_0": 32, "Q5_1": 32, "Q8_0": 32, "IQ4_NL": 32,
                  "BF16": 1, "F16": 1, "Q4_K": 256, "Q6_K": 256}
        block = BLOCKS[name]
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
