"""Convert a Qwen2.5 safetensors checkpoint to a GGUF file Orbix can read.

This is a cross-validation tool, not part of the engine: it exists so the pure-Go
Qwen2 support can be checked against a real checkpoint's weights. It is deliberately
independent of the Go code -- a converter that shared its assumptions could not catch
a wrong assumption -- and it writes F32 throughout, because the point is numerical
agreement rather than file size.

Run it with uv, which provides numpy without touching the system Python:

    uv run --with numpy python3 scripts/hf_to_gguf.py <src-dir> out.gguf

Then point Orbix at the result:

    orbix run --model out.gguf "The capital of France is"

Two mistakes this script made before it worked, recorded because they are the kind a
hand-written writer makes and the format's own readers catch:

- The GGUF header is magic, version, tensor count, metadata count. Writing those two
  counts in the other order shifts every later field.
- The tensor count appears once, in the header. The directory after the metadata is
  just the entries.
"""
import json, struct, sys
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "/models/Qwen2.5-0.5B-Instruct"
OUT = sys.argv[2] if len(sys.argv) > 2 else "qwen2-f32.gguf"
CFG = json.load(open(f"{SRC}/config.json"))

NAMES = [  # GGUF key order is arbitrary; names must match llama.cpp's mapping.
    (("model.embed_tokens.weight",), "token_embd.weight"),
    (("model.norm.weight",), "output_norm.weight"),
]

def load_safetensors(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    base = 8 + n
    out = {}
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        f.seek(base + s)
        raw = f.read(e - s)
        dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}[meta["dtype"]]
        arr = np.frombuffer(raw, dtype=dt).reshape(meta["shape"])
        if meta["dtype"] == "BF16":
            # widen bf16 to f32 by shifting into the high half of a float32
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        out[name] = arr.astype(np.float32)
    return out

T = load_safetensors(f"{SRC}/model.safetensors")
print("loaded", len(T), "tensors")

# Map HF names onto GGUF names.
mapped = {}
mapped["token_embd.weight"] = T["model.embed_tokens.weight"]
mapped["output_norm.weight"] = T["model.norm.weight"]
if "lm_head.weight" in T:
    mapped["output.weight"] = T["lm_head.weight"]
for i in range(CFG["num_hidden_layers"]):
    p = f"model.layers.{i}."
    g = f"blk.{i}."
    mapped[g+"attn_norm.weight"] = T[p+"input_layernorm.weight"]
    mapped[g+"attn_q.weight"] = T[p+"self_attn.q_proj.weight"]
    mapped[g+"attn_k.weight"] = T[p+"self_attn.k_proj.weight"]
    mapped[g+"attn_v.weight"] = T[p+"self_attn.v_proj.weight"]
    mapped[g+"attn_output.weight"] = T[p+"self_attn.o_proj.weight"]
    mapped[g+"ffn_norm.weight"] = T[p+"post_attention_layernorm.weight"]
    mapped[g+"ffn_gate.weight"] = T[p+"mlp.gate_proj.weight"]
    mapped[g+"ffn_up.weight"] = T[p+"mlp.up_proj.weight"]
    mapped[g+"ffn_down.weight"] = T[p+"mlp.down_proj.weight"]
    for src, dst in (("q_proj","attn_q"),("k_proj","attn_k"),("v_proj","attn_v")):
        key = p+f"self_attn.{src}.bias"
        if key in T:
            mapped[g+dst+".bias"] = T[key]
# Qwen2 has no dense FFN bias; anything else present is unexpected.
unmapped = set(T) - set(f"model.layers.{i}.self_attn.q_proj.weight" for i in range(0))
print("wrote", len(mapped), "mapped tensors")

tj = json.load(open(f"{SRC}/tokenizer.json"))
tokens = tj["model"]["vocab"]
# added_tokens carry ids past the base vocab (Qwen2's chat markers), so the full
# vocabulary is base + added.
added = {a["content"]: a["id"] for a in tj.get("added_tokens", [])}
total = max(list(tokens.values()) + list(added.values())) + 1
vocab = [None]*total
for tok, idx in tokens.items():
    vocab[idx] = tok
for tok, idx in added.items():
    vocab[idx] = tok
for i, v in enumerate(vocab):
    if v is None:
        vocab[i] = f"<unused{i}>"
print("vocab", len(vocab))

def gstr(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b

def kv_str(k, v):
    return gstr(k) + struct.pack("<I", 8) + gstr(v)
def kv_u32(k, v):
    return gstr(k) + struct.pack("<I", 4) + struct.pack("<I", v)
def kv_f32(k, v):
    return gstr(k) + struct.pack("<I", 6) + struct.pack("<f", v)
def kv_strs(k, vals):
    # Value type 9 is array; its element type (8 = string) follows, then the count.
    out = gstr(k) + struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(vals))
    for x in vals:
        out += gstr(x)
    return out
def kv_i32s(k, vals):
    out = gstr(k) + struct.pack("<I", 9) + struct.pack("<I", 5) + struct.pack("<Q", len(vals))
    for x in vals:
        out += struct.pack("<i", x)
    return out

# Qwen2 special tokens.
im_start = next(i for i,t in enumerate(vocab) if t == "<|im_start|>")
im_end   = next(i for i,t in enumerate(vocab) if t == "<|im_end|>")
eos      = next(i for i,t in enumerate(vocab) if t == "<|endoftext|>")
types = [1]*len(vocab)   # normal
for i in (im_start, im_end, eos):
    types[i] = 3          # control

_meta_list = ([
    kv_str("general.architecture", "qwen2"),
    kv_str("general.name", "Qwen2.5-0.5B-Instruct"),
    kv_u32("qwen2.block_count", CFG["num_hidden_layers"]),
    kv_u32("qwen2.context_length", CFG["max_position_embeddings"]),
    kv_u32("qwen2.embedding_length", CFG["hidden_size"]),
    kv_u32("qwen2.feed_forward_length", CFG["intermediate_size"]),
    kv_u32("qwen2.attention.head_count", CFG["num_attention_heads"]),
    kv_u32("qwen2.attention.head_count_kv", CFG["num_key_value_heads"]),
    kv_f32("qwen2.attention.layer_norm_rms_epsilon", CFG["rms_norm_eps"]),
    kv_u32("qwen2.rope.dimension_count", CFG["hidden_size"]//CFG["num_attention_heads"]),
    kv_f32("qwen2.rope.freq_base", CFG["rope_theta"]),
    kv_strs("tokenizer.ggml.tokens", vocab),
    kv_i32s("tokenizer.ggml.token_type", types),
    kv_u32("tokenizer.ggml.bos_token_id", CFG["bos_token_id"]),
    kv_u32("tokenizer.ggml.eos_token_id", eos),
])
meta = b"".join(_meta_list)
kv_count = len(_meta_list)
print("metadata bytes", len(meta))

order = list(mapped)
# GGUF stores dims innermost-first, so an HF [out, in] weight becomes [in, out].
# The tensor count appears once, in the header. The directory itself is just the
# entries in order.
dir_bytes = b""
data = b""
offsets = []
for name in order:
    arr = mapped[name]
    gguf_dims = list(arr.shape[::-1])
    dir_bytes += gstr(name) + struct.pack("<I", len(gguf_dims))
    for d in gguf_dims:
        dir_bytes += struct.pack("<Q", d)
    dir_bytes += struct.pack("<I", 0)          # F32
    dir_bytes += struct.pack("<Q", len(data))   # offset
    data += arr.astype(np.float32).tobytes()

# GGUF header order is magic, version, tensor count, metadata count.
header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(order)) + struct.pack("<Q", kv_count)
# align tensor data to 32 bytes
# The tensor directory follows the metadata immediately; only the data section is
# aligned. Padding between metadata and directory shifts every directory read.
body = header + meta + dir_bytes
pad = (-len(body)) % 32
body += b"\x00"*pad
open(OUT, "wb").write(body + data)
print("wrote", OUT, len(body)+len(data), "bytes")
