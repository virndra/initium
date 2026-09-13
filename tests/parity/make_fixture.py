#!/usr/bin/env python3
"""Build a tiny synthetic llama2.c-format .bin model + tokenizer.bin
so we can run initium's real forward pass and compare it against the
project's own Python reference (tests/parity/llama2c_ref.py).

No downloads, no torch, no transformers — just numpy.
Fixed RNG seed for full determinism.
"""
import struct
import numpy as np
from pathlib import Path
import argparse

rng = np.random.default_rng(42)

dim = 8
hidden_dim = 16
n_layers = 2
n_heads = 2
n_kv_heads = 1          # GQA: 2 query heads share 1 kv head
vocab_size = 260         # 3 special + 256 byte tokens + 1 merge token "hi"
seq_len = 32
hd = dim // n_heads
kv_dim = n_kv_heads * hd


def randf(*shape):
    return rng.standard_normal(shape).astype(np.float32) * 0.1


def write_model(path: Path):
    with open(path, "wb") as f:
        # header: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size(+shared), seq_len
        f.write(struct.pack("<7i", dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len))

        f.write(randf(vocab_size, dim).tobytes())                     # token_embedding
        for _ in range(n_layers): f.write((np.ones(dim, dtype=np.float32)).tobytes())   # rms_att (=1 for simplicity)
        for _ in range(n_layers): f.write(randf(dim, dim).tobytes())                    # wq
        for _ in range(n_layers): f.write(randf(kv_dim, dim).tobytes())                 # wk
        for _ in range(n_layers): f.write(randf(kv_dim, dim).tobytes())                 # wv
        for _ in range(n_layers): f.write(randf(dim, dim).tobytes())                    # wo
        for _ in range(n_layers): f.write((np.ones(dim, dtype=np.float32)).tobytes())   # rms_ffn
        for _ in range(n_layers): f.write(randf(hidden_dim, dim).tobytes())             # w1 (gate)
        for _ in range(n_layers): f.write(randf(dim, hidden_dim).tobytes())             # w2 (down)
        for _ in range(n_layers): f.write(randf(hidden_dim, dim).tobytes())             # w3 (up)
        f.write(np.ones(dim, dtype=np.float32).tobytes())              # rms_final
        f.write(np.zeros(seq_len * (hd // 2), dtype=np.float32).tobytes())  # freq_cis_real (unused by either impl)
        f.write(np.zeros(seq_len * (hd // 2), dtype=np.float32).tobytes())  # freq_cis_imag
        # shared embeddings -> no separate wcls block

    print(f"wrote {path} (dim={dim} hid={hidden_dim} L={n_layers} H={n_heads} KV={n_kv_heads} V={vocab_size})")


def write_tokenizer(path: Path):
    # tokenizer.bin: llama2.c format
    # 0 <unk> 1 <s> 2 </s>, then 256 single-byte tokens (id = byte + 3), then one merge token "hi"
    with open(path, "wb") as f:
        f.write(struct.pack("<I", 8))  # max_token_length (generous)
        entries = []
        entries.append((-1e9, b"<unk>"))
        entries.append((-1e9, b"<s>"))
        entries.append((-1e9, b"</s>"))
        for b in range(256):
            entries.append((-1e6, bytes([b])))  # low score: never a preferred BPE merge target
        entries.append((10.0, b"hi"))  # merge token, high score so "h"+"i" -> "hi"
        assert len(entries) == vocab_size
        for score, piece in entries:
            f.write(struct.pack("<f", score))
            f.write(struct.pack("<I", len(piece)))
            f.write(piece)

    print(f"wrote {path} (vocab_size={vocab_size})")


def main():
    ap = argparse.ArgumentParser(description="Generate tiny synthetic model for offline parity testing")
    ap.add_argument("--model-out", type=Path, default=Path("testdata/tiny.bin"))
    ap.add_argument("--tok-out", type=Path, default=Path("testdata/tiny_tok.bin"))
    args = ap.parse_args()

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.tok_out.parent.mkdir(parents=True, exist_ok=True)

    write_model(args.model_out)
    write_tokenizer(args.tok_out)


if __name__ == "__main__":
    main()
