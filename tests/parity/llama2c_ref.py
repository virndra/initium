#!/usr/bin/env python3
"""Pure-Python reference matching karpathy/llama2.c (fp32) for Initium parity.

Loads stories260K-style .bin + tokenizer.bin, greedy-decodes, writes INI1 dumps.
"""
from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np

# ---- INI1 dump (same as dump_reference.py / src/verify.h) ----

def write_INI1(path: Path, prompt_tokens: list[int], vocab_size: int,
               steps: list[tuple[np.ndarray, int]]) -> None:
    with path.open("wb") as f:
        f.write(b"INI1")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", len(prompt_tokens)))
        f.write(struct.pack("<I", len(steps)))
        if prompt_tokens:
            f.write(struct.pack(f"<{len(prompt_tokens)}I", *prompt_tokens))
        for logits, chosen in steps:
            assert logits.dtype == np.float32 and logits.shape == (vocab_size,)
            f.write(logits.tobytes())
            f.write(struct.pack("<I", int(chosen)))


# ---- kernels (mirror llama2.c) ----

def rmsnorm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    ss = float(np.dot(x, x)) / x.shape[0]
    ss = 1.0 / math.sqrt(ss + 1e-5)
    return (weight * (ss * x)).astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    m = float(np.max(x))
    e = np.exp(x - m)
    return (e / e.sum()).astype(np.float32)


def matmul(x: np.ndarray, w: np.ndarray, n: int, d: int) -> np.ndarray:
    """W (d,n) @ x (n,) -> (d,)  — row-major W"""
    return (w.reshape(d, n) @ x).astype(np.float32)


# ---- model ----

class Config:
    def __init__(self, dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.seq_len = seq_len


class Transformer:
    def __init__(self, path: Path):
        data = path.read_bytes()
        raw = struct.unpack_from("<7i", data, 0)
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = raw
        shared = vocab_size > 0
        vocab_size = abs(vocab_size)
        self.cfg = Config(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)
        self.shared = shared

        # weights after 7 ints
        off = 28
        floats = np.frombuffer(data, dtype=np.float32, offset=off).copy()
        p = 0
        c = self.cfg
        head_size = c.dim // c.n_heads
        kv_dim = c.n_kv_heads * head_size

        def take(n):
            nonlocal p
            sl = floats[p:p + n]
            p += n
            return sl

        self.token_embedding = take(c.vocab_size * c.dim).reshape(c.vocab_size, c.dim)
        self.rms_att = take(c.n_layers * c.dim).reshape(c.n_layers, c.dim)
        self.wq = take(c.n_layers * c.dim * c.dim)
        self.wk = take(c.n_layers * c.dim * kv_dim)
        self.wv = take(c.n_layers * c.dim * kv_dim)
        self.wo = take(c.n_layers * c.dim * c.dim)
        self.rms_ffn = take(c.n_layers * c.dim).reshape(c.n_layers, c.dim)
        self.w1 = take(c.n_layers * c.dim * c.hidden_dim)
        self.w2 = take(c.n_layers * c.hidden_dim * c.dim)
        self.w3 = take(c.n_layers * c.dim * c.hidden_dim)
        self.rms_final = take(c.dim)
        # skip freq_cis
        take(c.seq_len * head_size // 2)
        take(c.seq_len * head_size // 2)
        if shared:
            self.wcls = self.token_embedding.reshape(-1)
        else:
            self.wcls = take(c.vocab_size * c.dim)

        self.kv_dim = kv_dim
        self.head_size = head_size
        self.kv_mul = c.n_heads // c.n_kv_heads

        self.key_cache = np.zeros((c.n_layers, c.seq_len, kv_dim), dtype=np.float32)
        self.value_cache = np.zeros((c.n_layers, c.seq_len, kv_dim), dtype=np.float32)

        expected = p
        if p > floats.size:
            raise RuntimeError(f"weight overrun: need {p} have {floats.size}")

    def forward(self, token: int, pos: int) -> np.ndarray:
        c = self.cfg
        dim = c.dim
        kv_dim = self.kv_dim
        head_size = self.head_size
        kv_mul = self.kv_mul
        hidden_dim = c.hidden_dim

        x = self.token_embedding[token].copy()

        for l in range(c.n_layers):
            xb = rmsnorm(x, self.rms_att[l])

            wq = self.wq[l * dim * dim:(l + 1) * dim * dim]
            wk = self.wk[l * dim * kv_dim:(l + 1) * dim * kv_dim]
            wv = self.wv[l * dim * kv_dim:(l + 1) * dim * kv_dim]
            wo = self.wo[l * dim * dim:(l + 1) * dim * dim]

            q = matmul(xb, wq, dim, dim)
            k = matmul(xb, wk, dim, kv_dim)
            v = matmul(xb, wv, dim, kv_dim)

            # RoPE exactly as llama2.c: over concatenated dim indices
            for i in range(0, dim, 2):
                head_dim = i % head_size
                freq = 1.0 / (10000.0 ** (head_dim / float(head_size)))
                val = pos * freq
                fcr = math.cos(val)
                fci = math.sin(val)
                rotn = 2 if i < kv_dim else 1
                for vv in range(rotn):
                    vec = q if vv == 0 else k
                    v0, v1 = float(vec[i]), float(vec[i + 1])
                    vec[i] = v0 * fcr - v1 * fci
                    vec[i + 1] = v0 * fci + v1 * fcr

            self.key_cache[l, pos] = k
            self.value_cache[l, pos] = v

            xb_out = np.zeros(dim, dtype=np.float32)
            for h in range(c.n_heads):
                qh = q[h * head_size:(h + 1) * head_size]
                scores = np.empty(pos + 1, dtype=np.float32)
                for t in range(pos + 1):
                    kh = self.key_cache[l, t, (h // kv_mul) * head_size:
                                       (h // kv_mul) * head_size + head_size]
                    scores[t] = float(np.dot(qh, kh)) / math.sqrt(head_size)
                scores = softmax(scores)
                oh = np.zeros(head_size, dtype=np.float32)
                for t in range(pos + 1):
                    vh = self.value_cache[l, t, (h // kv_mul) * head_size:
                                          (h // kv_mul) * head_size + head_size]
                    oh += scores[t] * vh
                xb_out[h * head_size:(h + 1) * head_size] = oh

            x = x + matmul(xb_out, wo, dim, dim)

            xb = rmsnorm(x, self.rms_ffn[l])
            w1 = self.w1[l * dim * hidden_dim:(l + 1) * dim * hidden_dim]
            w2 = self.w2[l * hidden_dim * dim:(l + 1) * hidden_dim * dim]
            w3 = self.w3[l * dim * hidden_dim:(l + 1) * dim * hidden_dim]
            hb = matmul(xb, w1, dim, hidden_dim)
            hb2 = matmul(xb, w3, dim, hidden_dim)
            # silu * hb2
            hb = hb * (1.0 / (1.0 + np.exp(-hb)))
            hb = (hb * hb2).astype(np.float32)
            x = x + matmul(hb, w2, hidden_dim, dim)

        x = rmsnorm(x, self.rms_final)
        logits = matmul(x, self.wcls, dim, c.vocab_size)
        return logits


# ---- tokenizer (llama2.c encode) ----

class Tokenizer:
    def __init__(self, path: Path, vocab_size: int):
        data = path.read_bytes()
        (max_token_length,) = struct.unpack_from("<I", data, 0)
        self.max_token_length = max_token_length
        self.vocab: list[str] = []
        self.scores: list[float] = []
        off = 4
        for _ in range(vocab_size):
            score = struct.unpack_from("<f", data, off)[0]
            off += 4
            (length,) = struct.unpack_from("<I", data, off)
            off += 4
            piece = data[off:off + length]
            off += length
            # raw bytes as latin-1 so 1:1 with C char*
            self.vocab.append(piece.decode("latin-1"))
            self.scores.append(score)
        self.sorted = sorted([(s, i) for i, s in enumerate(self.vocab)], key=lambda x: x[0])
        self.sorted_map = {s: i for s, i in self.sorted}

    def lookup(self, s: str) -> int:
        return self.sorted_map.get(s, -1)

    def encode(self, text: str, bos: bool = True) -> list[int]:
        tokens: list[int] = []
        if bos:
            tokens.append(1)
        if text:
            dummy = self.lookup(" ")
            if dummy >= 0:
                tokens.append(dummy)

        # process UTF-8 codepoints (text is str; encode to utf-8 bytes then decode codepoints)
        raw = text.encode("utf-8")
        i = 0
        while i < len(raw):
            # gather one UTF-8 codepoint
            c0 = raw[i]
            if c0 < 0x80:
                n = 1
            elif (c0 & 0xE0) == 0xC0:
                n = 2
            elif (c0 & 0xF0) == 0xE0:
                n = 3
            elif (c0 & 0xF8) == 0xF0:
                n = 4
            else:
                n = 1
            piece = raw[i:i + n].decode("latin-1")
            i += n
            tid = self.lookup(piece)
            if tid != -1:
                tokens.append(tid)
            else:
                for b in piece.encode("latin-1"):
                    tokens.append(b + 3)

        # BPE merges
        while True:
            best_score = -1e10
            best_id = -1
            best_idx = -1
            for j in range(len(tokens) - 1):
                merged = self.vocab[tokens[j]] + self.vocab[tokens[j + 1]]
                mid = self.lookup(merged)
                if mid != -1 and self.scores[mid] > best_score:
                    best_score = self.scores[mid]
                    best_id = mid
                    best_idx = j
            if best_idx < 0:
                break
            tokens[best_idx] = best_id
            del tokens[best_idx + 1]
        return tokens

    def decode_piece(self, prev: int, token: int) -> str:
        piece = self.vocab[token]
        if prev == 1 and piece.startswith(" "):
            piece = piece[1:]
        return piece


def greedy_dump(model: Transformer, tok: Tokenizer, prompt: str, n_steps: int,
                out: Path) -> None:
    prompt_tokens = tok.encode(prompt, bos=True)
    steps = []
    token = prompt_tokens[0]
    pos = 0
    # reset caches
    model.key_cache[:] = 0
    model.value_cache[:] = 0

    gen_count = 0
    while gen_count < n_steps:
        logits = model.forward(token, pos)
        if pos < len(prompt_tokens) - 1:
            nxt = prompt_tokens[pos + 1]
        else:
            chosen = int(np.argmax(logits))
            steps.append((logits.copy(), chosen))
            nxt = chosen
            gen_count += 1
            # print
            piece = tok.decode_piece(token, nxt)
            print(piece, end="", flush=True)
            if nxt == 1:  # BOS ends like llama2.c
                break
        pos += 1
        token = nxt
        if pos >= model.cfg.seq_len:
            break
    print()
    write_INI1(out, prompt_tokens, model.cfg.vocab_size, steps)
    print(f"wrote {out}: prompt_tokens={len(prompt_tokens)} steps={len(steps)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=Path, required=True)
    ap.add_argument("-z", "--tokenizer", type=Path, required=True)
    ap.add_argument("-p", "--prompt", type=str, default="Once upon a time")
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("testdata/tiny_ref.bin"))
    args = ap.parse_args()

    model = Transformer(args.model)
    tok = Tokenizer(args.tokenizer, model.cfg.vocab_size)
    print(f"cfg dim={model.cfg.dim} layers={model.cfg.n_layers} vocab={model.cfg.vocab_size}",
          flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    greedy_dump(model, tok, args.prompt, args.n, args.out)


if __name__ == "__main__":
    main()
