#!/usr/bin/env python3
"""Dump reference token IDs + per-step fp32 logits for Initium parity tests.

Formats a flat binary "INI1" file (see src/verify.h).

Modes:
  --self-test   Write a synthetic dump and exit (M0 plumbing gate; no torch).
  --model HF    Use transformers/torch to dump greedy logits (requires deps).
  --llama2c     Use a tiny pure-Python llama2c-style forward if available.

M0 gate: --self-test produces a dump; compare.py dump dump must pass.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


MAGIC = b"INI1"
VERSION = 1


def write_dump(
    path: Path,
    prompt_tokens: list[int],
    vocab_size: int,
    steps: list[tuple[list[float], int]],
) -> None:
    """steps: list of (logits[vocab], chosen_token)."""
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", len(prompt_tokens)))
        f.write(struct.pack("<I", len(steps)))
        if prompt_tokens:
            f.write(struct.pack(f"<{len(prompt_tokens)}I", *prompt_tokens))
        for logits, chosen in steps:
            assert len(logits) == vocab_size
            f.write(struct.pack(f"<{vocab_size}f", *logits))
            f.write(struct.pack("<I", chosen))


def load_dump(path: Path):
    with path.open("rb") as f:
        magic = f.read(4)
        assert magic == MAGIC, magic
        (version,) = struct.unpack("<I", f.read(4))
        assert version == VERSION
        vocab_size, n_prompt, n_steps = struct.unpack("<III", f.read(12))
        prompt = list(struct.unpack(f"<{n_prompt}I", f.read(4 * n_prompt))) if n_prompt else []
        steps = []
        for _ in range(n_steps):
            logits = list(struct.unpack(f"<{vocab_size}f", f.read(4 * vocab_size)))
            (chosen,) = struct.unpack("<I", f.read(4))
            steps.append((logits, chosen))
        return {
            "vocab_size": vocab_size,
            "prompt": prompt,
            "steps": steps,
        }


def self_test(out: Path) -> None:
    """Synthetic dump for M0 plumbing — no ML deps."""
    vocab = 32
    prompt = [1, 5, 7]
    steps = []
    for s in range(8):
        logits = [0.0] * vocab
        # peak at token (10 + s) % vocab
        peak = (10 + s) % vocab
        logits[peak] = 5.0 + s * 0.1
        logits[(peak + 1) % vocab] = 1.0
        steps.append((logits, peak))
    write_dump(out, prompt, vocab, steps)
    # round-trip
    d = load_dump(out)
    assert d["vocab_size"] == vocab
    assert d["prompt"] == prompt
    assert len(d["steps"]) == 8
    print(f"self-test dump written: {out} ({vocab} vocab, {len(steps)} steps)")


def dump_hf(model_id: str, prompt: str, n_steps: int, out: Path) -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print("transformers/torch required for --model mode:", e, file=sys.stderr)
        sys.exit(2)

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    inputs = tok(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"][0].tolist()
    vocab = model.config.vocab_size
    steps = []
    ids = list(input_ids)

    with torch.no_grad():
        for _ in range(n_steps):
            t = torch.tensor([ids], dtype=torch.long)
            out_m = model(t)
            logits = out_m.logits[0, -1].float().cpu().tolist()
            # greedy
            chosen = max(range(len(logits)), key=lambda i: logits[i])
            steps.append((logits, chosen))
            ids.append(chosen)

    write_dump(out, input_ids, vocab, steps)
    print(f"HF dump: model={model_id} prompt_tokens={len(input_ids)} steps={n_steps} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Initium parity reference dump")
    ap.add_argument("--out", type=Path, default=Path("testdata/ref.bin"))
    ap.add_argument("--self-test", action="store_true", help="M0 synthetic dump")
    ap.add_argument("--model", type=str, default=None, help="HF model id")
    ap.add_argument("--prompt", type=str, default="Once upon a time")
    ap.add_argument("-n", type=int, default=64, help="generation steps")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.self_test:
        self_test(args.out)
        return
    if args.model:
        dump_hf(args.model, args.prompt, args.n, args.out)
        return

    print("Specify --self-test or --model <hf_id>", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
