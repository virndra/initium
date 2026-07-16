#!/usr/bin/env python3
"""Compare two INI1 reference dumps (or dump vs itself for M0 gate).

Tier A: max |dlogit| < 1e-3 AND argmax match every step.
Tier B: argmax match ≥ 63/64 and top-5 set match every step.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


def load(path: Path):
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"INI1":
            raise SystemExit(f"bad magic in {path}: {magic!r}")
        (version,) = struct.unpack("<I", f.read(4))
        vocab_size, n_prompt, n_steps = struct.unpack("<III", f.read(12))
        prompt = list(struct.unpack(f"<{n_prompt}I", f.read(4 * n_prompt))) if n_prompt else []
        steps = []
        for _ in range(n_steps):
            logits = list(struct.unpack(f"<{vocab_size}f", f.read(4 * vocab_size)))
            (chosen,) = struct.unpack("<I", f.read(4))
            steps.append((logits, chosen))
        return vocab_size, prompt, steps


def argmax(xs):
    best_i, best_v = 0, xs[0]
    for i, v in enumerate(xs):
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def top5(xs):
    idx = sorted(range(len(xs)), key=lambda i: xs[i], reverse=True)[:5]
    return set(idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--tier", choices=["A", "B"], default="A")
    args = ap.parse_args()

    va, pa, sa = load(args.a)
    vb, pb, sb = load(args.b)
    if va != vb:
        print(f"vocab mismatch {va} vs {vb}", file=sys.stderr)
        sys.exit(1)
    n = min(len(sa), len(sb))
    if n == 0:
        print("no steps", file=sys.stderr)
        sys.exit(1)

    worst = 0.0
    argmax_ok = 0
    top5_ok = 0
    for i in range(n):
        la, _ = sa[i]
        lb, _ = sb[i]
        mad = max(abs(x - y) for x, y in zip(la, lb))
        if mad > worst:
            worst = mad
        aa, ab = argmax(la), argmax(lb)
        if aa == ab:
            argmax_ok += 1
        if top5(la) == top5(lb):
            top5_ok += 1

    print(f"steps={n} worst_max_abs={worst:.6g} argmax_match={argmax_ok}/{n} top5_match={top5_ok}/{n}")

    if args.tier == "A":
        ok = worst < 1e-3 and argmax_ok == n
    else:
        # ≥ 63/64 argmax if n>=64 else all; top5 all
        need = n if n < 64 else n - 1
        ok = argmax_ok >= need and top5_ok == n

    if ok:
        print(f"PASS tier {args.tier}")
        sys.exit(0)
    print(f"FAIL tier {args.tier}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
