# Initium

**From-scratch LLM inference engine in C11** — loads real Llama-family models and generates text with correctness gated against a reference harness.

> **Prime directive:** Correctness is non-negotiable. Speed is earned incrementally.

## What is this? (ELI5)

Imagine a huge brain (the *model*) stored as numbers in a file. Initium is a small, careful program that:

1. **Reads** those numbers (weights)
2. **Turns your words into numbers** (tokenizer)
3. **Runs the math** layer by layer (attention, feed-forward)
4. **Picks the next word** again and again (sampler)

No PyTorch. No CUDA. Just C, libc, libm, and pthreads.

## Build

```bash
# Linux / macOS / MinGW
make

# Windows (LLVM-MinGW) — use mingw32-make if `make` is missing:
mingw32-make
```

Targets:

| Target         | Meaning                          |
|----------------|----------------------------------|
| `initium`        | Optimized binary                 |
| `initium_debug`  | ASan + UBSan debug build         |
| `test`         | Unit tests + M0 parity self-check|
| `bench`        | Hint for `--stats` runs          |
| `clean`        | Remove build products            |

## Download models

```bash
# bash
bash scripts/download_models.sh

# PowerShell
powershell -File scripts/download_models.ps1
```

This fetches **TinyStories-260K** (`stories260K.bin` + `tok512.bin`) — small, fast, perfect for M1 gates.

## Run

```bash
./initium -m models/stories260K.bin --tokenizer models/tok512.bin \
  -p "Once upon a time" -n 64 --greedy --stats

./initium -m models/stories260K.bin --tokenizer models/tok512.bin --chat

./initium -m models/stories260K.bin --tokenizer models/tok512.bin \
  -p "..." --greedy --verify testdata/ref.bin
```

## CLI

| Flag | Description |
|------|-------------|
| `-m <path>` | Model (llama2.c `.bin` or GGUF) |
| `--tokenizer <path>` | `tokenizer.bin` for `.bin` models |
| `-p <string>` | Prompt |
| `--chat` | Interactive REPL |
| `-n <int>` | Max new tokens (default 256) |
| `-t <float>` | Temperature (0 = greedy) |
| `--top-p` / `--top-k` | Nucleus / top-k sampling |
| `--seed` | RNG seed (reproducible sampling) |
| `--ctx` | Context length override |
| `--threads` | Worker count |
| `--greedy` | Force argmax |
| `--verify <path>` | Compare per-step logits to INI1 dump |
| `--stats` | Prefill/decode tok/s on stderr |
| `--no-simd` | Scalar kernels only |

## Milestones

| Gate | Status | Notes |
|------|--------|-------|
| M0 Scaffold + parity harness | **PASS** | INI1 self-compare; unit tests green |
| M1 fp32 forward TinyStories | **PASS** | tier A vs `llama2c_ref.py`: 64/64 steps, max‖Δ‖≈1.7e-5 |
| M2 KV cache | **PASS** | same path as reference (cache is pure opt; parity proves correctness) |
| M3 Tokenizer + sampler | **partial** | llama2.c BPE + merges; greedy decode spacing fixed |
| M4 Full GGUF + fp16 | parser done | weight wiring still TODO |
| M5 Thread pool | API ready | matmul not yet parallel |
| M6 Q8_0 / Q4_0 | dequant stubs | block layouts present |
| M7 SIMD | pending | NEON on this ARM64 box |
| M8 Polish | pending | README results table |

### M1 parity numbers (this machine, ARM64 Windows)

| Prompt | Steps | Tier A | worst max_abs |
|--------|------:|--------|---------------|
| Once upon a time | 64 | 64/64 | 1.72e-5 |
| Hello | 32 | 32/32 | 1.72e-5 |
| The quick brown fox | 32 | 32/32 | 1.43e-5 |
| Once upon a time there was | 32 | 32/32 | 1.24e-5 |

Reference dumper: `python tests/parity/llama2c_ref.py -m models/stories260K.bin -z models/tok512.bin -p "..." -n 64 --out testdata/ref.bin`

## Parity harness

```bash
python tests/parity/dump_reference.py --self-test --out testdata/self_ref.bin
python tests/parity/compare.py testdata/self_ref.bin testdata/self_ref.bin
```

Reference dump format **INI1** is documented in `src/verify.h`.

## Layout

See the PRD. Math lives **only** in `src/kernels.c`.

## License

MIT (engine). Models retain their own licenses.
