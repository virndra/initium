# initium

a llm inference engine written from scratch in c11. you give it a llama style model file and a prompt, and it generates text on your cpu. that is the whole product.

there is no pytorch, no cuda, no external libraries at all. the only things it links against are the c math library and pthreads. every piece of the pipeline, the file parsing, the tokenizer, the transformer math, the sampling, the threading, is written by hand in this repo. the point of the project is to understand how llm inference actually works at the lowest level, and the best way to prove you understand something is to build it and then test it against a known good reference until the numbers match exactly.

## what it can do

- load and run llama2.c style .bin models (the karpathy format, like the tinystories models)
- load and run gguf models, the format llama.cpp uses. tested end to end with tinyllama 1.1b chat
- handle q4_0 and q6_k quantized weights. they get dequantized to float32 at load time so the forward pass stays simple
- tokenize text with a byte pair encoding tokenizer written from scratch, sentencepiece style, with byte fallback for characters that are not in the vocab. output is verified token by token against the reference implementation
- sample with greedy argmax, temperature, top-p (nucleus), and top-k. the rng is seedable, so the same seed and same prompt always give the same output
- run the heavy matmuls across multiple threads with a simd path, and fall back to plain scalar c if you ask it to
- verify itself. there is a parity mode that compares the logits it produces against a reference dump, number by number. correctness here is measured, not assumed

## how it works, the short version

when you run it, this is what happens:

1. the model file is opened and parsed. for gguf, the metadata (dimensions, layer count, head count, rope theta, and the whole tokenizer vocab) is read out of the file header. for .bin models, a separate tokenizer.bin is loaded
2. the weights are loaded into ram as float32. quantized tensors (q4_0, q6_k) are unpacked block by block into floats
3. your prompt is encoded into token ids by the bpe tokenizer. it starts from raw bytes and repeatedly merges the pair with the best score until no merges are left, same as sentencepiece
4. the transformer runs one token at a time. each step goes through every layer: rmsnorm, attention with rotary position embeddings (rope), the key value cache so old tokens are not recomputed, then the feed forward block (silu gate, elementwise multiply, down projection), with residual connections around both
5. the final logits go to the sampler, which picks the next token. that token gets decoded back to text, printed, and fed back in as input for the next step
6. repeat until you hit the token limit or the model emits end of sequence

grouped query attention is supported, which is why tinyllama (32 query heads, 4 kv heads) works.

## build

you need a c11 compiler and make. gcc and clang both work.

```bash
make
```

on windows with mingw:

```bash
mingw32-make
```

that produces the `initium` binary. there is also a debug build with address sanitizer and undefined behavior sanitizer turned on, useful if you are hacking on the code:

```bash
make initium_debug
```

## get a model

the easy way is the download script, which pulls the tinystories models. they are tiny (the smallest is 260k parameters) and perfect for testing that everything works:

```bash
powershell -file scripts/download_models.ps1
```

or on linux and mac:

```bash
bash scripts/download_models.sh
```

for a real model, download any q4_0 gguf from hugging face. tinyllama 1.1b chat is what this was tested with:

```
https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
```

drop the file in `models/` (or anywhere, you just pass the path).

## run

with a llama2.c .bin model you also need to pass the tokenizer file:

```bash
./initium -m models/stories260K.bin --tokenizer models/tok512.bin -p "once upon a time" -n 64 --greedy --stats
```

gguf files carry their own tokenizer inside, so it is just:

```bash
./initium -m models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf -p "hello" -n 32 --greedy
```

and there is an interactive chat loop:

```bash
./initium -m models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf --chat
```

## flags

| flag | what it does |
| --- | --- |
| `-m <path>` | model file, .bin or .gguf |
| `--tokenizer <path>` | tokenizer.bin, only needed for .bin models |
| `-p <text>` | the prompt |
| `-n <int>` | how many tokens to generate |
| `--chat` | interactive repl instead of one shot generation |
| `--temp <float>` | temperature. 0 means greedy |
| `--top-p <float>` | nucleus sampling cutoff, default 0.95 |
| `--top-k <int>` | keep only the k most likely tokens, default 40 |
| `--seed <int>` | rng seed, for repeatable output |
| `--greedy` | force argmax, ignore all sampling settings |
| `--ctx <int>` | override the context length |
| `--threads <int>` | worker threads for the matmul kernels |
| `--stats` | print tokens per second on stderr |
| `--no-simd` | force the scalar kernels, useful for debugging |
| `--verify <path>` | parity mode, compare logits against a reference dump |
| `-h` | full help |

## tests

```bash
make test
```

this builds and runs three things:

- kernel unit tests, which check the math functions (matmul, rmsnorm, softmax, rope, and the rest) against hand computed values
- tokenizer tests, which check encode and decode behavior including byte fallback and the leading space rule after bos
- a logit parity self check, which dumps reference logits and compares them back

the deeper correctness story lives in `tests/parity/`. there is a python reimplementation of llama2.c used as the ground truth. the scripts there dump reference logits for a set of prompts, and initium is run with `--verify` against those dumps. tokenizer output is also compared token by token against the reference. this is how the project was built milestone by milestone: get fp32 logits to match exactly, then get the tokenizer to match, then add gguf loading and quantization on top of a base that is known to be correct.

## code layout

everything is in `src/`, one file per concern:

```
main.c        cli parsing, generation loop, chat repl
model.c       the transformer itself: loading, forward pass, gqa attention
kernels.c     all the math: matmul, rmsnorm, softmax, silu, rope, dot, argmax
tokenizer.c   bpe encode and decode, byte fallback, utf-8 handling
sampler.c     greedy, temperature, top-p, top-k, the rng
gguf.c        gguf v3 file parsing, metadata and tensor directory
quant.c       q4_0 and q6_k block dequantization
kvcache.c     the key value cache
threadpool.c  worker threads that the matmul kernels fan out to
verify.c      the logit parity check machinery
```

one rule the codebase follows strictly: all math lives in `kernels.c`. nothing else is allowed to implement its own matmul or softmax. that keeps the numeric behavior in one place, which is what makes the parity testing meaningful.

## performance notes

- weights are held in ram as float32 even when the file is quantized. a 1.1b model takes a few gb of memory. this is a deliberate tradeoff: quantized math in the forward pass would be faster and smaller, but float32 everywhere makes the code easier to read and easier to verify
- it is cpu only, so generation speed is modest. tinystories models are instant, tinyllama is usable, anything much bigger will test your patience
- `--threads` and the simd kernels help a lot on the matmuls, which is where nearly all the time goes
- if numbers ever look wrong, `--no-simd` and `--greedy` with a fixed `--seed` are your friends for narrowing down where behavior diverges

## what it does not do

- no gpu support
- no quantized inference (quantized files load fine, but the math runs in float32)
- no chat templates. in chat mode the model sees your raw text, so instruct models behave better if you format the prompt the way they expect
- only llama style architectures. no mixture of experts, no other model families

these are not oversights, they are scope. this is a learning engine, not a llama.cpp competitor.

## license

mit for the code. models keep their own licenses.
