# initium

a small c11 program that runs llama style models on cpu.

no pytorch. no cuda. just c, math, and threads.

## build

```bash
make
```

on windows if make is missing:

```bash
mingw32-make
```

## get a model

```bash
powershell -file scripts/download_models.ps1
```

or:

```bash
bash scripts/download_models.sh
```

this pulls tinystories (small and good for testing).

## run

```bash
./initium -m models/stories260K.bin --tokenizer models/tok512.bin -p "once upon a time" -n 64 --greedy --stats
```

gguf models work too if you have one:

```bash
./initium -m models/your-model.gguf -p "hello" -n 32 --greedy
```

## notes

- weights get loaded into ram as float32
- big models will use a lot of memory and will be slow on cpu
- math lives in `src/kernels.c`

## license

mit for the code. models keep their own licenses.
