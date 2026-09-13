# Initium — LLM inference engine (C11)
# Uses system CC (clang/gcc). On Windows MinGW: mingw32-make

CC      ?= gcc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -Wpedantic -Wno-unused-parameter -D_POSIX_C_SOURCE=200809L
CFLAGS  += -Isrc
LDFLAGS ?= -lm -lpthread

# Debug / ASan
CFLAGS_DEBUG = -std=c11 -O0 -g -Wall -Wextra -Isrc -DDEBUG -fsanitize=address,undefined -D_POSIX_C_SOURCE=200809L
LDFLAGS_DEBUG = -lm -lpthread -fsanitize=address,undefined

# SIMD auto-detection: enable AVX2+FMA on x86_64, NEON is implicit on aarch64
UNAME_M := $(shell uname -m 2>/dev/null || echo unknown)
ifneq (,$(filter x86_64 AMD64,$(UNAME_M)))
  CFLAGS += -mavx2 -mfma
endif

SRC = \
	src/main.c \
	src/model.c \
	src/kernels.c \
	src/kvcache.c \
	src/tokenizer.c \
	src/sampler.c \
	src/gguf.c \
	src/quant.c \
	src/threadpool.c \
	src/verify.c

OBJ = $(SRC:.c=.o)

.PHONY: all initium initium_debug test test-parity-offline bench clean

all: initium

initium: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS)

initium_debug: CFLAGS := $(CFLAGS_DEBUG)
initium_debug: LDFLAGS := $(LDFLAGS_DEBUG)
initium_debug: $(SRC)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Unit tests (standalone)
tests/test_kernels: tests/test_kernels.c src/kernels.c src/threadpool.c
	$(CC) $(CFLAGS) -o $@ tests/test_kernels.c src/kernels.c src/threadpool.c $(LDFLAGS)

tests/test_tokenizer: tests/test_tokenizer.c src/tokenizer.c src/gguf.c
	$(CC) $(CFLAGS) -o $@ tests/test_tokenizer.c src/tokenizer.c src/gguf.c $(LDFLAGS)

test: tests/test_kernels tests/test_tokenizer initium test-parity-offline
	./tests/test_kernels
	./tests/test_tokenizer
	@echo "Running M0 parity self-check..."
	python tests/parity/dump_reference.py --self-test --out testdata/self_ref.bin
	python tests/parity/compare.py testdata/self_ref.bin testdata/self_ref.bin
	@echo "ALL TESTS PASSED"

test-parity-offline: initium
	@echo "Generating tiny synthetic model..."
	python tests/parity/make_fixture.py --model-out testdata/tiny.bin --tok-out testdata/tiny_tok.bin
	@echo "Generating Python reference dump..."
	python tests/parity/llama2c_ref.py -m testdata/tiny.bin -z testdata/tiny_tok.bin -p "hi there" -n 6 --out testdata/tiny_ref.bin
	@echo "Running initium --verify against reference..."
	./initium -m testdata/tiny.bin --tokenizer testdata/tiny_tok.bin -p "hi there" -n 6 --greedy --verify testdata/tiny_ref.bin --stats
	@echo "OFFLINE PARITY TEST PASSED"

bench: initium
	@echo "Use: ./initium -m <model> -p \"...\" -n 64 --greedy --stats"

clean:
	-rm -f initium initium_debug initium.exe initium_debug.exe $(OBJ) \
		tests/test_kernels tests/test_kernels.exe \
		tests/test_tokenizer tests/test_tokenizer.exe 2>/dev/null; true
	-rm -f src/*.o 2>/dev/null; true
	-del /Q initium.exe initium_debug.exe src\*.o tests\test_kernels.exe tests\test_tokenizer.exe 2>NUL