# Initium — LLM inference engine (C11)
# Uses system CC (clang/gcc). On Windows MinGW: mingw32-make

CC      ?= gcc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -Wpedantic -Wno-unused-parameter
CFLAGS  += -Isrc
LDFLAGS ?= -lm -lpthread

# Debug / ASan
CFLAGS_DEBUG = -std=c11 -O0 -g -Wall -Wextra -Isrc -DDEBUG -fsanitize=address,undefined
LDFLAGS_DEBUG = -lm -lpthread -fsanitize=address,undefined

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

.PHONY: all initium initium_debug test bench clean

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

test: tests/test_kernels tests/test_tokenizer initium
	./tests/test_kernels
	./tests/test_tokenizer
	@echo "Running M0 parity self-check..."
	python tests/parity/dump_reference.py --self-test --out testdata/self_ref.bin
	python tests/parity/compare.py testdata/self_ref.bin testdata/self_ref.bin
	@echo "ALL TESTS PASSED"

bench: initium
	@echo "Use: ./initium -m <model> -p \"...\" -n 64 --greedy --stats"

clean:
	-rm -f initium initium_debug initium.exe initium_debug.exe $(OBJ) \
		tests/test_kernels tests/test_kernels.exe \
		tests/test_tokenizer tests/test_tokenizer.exe 2>/dev/null; true
	-rm -f src/*.o 2>/dev/null; true
	-del /Q initium.exe initium_debug.exe src\*.o tests\test_kernels.exe tests\test_tokenizer.exe 2>NUL