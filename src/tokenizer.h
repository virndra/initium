#ifndef INITIUM_TOKENIZER_H
#define INITIUM_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int vocab_size;
    char **token_table;     /* id -> piece string (may contain raw bytes) */
    float *scores;          /* BPE merge scores (SentencePiece) */
    int *sorted_indices;    /* vocab sorted by piece for lookup */
    int bos_id;
    int eos_id;
    int pad_id;
    int add_bos;
    /* byte-fallback: piece "<0xNN>" style for SP */
    int byte_fallback;
    /* owned buffer for loading */
    void *arena;
    size_t arena_size;
} Tokenizer;

/* Load llama2.c tokenizer.bin (Karpathy format) */
int tokenizer_load_llama2c(Tokenizer *t, const char *path, int vocab_size);

/* Load from already-open GGUF (tokens + scores in metadata) */
int tokenizer_load_gguf(Tokenizer *t, const void *gguf_file /* GGUFFile* */);

void tokenizer_free(Tokenizer *t);

/* Encode text -> token ids. Returns count written (or needed if out is NULL).
 * max_tokens is capacity of out. */
int tokenizer_encode(const Tokenizer *t, const char *text, int *out, int max_tokens, int add_bos);

/* Decode token id into piece; returns pointer to internal string (valid until free) */
const char *tokenizer_decode_piece(const Tokenizer *t, int id);

/* Decode one token with llama2.c rules: after BOS (prev==bos_id), strip leading space.
 * byte tokens <0xNN> become raw bytes. Writes into out (NUL-terminated). */
void tokenizer_decode_token(const Tokenizer *t, int prev_token, int token,
                            char *out, int out_size);

/* Decode sequence to UTF-8 buffer. Handles partial multi-byte UTF-8.
 * Writes at most out_size-1 bytes + NUL. Returns bytes written. */
int tokenizer_decode(const Tokenizer *t, const int *ids, int n, char *out, int out_size);

#ifdef __cplusplus
}
#endif

#endif /* INITIUM_TOKENIZER_H */
