#include "tokenizer.h"
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef struct {
    char *str;
    int id;
} TokenIndex;

void tokenizer_free(Tokenizer *t) {
    if (!t) return;
    if (t->token_table) {
        for (int i = 0; i < t->vocab_size; i++) free(t->token_table[i]);
        free(t->token_table);
    }
    free(t->scores);
    free(t->sorted_indices);
    free(t->arena);
    memset(t, 0, sizeof(*t));
}

/* ---- llama2.c tokenizer.bin ---- */
int tokenizer_load_llama2c(Tokenizer *t, const char *path, int vocab_size) {
    memset(t, 0, sizeof(*t));
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "initium/tok: cannot open '%s'\n", path);
        return -1;
    }
    t->vocab_size = vocab_size;
    t->token_table = (char **)calloc((size_t)vocab_size, sizeof(char *));
    t->scores = (float *)calloc((size_t)vocab_size, sizeof(float));
    if (!t->token_table || !t->scores) {
        fclose(f);
        tokenizer_free(t);
        return -1;
    }

    unsigned int max_token_length = 0;
    if (fread(&max_token_length, sizeof(int), 1, f) != 1) {
        fclose(f);
        tokenizer_free(t);
        return -1;
    }
    /* stash max length in pad_id temporarily? use arena for it */
    t->pad_id = (int)max_token_length; /* reuse field as max_token_length storage */

    for (int i = 0; i < vocab_size; i++) {
        float score;
        unsigned int len;
        if (fread(&score, sizeof(float), 1, f) != 1) break;
        if (fread(&len, sizeof(unsigned int), 1, f) != 1) break;
        char *s = (char *)malloc(len + 1);
        if (!s) { fclose(f); tokenizer_free(t); return -1; }
        if (fread(s, 1, len, f) != len) { free(s); fclose(f); tokenizer_free(t); return -1; }
        s[len] = '\0';
        t->token_table[i] = s;
        t->scores[i] = score;
    }
    fclose(f);

    /* sorted indices by string for binary search */
    t->sorted_indices = (int *)malloc((size_t)vocab_size * sizeof(int));
    if (!t->sorted_indices) { tokenizer_free(t); return -1; }
    for (int i = 0; i < vocab_size; i++) t->sorted_indices[i] = i;
    /* insertion sort is fine for 512; for larger use qsort */
    for (int i = 1; i < vocab_size; i++) {
        int key = t->sorted_indices[i];
        int j = i - 1;
        while (j >= 0 && strcmp(t->token_table[t->sorted_indices[j]], t->token_table[key]) > 0) {
            t->sorted_indices[j + 1] = t->sorted_indices[j];
            j--;
        }
        t->sorted_indices[j + 1] = key;
    }

    t->bos_id = 1;
    t->eos_id = 2;
    t->add_bos = 1;
    t->byte_fallback = 1;
    return 0;
}

int tokenizer_load_gguf(Tokenizer *t, const void *gguf_file) {
    const GGUFFile *gf = (const GGUFFile *)gguf_file;
    memset(t, 0, sizeof(*t));

    const GGUFKV *tok = gguf_find_kv(gf, "tokenizer.ggml.tokens");
    const GGUFKV *scores = gguf_find_kv(gf, "tokenizer.ggml.scores");
    if (!tok || tok->type != GGUF_TYPE_ARRAY || tok->arr_type != GGUF_TYPE_STRING) {
        fprintf(stderr, "initium/tok: GGUF missing tokenizer.ggml.tokens\n");
        return -1;
    }
    int n = (int)tok->arr_len;
    t->vocab_size = n;
    t->token_table = (char **)calloc((size_t)n, sizeof(char *));
    t->scores = (float *)calloc((size_t)n, sizeof(float));
    if (!t->token_table || !t->scores) {
        tokenizer_free(t);
        return -1;
    }
    char **arr = (char **)tok->v.arr;
    for (int i = 0; i < n; i++) {
        size_t len = strlen(arr[i]);
        t->token_table[i] = (char *)malloc(len + 1);
        if (!t->token_table[i]) { tokenizer_free(t); return -1; }
        memcpy(t->token_table[i], arr[i], len + 1);
    }
    if (scores && scores->type == GGUF_TYPE_ARRAY && scores->arr_type == GGUF_TYPE_FLOAT32) {
        memcpy(t->scores, scores->v.arr, (size_t)n * sizeof(float));
    }

    uint32_t bos = 1, eos = 2;
    gguf_get_u32(gf, "tokenizer.ggml.bos_token_id", &bos);
    gguf_get_u32(gf, "tokenizer.ggml.eos_token_id", &eos);
    t->bos_id = (int)bos;
    t->eos_id = (int)eos;
    t->add_bos = 1;
    t->byte_fallback = 1;
    t->pad_id = 256;
    /* sorted vocab for BPE lookup */
    t->sorted_indices = (int *)malloc((size_t)n * sizeof(int));
    if (!t->sorted_indices) { tokenizer_free(t); return -1; }
    for (int i = 0; i < n; i++) t->sorted_indices[i] = i;
    for (int i = 1; i < n; i++) {
        int key = t->sorted_indices[i];
        int j = i - 1;
        while (j >= 0 && strcmp(t->token_table[t->sorted_indices[j]], t->token_table[key]) > 0) {
            t->sorted_indices[j + 1] = t->sorted_indices[j];
            j--;
        }
        t->sorted_indices[j + 1] = key;
    }
    return 0;
}

const char *tokenizer_decode_piece(const Tokenizer *t, int id) {
    if (id < 0 || id >= t->vocab_size) return "";
    return t->token_table[id] ? t->token_table[id] : "";
}

static int str_lookup(const Tokenizer *t, const char *str) {
    /* binary search on sorted_indices */
    if (!t->sorted_indices) {
        for (int i = 0; i < t->vocab_size; i++) {
            if (t->token_table[i] && strcmp(t->token_table[i], str) == 0) return i;
        }
        return -1;
    }
    int lo = 0, hi = t->vocab_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int id = t->sorted_indices[mid];
        int cmp = strcmp(t->token_table[id], str);
        if (cmp == 0) return id;
        if (cmp < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

/* Encode: llama2.c path for tok512, SPM path for GGUF (spaces -> U+2581 ▁). */
int tokenizer_encode(const Tokenizer *t, const char *text, int *out, int max_tokens, int add_bos) {
    if (!text) text = "";
    int cap = max_tokens > 0 ? max_tokens : 8192;
    int *tokens = out;
    int owned = 0;
    if (!tokens) {
        tokens = (int *)malloc((size_t)cap * sizeof(int));
        owned = 1;
        if (!tokens) return -1;
    }
    int n = 0;
    if (add_bos && t->add_bos) {
        if (n < cap) tokens[n] = t->bos_id;
        n++;
    }

    /* Detect SPM-style vocab (has ▁ token or no plain space token) */
    int spm = (str_lookup(t, "\xE2\x96\x81") >= 0) || (str_lookup(t, " ") < 0);

    /* Build processed UTF-8: SPM replaces ' ' with ▁; llama2.c keeps spaces + dummy prefix */
    size_t tlen = strlen(text);
    char *proc = (char *)malloc(tlen * 3 + 8);
    if (!proc) { if (owned) free(tokens); return -1; }
    size_t po = 0;
    if (spm) {
        /* leading ▁ if non-empty (SPM add_dummy_prefix) */
        if (tlen > 0) {
            proc[po++] = (char)0xE2; proc[po++] = (char)0x96; proc[po++] = (char)0x81;
        }
        for (size_t i = 0; i < tlen; i++) {
            if (text[i] == ' ') {
                proc[po++] = (char)0xE2; proc[po++] = (char)0x96; proc[po++] = (char)0x81;
            } else {
                proc[po++] = text[i];
            }
        }
    } else {
        if (tlen > 0) {
            int dummy = str_lookup(t, " ");
            if (dummy >= 0) {
                if (n < cap) tokens[n] = dummy;
                n++;
            }
        }
        memcpy(proc, text, tlen);
        po = tlen;
    }
    proc[po] = '\0';

    /* process raw UTF-8 codepoints of proc */
    const unsigned char *c = (const unsigned char *)proc;
    char str_buffer[256];
    size_t str_len = 0;
    while (*c) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        if (str_len + 1 < sizeof(str_buffer)) str_buffer[str_len++] = (char)(*c);
        str_buffer[str_len] = '\0';
        if ((c[1] & 0xC0) == 0x80 && str_len < 4) { c++; continue; }
        int id = str_lookup(t, str_buffer);
        if (id != -1) {
            if (n < cap) tokens[n] = id;
            n++;
        } else {
            for (size_t i = 0; i < str_len; i++) {
                /* try <0xNN> byte tokens (GGUF), else llama2.c byte ids (byte+3) */
                char hex[8];
                snprintf(hex, sizeof(hex), "<0x%02X>", (unsigned char)str_buffer[i]);
                int bid = str_lookup(t, hex);
                if (bid < 0 && t->byte_fallback) {
                    int cand = (int)(unsigned char)str_buffer[i] + 3;
                    if (cand >= 0 && cand < t->vocab_size) bid = cand;
                }
                if (bid < 0 || bid >= t->vocab_size) continue;
                if (n < cap) tokens[n] = bid;
                n++;
            }
        }
        str_len = 0;
        c++;
    }
    free(proc);

    /* BPE merges — only if we have a writable tokens buffer within capacity */
    if (tokens && n <= cap) {
        int max_tl = t->pad_id > 0 ? t->pad_id : 64;
        char *merge_buf = (char *)malloc((size_t)max_tl * 2 + 8);
        if (merge_buf) {
            while (1) {
                float best_score = -1e10f;
                int best_id = -1;
                int best_idx = -1;
                for (int i = 0; i < n - 1; i++) {
                    int ta = tokens[i], tb = tokens[i + 1];
                    if (ta < 0 || tb < 0 || ta >= t->vocab_size || tb >= t->vocab_size)
                        continue;
                    const char *a = t->token_table[ta];
                    const char *b = t->token_table[tb];
                    if (!a || !b) continue;
                    size_t la = strlen(a), lb = strlen(b);
                    if (la + lb + 1 > (size_t)max_tl * 2 + 8) continue;
                    memcpy(merge_buf, a, la);
                    memcpy(merge_buf + la, b, lb);
                    merge_buf[la + lb] = '\0';
                    int id = str_lookup(t, merge_buf);
                    if (id != -1 && t->scores && t->scores[id] > best_score) {
                        best_score = t->scores[id];
                        best_id = id;
                        best_idx = i;
                    }
                }
                if (best_idx == -1) break;
                tokens[best_idx] = best_id;
                for (int i = best_idx + 1; i < n - 1; i++) tokens[i] = tokens[i + 1];
                n--;
            }
            free(merge_buf);
        }
    }

    if (owned) {
        free(tokens);
    }
    return n;
}

void tokenizer_decode_token(const Tokenizer *t, int prev_token, int token,
                            char *out, int out_size) {
    if (out_size <= 0) return;
    out[0] = '\0';
    const char *p = tokenizer_decode_piece(t, token);
    if (!p) return;
    /* following BOS, sentencepiece strips leading whitespace / ▁ */
    if (prev_token == t->bos_id) {
        if (p[0] == ' ') p++;
        else if ((unsigned char)p[0] == 0xE2 && (unsigned char)p[1] == 0x96 &&
                 (unsigned char)p[2] == 0x81) {
            p += 3;
        }
    }
    unsigned char byte_val = 0;
    if (sscanf(p, "<0x%02hhX>", &byte_val) == 1) {
        if (out_size >= 2) {
            out[0] = (char)byte_val;
            out[1] = '\0';
        }
        return;
    }
    /* convert SPM ▁ (U+2581) to space while copying */
    int o = 0;
    for (const unsigned char *c = (const unsigned char *)p; *c && o + 1 < out_size; ) {
        if (c[0] == 0xE2 && c[1] == 0x96 && c[2] == 0x81) {
            out[o++] = ' ';
            c += 3;
        } else {
            out[o++] = (char)(*c++);
        }
    }
    out[o] = '\0';
}

int tokenizer_decode(const Tokenizer *t, const int *ids, int n, char *out, int out_size) {
    if (out_size <= 0) return 0;
    int o = 0;
    out[0] = '\0';
    int prev = t->bos_id;
    for (int i = 0; i < n; i++) {
        char piece[256];
        tokenizer_decode_token(t, prev, ids[i], piece, sizeof(piece));
        size_t plen = strlen(piece);
        for (size_t j = 0; j < plen && o + 1 < out_size; j++) {
            out[o++] = piece[j];
        }
        prev = ids[i];
    }
    if (o < out_size) out[o] = '\0';
    else out[out_size - 1] = '\0';
    return o;
}
