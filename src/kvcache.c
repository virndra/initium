#include "kvcache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Layer stride: max_seq_len * kv_dim elements per layer */
static size_t layer_stride(const KVCache *c) {
    return (size_t)c->max_seq_len * (size_t)c->kv_dim;
}

int kvcache_init(KVCache *c, int n_layers, int n_kv_heads, int head_dim, int max_seq_len) {
    memset(c, 0, sizeof(*c));
    c->n_layers = n_layers;
    c->n_kv_heads = n_kv_heads;
    c->head_dim = head_dim;
    c->kv_dim = n_kv_heads * head_dim;
    c->max_seq_len = max_seq_len;
    size_t n = (size_t)n_layers * layer_stride(c);
    c->k = (float *)calloc(n, sizeof(float));
    c->v = (float *)calloc(n, sizeof(float));
    if (!c->k || !c->v) {
        fprintf(stderr, "initium: kvcache_init: out of memory (need ~%zu MB)\n",
                (n * 2 * sizeof(float)) / (1024 * 1024));
        kvcache_free(c);
        return -1;
    }
    return 0;
}

void kvcache_free(KVCache *c) {
    free(c->k);
    free(c->v);
    memset(c, 0, sizeof(*c));
}

void kvcache_clear(KVCache *c) {
    if (!c->k) return;
    size_t n = (size_t)c->n_layers * layer_stride(c);
    memset(c->k, 0, n * sizeof(float));
    memset(c->v, 0, n * sizeof(float));
}

float *kvcache_k_row(KVCache *c, int layer, int pos) {
    size_t off = (size_t)layer * layer_stride(c) + (size_t)pos * (size_t)c->kv_dim;
    return c->k + off;
}

float *kvcache_v_row(KVCache *c, int layer, int pos) {
    size_t off = (size_t)layer * layer_stride(c) + (size_t)pos * (size_t)c->kv_dim;
    return c->v + off;
}

float *kvcache_k_head(KVCache *c, int layer, int kv_head, int pos) {
    return kvcache_k_row(c, layer, pos) + (size_t)kv_head * (size_t)c->head_dim;
}

float *kvcache_v_head(KVCache *c, int layer, int kv_head, int pos) {
    return kvcache_v_row(c, layer, pos) + (size_t)kv_head * (size_t)c->head_dim;
}

void kvcache_append(KVCache *c, int layer, int pos, const float *k_row, const float *v_row) {
    if (pos < 0 || pos >= c->max_seq_len) {
        fprintf(stderr, "initium: kvcache_append: pos %d exceeds max_seq_len %d\n",
                pos, c->max_seq_len);
        return;
    }
    float *dk = kvcache_k_row(c, layer, pos);
    float *dv = kvcache_v_row(c, layer, pos);
    memcpy(dk, k_row, (size_t)c->kv_dim * sizeof(float));
    memcpy(dv, v_row, (size_t)c->kv_dim * sizeof(float));
}
