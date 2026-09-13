#ifndef INITIUM_KVCACHE_H
#define INITIUM_KVCACHE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Layout: cache_k[layer * seq_len * kv_dim + pos * kv_dim + d]
 * where kv_dim = n_kv_heads * head_dim.
 * Contiguous in kv_dim (all KV heads at a position), then pos, then layer.
 * This matches the original model.c inline cache layout exactly. */

typedef struct {
    int n_layers;
    int n_kv_heads;
    int head_dim;
    int kv_dim;         /* n_kv_heads * head_dim (cached for convenience) */
    int max_seq_len;
    float *k; /* size: n_layers * max_seq_len * kv_dim */
    float *v;
} KVCache;

int  kvcache_init(KVCache *c, int n_layers, int n_kv_heads, int head_dim, int max_seq_len);
void kvcache_free(KVCache *c);
void kvcache_clear(KVCache *c);

/* Pointer to K (or V) row of kv_dim floats at (layer, pos).
 * This is the full KV vector for all kv heads at that position. */
float *kvcache_k_row(KVCache *c, int layer, int pos);
float *kvcache_v_row(KVCache *c, int layer, int pos);

/* Pointer to K (or V) vector of length head_dim for a specific (layer, kv_head, pos).
 * Equivalent to kvcache_k_row(c, layer, pos) + kv_head * head_dim. */
float *kvcache_k_head(KVCache *c, int layer, int kv_head, int pos);
float *kvcache_v_head(KVCache *c, int layer, int kv_head, int pos);

/* Copy k_row / v_row (kv_dim floats) into cache at position pos for layer. */
void kvcache_append(KVCache *c, int layer, int pos, const float *k_row, const float *v_row);

#ifdef __cplusplus
}
#endif

#endif /* INITIUM_KVCACHE_H */
