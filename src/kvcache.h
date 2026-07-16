#ifndef INITIUM_KVCACHE_H
#define INITIUM_KVCACHE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Layout: cache_k[layer][kv_head * max_seq * hd + pos * hd + d]
 * Contiguous in hd, then pos — attention streams K/V sequentially per head. */

typedef struct {
    int n_layers;
    int n_kv_heads;
    int head_dim;
    int max_seq_len;
    float *k; /* size: n_layers * n_kv_heads * max_seq_len * head_dim */
    float *v;
} KVCache;

int  kvcache_init(KVCache *c, int n_layers, int n_kv_heads, int head_dim, int max_seq_len);
void kvcache_free(KVCache *c);
void kvcache_clear(KVCache *c);

/* Pointer to K (or V) vector of length head_dim at (layer, kv_head, pos) */
float *kvcache_k_ptr(KVCache *c, int layer, int kv_head, int pos);
float *kvcache_v_ptr(KVCache *c, int layer, int kv_head, int pos);

/* Append (copy) k_row / v_row (n_kv_heads * head_dim) at position pos for layer */
void kvcache_append(KVCache *c, int layer, int pos, const float *k_row, const float *v_row);

#ifdef __cplusplus
}
#endif

#endif /* INITIUM_KVCACHE_H */
