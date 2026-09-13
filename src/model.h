#ifndef INITIUM_MODEL_H
#define INITIUM_MODEL_H

#include "kvcache.h"
#include "threadpool.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int dim;           /* transformer dimension */
    int hidden_dim;    /* FFN intermediate */
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
    float rope_theta;  /* default 10000 */
    float norm_eps;    /* default 1e-5 */
    int sliding_window;/* 0 = disabled */
    int tied_embeddings;
} ModelConfig;

/* Weight pointers (fp32 after load; may point into mmap or owned buffers) */
typedef struct {
    float *token_embedding; /* vocab x dim */
    /* per layer */
    float **rms_att_weight; /* [layer][dim] */
    float **rms_ffn_weight;
    float **wq; /* [layer][dim x dim] — for GQA: dim x (n_heads*hd) same as dim x dim if n_heads*hd=dim */
    float **wk; /* [layer][n_kv_heads*hd x dim] stored as (n_kv_heads*hd) rows, dim cols */
    float **wv;
    float **wo;
    float **w1; /* gate: hidden_dim x dim */
    float **w2; /* down: dim x hidden_dim */
    float **w3; /* up:   hidden_dim x dim */
    float *rms_final;
    float *wcls; /* vocab x dim (or == token_embedding if tied) */

    /* owned heap (llama2.c load) */
    float *blob;
    size_t blob_size;
    int owns_blob;
} ModelWeights;

typedef struct {
    ModelConfig cfg;
    ModelWeights w;
    KVCache kv;
    ThreadPool *pool;
    int use_kv_cache; /* 1 after M2 */

    /* run-state buffers */
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
} Transformer;

int  model_load_llama2c_bin(Transformer *t, const char *path, int ctx_override);
int  model_load_gguf(Transformer *t, const char *path, int ctx_override);
void model_free(Transformer *t);

/* Forward one token at position pos. Writes vocab logits into t->logits.
 * If use_kv_cache, only processes this position (decode).
 * If not, recomputes full sequence when pos grows (M1 path uses cache arrays anyway). */
float *model_forward(Transformer *t, int token, int pos);

/* Accessors */
const ModelConfig *model_config(const Transformer *t);
float *model_logits(Transformer *t);

#ifdef __cplusplus
}
#endif

#endif /* INITIUM_MODEL_H */
