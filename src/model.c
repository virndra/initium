#include "model.h"
#include "kernels.h"
#include "gguf.h"
#include "quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

const ModelConfig *model_config(const Transformer *t) { return &t->cfg; }
float *model_logits(Transformer *t) { return t->logits; }

static int alloc_run_state(Transformer *t) {
    ModelConfig *p = &t->cfg;
    int dim = p->dim;
    int hd = dim / p->n_heads;
    int kv_dim = hd * p->n_kv_heads;
    int hidden = p->hidden_dim;

    t->x      = calloc((size_t)dim, sizeof(float));
    t->xb     = calloc((size_t)dim, sizeof(float));
    t->xb2    = calloc((size_t)dim, sizeof(float));
    t->hb     = calloc((size_t)hidden, sizeof(float));
    t->hb2    = calloc((size_t)hidden, sizeof(float));
    t->q      = calloc((size_t)dim, sizeof(float));
    t->k      = calloc((size_t)kv_dim, sizeof(float));
    t->v      = calloc((size_t)kv_dim, sizeof(float));
    t->att    = calloc((size_t)p->n_heads * (size_t)p->seq_len, sizeof(float));
    t->logits = calloc((size_t)p->vocab_size, sizeof(float));

    if (!t->x || !t->xb || !t->xb2 || !t->hb || !t->hb2 || !t->q || !t->k || !t->v ||
        !t->att || !t->logits) {
        fprintf(stderr, "initium/model: OOM allocating run state\n");
        return -1;
    }

    if (kvcache_init(&t->kv, p->n_layers, p->n_kv_heads, hd, p->seq_len) != 0) {
        return -1;
    }

    return 0;
}

static void free_run_state(Transformer *t) {
    free(t->x); free(t->xb); free(t->xb2); free(t->hb); free(t->hb2);
    free(t->q); free(t->k); free(t->v); free(t->att); free(t->logits);
    t->x = t->xb = t->xb2 = t->hb = t->hb2 = NULL;
    t->q = t->k = t->v = t->att = t->logits = NULL;
}

static int alloc_weight_ptrs(ModelWeights *w, int n_layers) {
    w->rms_att_weight = calloc((size_t)n_layers, sizeof(float *));
    w->rms_ffn_weight = calloc((size_t)n_layers, sizeof(float *));
    w->wq = calloc((size_t)n_layers, sizeof(float *));
    w->wk = calloc((size_t)n_layers, sizeof(float *));
    w->wv = calloc((size_t)n_layers, sizeof(float *));
    w->wo = calloc((size_t)n_layers, sizeof(float *));
    w->w1 = calloc((size_t)n_layers, sizeof(float *));
    w->w2 = calloc((size_t)n_layers, sizeof(float *));
    w->w3 = calloc((size_t)n_layers, sizeof(float *));
    if (!w->rms_att_weight || !w->rms_ffn_weight || !w->wq || !w->wk || !w->wv ||
        !w->wo || !w->w1 || !w->w2 || !w->w3) return -1;
    return 0;
}

/* llama2.c export.bin layout (karpathy) — all float32 */
int model_load_llama2c_bin(Transformer *t, const char *path, int ctx_override) {
    memset(t, 0, sizeof(*t));
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "initium/model: cannot open '%s'\n", path);
        return -1;
    }

    /* Config is 7 ints */
    int raw[7];
    if (fread(raw, sizeof(int), 7, f) != 7) {
        fprintf(stderr, "initium/model: truncated config header\n");
        fclose(f);
        return -1;
    }
    ModelConfig *p = &t->cfg;
    p->dim = raw[0];
    p->hidden_dim = raw[1];
    p->n_layers = raw[2];
    p->n_heads = raw[3];
    p->n_kv_heads = raw[4];
    p->vocab_size = raw[5];
    p->seq_len = raw[6];
    /* llama2.c: negative vocab_size means shared classifier */
    int shared = p->vocab_size > 0 ? 1 : 0;
    if (p->vocab_size < 0) p->vocab_size = -p->vocab_size;
    p->tied_embeddings = shared;
    p->rope_theta = 10000.0f;
    p->norm_eps = 1e-5f;
    p->sliding_window = 0;
    if (ctx_override > 0) p->seq_len = ctx_override;

    if (p->dim % p->n_heads != 0) {
        fprintf(stderr, "initium/model: dim %d not divisible by n_heads %d\n", p->dim, p->n_heads);
        fclose(f);
        return -1;
    }

    /* file size of weights */
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long file_size = ftell(f);
    long header = 7 * (long)sizeof(int);
    size_t nbytes = (size_t)(file_size - header);
    if (fseek(f, header, SEEK_SET) != 0) { fclose(f); return -1; }

    t->w.blob = (float *)malloc(nbytes);
    if (!t->w.blob) {
        fprintf(stderr, "initium/model: OOM reading weights (%zu bytes)\n", nbytes);
        fclose(f);
        return -1;
    }
    t->w.blob_size = nbytes;
    t->w.owns_blob = 1;
    if (fread(t->w.blob, 1, nbytes, f) != nbytes) {
        fprintf(stderr, "initium/model: short read of weights\n");
        fclose(f);
        model_free(t);
        return -1;
    }
    fclose(f);

    if (alloc_weight_ptrs(&t->w, p->n_layers) != 0) {
        model_free(t);
        return -1;
    }

    float *ptr = t->w.blob;
    int dim = p->dim;
    int hidden = p->hidden_dim;
    int n_layers = p->n_layers;
    int vocab = p->vocab_size;
    int hd = dim / p->n_heads;
    int kv_dim = p->n_kv_heads * hd;

    t->w.token_embedding = ptr;
    ptr += (size_t)vocab * dim;

    for (int L = 0; L < n_layers; L++) {
        t->w.rms_att_weight[L] = ptr;
        ptr += dim;
    }
    for (int L = 0; L < n_layers; L++) {
        t->w.wq[L] = ptr;
        ptr += (size_t)dim * dim;
    }
    for (int L = 0; L < n_layers; L++) {
        t->w.wk[L] = ptr;
        ptr += (size_t)dim * kv_dim; /* note: llama2.c stores as dim * kv_dim? */
    }
    /* llama2.c map: wk is (dim, kv_dim) in matmul as y[kv_dim] = W[kv_dim, dim] * x
     * Weight count: n_layers * dim * kv_dim for each of wk,wv — check karpathy:
     *   s->wk = ptr; ptr += n_layers * dim * kv_dim;  YES
     */
    for (int L = 0; L < n_layers; L++) {
        t->w.wv[L] = ptr;
        ptr += (size_t)dim * kv_dim;
    }
    for (int L = 0; L < n_layers; L++) {
        t->w.wo[L] = ptr;
        ptr += (size_t)dim * dim;
    }
    for (int L = 0; L < n_layers; L++) {
        t->w.rms_ffn_weight[L] = ptr;
        ptr += dim;
    }
    for (int L = 0; L < n_layers; L++) {
        t->w.w1[L] = ptr; /* gate w1: hidden x dim */
        ptr += (size_t)dim * hidden;
    }
    for (int L = 0; L < n_layers; L++) {
        t->w.w2[L] = ptr; /* down w2: dim x hidden */
        ptr += (size_t)hidden * dim;
    }
    for (int L = 0; L < n_layers; L++) {
        t->w.w3[L] = ptr; /* up w3: hidden x dim */
        ptr += (size_t)dim * hidden;
    }
    t->w.rms_final = ptr;
    ptr += dim;

    /* freq_cis_real / imag skipped in newer exports? llama2.c still has them for rope table
     * stories260K uses on-the-fly rope in some forks; classic llama2.c includes:
     *   ptr += seq_len * head_size / 2; // real
     *   ptr += seq_len * head_size / 2; // imag
     * Then wcls if not shared.
     */
    ptr += (size_t)p->seq_len * (size_t)(hd / 2); /* real */
    ptr += (size_t)p->seq_len * (size_t)(hd / 2); /* imag */

    if (shared) {
        t->w.wcls = t->w.token_embedding;
        t->cfg.tied_embeddings = 1;
    } else {
        t->w.wcls = ptr;
        t->cfg.tied_embeddings = 0;
    }

    /* Fix wk/wv pointers — we advanced wrong in the loop above.
     * Re-walk carefully from blob. */
    {
        float *p2 = t->w.blob;
        p2 += (size_t)vocab * dim;
        for (int L = 0; L < n_layers; L++) { t->w.rms_att_weight[L] = p2; p2 += dim; }
        for (int L = 0; L < n_layers; L++) { t->w.wq[L] = p2; p2 += (size_t)dim * dim; }
        for (int L = 0; L < n_layers; L++) { t->w.wk[L] = p2; p2 += (size_t)dim * kv_dim; }
        for (int L = 0; L < n_layers; L++) { t->w.wv[L] = p2; p2 += (size_t)dim * kv_dim; }
        for (int L = 0; L < n_layers; L++) { t->w.wo[L] = p2; p2 += (size_t)dim * dim; }
        for (int L = 0; L < n_layers; L++) { t->w.rms_ffn_weight[L] = p2; p2 += dim; }
        for (int L = 0; L < n_layers; L++) { t->w.w1[L] = p2; p2 += (size_t)dim * hidden; }
        for (int L = 0; L < n_layers; L++) { t->w.w2[L] = p2; p2 += (size_t)hidden * dim; }
        for (int L = 0; L < n_layers; L++) { t->w.w3[L] = p2; p2 += (size_t)dim * hidden; }
        t->w.rms_final = p2; p2 += dim;
        p2 += (size_t)p->seq_len * (size_t)(hd / 2);
        p2 += (size_t)p->seq_len * (size_t)(hd / 2);
        if (!shared) t->w.wcls = p2;
        else t->w.wcls = t->w.token_embedding;
    }

    t->use_kv_cache = 1;

    if (alloc_run_state(t) != 0) {
        model_free(t);
        return -1;
    }

    fprintf(stderr, "initium: loaded llama2c bin dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d seq=%d tied=%d\n",
            p->dim, p->hidden_dim, p->n_layers, p->n_heads, p->n_kv_heads, p->vocab_size, p->seq_len,
            p->tied_embeddings);
    return 0;
}

/* Load GGUF tensor -> newly malloc'd f32. ne0=cols (in), ne1=rows (out) for 2D. */
static float *load_tensor_f32(const GGUFFile *gf, const char *name,
                              uint64_t expect_ne0, uint64_t expect_ne1, int need_2d) {
    const GGUFTensor *t = gguf_find_tensor(gf, name);
    if (!t) {
        fprintf(stderr, "initium/gguf: missing tensor '%s'\n", name);
        return NULL;
    }
    if (need_2d) {
        if (t->ne[0] != expect_ne0 || t->ne[1] != expect_ne1) {
            fprintf(stderr, "initium/gguf: tensor '%s' shape [%llu,%llu] want [%llu,%llu]\n",
                    name,
                    (unsigned long long)t->ne[0], (unsigned long long)t->ne[1],
                    (unsigned long long)expect_ne0, (unsigned long long)expect_ne1);
            return NULL;
        }
    } else {
        if (t->ne[0] != expect_ne0) {
            fprintf(stderr, "initium/gguf: tensor '%s' ne0=%llu want %llu\n",
                    name, (unsigned long long)t->ne[0], (unsigned long long)expect_ne0);
            return NULL;
        }
    }
    uint64_t n = 1;
    for (int i = 0; i < t->n_dims; i++) n *= t->ne[i];
    float *out = (float *)malloc((size_t)n * sizeof(float));
    if (!out) return NULL;
    if (quant_dequantize_tensor(out, t->data, t->type, t->ne, t->n_dims) != 0) {
        free(out);
        return NULL;
    }
    return out;
}

/* GGUF: dequant all weights to fp32 (F16/Q4_0/Q8_0/F32). Correctness first. */
int model_load_gguf(Transformer *t, const char *path, int ctx_override) {
    memset(t, 0, sizeof(*t));
    GGUFFile gf;
    if (gguf_open(path, &gf) != 0) return -1;

    ModelConfig *p = &t->cfg;
    uint32_t u;
    const char *arch = NULL;
    gguf_get_str(&gf, "general.architecture", &arch);
    char keybuf[128];
    const char *prefix = arch ? arch : "llama";

    #define GET_U32(suffix, field, def) do { \
        snprintf(keybuf, sizeof(keybuf), "%s." suffix, prefix); \
        if (gguf_get_u32(&gf, keybuf, &u) == 0) p->field = (int)u; \
        else p->field = (def); \
    } while (0)

    GET_U32("embedding_length", dim, 0);
    GET_U32("feed_forward_length", hidden_dim, 0);
    GET_U32("block_count", n_layers, 0);
    GET_U32("attention.head_count", n_heads, 0);
    GET_U32("attention.head_count_kv", n_kv_heads, p->n_heads);
    GET_U32("context_length", seq_len, 2048);
    GET_U32("vocab_size", vocab_size, 0);
    #undef GET_U32

    float f;
    snprintf(keybuf, sizeof(keybuf), "%s.rope.freq_base", prefix);
    if (gguf_get_f32(&gf, keybuf, &f) == 0) p->rope_theta = f;
    else p->rope_theta = 10000.0f;
    snprintf(keybuf, sizeof(keybuf), "%s.attention.layer_norm_rms_epsilon", prefix);
    if (gguf_get_f32(&gf, keybuf, &f) == 0) p->norm_eps = f;
    else p->norm_eps = 1e-5f;
    p->sliding_window = 0;
    snprintf(keybuf, sizeof(keybuf), "%s.attention.sliding_window", prefix);
    if (gguf_get_u32(&gf, keybuf, &u) == 0) p->sliding_window = (int)u;

    if (p->vocab_size == 0) {
        const GGUFKV *tok = gguf_find_kv(&gf, "tokenizer.ggml.tokens");
        if (tok) p->vocab_size = (int)tok->arr_len;
    }
    if (ctx_override > 0 && ctx_override < p->seq_len) p->seq_len = ctx_override;
    /* Cap ctx for RAM — user can raise with --ctx */
    if (ctx_override <= 0 && p->seq_len > 2048) p->seq_len = 2048;

    if (p->dim == 0 || p->n_layers == 0 || p->n_heads == 0 || p->vocab_size == 0) {
        fprintf(stderr, "initium/model: bad GGUF config dim=%d layers=%d heads=%d vocab=%d\n",
                p->dim, p->n_layers, p->n_heads, p->vocab_size);
        gguf_close(&gf);
        return -1;
    }
    if (p->dim % p->n_heads != 0) {
        fprintf(stderr, "initium/model: dim not divisible by n_heads\n");
        gguf_close(&gf);
        return -1;
    }

    int dim = p->dim, hidden = p->hidden_dim, n_layers = p->n_layers;
    int hd = dim / p->n_heads;
    int kv_dim = p->n_kv_heads * hd;
    int vocab = p->vocab_size;

    fprintf(stderr, "initium: loading GGUF arch=%s dim=%d hid=%d L=%d H=%d KV=%d V=%d theta=%.0f (dequant->f32)...\n",
            arch ? arch : "?", dim, hidden, n_layers, p->n_heads, p->n_kv_heads, vocab, p->rope_theta);

    if (alloc_weight_ptrs(&t->w, n_layers) != 0) {
        gguf_close(&gf);
        return -1;
    }

    /* Collect all f32 weight slabs in one freeable list via blob = first emb + free individually */
    /* We allocate each tensor separately; free in model_free via owns_blob special path. */
    t->w.token_embedding = load_tensor_f32(&gf, "token_embd.weight", (uint64_t)dim, (uint64_t)vocab, 1);
    if (!t->w.token_embedding) {
        /* some models use ne = [vocab, dim] */
        const GGUFTensor *te = gguf_find_tensor(&gf, "token_embd.weight");
        if (te && te->ne[0] == (uint64_t)vocab && te->ne[1] == (uint64_t)dim) {
            t->w.token_embedding = load_tensor_f32(&gf, "token_embd.weight", (uint64_t)vocab, (uint64_t)dim, 1);
            /* transpose vocab x dim -> need dim-major for embed row? Our embed is token * dim.
             * If stored [vocab, dim] with ne0=vocab, that's wrong for row token*dim.
             * ggml usually ne0=n_embd, ne1=n_vocab so row j is token j at data[j*dim + i] if we
             * interpret as [ne1][ne0] = [vocab][dim]. Yes: index = token * ne0 + d = token * dim + d. */
        }
    }
    if (!t->w.token_embedding) { gguf_close(&gf); model_free(t); return -1; }

    /* ggml layout: ne0 = n_embd, ne1 = n_vocab → dequant gives [vocab rows of dim] contiguous. OK */

    char name[96];
    for (int L = 0; L < n_layers; L++) {
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", L);
        t->w.rms_att_weight[L] = load_tensor_f32(&gf, name, (uint64_t)dim, 0, 0);
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", L);
        t->w.rms_ffn_weight[L] = load_tensor_f32(&gf, name, (uint64_t)dim, 0, 0);

        /* weights: ne0 = in (cols), ne1 = out (rows) → matmul(y,x,W,rows=ne1,cols=ne0) */
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", L);
        t->w.wq[L] = load_tensor_f32(&gf, name, (uint64_t)dim, (uint64_t)dim, 1);
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", L);
        t->w.wk[L] = load_tensor_f32(&gf, name, (uint64_t)dim, (uint64_t)kv_dim, 1);
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", L);
        t->w.wv[L] = load_tensor_f32(&gf, name, (uint64_t)dim, (uint64_t)kv_dim, 1);
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", L);
        t->w.wo[L] = load_tensor_f32(&gf, name, (uint64_t)dim, (uint64_t)dim, 1);

        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", L);
        t->w.w1[L] = load_tensor_f32(&gf, name, (uint64_t)dim, (uint64_t)hidden, 1);
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", L);
        t->w.w2[L] = load_tensor_f32(&gf, name, (uint64_t)hidden, (uint64_t)dim, 1);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", L);
        t->w.w3[L] = load_tensor_f32(&gf, name, (uint64_t)dim, (uint64_t)hidden, 1);

        if (!t->w.rms_att_weight[L] || !t->w.rms_ffn_weight[L] || !t->w.wq[L] || !t->w.wk[L] ||
            !t->w.wv[L] || !t->w.wo[L] || !t->w.w1[L] || !t->w.w2[L] || !t->w.w3[L]) {
            fprintf(stderr, "initium/gguf: failed layer %d\n", L);
            gguf_close(&gf);
            model_free(t);
            return -1;
        }
    }

    t->w.rms_final = load_tensor_f32(&gf, "output_norm.weight", (uint64_t)dim, 0, 0);
    if (!t->w.rms_final) { gguf_close(&gf); model_free(t); return -1; }

    /* output.weight may be absent if tied */
    t->w.wcls = load_tensor_f32(&gf, "output.weight", (uint64_t)dim, (uint64_t)vocab, 1);
    if (!t->w.wcls) {
        t->w.wcls = t->w.token_embedding;
        p->tied_embeddings = 1;
        fprintf(stderr, "initium: output tied to token_embd\n");
    } else {
        p->tied_embeddings = 0;
    }

    t->w.owns_blob = 2; /* special: free each pointer separately */
    t->use_kv_cache = 1;

    if (alloc_run_state(t) != 0) {
        gguf_close(&gf);
        model_free(t);
        return -1;
    }

    fprintf(stderr, "initium: GGUF ready (fp32 weights in RAM)\n");
    gguf_close(&gf);
    return 0;
}

void model_free(Transformer *t) {
    if (!t) return;
    free_run_state(t);
    kvcache_free(&t->kv);
    threadpool_destroy(t->pool);
    if (t->w.owns_blob == 2) {
        /* GGUF path: each tensor malloc'd */
        free(t->w.token_embedding);
        if (t->w.wcls && t->w.wcls != t->w.token_embedding) free(t->w.wcls);
        free(t->w.rms_final);
        if (t->w.wq) {
            for (int L = 0; L < t->cfg.n_layers; L++) {
                free(t->w.rms_att_weight ? t->w.rms_att_weight[L] : NULL);
                free(t->w.rms_ffn_weight ? t->w.rms_ffn_weight[L] : NULL);
                free(t->w.wq[L]); free(t->w.wk[L]); free(t->w.wv[L]); free(t->w.wo[L]);
                free(t->w.w1[L]); free(t->w.w2[L]); free(t->w.w3[L]);
            }
        }
    } else if (t->w.owns_blob == 1) {
        free(t->w.blob);
    }
    free(t->w.rms_att_weight);
    free(t->w.rms_ffn_weight);
    free(t->w.wq); free(t->w.wk); free(t->w.wv); free(t->w.wo);
    free(t->w.w1); free(t->w.w2); free(t->w.w3);
    memset(t, 0, sizeof(*t));
}


float *model_forward(Transformer *t, int token, int pos) {
    ModelConfig *p = &t->cfg;
    ModelWeights *w = &t->w;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int n_heads = p->n_heads;
    int n_kv = p->n_kv_heads;
    int hd = dim / n_heads;
    int kv_dim = n_kv * hd;
    int kv_mul = n_heads / n_kv; /* GQA: how many Q heads share one KV head */

    if (pos < 0 || pos >= p->seq_len) {
        fprintf(stderr, "initium: position %d exceeds context %d\n", pos, p->seq_len);
        return NULL;
    }

    /* embed */
    float *content = w->token_embedding + (size_t)token * dim;
    vec_copy(t->x, content, dim);

    for (int l = 0; l < p->n_layers; l++) {
        /* attn norm */
        rmsnorm(t->xb, t->x, w->rms_att_weight[l], dim, p->norm_eps);

        /* qkv — match llama2.c: matmul(out, x, W, n=cols, d=rows) */
        matmul(t->q, t->xb, w->wq[l], dim, dim);
        matmul(t->k, t->xb, w->wk[l], kv_dim, dim);
        matmul(t->v, t->xb, w->wv[l], kv_dim, dim);

        rope_llama2c(t->q, t->k, dim, kv_dim, hd, pos, p->rope_theta);

        /* write key/value cache for this layer at pos */
        kvcache_append(&t->kv, l, pos, t->k, t->v);

        /* multi-head attention — head outputs into xb (llama2.c layout), then wo -> xb2 */
        for (int h = 0; h < n_heads; h++) {
            float *q = t->q + h * hd;
            float *att = t->att + h * p->seq_len;
            int kv_head = h / kv_mul;

            int start = 0;
            if (p->sliding_window > 0 && (pos + 1) > p->sliding_window) {
                start = (pos + 1) - p->sliding_window;
            }

            for (int tpos = start; tpos <= pos; tpos++) {
                float *krow = kvcache_k_head(&t->kv, l, kv_head, tpos);
                float score = dot(q, krow, hd) / sqrtf((float)hd);
                att[tpos] = score;
            }
            /* llama2.c softmaxs att[0..pos] when no sliding window */
            if (start == 0) {
                softmax(att, pos + 1);
            } else {
                softmax(att + start, pos - start + 1);
                for (int tpos = 0; tpos < start; tpos++) att[tpos] = 0.0f;
            }

            float *xb = t->xb + h * hd;
            memset(xb, 0, (size_t)hd * sizeof(float));
            for (int tpos = start; tpos <= pos; tpos++) {
                float *vrow = kvcache_v_head(&t->kv, l, kv_head, tpos);
                float a = att[tpos];
                for (int i = 0; i < hd; i++) {
                    xb[i] += a * vrow[i];
                }
            }
        }

        matmul(t->xb2, t->xb, w->wo[l], dim, dim);
        residual_add(t->x, t->x, t->xb2, dim);

        /* FFN SwiGLU */
        rmsnorm(t->xb, t->x, w->rms_ffn_weight[l], dim, p->norm_eps);
        matmul(t->hb, t->xb, w->w1[l], hidden_dim, dim);  /* gate */
        matmul(t->hb2, t->xb, w->w3[l], hidden_dim, dim); /* up */
        silu(t->hb, hidden_dim);
        elem_mul(t->hb, t->hb2, hidden_dim);
        matmul(t->xb, t->hb, w->w2[l], dim, hidden_dim);  /* down */
        residual_add(t->x, t->x, t->xb, dim);
    }

    rmsnorm(t->x, t->x, w->rms_final, dim, p->norm_eps);
    matmul(t->logits, t->x, w->wcls, p->vocab_size, dim);
    return t->logits;
}
