/* kernels.h — ALL math lives in kernels.c. Nothing else may implement matmul/softmax/etc. */
#ifndef INITIUM_KERNELS_H
#define INITIUM_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* y[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]  — mean accumulated in float */
void rmsnorm(float *y, const float *x, const float *weight, int n, float eps);

/* y = W * x  where W is (rows x cols) row-major, x is cols, y is rows */
void matmul(float *y, const float *x, const float *w, int rows, int cols);

/* Numerically stable softmax in-place over n elements */
void softmax(float *x, int n);

/* silu(x) = x * sigmoid(x)  in-place */
void silu(float *x, int n);

/* Element-wise product: a[i] *= b[i] */
void elem_mul(float *a, const float *b, int n);

/* y[i] = a[i] + b[i] */
void residual_add(float *y, const float *a, const float *b, int n);

/* Copy n floats */
void vec_copy(float *dst, const float *src, int n);

/* RoPE — karpathy/llama2.c convention (production path for llama2.c .bin models).
 * Iterates over the full concatenated dim, computing freq from i % head_size.
 * Rotates q (dim floats) always; rotates k only when i < kv_dim (for GQA).
 * head_size must be even. */
void rope_llama2c(float *q, float *k, int dim, int kv_dim, int head_size,
                  int pos, float theta_base);

/* RoPE — interleaved-pair convention (GGUF/NEOX/llama.cpp style).
 * Operates on a single head of head_dim floats.
 * For pair (2i, 2i+1): angle = pos * theta_base^(-2i/head_dim).
 * head_dim must be even. */
void rope_neox(float *q, float *k, int head_dim, int pos, float theta_base);

/* Apply per-head NEOX RoPE to all query heads and KV heads.
 * q: n_heads * head_dim, k: n_kv_heads * head_dim */
void rope_neox_all(float *q, float *k, int n_heads, int n_kv_heads, int head_dim,
                   int pos, float theta_base);

/* Dot product of two length-n vectors */
float dot(const float *a, const float *b, int n);

/* argmax of n floats; returns index */
int argmax_f32(const float *x, int n);

/* Global flag: when non-zero, optimized kernels fall back to scalar path */
extern int g_initium_no_simd;

/* Optional thread pool for matmul (NULL / 1 thread => scalar). Opaque void* avoids
 * circular includes; pass ThreadPool* from model load. */
void kernels_set_threadpool(void *threadpool);

#ifdef __cplusplus
}
#endif

#endif /* INITIUM_KERNELS_H */
