#include "kernels.h"
#include "threadpool.h"

#include <math.h>
#include <string.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define INITIUM_HAS_AVX2 1
#else
#define INITIUM_HAS_AVX2 0
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#define INITIUM_HAS_NEON 1
#else
#define INITIUM_HAS_NEON 0
#endif

int g_initium_no_simd = 0;
static ThreadPool *g_pool = NULL;

void kernels_set_threadpool(void *threadpool) {
    g_pool = (ThreadPool *)threadpool;
}

void rmsnorm(float *y, const float *x, const float *weight, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) {
        ss += (double)x[i] * (double)x[i];
    }
    float mean = (float)(ss / (double)n);
    float scale = 1.0f / sqrtf(mean + eps);
    for (int i = 0; i < n; i++) {
        y[i] = x[i] * scale * weight[i];
    }
}

typedef struct { float *y; const float *x; const float *w; int cols; } MatmulArgs;

static void matmul_slice_scalar(int start, int end, void *ud) {
    MatmulArgs *a = (MatmulArgs *)ud;
    for (int r = start; r < end; r++) {
        const float *row = a->w + (size_t)r * (size_t)a->cols;
        float sum = 0.0f;
        for (int c = 0; c < a->cols; c++) sum += row[c] * a->x[c];
        a->y[r] = sum;
    }
}

#if INITIUM_HAS_AVX2
static void matmul_slice_avx2(int start, int end, void *ud) {
    MatmulArgs *a = (MatmulArgs *)ud;
    int cols = a->cols;
    for (int r = start; r < end; r++) {
        const float *row = a->w + (size_t)r * (size_t)cols;
        __m256 sum = _mm256_setzero_ps();
        int c = 0;
        for (; c + 7 < cols; c += 8) {
            __m256 vw = _mm256_loadu_ps(row + c);
            __m256 vx = _mm256_loadu_ps(a->x + c);
            sum = _mm256_fmadd_ps(vw, vx, sum);
        }
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        float s = _mm_cvtss_f32(lo);
        for (; c < cols; c++) s += row[c] * a->x[c];
        a->y[r] = s;
    }
}
#endif

#if INITIUM_HAS_NEON
static void matmul_slice_neon(int start, int end, void *ud) {
    MatmulArgs *a = (MatmulArgs *)ud;
    int cols = a->cols;
    for (int r = start; r < end; r++) {
        const float *row = a->w + (size_t)r * (size_t)cols;
        float32x4_t sum = vdupq_n_f32(0.0f);
        int c = 0;
        for (; c + 3 < cols; c += 4) {
            float32x4_t vw = vld1q_f32(row + c);
            float32x4_t vx = vld1q_f32(a->x + c);
            sum = vfmaq_f32(sum, vw, vx);
        }
        float s = vaddvq_f32(sum);
        for (; c < cols; c++) s += row[c] * a->x[c];
        a->y[r] = s;
    }
}
#endif

static void matmul_slice(int start, int end, void *ud) {
    if (!g_initium_no_simd) {
#if INITIUM_HAS_AVX2
        matmul_slice_avx2(start, end, ud);
        return;
#elif INITIUM_HAS_NEON
        matmul_slice_neon(start, end, ud);
        return;
#endif
    }
    matmul_slice_scalar(start, end, ud);
}

void matmul(float *y, const float *x, const float *w, int rows, int cols) {
    MatmulArgs args = { y, x, w, cols };
    if (g_pool && threadpool_num_threads(g_pool) > 1 && rows >= 64) {
        threadpool_parallel_for(g_pool, rows, matmul_slice, &args);
        return;
    }
    matmul_slice(0, rows, &args);
}

void softmax(float *x, int n) {
    float max_v = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_v) max_v = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_v);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        x[i] *= inv;
    }
}

void silu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

void elem_mul(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] *= b[i];
    }
}

void residual_add(float *y, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = a[i] + b[i];
    }
}

void vec_copy(float *dst, const float *src, int n) {
    memcpy(dst, src, (size_t)n * sizeof(float));
}

static float dot_scalar(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

#if INITIUM_HAS_AVX2
static float dot_avx2(const float *a, const float *b, int n) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    /* horizontal sum of 8 floats */
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    /* scalar tail */
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}
#endif

#if INITIUM_HAS_NEON
static float dot_neon(const float *a, const float *b, int n) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vfmaq_f32(sum, va, vb);
    }
    float s = vaddvq_f32(sum);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}
#endif

float dot(const float *a, const float *b, int n) {
    if (!g_initium_no_simd) {
#if INITIUM_HAS_AVX2
        return dot_avx2(a, b, n);
#elif INITIUM_HAS_NEON
        return dot_neon(a, b, n);
#endif
    }
    return dot_scalar(a, b, n);
}

int argmax_f32(const float *x, int n) {
    int best = 0;
    float best_v = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > best_v) {
            best_v = x[i];
            best = i;
        }
    }
    return best;
}

/* RoPE — karpathy/llama2.c convention (production path).
 * Iterates over the full concatenated dim, computing freq from i % head_size.
 * Rotates q always; rotates k only when i < kv_dim (for GQA). */
void rope_llama2c(float *q, float *k, int dim, int kv_dim, int head_size,
                  int pos, float theta_base) {
    for (int i = 0; i < dim; i += 2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(theta_base, (float)head_dim / (float)head_size);
        float val = (float)pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn = (i < kv_dim) ? 2 : 1;
        for (int v = 0; v < rotn; v++) {
            float *vec = (v == 0) ? q : k;
            float v0 = vec[i];
            float v1 = vec[i + 1];
            vec[i]     = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }
    }
}

/* RoPE — interleaved-pair convention (GGUF/NEOX/llama.cpp style).
 * Operates on a single head of head_dim floats.
 * For pair (2i, 2i+1): angle = pos * theta_base^(-2i/head_dim).
 * Retained for future GGUF models that use this convention. */
void rope_neox(float *q, float *k, int head_dim, int pos, float theta_base) {
    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / powf(theta_base, ((float)i) / (float)head_dim);
        float val = (float)pos * freq;
        float c = cosf(val);
        float s = sinf(val);
        float q0 = q[i], q1 = q[i + 1];
        q[i]     = q0 * c - q1 * s;
        q[i + 1] = q0 * s + q1 * c;
        if (k) {
            float k0 = k[i], k1 = k[i + 1];
            k[i]     = k0 * c - k1 * s;
            k[i + 1] = k0 * s + k1 * c;
        }
    }
}

void rope_neox_all(float *q, float *k, int n_heads, int n_kv_heads, int head_dim,
                   int pos, float theta_base) {
    for (int h = 0; h < n_heads; h++) {
        rope_neox(q + h * head_dim, NULL, head_dim, pos, theta_base);
    }
    for (int h = 0; h < n_kv_heads; h++) {
        rope_neox(k + h * head_dim, NULL, head_dim, pos, theta_base);
    }
}
