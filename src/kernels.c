#include "kernels.h"
#include "threadpool.h"

#include <math.h>
#include <string.h>

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

static void matmul_slice(int start, int end, void *ud) {
    MatmulArgs *a = (MatmulArgs *)ud;
    for (int r = start; r < end; r++) {
        const float *row = a->w + (size_t)r * (size_t)a->cols;
        float sum = 0.0f;
        for (int c = 0; c < a->cols; c++) sum += row[c] * a->x[c];
        a->y[r] = sum;
    }
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

float dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
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

/* Interleaved-pair RoPE (llama.cpp / GGUF convention after conversion).
 * For pair (2i, 2i+1): angle = pos * theta_base^(-2i/head_dim) */
void rope(float *q, float *k, int head_dim, int pos, float theta_base) {
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

void rope_all(float *q, float *k, int n_heads, int n_kv_heads, int head_dim,
              int pos, float theta_base) {
    for (int h = 0; h < n_heads; h++) {
        rope(q + h * head_dim, NULL, head_dim, pos, theta_base);
    }
    for (int h = 0; h < n_kv_heads; h++) {
        rope(k + h * head_dim, NULL, head_dim, pos, theta_base);
    }
}
