#include "quant.h"
#include "gguf.h"
#include "kernels.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* IEEE half <-> float (bit manipulation, no dependency) */
float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)(h & 0x3ffu);
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            exp++;
            mant &= 0x3ffu;
            f = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7f800000u | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t t = (uint32_t)(14 - exp);
        uint32_t m = mant >> t;
        return (uint16_t)(sign | m);
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

void dequant_q8_0_block(const block_q8_0 *b, float *out) {
    float d = fp16_to_fp32(b->d);
    for (int i = 0; i < QK8_0; i++) {
        out[i] = d * (float)b->qs[i];
    }
}

void dequant_q4_0_block(const block_q4_0 *b, float *out) {
    float d = fp16_to_fp32(b->d);
    for (int i = 0; i < QK4_0 / 2; i++) {
        int x0 = (b->qs[i] & 0x0f) - 8;
        int x1 = (b->qs[i] >> 4) - 8;
        out[i]             = d * (float)x0;
        out[i + QK4_0 / 2] = d * (float)x1;
    }
}

void matmul_q8_0(float *y, const float *x, const void *w, int rows, int cols) {
    const int nb = cols / QK8_0;
    const block_q8_0 *bw = (const block_q8_0 *)w;
    for (int r = 0; r < rows; r++) {
        const block_q8_0 *row = bw + (size_t)r * (size_t)nb;
        float sum = 0.0f;
        for (int ib = 0; ib < nb; ib++) {
            float wtmp[QK8_0];
            dequant_q8_0_block(&row[ib], wtmp);
            const float *xb = x + ib * QK8_0;
            for (int i = 0; i < QK8_0; i++) sum += wtmp[i] * xb[i];
        }
        y[r] = sum;
    }
}

void matmul_q4_0(float *y, const float *x, const void *w, int rows, int cols) {
    const int nb = cols / QK4_0;
    const block_q4_0 *bw = (const block_q4_0 *)w;
    for (int r = 0; r < rows; r++) {
        const block_q4_0 *row = bw + (size_t)r * (size_t)nb;
        float sum = 0.0f;
        for (int ib = 0; ib < nb; ib++) {
            float wtmp[QK4_0];
            dequant_q4_0_block(&row[ib], wtmp);
            const float *xb = x + ib * QK4_0;
            for (int i = 0; i < QK4_0; i++) sum += wtmp[i] * xb[i];
        }
        y[r] = sum;
    }
}

static uint64_t n_elements(const uint64_t ne[4], int n_dims) {
    uint64_t n = 1;
    for (int i = 0; i < n_dims; i++) n *= ne[i];
    return n;
}

int quant_dequantize_tensor(float *out, const void *data, int type,
                            const uint64_t ne[4], int n_dims) {
    uint64_t n = n_elements(ne, n_dims);
    if (type == GGML_TYPE_F32) {
        memcpy(out, data, (size_t)n * sizeof(float));
        return 0;
    }
    if (type == GGML_TYPE_F16) {
        const uint16_t *h = (const uint16_t *)data;
        for (uint64_t i = 0; i < n; i++) out[i] = fp16_to_fp32(h[i]);
        return 0;
    }
    if (type == GGML_TYPE_Q8_0) {
        if (ne[0] % QK8_0 != 0) {
            fprintf(stderr, "initium/quant: Q8_0 ne0=%llu not multiple of 32\n",
                    (unsigned long long)ne[0]);
            return -1;
        }
        const block_q8_0 *blocks = (const block_q8_0 *)data;
        uint64_t nb0 = ne[0] / QK8_0;
        uint64_t nblocks = n / QK8_0;
        for (uint64_t b = 0; b < nblocks; b++) {
            dequant_q8_0_block(&blocks[b], out + b * QK8_0);
        }
        (void)nb0;
        return 0;
    }
    if (type == GGML_TYPE_Q4_0) {
        if (ne[0] % QK4_0 != 0) {
            fprintf(stderr, "initium/quant: Q4_0 ne0=%llu not multiple of 32\n",
                    (unsigned long long)ne[0]);
            return -1;
        }
        const block_q4_0 *blocks = (const block_q4_0 *)data;
        uint64_t nblocks = n / QK4_0;
        for (uint64_t b = 0; b < nblocks; b++) {
            dequant_q4_0_block(&blocks[b], out + b * QK4_0);
        }
        return 0;
    }
    if (type == GGML_TYPE_Q6_K) {
        /* llama.cpp dequantize_row_q6_K */
        enum { QK_K = 256 };
        typedef struct {
            uint8_t ql[QK_K / 2];
            uint8_t qh[QK_K / 4];
            int8_t  scales[QK_K / 16];
            uint16_t d;
        } block_q6_K;
        if (n % QK_K != 0) {
            fprintf(stderr, "initium/quant: Q6_K n=%llu not multiple of 256\n",
                    (unsigned long long)n);
            return -1;
        }
        const block_q6_K *blocks = (const block_q6_K *)data;
        uint64_t nb = n / QK_K;
        float *y = out;
        for (uint64_t i = 0; i < nb; i++) {
            float d = fp16_to_fp32(blocks[i].d);
            const uint8_t *ql = blocks[i].ql;
            const uint8_t *qh = blocks[i].qh;
            const int8_t *sc = blocks[i].scales;
            for (int nblk = 0; nblk < QK_K; nblk += 128) {
                for (int l = 0; l < 32; ++l) {
                    int is = l / 16;
                    int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    y[l + 0]  = d * (float)sc[is + 0] * (float)q1;
                    y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                    y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                    y[l + 96] = d * (float)sc[is + 6] * (float)q4;
                }
                y += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        return 0;
    }
    fprintf(stderr, "initium/quant: unsupported ggml type %d\n", type);
    return -1;
}
