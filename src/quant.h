#ifndef INITIUM_QUANT_H
#define INITIUM_QUANT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* llama.cpp-compatible block formats */

#define QK8_0 32
#define QK4_0 32

typedef struct {
    uint16_t d;      /* fp16 scale */
    int8_t   qs[QK8_0];
} block_q8_0;

typedef struct {
    uint16_t d;           /* fp16 scale */
    uint8_t  qs[QK4_0 / 2]; /* nibbles */
} block_q4_0;

/* fp16 <-> fp32 helpers */
float    fp16_to_fp32(uint16_t h);
uint16_t fp32_to_fp16(float f);

/* Dequantize one block into 32 floats */
void dequant_q8_0_block(const block_q8_0 *b, float *out);
void dequant_q4_0_block(const block_q4_0 *b, float *out);

/* Quantized matmul: y[rows] = W[rows x cols] * x[cols]
 * W is packed Q8_0 or Q4_0 row-major blocks along cols. */
void matmul_q8_0(float *y, const float *x, const void *w, int rows, int cols);
void matmul_q4_0(float *y, const float *x, const void *w, int rows, int cols);

/* Dequantize a full ggml/GGUF tensor into caller-allocated f32[n_elem].
 * type is GGMLType; ne[0..n_dims-1] element counts; data points at tensor bytes.
 * Layout: elements contiguous with ne[0] fastest (ggml). Returns 0 on success. */
int quant_dequantize_tensor(float *out, const void *data, int type,
                            const uint64_t ne[4], int n_dims);

#ifdef __cplusplus
}
#endif

#endif /* INITIUM_QUANT_H */
