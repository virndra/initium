#ifndef INITIUM_SAMPLER_H
#define INITIUM_SAMPLER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float temperature; /* 0 => greedy */
    float top_p;
    int   top_k;
    uint64_t rng_state;
    int greedy; /* force argmax */
} Sampler;

void sampler_init(Sampler *s, float temperature, float top_p, int top_k, uint64_t seed, int greedy);

/* Sample from logits[vocab]. May mutate a working buffer internally — pass logits as mutable. */
int sampler_sample(Sampler *s, float *logits, int vocab_size);

/* Pure greedy argmax (does not mutate logits beyond a scan) */
int sampler_greedy(const float *logits, int vocab_size);

#ifdef __cplusplus
}
#endif

#endif /* INITIUM_SAMPLER_H */
