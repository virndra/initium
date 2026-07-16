#include "sampler.h"
#include "kernels.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* xorshift64* */
static uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static float rng_f32(uint64_t *state) {
    /* [0, 1) */
    return (rng_next(state) >> 11) * (1.0f / 9007199254740992.0f);
}

void sampler_init(Sampler *s, float temperature, float top_p, int top_k, uint64_t seed, int greedy) {
    s->temperature = temperature;
    s->top_p = top_p;
    s->top_k = top_k;
    s->greedy = greedy || temperature <= 0.0f;
    if (seed == 0) {
        seed = (uint64_t)time(NULL) ^ ((uint64_t)(uintptr_t)s << 16);
        if (seed == 0) seed = 1;
    }
    s->rng_state = seed;
}

int sampler_greedy(const float *logits, int vocab_size) {
    return argmax_f32(logits, vocab_size);
}

typedef struct {
    float logit;
    int   idx;
} ProbIndex;

static int cmp_prob_desc(const void *a, const void *b) {
    float da = ((const ProbIndex *)a)->logit;
    float db = ((const ProbIndex *)b)->logit;
    if (da > db) return -1;
    if (da < db) return 1;
    return 0;
}

int sampler_sample(Sampler *s, float *logits, int vocab_size) {
    if (s->greedy) {
        return sampler_greedy(logits, vocab_size);
    }

    /* Temperature */
    float temp = s->temperature;
    if (temp != 1.0f) {
        float inv = 1.0f / temp;
        for (int i = 0; i < vocab_size; i++) {
            logits[i] *= inv;
        }
    }

    /* Softmax -> probabilities in-place */
    softmax(logits, vocab_size);

    /* Build index list for top-k / top-p */
    ProbIndex *pi = (ProbIndex *)malloc((size_t)vocab_size * sizeof(ProbIndex));
    if (!pi) return sampler_greedy(logits, vocab_size);

    for (int i = 0; i < vocab_size; i++) {
        pi[i].logit = logits[i];
        pi[i].idx = i;
    }
    qsort(pi, (size_t)vocab_size, sizeof(ProbIndex), cmp_prob_desc);

    int n = vocab_size;
    if (s->top_k > 0 && s->top_k < n) {
        n = s->top_k;
    }

    /* Nucleus (top-p) */
    if (s->top_p < 1.0f && s->top_p > 0.0f) {
        float cum = 0.0f;
        int cutoff = n;
        for (int i = 0; i < n; i++) {
            cum += pi[i].logit;
            if (cum >= s->top_p) {
                cutoff = i + 1;
                break;
            }
        }
        n = cutoff;
    }

    /* Renormalize over remaining */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += pi[i].logit;
    float r = rng_f32(&s->rng_state) * sum;
    float cdf = 0.0f;
    int choice = pi[n - 1].idx;
    for (int i = 0; i < n; i++) {
        cdf += pi[i].logit;
        if (r < cdf) {
            choice = pi[i].idx;
            break;
        }
    }
    free(pi);
    return choice;
}
