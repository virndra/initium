#ifndef INITIUM_VERIFY_H
#define INITIUM_VERIFY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initium reference dump format "INI1"
 * Header:
 *   magic[4] = 'I','N','I','1'
 *   version  u32 = 1
 *   vocab_size u32
 *   n_prompt_tokens u32
 *   n_gen_steps u32
 * Body:
 *   prompt_tokens: u32[n_prompt_tokens]
 *   for each step in 0..n_gen_steps-1:
 *     logits: f32[vocab_size]
 *     chosen_token: u32
 */

typedef struct {
    uint32_t vocab_size;
    uint32_t n_prompt;
    uint32_t n_steps;
    uint32_t *prompt_tokens;
    float    *logits;       /* n_steps * vocab_size */
    uint32_t *chosen;       /* n_steps */
    void     *blob;         /* owned */
} RefDump;

int  verify_load(const char *path, RefDump *d);
void verify_free(RefDump *d);

/* Compare engine logits to reference step. Returns 0 if within tier A/B.
 * tier: 0 = A (max abs < 1e-3 + argmax match), 1 = B (argmax + top5)
 * Prints diagnostics to stderr. */
int verify_compare_step(const RefDump *d, int step, const float *logits,
                        int vocab_size, int tier, float *out_max_abs);

/* Write a dump (for self-test / engine export) */
int verify_write(const char *path, const uint32_t *prompt, uint32_t n_prompt,
                 uint32_t vocab_size, uint32_t n_steps,
                 const float *logits_steps, const uint32_t *chosen);

#ifdef __cplusplus
}
#endif

#endif /* INITIUM_VERIFY_H */
