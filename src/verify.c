#include "verify.h"
#include "kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int verify_load(const char *path, RefDump *d) {
    memset(d, 0, sizeof(*d));
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "initium/verify: cannot open '%s'\n", path);
        return -1;
    }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "INI1", 4) != 0) {
        fprintf(stderr, "initium/verify: bad magic in '%s'\n", path);
        fclose(f);
        return -1;
    }
    uint32_t version = 0;
    if (fread(&version, 4, 1, f) != 1 || version != 1) {
        fprintf(stderr, "initium/verify: unsupported version\n");
        fclose(f);
        return -1;
    }
    if (fread(&d->vocab_size, 4, 1, f) != 1 ||
        fread(&d->n_prompt, 4, 1, f) != 1 ||
        fread(&d->n_steps, 4, 1, f) != 1) {
        fclose(f);
        return -1;
    }

    size_t prompt_bytes = (size_t)d->n_prompt * 4;
    size_t logits_bytes = (size_t)d->n_steps * (size_t)d->vocab_size * sizeof(float);
    size_t chosen_bytes = (size_t)d->n_steps * 4;
    size_t total = prompt_bytes + logits_bytes + chosen_bytes;
    d->blob = malloc(total);
    if (!d->blob) { fclose(f); return -1; }

    d->prompt_tokens = (uint32_t *)d->blob;
    d->logits = (float *)((uint8_t *)d->blob + prompt_bytes);
    d->chosen = (uint32_t *)((uint8_t *)d->blob + prompt_bytes + logits_bytes);

    if (d->n_prompt && fread(d->prompt_tokens, 4, d->n_prompt, f) != d->n_prompt) {
        verify_free(d); fclose(f); return -1;
    }
    for (uint32_t s = 0; s < d->n_steps; s++) {
        if (fread(d->logits + (size_t)s * d->vocab_size, sizeof(float), d->vocab_size, f)
            != d->vocab_size) {
            verify_free(d); fclose(f); return -1;
        }
        if (fread(&d->chosen[s], 4, 1, f) != 1) {
            verify_free(d); fclose(f); return -1;
        }
    }
    fclose(f);
    return 0;
}

void verify_free(RefDump *d) {
    free(d->blob);
    memset(d, 0, sizeof(*d));
}

int verify_write(const char *path, const uint32_t *prompt, uint32_t n_prompt,
                 uint32_t vocab_size, uint32_t n_steps,
                 const float *logits_steps, const uint32_t *chosen) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite("INI1", 1, 4, f);
    uint32_t version = 1;
    fwrite(&version, 4, 1, f);
    fwrite(&vocab_size, 4, 1, f);
    fwrite(&n_prompt, 4, 1, f);
    fwrite(&n_steps, 4, 1, f);
    if (n_prompt) fwrite(prompt, 4, n_prompt, f);
    for (uint32_t s = 0; s < n_steps; s++) {
        fwrite(logits_steps + (size_t)s * vocab_size, sizeof(float), vocab_size, f);
        fwrite(&chosen[s], 4, 1, f);
    }
    fclose(f);
    return 0;
}

static void top5(const float *logits, int n, int out[5]) {
    for (int k = 0; k < 5; k++) out[k] = -1;
    float bestv[5];
    for (int k = 0; k < 5; k++) bestv[k] = -1e30f;
    for (int i = 0; i < n; i++) {
        float v = logits[i];
        for (int k = 0; k < 5; k++) {
            if (v > bestv[k]) {
                for (int m = 4; m > k; m--) {
                    bestv[m] = bestv[m - 1];
                    out[m] = out[m - 1];
                }
                bestv[k] = v;
                out[k] = i;
                break;
            }
        }
    }
}

int verify_compare_step(const RefDump *d, int step, const float *logits,
                        int vocab_size, int tier, float *out_max_abs) {
    if (step < 0 || (uint32_t)step >= d->n_steps) {
        fprintf(stderr, "verify: step %d out of range (n_steps=%u)\n", step, d->n_steps);
        return -1;
    }
    if ((uint32_t)vocab_size != d->vocab_size) {
        fprintf(stderr, "verify: vocab mismatch engine=%d ref=%u\n", vocab_size, d->vocab_size);
        return -1;
    }
    const float *ref = d->logits + (size_t)step * d->vocab_size;
    float max_abs = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        float a = fabsf(logits[i] - ref[i]);
        if (a > max_abs) max_abs = a;
    }
    if (out_max_abs) *out_max_abs = max_abs;

    int eng_tok = argmax_f32(logits, vocab_size);
    int ref_tok = argmax_f32(ref, vocab_size);
    int chosen = (int)d->chosen[step];

    if (tier == 0) {
        /* Tier A */
        int ok = (max_abs < 1e-3f) && (eng_tok == ref_tok);
        if (!ok) {
            fprintf(stderr, "verify step %d FAIL tierA: max_abs=%.6g eng_argmax=%d ref_argmax=%d chosen=%d\n",
                    step, max_abs, eng_tok, ref_tok, chosen);
        }
        return ok ? 0 : -1;
    } else {
        /* Tier B: argmax + top5 set */
        int e5[5], r5[5];
        top5(logits, vocab_size, e5);
        top5(ref, vocab_size, r5);
        int top5_ok = 1;
        for (int k = 0; k < 5; k++) {
            int found = 0;
            for (int j = 0; j < 5; j++) if (e5[k] == r5[j]) found = 1;
            if (!found) top5_ok = 0;
        }
        int ok = (eng_tok == ref_tok) && top5_ok;
        if (!ok) {
            fprintf(stderr, "verify step %d FAIL tierB: eng_argmax=%d ref_argmax=%d max_abs=%.6g\n",
                    step, eng_tok, ref_tok, max_abs);
        }
        return ok ? 0 : -1;
    }
}
