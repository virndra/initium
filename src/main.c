#include "model.h"
#include "tokenizer.h"
#include "sampler.h"
#include "kernels.h"
#include "verify.h"
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
static double now_sec(void) {
    static LARGE_INTEGER freq;
    static int init = 0;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

typedef struct {
    const char *model_path;
    const char *tokenizer_path;
    const char *prompt;
    const char *verify_path;
    int max_new;
    float temperature;
    float top_p;
    int top_k;
    uint64_t seed;
    int seed_set;
    int ctx;
    int threads;
    int greedy;
    int chat;
    int stats;
    int no_simd;
} Options;

static void usage(const char *argv0) {
    fprintf(stderr,
        "Initium — from-scratch LLM inference engine\n"
        "Usage: %s -m <model> [options]\n"
        "  -m <path>        model (.bin llama2c or .gguf) [required]\n"
        "  --tokenizer <p>  tokenizer.bin (required for llama2c .bin models)\n"
        "  -p <string>      prompt (one-shot)\n"
        "  --chat           interactive REPL\n"
        "  -n <int>         max new tokens (default 256)\n"
        "  -t <float>       temperature (default 0.8; 0 = greedy)\n"
        "  --top-p <float>  nucleus (default 0.95)\n"
        "  --top-k <int>    (default 40)\n"
        "  --seed <int>     RNG seed\n"
        "  --ctx <int>      context override\n"
        "  --threads <int>  (default 1 for now)\n"
        "  --greedy         force argmax\n"
        "  --verify <path>  parity mode vs reference dump\n"
        "  --stats          tokens/sec on stderr\n"
        "  --no-simd        force scalar kernels\n",
        argv0);
}

static int ends_with(const char *s, const char *suf) {
    size_t n = strlen(s), m = strlen(suf);
    return n >= m && strcmp(s + n - m, suf) == 0;
}

static int parse_args(int argc, char **argv, Options *o) {
    memset(o, 0, sizeof(*o));
    o->max_new = 256;
    o->temperature = 0.8f;
    o->top_p = 0.95f;
    o->top_k = 40;
    o->threads = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) o->model_path = argv[++i];
        else if (!strcmp(argv[i], "--tokenizer") && i + 1 < argc) o->tokenizer_path = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) o->prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) o->max_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) o->temperature = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) o->top_p = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) o->top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            o->seed = (uint64_t)strtoull(argv[++i], NULL, 10);
            o->seed_set = 1;
        }
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) o->ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) o->threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--greedy")) o->greedy = 1;
        else if (!strcmp(argv[i], "--chat")) o->chat = 1;
        else if (!strcmp(argv[i], "--stats")) o->stats = 1;
        else if (!strcmp(argv[i], "--no-simd")) o->no_simd = 1;
        else if (!strcmp(argv[i], "--verify") && i + 1 < argc) o->verify_path = argv[++i];
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 1;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            usage(argv[0]);
            return -1;
        }
    }
    if (!o->model_path) {
        usage(argv[0]);
        return -1;
    }
    return 0;
}

static int generate(Transformer *m, Tokenizer *tok, Sampler *samp, Options *opt,
                    const char *prompt) {
    const ModelConfig *cfg = model_config(m);
    int *tokens = (int *)malloc((size_t)(cfg->seq_len + 8) * sizeof(int));
    if (!tokens) return -1;

    int n_prompt = tokenizer_encode(tok, prompt ? prompt : "", tokens, cfg->seq_len, 1);
    if (n_prompt < 1) {
        fprintf(stderr, "initium: encode produced no tokens\n");
        free(tokens);
        return -1;
    }
    if (n_prompt >= cfg->seq_len) {
        fprintf(stderr, "initium: prompt length %d exceeds context %d\n", n_prompt, cfg->seq_len);
        free(tokens);
        return -1;
    }

    RefDump ref;
    int do_verify = 0;
    if (opt->verify_path) {
        if (verify_load(opt->verify_path, &ref) != 0) {
            free(tokens);
            return -1;
        }
        do_verify = 1;
        /* Prefer prompt tokens from reference for exact parity */
        if (ref.n_prompt > 0 && (int)ref.n_prompt <= cfg->seq_len) {
            n_prompt = (int)ref.n_prompt;
            for (int i = 0; i < n_prompt; i++) tokens[i] = (int)ref.prompt_tokens[i];
        }
    }

    double t0 = now_sec();
    double t_prefill_end = t0;
    int pos = 0;
    int token = tokens[0];
    int steps_ok = 0;
    int steps_total = 0;
    float worst_abs = 0.0f;

    /* Prefill + generate */
    int max_steps = opt->max_new;
    if (do_verify && (int)ref.n_steps < max_steps) max_steps = (int)ref.n_steps;

    int next = 0;
    int prev_token = token;
    for (pos = 0; pos < n_prompt + max_steps; pos++) {
        float *logits = model_forward(m, token, pos);
        if (!logits) break;

        if (pos < n_prompt - 1) {
            /* still in prompt: teacher force */
            next = tokens[pos + 1];
        } else {
            if (pos == n_prompt - 1) t_prefill_end = now_sec();

            if (do_verify) {
                int step = pos - (n_prompt - 1);
                if (step >= 0 && step < (int)ref.n_steps) {
                    float mabs = 0;
                    int tier = 0; /* A for fp32 */
                    int rc = verify_compare_step(&ref, step, logits, cfg->vocab_size, tier, &mabs);
                    if (mabs > worst_abs) worst_abs = mabs;
                    steps_total++;
                    if (rc == 0) steps_ok++;
                }
            }

            /* sample working copy */
            float *work = (float *)malloc((size_t)cfg->vocab_size * sizeof(float));
            if (!work) break;
            memcpy(work, logits, (size_t)cfg->vocab_size * sizeof(float));
            next = sampler_sample(samp, work, cfg->vocab_size);
            free(work);

            /* decode and print (llama2.c: strip leading space only after BOS) */
            char piece[256];
            tokenizer_decode_token(tok, token, next, piece, sizeof(piece));
            fputs(piece, stdout);
            fflush(stdout);

            /* llama2.c ends generation on BOS(=1) as well as EOS */
            if (next == tok->eos_id || next == tok->bos_id) {
                pos++;
                break;
            }
            if (pos + 1 >= n_prompt + max_steps) {
                pos++;
                break;
            }
        }
        prev_token = token;
        (void)prev_token;
        token = next;
    }
    fputc('\n', stdout);

    double t1 = now_sec();
    if (opt->stats) {
        int gen_tokens = pos - (n_prompt - 1);
        if (gen_tokens < 0) gen_tokens = 0;
        double prefill_s = t_prefill_end - t0;
        double decode_s = t1 - t_prefill_end;
        fprintf(stderr, "stats: prompt_tokens=%d gen_tokens=%d prefill=%.2f tok/s decode=%.2f tok/s\n",
                n_prompt,
                gen_tokens,
                prefill_s > 0 ? (n_prompt / prefill_s) : 0.0,
                decode_s > 0 ? (gen_tokens / decode_s) : 0.0);
    }

    if (do_verify) {
        fprintf(stderr, "verify: %d/%d steps tierA ok, worst max_abs=%.6g\n",
                steps_ok, steps_total, worst_abs);
        verify_free(&ref);
        free(tokens);
        return (steps_ok == steps_total && steps_total > 0) ? 0 : 2;
    }

    free(tokens);
    return 0;
}

int main(int argc, char **argv) {
    Options opt;
    int pr = parse_args(argc, argv, &opt);
    if (pr != 0) return pr < 0 ? 1 : 0;

    g_initium_no_simd = opt.no_simd;

    Transformer model;
    int rc;
    if (ends_with(opt.model_path, ".gguf") || ends_with(opt.model_path, ".GGUF")) {
        rc = model_load_gguf(&model, opt.model_path, opt.ctx);
    } else {
        rc = model_load_llama2c_bin(&model, opt.model_path, opt.ctx);
    }
    if (rc != 0) {
        fprintf(stderr, "initium: failed to load model\n");
        return 1;
    }

    Tokenizer tok;
    memset(&tok, 0, sizeof(tok));
    if (ends_with(opt.model_path, ".gguf") || ends_with(opt.model_path, ".GGUF")) {
        /* tokenizer lives inside GGUF metadata */
        GGUFFile gftok;
        if (gguf_open(opt.model_path, &gftok) != 0 ||
            tokenizer_load_gguf(&tok, &gftok) != 0) {
            fprintf(stderr, "initium: failed to load tokenizer from GGUF\n");
            gguf_close(&gftok);
            model_free(&model);
            return 1;
        }
        gguf_close(&gftok);
    } else if (opt.tokenizer_path) {
        if (tokenizer_load_llama2c(&tok, opt.tokenizer_path, model.cfg.vocab_size) != 0) {
            model_free(&model);
            return 1;
        }
    } else {
        char guess[1024];
        snprintf(guess, sizeof(guess), "tokenizer.bin");
        if (tokenizer_load_llama2c(&tok, guess, model.cfg.vocab_size) != 0) {
            fprintf(stderr, "initium: need --tokenizer for .bin models\n");
            model_free(&model);
            return 1;
        }
    }

    Sampler samp;
    uint64_t seed = opt.seed_set ? opt.seed : 0;
    sampler_init(&samp, opt.temperature, opt.top_p, opt.top_k, seed, opt.greedy);

    if (opt.chat) {
        char line[4096];
        fprintf(stderr, "initium chat (empty line to quit)\n");
        while (1) {
            fprintf(stderr, "> ");
            if (!fgets(line, sizeof(line), stdin)) break;
            size_t n = strlen(line);
            while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
            if (n == 0) break;
            generate(&model, &tok, &samp, &opt, line);
            /* reset caches between turns for simplicity */
            memset(model.key_cache, 0, (size_t)model.cfg.n_layers * model.cfg.seq_len *
                   (model.cfg.dim / model.cfg.n_heads) * model.cfg.n_kv_heads * sizeof(float));
            memset(model.value_cache, 0, (size_t)model.cfg.n_layers * model.cfg.seq_len *
                   (model.cfg.dim / model.cfg.n_heads) * model.cfg.n_kv_heads * sizeof(float));
        }
    } else {
        rc = generate(&model, &tok, &samp, &opt, opt.prompt ? opt.prompt : "");
    }

    tokenizer_free(&tok);
    model_free(&model);
    return rc;
}
