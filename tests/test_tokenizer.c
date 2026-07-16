#include "../src/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Minimal smoke test without model files: constructs a tiny fake vocab. */
int main(void) {
    Tokenizer t;
    memset(&t, 0, sizeof(t));
    t.vocab_size = 5;
    t.token_table = (char **)calloc(5, sizeof(char *));
    t.scores = (float *)calloc(5, sizeof(float));
    t.token_table[0] = strdup("<unk>");
    t.token_table[1] = strdup("<s>");
    t.token_table[2] = strdup("</s>");
    /* ▁hi */
    t.token_table[3] = strdup("\xE2\x96\x81""hi");
    t.token_table[4] = strdup("!");
    t.bos_id = 1;
    t.eos_id = 2;
    t.add_bos = 1;

    int ids[16];
    int n = tokenizer_encode(&t, "hi!", ids, 16, 1);
    if (n < 2) {
        fprintf(stderr, "encode too short: %d\n", n);
        return 1;
    }
    if (ids[0] != 1) {
        fprintf(stderr, "expected BOS first, got %d\n", ids[0]);
        return 1;
    }

    char out[64];
    int dec_ids[] = {3, 4};
    tokenizer_decode(&t, dec_ids, 2, out, sizeof(out));
    printf("decoded='%s'\n", out);
    printf("test_tokenizer: OK (smoke)\n");

    tokenizer_free(&t);
    return 0;
}
