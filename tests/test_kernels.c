#include "../src/kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

static void expect_close(const char *name, float a, float b, float tol) {
    if (fabsf(a - b) > tol) {
        fprintf(stderr, "FAIL %s: got %.8g want %.8g (tol %g)\n", name, a, b, tol);
        failures++;
    }
}

static void test_rmsnorm(void) {
    float x[4] = {1, 2, 3, 4};
    float w[4] = {1, 1, 1, 1};
    float y[4];
    rmsnorm(y, x, w, 4, 1e-5f);
    float mean = (1 + 4 + 9 + 16) / 4.0f;
    float scale = 1.0f / sqrtf(mean + 1e-5f);
    for (int i = 0; i < 4; i++) {
        expect_close("rmsnorm", y[i], x[i] * scale, 1e-5f);
    }
}

static void test_softmax(void) {
    float x[3] = {1, 2, 3};
    softmax(x, 3);
    float s = x[0] + x[1] + x[2];
    expect_close("softmax_sum", s, 1.0f, 1e-5f);
    if (!(x[2] > x[1] && x[1] > x[0])) {
        fprintf(stderr, "FAIL softmax ordering\n");
        failures++;
    }
}

static void test_matmul(void) {
    /* W 2x3, x 3 -> y 2 */
    float w[6] = {1, 2, 3, 4, 5, 6};
    float x[3] = {1, 1, 1};
    float y[2];
    matmul(y, x, w, 2, 3);
    expect_close("matmul0", y[0], 6.0f, 1e-6f);
    expect_close("matmul1", y[1], 15.0f, 1e-6f);
}

static void test_argmax(void) {
    float x[5] = {0.1f, 0.5f, 0.2f, 0.9f, 0.3f};
    int i = argmax_f32(x, 5);
    if (i != 3) {
        fprintf(stderr, "FAIL argmax got %d\n", i);
        failures++;
    }
}

int main(void) {
    test_rmsnorm();
    test_softmax();
    test_matmul();
    test_argmax();
    if (failures) {
        fprintf(stderr, "%d kernel tests failed\n", failures);
        return 1;
    }
    printf("test_kernels: OK\n");
    return 0;
}
