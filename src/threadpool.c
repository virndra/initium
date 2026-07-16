#include "threadpool.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static int hw_concurrency(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors > 0 ? (int)si.dwNumberOfProcessors : 1;
}
#else
#include <unistd.h>
static int hw_concurrency(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
}
#endif

typedef struct {
    initium_task_fn fn;
    void *userdata;
    int start;
    int end;
} WorkSlice;

struct ThreadPool {
    int n_threads;
    /* For M5 we use simple fork-join per parallel_for via temporary threads.
     * Persistent workers come in when matmul is hot — same API. */
    int persistent; /* reserved */
};

ThreadPool *threadpool_create(int n_threads) {
    ThreadPool *tp = (ThreadPool *)calloc(1, sizeof(ThreadPool));
    if (!tp) return NULL;
    if (n_threads <= 0) n_threads = hw_concurrency();
    if (n_threads < 1) n_threads = 1;
    tp->n_threads = n_threads;
    return tp;
}

void threadpool_destroy(ThreadPool *tp) {
    free(tp);
}

int threadpool_num_threads(const ThreadPool *tp) {
    return tp ? tp->n_threads : 1;
}

typedef struct {
    initium_task_fn fn;
    void *userdata;
    int start;
    int end;
} ThreadArg;

static void *worker_entry(void *arg) {
    ThreadArg *a = (ThreadArg *)arg;
    a->fn(a->start, a->end, a->userdata);
    return NULL;
}

void threadpool_parallel_for(ThreadPool *tp, int n, initium_task_fn fn, void *userdata) {
    if (n <= 0) return;
    int nt = tp ? tp->n_threads : 1;
    if (nt <= 1 || n < 2) {
        fn(0, n, userdata);
        return;
    }
    if (nt > n) nt = n;

    pthread_t *threads = (pthread_t *)calloc((size_t)nt, sizeof(pthread_t));
    ThreadArg *args = (ThreadArg *)calloc((size_t)nt, sizeof(ThreadArg));
    if (!threads || !args) {
        free(threads);
        free(args);
        fn(0, n, userdata);
        return;
    }

    int chunk = (n + nt - 1) / nt;
    for (int t = 0; t < nt; t++) {
        int start = t * chunk;
        int end = start + chunk;
        if (start >= n) break;
        if (end > n) end = n;
        args[t].fn = fn;
        args[t].userdata = userdata;
        args[t].start = start;
        args[t].end = end;
        if (t == 0) {
            /* run first slice on current thread after launching others */
            continue;
        }
        if (pthread_create(&threads[t], NULL, worker_entry, &args[t]) != 0) {
            /* fallback: run this slice inline later */
            args[t].fn = NULL;
        }
    }

    /* main thread does slice 0 */
    args[0].fn(args[0].start, args[0].end, args[0].userdata);

    for (int t = 1; t < nt; t++) {
        if (args[t].fn == NULL && args[t].start < args[t].end) {
            /* failed to launch — run inline */
            fn(args[t].start, args[t].end, userdata);
        } else if (args[t].start < n && args[t].fn != NULL) {
            pthread_join(threads[t], NULL);
        }
    }

    free(threads);
    free(args);
}
