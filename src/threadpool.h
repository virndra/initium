#ifndef INITIUM_THREADPOOL_H
#define INITIUM_THREADPOOL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*initium_task_fn)(int start, int end, void *userdata);

typedef struct ThreadPool ThreadPool;

/* Create a pool with n_threads workers (n_threads <= 0 => hardware concurrency). */
ThreadPool *threadpool_create(int n_threads);
void        threadpool_destroy(ThreadPool *tp);

/* Parallel-for over [0, n) partitioned into static row blocks across workers.
 * Blocks until all work completes. Safe to call with n_threads==1 (runs inline). */
void threadpool_parallel_for(ThreadPool *tp, int n, initium_task_fn fn, void *userdata);

int threadpool_num_threads(const ThreadPool *tp);

#ifdef __cplusplus
}
#endif

#endif /* INITIUM_THREADPOOL_H */
