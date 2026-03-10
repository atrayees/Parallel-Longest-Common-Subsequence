#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define BLOCK_SIZE 256
typedef unsigned long long Key;

/* ================= TIMING ================= */

double now_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ================= CUDA KERNELS ================= */

__global__ void init_rank_kernel(
    const unsigned char *s,
    int *rank,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        rank[i] = s[i];
}

__global__ void build_keys_kernel(
    const int *__restrict__ rank,
    Key *__restrict__ keys,
    int *__restrict__ idx,
    int n,
    int k
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned int r1 = __ldg(&rank[i]);
        unsigned int r2 = (i + k < n) ? __ldg(&rank[i + k]) : 0u;
        keys[i] = (Key(r1) << 32) | Key(r2);
        idx[i]  = i;
    }
}

/* ================= MAIN ================= */

int main(int argc, char **argv) {

    if (argc != 2) {
        printf("Usage: %s <N>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    printf("Using N = %d\n", N);

    srand(1);

    double t0, t1, t2, t3;

    /* ---------- Generate input ---------- */
    char *s1 = (char *)malloc(N + 1);
    char *s2 = (char *)malloc(N + 1);

    for (int i = 0; i < N; i++) {
        s1[i] = "abcdefghijklmnopqrstuvwxyz"[rand() % 26];
        s2[i] = "abcdefghijklmnopqrstuvwxyz"[rand() % 26];
    }
    s1[N] = s2[N] = '\0';

    int n = 2 * N + 2;

    unsigned char *T = (unsigned char *)malloc(n);

    memcpy(T, s1, N);
    T[N] = 1;                    // separator
    memcpy(T + N + 1, s2, N);
    T[n - 1] = 0;                // terminal

    printf("Total length = %d\n", n);

    /* ---------- Device memory ---------- */

    unsigned char *d_T;
    int *d_rank;
    Key *d_keys;
    int *d_idx;

    cudaMalloc(&d_T,    n * sizeof(unsigned char));
    cudaMalloc(&d_rank, n * sizeof(int));
    cudaMalloc(&d_keys, n * sizeof(Key));
    cudaMalloc(&d_idx,  n * sizeof(int));

    cudaMemcpy(d_T, T,
               n * sizeof(unsigned char),
               cudaMemcpyHostToDevice);

    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* ---------- Initialize ranks ---------- */

    init_rank_kernel<<<grid, BLOCK_SIZE>>>(d_T, d_rank, n);
    cudaDeviceSynchronize();

    /* ---------- Host buffers ---------- */

    int *h_idx  = (int *)malloc(n * sizeof(int));
    int *rank   = (int *)malloc(n * sizeof(int));
    int *tmp    = (int *)malloc(n * sizeof(int));
    Key *h_keys = (Key *)malloc(n * sizeof(Key));

    /* ================= SUFFIX ARRAY ================= */

    printf("Building suffix array (radix GPU)...\n");
    t0 = now_seconds();

    for (int k = 1; k < n; k <<= 1) {

        build_keys_kernel<<<grid, BLOCK_SIZE>>>(
            d_rank, d_keys, d_idx, n, k
        );
        cudaDeviceSynchronize();

        thrust::device_ptr<Key> d_keys_ptr =
            thrust::device_pointer_cast(d_keys);
        thrust::device_ptr<int> d_idx_ptr =
            thrust::device_pointer_cast(d_idx);

        thrust::sort_by_key(d_keys_ptr,
                            d_keys_ptr + n,
                            d_idx_ptr);

        cudaMemcpy(h_idx, d_idx,
                   n * sizeof(int),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(h_keys, d_keys,
                   n * sizeof(Key),
                   cudaMemcpyDeviceToHost);

        tmp[h_idx[0]] = 0;
        for (int i = 1; i < n; i++) {
            tmp[h_idx[i]] =
                tmp[h_idx[i - 1]] +
                (h_keys[i] != h_keys[i - 1]);
        }

        memcpy(rank, tmp, n * sizeof(int));

        cudaMemcpy(d_rank, rank,
                   n * sizeof(int),
                   cudaMemcpyHostToDevice);

        if (rank[h_idx[n - 1]] == n - 1)
            break;
    }

    t1 = now_seconds();
    printf("Suffix array time : %.3f s\n", t1 - t0);

    /* ================= BUILD SA ================= */

    int *sa = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
        sa[rank[i]] = i;

    /* ================= LCP (KASAI) ================= */

    printf("Building LCP...\n");
    t2 = now_seconds();

    int *lcp = (int *)malloc(n * sizeof(int));
    int *rnk = (int *)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++)
        rnk[sa[i]] = i;

    int h = 0;
    lcp[0] = 0;

    for (int i = 0; i < n; i++) {
        int ri = rnk[i];
        if (ri > 0) {
            int j = sa[ri - 1];
            while (T[i + h] == T[j + h]) h++;
            lcp[ri] = h;
            if (h > 0) h--;
        }
    }

    t3 = now_seconds();
    printf("LCP time          : %.3f s\n", t3 - t2);

    /* ================= LCS ================= */

    printf("Computing LCS...\n");

    int best = 0, pos = -1;

    for (int i = 1; i < n; i++) {
        int a = sa[i];
        int b = sa[i - 1];
        if ((a < N) != (b < N)) {
            if (lcp[i] > best) {
                best = lcp[i];
                pos = sa[i];
            }
        }
    }

    printf("\n=== RESULTS ===\n");
    printf("LCS length : %d\n", best);

    if (best > 0) {
        printf("LCS sample: ");
        for (int i = 0; i < best; i++)
            putchar(T[pos + i]);
        putchar('\n');
    }

    printf("Total SA + LCP time : %.3f s\n",
           (t1 - t0) + (t3 - t2));

    /* ================= CLEANUP ================= */

    free(s1); free(s2); free(T);
    free(h_idx); free(h_keys);
    free(rank); free(tmp);
    free(sa); free(lcp); free(rnk);

    cudaFree(d_T);
    cudaFree(d_rank);
    cudaFree(d_keys);
    cudaFree(d_idx);

    return 0;
}