#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define N 100000000
#define ALPHABET 256

double now_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ================= RANDOM STRING ================= */

void random_string(char *s, int n) {
    const char alpha[] = "abcdefghijklmnopqrstuvwxyz";
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        s[i] = alpha[rand() % 26];
    s[n] = '\0';
}

/* ================= SUFFIX ARRAY ================= */

typedef struct {
    int first, second, idx;
} Tuple;

int cmp_tuple(const void *a, const void *b) {
    const Tuple *x = (const Tuple *)a;
    const Tuple *y = (const Tuple *)b;
    if (x->first != y->first) return x->first - y->first;
    return x->second - y->second;
}

void counting_sort(Tuple *arr, Tuple *out, int n, int max_rank, int by_second) {
    int *cnt = (int *)calloc(max_rank + 2, sizeof(int));

    // Count
    for (int i = 0; i < n; i++) {
        int key = by_second ? arr[i].second : arr[i].first;
        key++;                    // shift because -1 exists
        cnt[key]++;
    }

    // Prefix sum
    for (int i = 1; i <= max_rank + 1; i++)
        cnt[i] += cnt[i - 1];

    // Stable placement (reverse traversal!)
    for (int i = n - 1; i >= 0; i--) {
        int key = by_second ? arr[i].second : arr[i].first;
        key++;
        out[--cnt[key]] = arr[i];
    }

    free(cnt);
}
void build_sa(const unsigned char *s, int *sa, int n) {
    int *rank = (int *)malloc(n * sizeof(int));
    int *tmp  = (int *)malloc(n * sizeof(int));
    Tuple *arr  = (Tuple *)malloc(n * sizeof(Tuple));
    Tuple *out  = (Tuple *)malloc(n * sizeof(Tuple));

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        rank[i] = s[i];
    }

    for (int k = 1; k < n; k <<= 1) {

        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            arr[i].first  = rank[i];
            arr[i].second = (i + k < n) ? rank[i + k] : -1;
            arr[i].idx    = i;
        }

        int max_rank = n;

        // Radix: sort by second, then first
        counting_sort(arr, out, n, max_rank, 1);
        counting_sort(out, arr, n, max_rank, 0);

        tmp[arr[0].idx] = 0;

        for (int i = 1; i < n; i++) {
            tmp[arr[i].idx] = tmp[arr[i - 1].idx] +
                (arr[i].first != arr[i - 1].first ||
                 arr[i].second != arr[i - 1].second);
        }

        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            rank[i] = tmp[i];

        if (rank[arr[n - 1].idx] == n - 1)
            break;
    }

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        sa[rank[i]] = i;

    free(rank);
    free(tmp);
    free(arr);
    free(out);
}

/* ================= LCP ================= */

void build_lcp(const unsigned char *s, int *sa, int *lcp, int n) {
    int *rank = (int *)malloc(n * sizeof(int));

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        rank[sa[i]] = i;

    int h = 0;
    lcp[0] = 0;

    for (int i = 0; i < n; i++) {
        if (rank[i] > 0) {
            int j = sa[rank[i] - 1];
            while (s[i + h] == s[j + h]) h++;
            lcp[rank[i]] = h;
            if (h > 0) h--;
        }
    }
    free(rank);
}

/* ================= MAIN ================= */

int main() {
    srand(1);

    double t0 = now_seconds();

    char *s1 = malloc(N + 1);
    char *s2 = malloc(N + 1);

    random_string(s1, N);
    random_string(s2, N);

    int n = 2 * N + 2;
    unsigned char *T = malloc(n);

    memcpy(T, s1, N);
    T[N] = 1;
    memcpy(T + N + 1, s2, N);
    T[n - 1] = 0;

    int *sa = malloc(n * sizeof(int));
    int *lcp = malloc(n * sizeof(int));

    build_sa(T, sa, n);
    build_lcp(T, sa, lcp, n);

    int best = 0, pos = 0;

    #pragma omp parallel
    {
        int local_best = 0, local_pos = 0;

        #pragma omp for nowait
        for (int i = 1; i < n; i++) {
            int a = sa[i];
            int b = sa[i - 1];
            if ((a < N) != (b < N)) {
                if (lcp[i] > local_best) {
                    local_best = lcp[i];
                    local_pos = sa[i];
                }
            }
        }

        #pragma omp critical
        {
            if (local_best > best) {
                best = local_best;
                pos = local_pos;
            }
        }
    }

    double t1 = now_seconds();
    printf("LCS length = %d\n", best);
    printf("Total time = %.2f s\n", t1 - t0);

    free(s1); free(s2); free(T); free(sa); free(lcp);
    return 0;
}