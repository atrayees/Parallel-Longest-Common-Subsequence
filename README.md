# GPU Longest Common Substring using Suffix Arrays (CUDA)

This project implements a **GPU-accelerated suffix array algorithm** to compute the **Longest Common Substring (LCS)** between two strings.  
The suffix array construction is partially parallelized on the GPU using **CUDA**, while the **LCP computation and LCS detection run on the CPU**.

The implementation uses the **prefix-doubling suffix array algorithm** with **GPU key generation** and **Thrust GPU sorting**.

---

# Overview

Given two strings:

```
S1
S2
```

we compute the **longest substring that appears in both strings**.

Example:

```
S1 = banana
S2 = ananas

LCS = anana
```

This implementation:

1. Concatenates the strings
2. Builds a **suffix array using the GPU**
3. Computes the **LCP array (Kasai algorithm)**
4. Finds the **longest common substring**

---

# Algorithm

The pipeline consists of the following steps.

## 1. String Concatenation

We build a combined string:

```
T = S1 + separator + S2 + terminal
```

Example:

```
banana#ananas$
```

Special characters ensure suffixes do not cross boundaries.

---

## 2. GPU Suffix Array Construction

The suffix array is built using the **prefix-doubling algorithm**.

At iteration `k`:

Each suffix is represented by a key:

```
(rank[i], rank[i + k])
```

This pair is packed into a **64-bit key**:

```
key = (rank[i] << 32) | rank[i+k]
```

GPU kernels generate these keys in parallel.

### GPU Kernels

**1️⃣ Initialize ranks**

Each suffix rank is initialized from the character values.

```
rank[i] = T[i]
```

**2️⃣ Build keys**

For every suffix:

```
key[i] = (rank[i], rank[i+k])
```

---

## 3. GPU Sorting

The suffixes are sorted using:

```
thrust::sort_by_key
```

This sorts suffix indices according to their `(rank[i], rank[i+k])` pairs.

---

## 4. Rank Update (CPU)

After sorting:

```
rank[new_suffix] = new rank
```

Ranks are recomputed until all suffixes become uniquely ranked.

---

## 5. LCP Construction (Kasai Algorithm)

Once the suffix array is built, we compute the **LCP (Longest Common Prefix)** array.

```
LCP[i] = length of longest common prefix
         between SA[i] and SA[i-1]
```

This step runs on the **CPU**.

---

## 6. Longest Common Substring

To find the LCS:

We scan adjacent suffixes in the suffix array.

If two suffixes come from **different input strings**, we consider their LCP.

The maximum such LCP is the **longest common substring**.

---

# File Structure

Example repository layout:

```
.
├── lcs_gpu.cu
├── README.md
```

---

# Requirements

You need:

- NVIDIA GPU
- CUDA Toolkit
- GCC / g++

Tested with:

```
CUDA 11+
```

---

# Compilation

Compile using `nvcc`.

```
nvcc -O3 lcs_gpu.cu -o lcs_gpu
```

---

# Usage

Run the program by specifying the length of the random strings:

```
./lcs_gpu N
```

Example:

```
./lcs_gpu 100000
```

Output example:

```
Using N = 100000
Total length = 200002

Building suffix array (radix GPU)...
Suffix array time : 0.423 s

Building LCP...
LCP time          : 0.017 s

Computing LCS...

=== RESULTS ===
LCS length : 9
LCS sample: abcdefghi

Total SA + LCP time : 0.440 s
```

---

# Performance Notes

The GPU is used for:

- Key generation
- Sorting suffix pairs

The CPU handles:

- Rank updates
- LCP computation
- Final LCS detection

This **hybrid GPU–CPU approach** reduces suffix array construction time significantly for large inputs.

---

# Complexity

Suffix Array Construction:

```
O(n log n)
```

LCP Construction:

```
O(n)
```

Overall:

```
O(n log n)
```

but with significant GPU acceleration.

---

# Implementation Details

Key optimizations used:

- CUDA kernels for parallel key construction
- 64-bit packed keys
- Thrust GPU sorting
- Shared prefix-doubling algorithm
- Kasai algorithm for LCP

---

# Future Improvements

Possible extensions:

- Fully GPU-based suffix array construction
- GPU LCP computation
- Handling real input strings instead of random generation
- Memory optimizations for very large datasets
- Multi-GPU scaling

---

# License

MIT License.
