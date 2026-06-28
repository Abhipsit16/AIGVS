# AIGVS Architecture & Optimization Guide

This document catalogs the specific architectural tricks, algorithmic bypasses, and mathematical optimizations we implemented to achieve a 10x ingestion speedup over FAISS while matching its 98% recall and sub-millisecond latency at the 100k scale.

## 1. The "Flat Trunk" Macro-Attractor Bypass
*The Secret to 10x Faster Ingestion (8 seconds vs 84 seconds)*

**The Problem:** Standard vector databases like FAISS build a single, monolithic HNSW graph. Every new vector must search the entire graph to find neighbors, creating an $O(N \log N)$ bottleneck that scales terribly as the dataset grows.
**The AIGVS Trick:** We implemented a "Divide and Conquer" approach:
1. Instead of building one graph, we run an ultra-fast, low-iteration **Spherical K-Means** algorithm to find a few "Macro Attractors" (e.g., 4 buckets for 100k vectors).
2. The dataset is instantly partitioned into these 4 independent buckets ($O(1)$ time).
3. The mathematical complexity is reduced from $1 \times (100,000 \log 100,000)$ to $4 \times (25,000 \log 25,000)$, bypassing the global search bottleneck entirely.

## 2. Polysemic Boundary Assignments (Soft Routing)
*The Secret to 98.0% Recall*

**The Problem:** When you cut a dataset into buckets (like K-Means), vectors sitting right on the boundary line can easily get routed to the wrong bucket during a query, destroying recall.
**The AIGVS Trick:** We use **Polysemy** (duplication). During ingestion, if a vector's similarity to a second attractor is within a tight threshold (`threshold = max_sims - 0.08`), we intentionally duplicate the vector into *both* buckets.
*   **Trade-off:** This inflates the memory footprint by ~46% (e.g., storing 146k vectors instead of 100k).
*   **Result:** It acts as a mathematical guarantee. The router can blindly pick the single highest-probability bucket and we are practically guaranteed that the golden document is inside it.

## 3. Scalar Quantization (SQ8) & Cache Locality
*The Secret to High-Speed Distance Math*

**The Problem:** 384-dimensional `f32` vectors require 1,536 bytes each. Fetching these from RAM into the CPU is slow and causes constant cache misses during graph traversal.
**The AIGVS Trick:** We convert the vectors to 8-bit integers (`i8`) inside the Rust backend before caching them.
1.  **Compression:** The vector shrinks from 1536 bytes to 384 bytes (75% compression).
2.  **L1/L2 Cache Magic:** Because the vectors are 4x smaller, the CPU can fit 4x more of them into its ultra-fast local cache, nearly eliminating RAM lookups.
3.  **Contiguous Memory:** Each "bucket" has its own dedicated, contiguous vector cache in Rust, meaning the CPU reads memory in a perfectly straight, predictable line.

## 4. SIMD-Accelerated Dot Products
*The Secret to Hardware Acceleration*

**The Problem:** Calculating the dot-product between two 384-dimensional vectors requires 384 multiplications and 384 additions. Doing this one by one is slow.
**The AIGVS Trick:** Because we converted the vectors to integers (`i8`), we can leverage **SIMD** (Single Instruction, Multiple Data). By compiling the Rust code with the `target-cpu=native` flag, we unlock advanced CPU instructions (AVX2/AVX-512). The CPU now processes up to 32 dimensions of the vector in a *single clock cycle*. 

## 5. Lock-Free Parallelization (Rayon)
*The Secret to Utilizing All CPU Cores*

**The Problem:** Python's Global Interpreter Lock (GIL) prevents true multithreading, making it impossible to build multiple graphs simultaneously in Python.
**The AIGVS Trick:** Because our K-Means algorithm partitions the data into 4 completely independent buckets, we pass all 4 buckets to Rust via FFI. Rust then uses the `Rayon` library to spawn 4 independent threads, mapping one bucket to each CPU core. Because the buckets don't share any memory, no thread-locks are required, and the CPU hits 100% utilization.

## 6. Hierarchical Softmax Routing
*The Secret to Sub-Millisecond Latency (0.71ms)*

**The Problem:** If a query evaluates too many paths, latency spikes.
**The AIGVS Trick:** The top-level Router uses Softmax to convert distances into probabilities. We implemented two dials to prune the search space:
*   **`beta` (Temperature):** Sharpens the contrast between good and bad branches.
*   **`target_confidence` (Cumulative Threshold):** The router sorts the branches by probability and picks them one by one until the total probability hits e.g. `0.95`. If the first branch has a 96% chance, it picks only 1 branch and ignores the rest of the database, granting true $O(1)$ search latency from the router's perspective.
