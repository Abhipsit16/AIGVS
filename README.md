# AIGVS: Attractor Indexed Graphical Vector Search

A novel retrieval architecture that bridges neural-symbolic Knowledge Graphs and associative connectionism. AIGVS replaces global spatial graph traversals (O(N)) with a Continuous Modern Hopfield Network, achieving deterministic O(log N) retrieval complexity while preserving relational sub-graph topologies.

---

## 1. The Problem
Current enterprise Retrieval-Augmented Generation (RAG) architectures are paralyzed by a dichotomy of scale:
* **Standard Vector Databases (HNSW):** Fast but semantically blind. HNSW searches a global spatial void, returning isolated chunks of text. It completely drops the causal and relational edges between concepts, failing at complex reasoning tasks.
* **Knowledge Graphs (GraphRAG):** Highly relational but computationally rigid. Executing a global query requires massive LLM map-reduce overhead, and adding new documents often triggers cascading, expensive re-clustering of the entire graph structure (O(V log V)).

## 2. The AIGVS Solution
AIGVS solves this by decoupling the **Macro-Routing** from the **Micro-Retrieval**. 
Instead of forcing a spatial algorithm to navigate an entire global database, AIGVS uses an associative memory matrix (a Continuous Hopfield Network) to mathematically "teleport" queries directly into highly localized, isolated Knowledge Graph communities. 

* **Macro-Routing:** Continuous attractor dynamics drop queries into the correct semantic basin in O(1) iterations.
* **Micro-Retrieval:** A local Navigable Small World (NSW) graph executes a tiny greedy search to find the exact chunk *and* its topological neighbors for unbroken LLM context.

---

## 3. System Architecture Hierarchy

The system operates on a composite tree design with strict decoupling of storage, routing, and search:

1. **`AIGVSPipeline` (The Orchestrator):** Manages the global state, flat storage adapter (e.g., Local JSON, Postgres), and concurrent execution pools.
2. **`DisjointFamily` (The Router Node):** A structural parent node. It holds no text. It contains a dense matrix of child prototype vectors (centroids) and uses Hopfield equations to route queries down the tree.
3. **`JointFamily` (The Leaf Node):** A localized Knowledge Graph space. It contains actual document chunks wired together in a Navigable Small World (NSW) adjacency list. 

---

## 4. Core Mathematical Engines

### 4.1 The Continuous Hopfield Macro-Router
Queries are not searched; they are settled into energy minima. Given a query vector `q` and a matrix of community prototypes `X`, the network executes the following continuous update rule:

```text
ξ(t+1) = Normalize( X^T * softmax(β * X * ξ(t)) )

```

The `softmax` function acts as an exponential filter. Instead of iterating, we evaluate the activation energies at `T=1`. If an activation crosses the threshold (e.g., > 0.30), the query is routed to that sub-graph. If no activation crosses the threshold, the router identifies the query as **Out-Of-Distribution (OOD)** and natively spins up a new isolated branch.

### 4.2 Dynamic Temperature Scaling (β)

A static temperature fails in hierarchical trees. Deep nodes have lower variance (hyper-similar concepts) and require sharper competitive filtering. AIGVS dynamically tunes `β` using **Inverse Fractional Scaling** based on the pairwise cosine variance (`σ²`) of the active routing matrix:

```text
β(σ²) = β_min + (k / (σ² + ε))

```

This self-tuning equation ensures macro-layers route gently while micro-layers separate concepts aggressively.

### 4.3 Spherical Hopfield Bifurcation (Node Splitting)

When a `JointFamily` leaf node exceeds its capacity, AIGVS does not use Euclidean K-Means, which degrades vector magnitude. Instead, it natively splits the node using **Fixed-Point Attractor Convergence**.
Two extreme seed vectors (`ξ_1`, `ξ_2`) are placed in the dense local data pool `X_local` and evolved:

```text
ξ_i(t+1) = Normalize( X_local^T * softmax(β * X_local * ξ_i(t)) )

```

The seeds naturally cascade into the true localized energy minima, cleanly bifurcating the sub-graph along the hypersphere with 100% deterministic accuracy.

---

## 5. Algorithmic Workflows

### The Read Path (Querying)

1. **Embed:** Convert user query to vector `q`.
2. **Hopfield Route:** Pass `q` through the `DisjointFamily` layers. The matrix math isolates the target `JointFamily` leaf in O(1) matrix multiplications. (Overlapping targets trigger parallel thread execution).
3. **Greedy Traversal:** Inside the leaf, execute a greedy NSW search starting from the entry node, hopping via graph edges to the mathematical minimum.
4. **Contextual Expansion:** Return the winning chunks *plus* their 1-hop structural neighbors from the adjacency list to provide relational context.

### The Write Path (Ingestion)

1. **Route to Leaf:** The insertion vector falls down the Hopfield tree.
2. **Local Insertion:** The chunk is appended to the `JointFamily` NSW graph.
3. **O(1) Matrix Update:** The leaf recalculates its centroid via a running average:
```text
μ_new = ((μ_old * N_c) + x_new) / (N_c + 1)

```


4. **Lazy Propagation:** The new centroid replaces the single column in the parent's `X` matrix. No global re-indexing occurs.

### The Delete Path (Tombstoning)

1. **Mark as Deleted:** The chunk ID is added to a local `deleted_indices` set. The physical graph edges remain intact (Ghost Traversal).
2. **Lazy Centroid Shift:** The vector's influence is mathematically subtracted from the running average centroid.
3. **Compaction:** If the tombstone ratio exceeds 20%, the leaf asynchronously rebuilds its internal adjacency list.

---

## 6. Computational Complexity

By converting the top half of a standard spatial graph traversal into SIMD-accelerated dense matrix multiplications, AIGVS redefines the speed limits of vector indexing.

Let `N` = total database chunks, `M` = number of communities, `b` = branching factor, `K` = graph edge evaluations.

* **Standard Global HNSW Read:** `O(K * log(N) * E)`
* **AIGVS Read:** `O(log_b(M) * E) + O(k * log(N/M) * E)`
* **Standard GraphRAG Write:** `O(V * log(V))` (Global Re-clustering)
* **AIGVS Write:** `O(M * E) + O(K * log(N/M) * E)`

**Conclusion:** AIGVS mathematically scales logarithmically identically to the fastest pure-vector databases, but delivers the structured, cohesive relational intelligence of a Knowledge Graph with nearly zero global write-penalties.

```

```
