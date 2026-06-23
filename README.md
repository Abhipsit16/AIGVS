# AIGVS: Attractor Indexed Graphical Vector Search

A novel retrieval architecture that bridges neural-symbolic Knowledge Graphs and associative connectionism.

**Now entirely implemented in pure C for extreme bare-metal performance.**

---

## 1. The C Port Advantage

By stripping away Python overhead and strictly controlling memory layouts, the C port of AIGVS drastically reduces latency and memory bloat. The system relies entirely on static structs and arrays, enabling the processor to heavily cache consecutive memory chunks (like Continuous Hopfield `X_matrix` networks) for blistering SIMD/Vectorization operations.

### Key Changes
- **Contiguous Routing Maps:** The DisjointRouter now houses routing thresholds strictly as arrays.
- **Pointer Hierarchy:** Nodes, routers, and storage mechanisms have been transformed into standard memory-managed pointer configurations.
- **Dependency Elimination:** Zero external dependencies (cJSON is included natively).

---

## 2. API Usage

To start using AIGVS in your native projects:

```c
#include "aigvs.h"

int main() {
    // 1. Boot up the routing pipeline (Setting Vector Dimension to 384)
    AIGVSPipeline* pipeline = aigvs_pipeline_create(384, "./aigvs_data");

    // 2. Prepare Vector Payload
    TextChunk chunk;
    snprintf(chunk.chunk_id, MAX_CHUNK_ID_LEN, "document_xyz");
    chunk.text = "Continuous Attractor Networks rule.";
    chunk.embedding = (float*)malloc(384 * sizeof(float));
    chunk.embedding_dim = 384;

    // 3. Ingest (Will recursively Hopfield Route down the tree to the NSW leaf)
    aigvs_pipeline_ingest(pipeline, &chunk, 1);

    // 4. Clean Memory
    aigvs_pipeline_destroy(pipeline);
    return 0;
}
```

---

## 3. Building

The project ships with a strict `Makefile`.
```bash
make
./aigvs_run
```

To clean the build objects:
```bash
make clean
```
