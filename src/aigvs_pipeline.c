#include "../include/aigvs.h"
#include "../include/aigvs_nodes.h"
#include <stdlib.h>
#include <string.h>

AIGVSPipeline* aigvs_pipeline_create(size_t embedding_dim, const char* storage_path) {
    AIGVSPipeline* pipeline = malloc(sizeof(AIGVSPipeline));
    pipeline->embedding_dim = embedding_dim;
    pipeline->storage_path = storage_path ? strdup(storage_path) : NULL;

    pipeline->root = malloc(sizeof(BaseNode));
    disjoint_family_init(pipeline->root, "root_node", NULL, embedding_dim);

    pipeline->storage = storage_create_in_memory();
    pipeline->node_registry = NULL; // Stubs for hash maps
    pipeline->chunk_location_map = NULL;

    return pipeline;
}

void aigvs_pipeline_destroy(AIGVSPipeline* pipeline) {
    if (pipeline->storage_path) free(pipeline->storage_path);
    disjoint_family_destroy(pipeline->root);
    free(pipeline->root);
    storage_destroy_in_memory(pipeline->storage);
    free(pipeline);
}

void aigvs_pipeline_ingest(AIGVSPipeline* pipeline, TextChunk* chunks, size_t num_chunks) {
    for (size_t i = 0; i < num_chunks; ++i) {
        node_insert(pipeline->root, &chunks[i], pipeline);
    }
}

QueryResult* aigvs_pipeline_query(AIGVSPipeline* pipeline, const float* query_embedding, int top_k, int* num_results) {
    // Normalize query vector
    float* q_vec = malloc(pipeline->embedding_dim * sizeof(float));
    memcpy(q_vec, query_embedding, pipeline->embedding_dim * sizeof(float));
    normalize_vector(q_vec, pipeline->embedding_dim);

    QueryResult* results = node_query(pipeline->root, q_vec, top_k, num_results);
    free(q_vec);

    return results;
}
