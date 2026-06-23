#ifndef AIGVS_H
#define AIGVS_H

#include <stddef.h>
#include <stdbool.h>

#define MAX_CHUNK_ID_LEN 128
#define MAX_NODE_ID_LEN 128

// TextChunk structure
typedef struct {
    char chunk_id[MAX_CHUNK_ID_LEN];
    char* text;
    float* embedding;
    size_t embedding_dim;
    // We can add metadata here, or handle it via a cJSON string or object later
    char* metadata_json;
} TextChunk;

// Result from a query
typedef struct {
    char chunk_id[MAX_CHUNK_ID_LEN];
    float score; // Cosine similarity score
} QueryResult;

// Forward declarations
typedef struct BaseNode BaseNode;
typedef struct AIGVSPipeline AIGVSPipeline;

// Router structure
typedef struct {
    size_t embedding_dim;
    float beta_min;
    float k_scale;
    float activation_threshold;

    char** id_lookup;
    float** X_matrix;
    int* node_capacities;
    size_t num_attractors;
    size_t capacity;
} HopfieldRouter;

// BaseNode types
typedef enum {
    NODE_DISJOINT,
    NODE_JOINT
} NodeType;

// Local graph node (adjacency list)
typedef struct IntListNode {
    int value;
    struct IntListNode* next;
} IntListNode;

struct BaseNode {
    char node_id[MAX_NODE_ID_LEN];
    char parent_id[MAX_NODE_ID_LEN];
    NodeType type;
    float* centroid_vector;

    // For JointFamily (Leaf Node)
    char** chunk_ids;
    float** vector_cache;
    IntListNode** local_graph; // Adjacency list
    size_t max_connections;
    int entry_point_idx;
    int* deleted_indices; // Simple array or hash set. We'll use a boolean array or just a list of deleted ints.
    size_t num_deleted;
    size_t num_chunks;
    size_t chunks_capacity;

    // For DisjointFamily (Router Node)
    BaseNode** children; // Array of pointers to children
    char** children_ids;
    size_t num_children;
    size_t children_capacity;
    HopfieldRouter router;
    size_t max_leaf_capacity;
};

struct AIGVSPipeline {
    size_t embedding_dim;
    char* storage_path;
    BaseNode* root;
    // In C, node_registry and chunk_location_map could be hash maps
    // We will define opaque structs for them
    void* node_registry;
    void* chunk_location_map;
    void* storage; // Opaque pointer to InMemoryStorage or LocalJSONStorage
};

// Utility functions
float cosine_similarity(const float* vec_a, const float* vec_b, size_t dim);
void normalize_vector(float* vec, size_t dim);

// Pipeline API
AIGVSPipeline* aigvs_pipeline_create(size_t embedding_dim, const char* storage_path);
void aigvs_pipeline_destroy(AIGVSPipeline* pipeline);
void aigvs_pipeline_ingest(AIGVSPipeline* pipeline, TextChunk* chunks, size_t num_chunks);
QueryResult* aigvs_pipeline_query(AIGVSPipeline* pipeline, const float* query_embedding, int top_k, int* num_results);

// Storage API
void* storage_create_in_memory();
void storage_destroy_in_memory(void* storage);
void storage_write_chunk(void* storage, const TextChunk* chunk);
TextChunk* storage_read_chunk(void* storage, const char* chunk_id);

#endif // AIGVS_H
