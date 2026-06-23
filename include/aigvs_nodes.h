#ifndef AIGVS_NODES_H
#define AIGVS_NODES_H

#include "aigvs.h"

// Initialize functions
void joint_family_init(BaseNode* node, const char* node_id, const char* parent_id, size_t embedding_dim);
void disjoint_family_init(BaseNode* node, const char* node_id, const char* parent_id, size_t embedding_dim);

// Free functions
void joint_family_destroy(BaseNode* node);
void disjoint_family_destroy(BaseNode* node);

// Common Node Interface
void node_insert(BaseNode* node, TextChunk* chunk, AIGVSPipeline* pipeline);
QueryResult* node_query(BaseNode* node, const float* query_vector, int top_k, int* num_results);

// Internal declarations for node_router
void hopfield_router_init(HopfieldRouter* router, size_t embedding_dim);
void hopfield_router_destroy(HopfieldRouter* router);
void hopfield_router_add_or_update(HopfieldRouter* router, const char* node_id, const float* centroid, int current_count);
char** hopfield_router_route(HopfieldRouter* router, const float* query_vector, int beam_width, int* out_count);

#endif // AIGVS_NODES_H
