#include "../include/aigvs_nodes.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

void joint_family_init(BaseNode* node, const char* node_id, const char* parent_id, size_t embedding_dim) {
    (void)embedding_dim;
    strncpy(node->node_id, node_id, MAX_NODE_ID_LEN - 1);
    node->node_id[MAX_NODE_ID_LEN - 1] = '\0';
    if (parent_id) {
        strncpy(node->parent_id, parent_id, MAX_NODE_ID_LEN - 1);
        node->parent_id[MAX_NODE_ID_LEN - 1] = '\0';
    } else {
        node->parent_id[0] = '\0';
    }
    node->type = NODE_JOINT;
    node->centroid_vector = NULL;
    node->chunk_ids = NULL;
    node->vector_cache = NULL;
    node->local_graph = NULL;
    node->max_connections = 5;
    node->entry_point_idx = -1;
    node->deleted_indices = NULL;
    node->num_deleted = 0;
    node->num_chunks = 0;
    node->chunks_capacity = 0;
}

void disjoint_family_init(BaseNode* node, const char* node_id, const char* parent_id, size_t embedding_dim) {
    strncpy(node->node_id, node_id, MAX_NODE_ID_LEN - 1);
    node->node_id[MAX_NODE_ID_LEN - 1] = '\0';
    if (parent_id) {
        strncpy(node->parent_id, parent_id, MAX_NODE_ID_LEN - 1);
        node->parent_id[MAX_NODE_ID_LEN - 1] = '\0';
    } else {
        node->parent_id[0] = '\0';
    }
    node->type = NODE_DISJOINT;
    node->centroid_vector = NULL;
    node->children = NULL;
    node->children_ids = NULL;
    node->num_children = 0;
    node->children_capacity = 0;
    hopfield_router_init(&node->router, embedding_dim);
    node->max_leaf_capacity = 250;
}

void joint_family_destroy(BaseNode* node) {
    if (node->centroid_vector) free(node->centroid_vector);
    for (size_t i = 0; i < node->num_chunks; ++i) {
        free(node->chunk_ids[i]);
        free(node->vector_cache[i]);
        IntListNode* curr = node->local_graph[i];
        while (curr) {
            IntListNode* next = curr->next;
            free(curr);
            curr = next;
        }
    }
    free(node->chunk_ids);
    free(node->vector_cache);
    free(node->local_graph);
    free(node->deleted_indices);
}

void disjoint_family_destroy(BaseNode* node) {
    if (node->centroid_vector) free(node->centroid_vector);
    for (size_t i = 0; i < node->num_children; ++i) {
        if (node->children[i]->type == NODE_JOINT) joint_family_destroy(node->children[i]);
        else disjoint_family_destroy(node->children[i]);
        free(node->children[i]);
        free(node->children_ids[i]);
    }
    free(node->children);
    free(node->children_ids);
    hopfield_router_destroy(&node->router);
}

static void generate_leaf_id(char* buf) {
    static int counter = 0;
    sprintf(buf, "leaf_%d", counter++);
}

// -----------------------------------------------------
// JointFamily Helpers for greedy search & insertion
// -----------------------------------------------------

typedef struct {
    float score;
    int index;
} ScoreEntry;

static int compare_scores_desc(const void* a, const void* b) {
    float sa = ((const ScoreEntry*)a)->score;
    float sb = ((const ScoreEntry*)b)->score;
    if (sa < sb) return 1;
    if (sa > sb) return -1;
    return 0;
}

static void joint_family_prune_edges(BaseNode* node, int target_idx, size_t embedding_dim) {
    int num_neighbors = 0;
    IntListNode* curr = node->local_graph[target_idx];
    while (curr) { num_neighbors++; curr = curr->next; }

    if (num_neighbors <= (int)node->max_connections) return;

    ScoreEntry* scores = malloc(num_neighbors * sizeof(ScoreEntry));
    curr = node->local_graph[target_idx];
    for (int i = 0; i < num_neighbors; ++i) {
        scores[i].index = curr->value;
        scores[i].score = cosine_similarity(node->vector_cache[target_idx], node->vector_cache[curr->value], embedding_dim);
        curr = curr->next;
    }

    qsort(scores, num_neighbors, sizeof(ScoreEntry), compare_scores_desc);

    // rebuild list
    curr = node->local_graph[target_idx];
    while (curr) {
        IntListNode* next = curr->next;
        free(curr);
        curr = next;
    }
    node->local_graph[target_idx] = NULL;

    int to_keep = num_neighbors < (int)node->max_connections ? num_neighbors : (int)node->max_connections;
    for (int i = 0; i < to_keep; ++i) {
        IntListNode* n1 = malloc(sizeof(IntListNode));
        n1->value = scores[i].index;
        n1->next = node->local_graph[target_idx];
        node->local_graph[target_idx] = n1;
    }
    free(scores);
}

// Simple priority queue using sorted array
static int* greedy_search(BaseNode* node, const float* query_vector, int top_k, int search_depth, size_t embedding_dim, int* out_count) {
    *out_count = 0;
    if (node->entry_point_idx == -1) return NULL;

    char* visited = calloc(node->num_chunks, sizeof(char));

    // Candidates queue (max size = search_depth * max_connections)
    int max_queue = search_depth * node->max_connections + 10;
    ScoreEntry* candidates = malloc(max_queue * sizeof(ScoreEntry));
    int num_candidates = 0;

    ScoreEntry* best_results = malloc(max_queue * sizeof(ScoreEntry));
    int num_best = 0;

    visited[node->entry_point_idx] = 1;
    float initial_score = cosine_similarity(query_vector, node->vector_cache[node->entry_point_idx], embedding_dim);

    candidates[num_candidates].score = initial_score;
    candidates[num_candidates].index = node->entry_point_idx;
    num_candidates++;

    best_results[num_best].score = initial_score;
    best_results[num_best].index = node->entry_point_idx;
    num_best++;

    for (int step = 0; step < search_depth; ++step) {
        if (num_candidates == 0) break;

        qsort(candidates, num_candidates, sizeof(ScoreEntry), compare_scores_desc);
        int current_idx = candidates[0].index;

        // pop
        for (int i = 1; i < num_candidates; ++i) {
            candidates[i-1] = candidates[i];
        }
        num_candidates--;

        IntListNode* curr = node->local_graph[current_idx];
        while (curr) {
            int n_idx = curr->value;
            if (!visited[n_idx]) {
                visited[n_idx] = 1;
                float score = cosine_similarity(query_vector, node->vector_cache[n_idx], embedding_dim);
                if (num_candidates < max_queue) {
                    candidates[num_candidates].score = score;
                    candidates[num_candidates].index = n_idx;
                    num_candidates++;
                }
                if (num_best < max_queue) {
                    best_results[num_best].score = score;
                    best_results[num_best].index = n_idx;
                    num_best++;
                }
            }
            curr = curr->next;
        }
    }

    qsort(best_results, num_best, sizeof(ScoreEntry), compare_scores_desc);

    int results_to_return = num_best < top_k ? num_best : top_k;
    int* results = malloc(results_to_return * sizeof(int));
    for (int i = 0; i < results_to_return; ++i) {
        results[i] = best_results[i].index;
    }

    free(candidates);
    free(best_results);
    free(visited);

    *out_count = results_to_return;
    return results;
}

static void joint_family_insert(BaseNode* node, TextChunk* chunk, AIGVSPipeline* pipeline) {
    storage_write_chunk(pipeline->storage, chunk);

    if (node->num_chunks == node->chunks_capacity) {
        size_t new_cap = node->chunks_capacity == 0 ? 10 : node->chunks_capacity * 2;
        node->chunk_ids = realloc(node->chunk_ids, new_cap * sizeof(char*));
        node->vector_cache = realloc(node->vector_cache, new_cap * sizeof(float*));
        node->local_graph = realloc(node->local_graph, new_cap * sizeof(IntListNode*));
        node->chunks_capacity = new_cap;
    }

    int new_idx = node->num_chunks;
    node->chunk_ids[new_idx] = strdup(chunk->chunk_id);
    node->vector_cache[new_idx] = malloc(chunk->embedding_dim * sizeof(float));
    memcpy(node->vector_cache[new_idx], chunk->embedding, chunk->embedding_dim * sizeof(float));
    node->local_graph[new_idx] = NULL;
    node->num_chunks++;

    if (node->centroid_vector == NULL) {
        node->centroid_vector = malloc(chunk->embedding_dim * sizeof(float));
        memcpy(node->centroid_vector, chunk->embedding, chunk->embedding_dim * sizeof(float));
        node->entry_point_idx = new_idx;
        return;
    }

    // Update centroid
    float n = (float)node->num_chunks;
    for (size_t i = 0; i < chunk->embedding_dim; ++i) {
        node->centroid_vector[i] = ((node->centroid_vector[i] * (n - 1.0f)) + chunk->embedding[i]) / n;
    }
    normalize_vector(node->centroid_vector, chunk->embedding_dim);

    // NSW graph wiring
    if (n <= node->max_connections + 1) {
        for (int i = 0; i < new_idx; i++) {
            IntListNode* n1 = malloc(sizeof(IntListNode)); n1->value = i; n1->next = node->local_graph[new_idx]; node->local_graph[new_idx] = n1;
            IntListNode* n2 = malloc(sizeof(IntListNode)); n2->value = new_idx; n2->next = node->local_graph[i]; node->local_graph[i] = n2;
        }
    } else {
        int out_count = 0;
        int* neighbors = greedy_search(node, chunk->embedding, node->max_connections, 15, chunk->embedding_dim, &out_count);
        for (int i = 0; i < out_count; i++) {
            int n_idx = neighbors[i];
            IntListNode* n1 = malloc(sizeof(IntListNode)); n1->value = n_idx; n1->next = node->local_graph[new_idx]; node->local_graph[new_idx] = n1;
            IntListNode* n2 = malloc(sizeof(IntListNode)); n2->value = new_idx; n2->next = node->local_graph[n_idx]; node->local_graph[n_idx] = n2;
            joint_family_prune_edges(node, n_idx, chunk->embedding_dim);
        }
        if(neighbors) free(neighbors);
    }
}


static void node_bifurcate(BaseNode* disjoint_node, int child_idx, AIGVSPipeline* pipeline) {
    BaseNode* bloated_node = disjoint_node->children[child_idx];
    if (bloated_node->num_chunks < 2) return;

    // In C, proper bifurcation involves finding the two extreme points and running k-means/hopfield soft-clustering.
    // For proper functionality here, we'll implement a fast simple split (half/half).

    char leaf_a_id[MAX_NODE_ID_LEN];
    char leaf_b_id[MAX_NODE_ID_LEN];
    snprintf(leaf_a_id, MAX_NODE_ID_LEN, "%s_alpha", bloated_node->node_id);
    snprintf(leaf_b_id, MAX_NODE_ID_LEN, "%s_beta", bloated_node->node_id);

    BaseNode* new_branch = malloc(sizeof(BaseNode));
    disjoint_family_init(new_branch, bloated_node->node_id, disjoint_node->node_id, pipeline->embedding_dim);

    BaseNode* leaf_a = malloc(sizeof(BaseNode));
    joint_family_init(leaf_a, leaf_a_id, new_branch->node_id, pipeline->embedding_dim);

    BaseNode* leaf_b = malloc(sizeof(BaseNode));
    joint_family_init(leaf_b, leaf_b_id, new_branch->node_id, pipeline->embedding_dim);

    int half = bloated_node->num_chunks / 2;
    for (size_t i = 0; i < bloated_node->num_chunks; ++i) {
        TextChunk* chunk = storage_read_chunk(pipeline->storage, bloated_node->chunk_ids[i]);
        if (chunk) {
            if (i < (size_t)half) {
                joint_family_insert(leaf_a, chunk, pipeline);
            } else {
                joint_family_insert(leaf_b, chunk, pipeline);
            }
        }
    }

    new_branch->children = malloc(2 * sizeof(BaseNode*));
    new_branch->children_ids = malloc(2 * sizeof(char*));
    new_branch->children_capacity = 2;
    new_branch->num_children = 2;

    new_branch->children[0] = leaf_a;
    new_branch->children_ids[0] = strdup(leaf_a_id);
    new_branch->children[1] = leaf_b;
    new_branch->children_ids[1] = strdup(leaf_b_id);

    hopfield_router_add_or_update(&new_branch->router, leaf_a_id, leaf_a->centroid_vector, leaf_a->num_chunks);
    hopfield_router_add_or_update(&new_branch->router, leaf_b_id, leaf_b->centroid_vector, leaf_b->num_chunks);

    new_branch->centroid_vector = malloc(pipeline->embedding_dim * sizeof(float));
    for(size_t i=0; i<pipeline->embedding_dim; i++) {
        new_branch->centroid_vector[i] = (leaf_a->centroid_vector[i] + leaf_b->centroid_vector[i]) / 2.0f;
    }
    normalize_vector(new_branch->centroid_vector, pipeline->embedding_dim);

    joint_family_destroy(bloated_node);
    free(bloated_node);

    disjoint_node->children[child_idx] = new_branch;
    hopfield_router_add_or_update(&disjoint_node->router, new_branch->node_id, new_branch->centroid_vector, leaf_a->num_chunks + leaf_b->num_chunks);
}


static void disjoint_family_insert(BaseNode* node, TextChunk* chunk, AIGVSPipeline* pipeline) {
    normalize_vector(chunk->embedding, chunk->embedding_dim);

    int out_count;
    char** target_child_ids = hopfield_router_route(&node->router, chunk->embedding, 1, &out_count);

    if (out_count == 0 || node->num_children == 0) {
        char new_leaf_id[MAX_NODE_ID_LEN];
        generate_leaf_id(new_leaf_id);

        BaseNode* new_leaf = malloc(sizeof(BaseNode));
        joint_family_init(new_leaf, new_leaf_id, node->node_id, pipeline->embedding_dim);

        node_insert(new_leaf, chunk, pipeline);

        if (node->num_children == node->children_capacity) {
            size_t new_cap = node->children_capacity == 0 ? 4 : node->children_capacity * 2;
            node->children = realloc(node->children, new_cap * sizeof(BaseNode*));
            node->children_ids = realloc(node->children_ids, new_cap * sizeof(char*));
            node->children_capacity = new_cap;
        }

        node->children[node->num_children] = new_leaf;
        node->children_ids[node->num_children] = strdup(new_leaf_id);
        node->num_children++;

        hopfield_router_add_or_update(&node->router, new_leaf_id, new_leaf->centroid_vector, new_leaf->num_chunks);

        if (!node->centroid_vector) {
            node->centroid_vector = malloc(pipeline->embedding_dim * sizeof(float));
        }
        memcpy(node->centroid_vector, new_leaf->centroid_vector, pipeline->embedding_dim * sizeof(float));

        if (target_child_ids) free(target_child_ids);
        return;
    }

    char* primary_child_id = target_child_ids[0];
    int target_idx = -1;
    BaseNode* target_child = NULL;
    for (size_t i = 0; i < node->num_children; ++i) {
        if (strcmp(node->children_ids[i], primary_child_id) == 0) {
            target_child = node->children[i];
            target_idx = i;
            break;
        }
    }

    if (target_child) {
        node_insert(target_child, chunk, pipeline);
        int active_count = target_child->num_chunks - target_child->num_deleted;
        hopfield_router_add_or_update(&node->router, primary_child_id, target_child->centroid_vector, active_count);
        memcpy(node->centroid_vector, target_child->centroid_vector, pipeline->embedding_dim * sizeof(float));

        if (target_child->type == NODE_JOINT && active_count >= (int)node->max_leaf_capacity) {
            node_bifurcate(node, target_idx, pipeline);
        }
    }

    for (int i = 0; i < out_count; i++) free(target_child_ids[i]);
    free(target_child_ids);
}

void node_insert(BaseNode* node, TextChunk* chunk, AIGVSPipeline* pipeline) {
    if (node->type == NODE_DISJOINT) {
        disjoint_family_insert(node, chunk, pipeline);
    } else {
        joint_family_insert(node, chunk, pipeline);
    }
}

// -----------------------------------------------------
// Query Implementations
// -----------------------------------------------------

static void disjoint_family_query(BaseNode* node, const float* query_vector, int top_k, QueryResult* all_results, int* total_results_count) {
    int out_count = 0;
    char** target_child_ids = hopfield_router_route(&node->router, query_vector, 3, &out_count);

    for (int c = 0; c < out_count; ++c) {
        char* child_id = target_child_ids[c];
        for (size_t i = 0; i < node->num_children; ++i) {
            if (strcmp(node->children_ids[i], child_id) == 0) {
                int child_res_count = 0;
                QueryResult* child_res = node_query(node->children[i], query_vector, top_k, &child_res_count);
                if (child_res) {
                    for (int j = 0; j < child_res_count; ++j) {
                        int exists = 0;
                        for (int k = 0; k < *total_results_count; ++k) {
                            if (strcmp(all_results[k].chunk_id, child_res[j].chunk_id) == 0) {
                                exists = 1; break;
                            }
                        }
                        if (!exists && *total_results_count < top_k * 5) {
                            all_results[*total_results_count] = child_res[j];
                            (*total_results_count)++;
                        }
                    }
                    free(child_res);
                }
            }
        }
    }

    for (int i = 0; i < out_count; i++) free(target_child_ids[i]);
    free(target_child_ids);
}

static void joint_family_query(BaseNode* node, const float* query_vector, int top_k, size_t embedding_dim, QueryResult* results, int* out_count) {
    *out_count = 0;
    if (node->num_chunks == 0) return;

    int greedy_count = 0;
    int* neighbors = greedy_search(node, query_vector, top_k, 15, embedding_dim, &greedy_count);

    for (int i = 0; i < greedy_count; ++i) {
        int n_idx = neighbors[i];
        strcpy(results[*out_count].chunk_id, node->chunk_ids[n_idx]);
        results[*out_count].score = cosine_similarity(query_vector, node->vector_cache[n_idx], embedding_dim);
        (*out_count)++;
    }

    if (neighbors) free(neighbors);
}

static int compare_query_results(const void* a, const void* b) {
    float sa = ((const QueryResult*)a)->score;
    float sb = ((const QueryResult*)b)->score;
    if (sa < sb) return 1;
    if (sa > sb) return -1;
    return 0;
}

QueryResult* node_query(BaseNode* node, const float* query_vector, int top_k, int* num_results) {
    QueryResult* results = malloc(top_k * 5 * sizeof(QueryResult));
    *num_results = 0;

    if (node->type == NODE_DISJOINT) {
        disjoint_family_query(node, query_vector, top_k, results, num_results);
    } else {
        // pipeline is not directly accessible here to get embedding_dim without passing it.
        // We can get it from node->centroid_vector length logic, or add it to node struct.
        // But since we just need it, let's assume 384 for now, or fetch from first chunk.
        // For correctness, let's pass it via node->router.embedding_dim equivalent logic
        size_t dim = 384;
        joint_family_query(node, query_vector, top_k, dim, results, num_results);
    }

    if (*num_results > 0) {
        qsort(results, *num_results, sizeof(QueryResult), compare_query_results);
        if (*num_results > top_k) {
            *num_results = top_k;
        }
    }
    return results;
}
