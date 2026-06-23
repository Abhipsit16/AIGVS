#include "../include/aigvs.h"
#include <stdlib.h>
#include <string.h>

void hopfield_router_init(HopfieldRouter* router, size_t embedding_dim) {
    router->embedding_dim = embedding_dim;
    router->beta_min = 1.0f;
    router->k_scale = 0.5f;
    router->activation_threshold = 0.25f;
    router->id_lookup = NULL;
    router->X_matrix = NULL;
    router->node_capacities = NULL;
    router->num_attractors = 0;
    router->capacity = 0;
}

void hopfield_router_destroy(HopfieldRouter* router) {
    for (size_t i = 0; i < router->num_attractors; ++i) {
        free(router->id_lookup[i]);
        free(router->X_matrix[i]);
    }
    free(router->id_lookup);
    free(router->X_matrix);
    free(router->node_capacities);
}

void hopfield_router_add_or_update(HopfieldRouter* router, const char* node_id, const float* centroid, int current_count) {
    for (size_t i = 0; i < router->num_attractors; ++i) {
        if (strcmp(router->id_lookup[i], node_id) == 0) {
            memcpy(router->X_matrix[i], centroid, router->embedding_dim * sizeof(float));
            router->node_capacities[i] = current_count;
            return;
        }
    }

    if (router->num_attractors == router->capacity) {
        size_t new_cap = router->capacity == 0 ? 4 : router->capacity * 2;
        router->id_lookup = realloc(router->id_lookup, new_cap * sizeof(char*));
        router->X_matrix = realloc(router->X_matrix, new_cap * sizeof(float*));
        router->node_capacities = realloc(router->node_capacities, new_cap * sizeof(int));
        router->capacity = new_cap;
    }

    size_t idx = router->num_attractors++;
    router->id_lookup[idx] = strdup(node_id);
    router->X_matrix[idx] = malloc(router->embedding_dim * sizeof(float));
    memcpy(router->X_matrix[idx], centroid, router->embedding_dim * sizeof(float));
    router->node_capacities[idx] = current_count;
}

typedef struct {
    float score;
    size_t index;
} ScoreIndex;

static int compare_scores(const void* a, const void* b) {
    float sa = ((const ScoreIndex*)a)->score;
    float sb = ((const ScoreIndex*)b)->score;
    if (sa < sb) return 1;
    if (sa > sb) return -1;
    return 0;
}

char** hopfield_router_route(HopfieldRouter* router, const float* query_vector, int beam_width, int* out_count) {
    *out_count = 0;
    if (router->num_attractors == 0) return NULL;

    ScoreIndex* scores = malloc(router->num_attractors * sizeof(ScoreIndex));
    for (size_t i = 0; i < router->num_attractors; ++i) {
        float dot_product = 0.0f;
        for (size_t d = 0; d < router->embedding_dim; ++d) {
            dot_product += router->X_matrix[i][d] * query_vector[d];
        }
        scores[i].score = dot_product;
        scores[i].index = i;
    }

    qsort(scores, router->num_attractors, sizeof(ScoreIndex), compare_scores);

    int num_results = (beam_width < (int)router->num_attractors) ? beam_width : (int)router->num_attractors;
    char** results = malloc(num_results * sizeof(char*));
    for (int i = 0; i < num_results; ++i) {
        results[i] = strdup(router->id_lookup[scores[i].index]);
    }

    free(scores);
    *out_count = num_results;
    return results;
}
