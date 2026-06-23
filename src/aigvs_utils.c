#include "../include/aigvs.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

float cosine_similarity(const float* vec_a, const float* vec_b, size_t dim) {
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < dim; ++i) {
        dot_product += vec_a[i] * vec_b[i];
        norm_a += vec_a[i] * vec_a[i];
        norm_b += vec_b[i] * vec_b[i];
    }

    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    if (denom < 1e-9f) {
        return 0.0f;
    }

    return dot_product / denom;
}

void normalize_vector(float* vec, size_t dim) {
    float norm = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        norm += vec[i] * vec[i];
    }
    norm = sqrtf(norm);
    if (norm > 0.0f) {
        for (size_t i = 0; i < dim; ++i) {
            vec[i] /= norm;
        }
    }
}
