#include "../include/aigvs.h"
#include "../include/cJSON.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Simple in-memory hash map for storage caching
typedef struct StorageNode {
    char chunk_id[MAX_CHUNK_ID_LEN];
    TextChunk* chunk;
    struct StorageNode* next;
} StorageNode;

typedef struct {
    StorageNode** buckets;
    size_t num_buckets;
} InMemoryStorage;

static unsigned int hash_string(const char* str, size_t num_buckets) {
    unsigned int hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % num_buckets;
}

void* storage_create_in_memory() {
    InMemoryStorage* storage = malloc(sizeof(InMemoryStorage));
    storage->num_buckets = 1024;
    storage->buckets = calloc(storage->num_buckets, sizeof(StorageNode*));
    return storage;
}

void storage_destroy_in_memory(void* storage_ptr) {
    InMemoryStorage* storage = (InMemoryStorage*)storage_ptr;
    for (size_t i = 0; i < storage->num_buckets; ++i) {
        StorageNode* curr = storage->buckets[i];
        while (curr) {
            StorageNode* next = curr->next;
            // Note: In real system, we'd also free the chunk's text and embedding if we own it
            free(curr);
            curr = next;
        }
    }
    free(storage->buckets);
    free(storage);
}

void storage_write_chunk(void* storage_ptr, const TextChunk* chunk) {
    InMemoryStorage* storage = (InMemoryStorage*)storage_ptr;
    unsigned int bucket = hash_string(chunk->chunk_id, storage->num_buckets);

    StorageNode* curr = storage->buckets[bucket];
    while (curr) {
        if (strcmp(curr->chunk_id, chunk->chunk_id) == 0) {
            curr->chunk = (TextChunk*)chunk; // update
            return;
        }
        curr = curr->next;
    }

    StorageNode* new_node = malloc(sizeof(StorageNode));
    strncpy(new_node->chunk_id, chunk->chunk_id, MAX_CHUNK_ID_LEN - 1);
    new_node->chunk_id[MAX_CHUNK_ID_LEN - 1] = '\0';
    new_node->chunk = (TextChunk*)chunk;
    new_node->next = storage->buckets[bucket];
    storage->buckets[bucket] = new_node;
}

TextChunk* storage_read_chunk(void* storage_ptr, const char* chunk_id) {
    InMemoryStorage* storage = (InMemoryStorage*)storage_ptr;
    unsigned int bucket = hash_string(chunk_id, storage->num_buckets);

    StorageNode* curr = storage->buckets[bucket];
    while (curr) {
        if (strcmp(curr->chunk_id, chunk_id) == 0) {
            return curr->chunk;
        }
        curr = curr->next;
    }
    return NULL;
}
