#include "../include/aigvs.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("=== AIGVS C Core ===\n");
    printf("Initializing Pipeline...\n");

    AIGVSPipeline* pipeline = aigvs_pipeline_create(384, "./aigvs_data");

    printf("Creating mock chunks...\n");
    TextChunk chunks[2];

    // Chunk 1
    float vec1[384] = {0}; vec1[0] = 1.0f; vec1[1] = 0.5f;
    snprintf(chunks[0].chunk_id, MAX_CHUNK_ID_LEN, "doc_1");
    chunks[0].text = "Hello Hopfield!";
    chunks[0].embedding = vec1;
    chunks[0].embedding_dim = 384;
    chunks[0].metadata_json = NULL;

    // Chunk 2
    float vec2[384] = {0}; vec2[0] = 0.5f; vec2[1] = 1.0f;
    snprintf(chunks[1].chunk_id, MAX_CHUNK_ID_LEN, "doc_2");
    chunks[1].text = "Hello AIGVS in pure C!";
    chunks[1].embedding = vec2;
    chunks[1].embedding_dim = 384;
    chunks[1].metadata_json = NULL;

    printf("Ingesting vectors...\n");
    aigvs_pipeline_ingest(pipeline, chunks, 2);
    printf("Vectors ingested successfully into the Hopfield Router.\n");

    printf("Executing vector query for doc_1...\n");
    int out_results = 0;
    QueryResult* results = aigvs_pipeline_query(pipeline, vec1, 10, &out_results);

    printf("Results matched: %d\n", out_results);
    for(int i=0; i<out_results; i++){
        printf(" -> Result [%d]: ID=%s, Score=%.4f\n", i, results[i].chunk_id, results[i].score);
    }

    if(results) free(results);

    aigvs_pipeline_destroy(pipeline);
    printf("Pipeline memory cleanly freed.\n");

    return 0;
}
