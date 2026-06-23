#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <random>

using namespace std;

// Hyperparameters matching your AIGVS Python architecture
const int EMBEDDING_DIM = 384;
const int ROOT_BRANCHES = 40;
const int BEAM_WIDTH = 3;
const int CHUNKS_PER_LEAF = 250;
const int CANDIDATES = BEAM_WIDTH * CHUNKS_PER_LEAF; // 750 chunks to re-rank
const int TOP_K = 10;

// Helper: Generate random float vectors to simulate embeddings
vector<float> generate_random_vector(int size) {
    vector<float> vec(size);
    mt19937 gen(42); // Fixed seed for consistency
    uniform_real_distribution<float> dis(-1.0, 1.0);
    for(int i = 0; i < size; ++i) vec[i] = dis(gen);
    return vec;
}

int main() {
    cout << "=== AIGVS BARE-METAL C++ SIMULATOR ===" << endl;

    // 1. Allocate Contiguous C++ Memory (No Python Dictionaries!)
    vector<vector<float>> attractors(ROOT_BRANCHES, vector<float>(EMBEDDING_DIM));
    vector<vector<float>> leaf_chunks(CANDIDATES, vector<float>(EMBEDDING_DIM));
    vector<float> query = generate_random_vector(EMBEDDING_DIM);

    // Fill with dummy data
    for(int i=0; i<ROOT_BRANCHES; i++) attractors[i] = generate_random_vector(EMBEDDING_DIM);
    for(int i=0; i<CANDIDATES; i++) leaf_chunks[i] = generate_random_vector(EMBEDDING_DIM);

    cout << "[SYSTEM] Memory Allocated. Waking up CPU..." << endl;

    // --- START THE STOPWATCH ---
    auto start_time = chrono::high_resolution_clock::now();

    // PHASE 1: HOPFIELD ROUTING (Find top 3 branches out of 40)
    vector<pair<float, int>> routing_scores(ROOT_BRANCHES);
    for (int i = 0; i < ROOT_BRANCHES; ++i) {
        float dot_product = 0.0f;
        // The inner math loop
        for (int d = 0; d < EMBEDDING_DIM; ++d) {
            dot_product += attractors[i][d] * query[d];
        }
        routing_scores[i] = {dot_product, i};
    }

    // Sort to find the best branches (Beam Search)
    sort(routing_scores.rbegin(), routing_scores.rend());

    // PHASE 2: RE-RANKING (Scan the 750 chunks inside those branches)
    vector<pair<float, int>> rerank_scores(CANDIDATES);
    for (int i = 0; i < CANDIDATES; ++i) {
        float dot_product = 0.0f;
        for (int d = 0; d < EMBEDDING_DIM; ++d) {
            dot_product += leaf_chunks[i][d] * query[d];
        }
        rerank_scores[i] = {dot_product, i};
    }

    // Sort to get the absolute top 10
    sort(rerank_scores.rbegin(), rerank_scores.rend());

    // --- STOP THE STOPWATCH ---
    auto end_time = chrono::high_resolution_clock::now();
    
    // Calculate precise microsecond latency
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
    double latency_ms = duration / 1000.0;

    cout << "\n==============================================" << endl;
    cout << "          C++ MICRO-BENCHMARK RESULTS         " << endl;
    cout << "==============================================" << endl;
    cout << "Architecture: AIGVS (Hopfield Routing + Beam Search)" << endl;
    cout << "Operations:   " << ROOT_BRANCHES << " routing distances + " << CANDIDATES << " leaf distances" << endl;
    cout << "Latency:      " << latency_ms << " ms" << endl;
    cout << "==============================================" << endl;
    
    // Proof of execution
    cout << "\nTop Hit Score: " << rerank_scores[0].first << " (ID: " << rerank_scores[0].second << ")" << endl;

    return 0;
}