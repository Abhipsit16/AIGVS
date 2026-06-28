import aigvs_rust
import numpy as np

engine = aigvs_rust.AIGVSEngine(384)
engine.register_branch('root_node', 'root_node', True)

# Create 4 fake macro attractors, just like MSMARCO
np.random.seed(42)
centroids = np.random.randn(4, 384).astype(np.float32)
for i in range(4):
    centroids[i] /= np.linalg.norm(centroids[i])
    engine.update_attractor('root_node', f'trunk_{i}', centroids[i])

# Create 1 query vector (same distribution)
q = np.random.randn(384).astype(np.float32)
q /= np.linalg.norm(q)

print("Running with beta=5")
engine.unified_search(q, top_k=10, max_beam=10, target_confidence=0.95, beta=5)

print("\nRunning with beta=5.0")
engine.unified_search(q, top_k=10, max_beam=10, target_confidence=0.95, beta=5.0)

print("\nRunning with beta=0.0")
engine.unified_search(q, top_k=10, max_beam=10, target_confidence=0.95, beta=0.0)
