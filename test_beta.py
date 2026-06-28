import aigvs_rust
import numpy as np

engine = aigvs_rust.AIGVSEngine(384)
engine.register_branch('root_node', 'root_node', True)
engine.update_attractor('root_node', 'leaf0', np.ones(384, dtype=np.float32) * 0.1)
engine.update_attractor('root_node', 'leaf1', np.ones(384, dtype=np.float32) * 0.2)
q = np.ones(384, dtype=np.float32) * 0.15

# We will just print the output of query
print(engine.unified_search(q, top_k=10, max_beam=10, target_confidence=0.95, beta=5))
print(engine.unified_search(q, top_k=10, max_beam=10, target_confidence=0.95, beta=5.0))
