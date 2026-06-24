import numpy as np
from typing import List, Dict, Optional

class HopfieldRouter:
    """The Continuous Attractor Network that directs routing flow using soft thresholds."""
    def __init__(self, embedding_dim: int, beta_min: float = 1.0, k_scale: float = 0.5, activation_threshold: float = 0.25):
        self.embedding_dim = embedding_dim
        self.beta_min = beta_min
        self.k_scale = k_scale
        self.activation_threshold = activation_threshold
        
        self.id_lookup: List[str] = []
        self.X_matrix: Optional[np.ndarray] = None  # Shape: (M, E)
        self.node_capacities: Dict[str, int] = {}

    def add_or_update_attractor(self, node_id: str, centroid: np.ndarray, current_count: int) -> None:
        """Registers or shifts a vector coordinate inside the routing matrix space."""
        vector = centroid.reshape(1, self.embedding_dim)
        self.node_capacities[node_id] = current_count
        
        if node_id in self.id_lookup:
            idx = self.id_lookup.index(node_id)
            self.X_matrix[idx] = vector
        else:
            if self.X_matrix is None:
                self.X_matrix = vector
                self.id_lookup = [node_id]
            else:
                self.X_matrix = np.vstack([self.X_matrix, vector])
                self.id_lookup.append(node_id)

    def remove_attractor(self, node_id: str) -> None:
        """Removes a routing coordinate immediately following a node split or deletion."""
        if node_id in self.id_lookup:
            idx = self.id_lookup.index(node_id)
            self.id_lookup.pop(idx)
            self.X_matrix = np.delete(self.X_matrix, idx, axis=0)
            self.node_capacities.pop(node_id, None)
            if self.X_matrix.shape[0] == 0:
                self.X_matrix = None

    def calculate_dynamic_beta(self) -> float:
        """Calculates temperature using inverse fractional scaling over pairwise variance."""
        if self.X_matrix is None or self.X_matrix.shape[0] < 2:
            return self.beta_min
            
        norms = np.linalg.norm(self.X_matrix, axis=1, keepdims=True) + 1e-9
        normalized_X = self.X_matrix / norms
        similarity_matrix = np.dot(normalized_X, normalized_X.T)
        
        upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        if len(upper_tri) == 0:
            return self.beta_min
            
        variance = float(np.var(upper_tri))
        return self.beta_min + (self.k_scale / (variance + 1e-5))

    def route_query(self, query_vector: np.ndarray, max_beam: int = 15, target_confidence: float = 0.65) -> List[str]:
        if self._size == 0:
            return []

        X = self.X_matrix[:self._size]  # view, no copy
        beta = self.calculate_dynamic_beta()

        q = query_vector.astype(np.float32)
        q /= (np.linalg.norm(q) + 1e-9)

        norms = np.linalg.norm(X, axis=1) + 1e-9
        raw_scores = (X / norms[:, np.newaxis]) @ q

        # Softmax gives us true probability/confidence percentages
        scaled = beta * raw_scores
        scaled -= scaled.max()
        weights = np.exp(scaled)
        weights /= weights.sum() + 1e-9

        above = raw_scores >= self.activation_threshold
        if not np.any(above):
            return []  # OOD — DisjointFamily.insert handles the new branch spawn

        # Zero out the probabilities of anything below our minimum cosine threshold
        masked_weights = np.where(above, weights, 0.0)

        # Sort indices strictly by probability (highest first)
        sorted_indices = np.argsort(-masked_weights)

        dynamic_beam_ids = []
        cumulative_confidence = 0.0

        # --- NUCLEUS ROUTING LOGIC ---
        for idx in sorted_indices[:max_beam]:
            w = masked_weights[idx]
            if w == 0.0:
                break  # Stop immediately if we hit rejected attractors

            dynamic_beam_ids.append(self.idx_to_id[idx])
            cumulative_confidence += w

            # Dynamic Cutoff: If we are confident enough, stop adding branches!
            if cumulative_confidence >= target_confidence:
                break

        return dynamic_beam_ids