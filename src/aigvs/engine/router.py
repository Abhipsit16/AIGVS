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

    def route_query(self, query_vector: np.ndarray) -> List[str]:
        """Evaluates activation weights using continuous equations to locate target nodes."""
        if self.X_matrix is None:
            return []

        q = query_vector.reshape(self.embedding_dim)
        raw_scores = np.dot(self.X_matrix, q)
        beta = self.calculate_dynamic_beta()
        
        # Softmax evaluation with stability shift
        shifted = raw_scores - np.max(raw_scores)
        exp_scores = np.exp(beta * shifted)
        softmax_weights = exp_scores / np.sum(exp_scores)
        
        passing_indices = np.where(softmax_weights >= self.activation_threshold)[0]
        sorted_indices = passing_indices[np.argsort(-softmax_weights[passing_indices])]
        
        return [self.id_lookup[idx] for idx in sorted_indices]