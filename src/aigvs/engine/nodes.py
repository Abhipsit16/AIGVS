import uuid
import heapq
import numpy as np
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set, Optional, TYPE_CHECKING

from src.aigvs.core.models import TextChunk
from src.aigvs.storage.local_json import LocalJSONStorage
from src.aigvs.engine.router import HopfieldRouter

if TYPE_CHECKING:
    from src.aigvs.pipeline import AIGVSPipeline

class BaseNode(ABC):
    def __init__(self, node_id: str, parent_id: Optional[str] = None):
        self.node_id = node_id
        self.parent_id = parent_id
        self.centroid_vector: Optional[np.ndarray] = None

    @abstractmethod
    def query(self, query_vector: np.ndarray, top_k: int, storage: LocalJSONStorage) -> List[str]:
        pass

    @abstractmethod
    def insert(self, chunk: TextChunk, storage: LocalJSONStorage, pipeline_ref: 'AIGVSPipeline') -> None:
        pass


class JointFamily(BaseNode):
    """ Leaf Node: Navigable Small World (NSW) with Tombstoning."""
    def __init__(self, node_id: str, parent_id: Optional[str] = None, max_connections: int = 5):
        super().__init__(node_id, parent_id)
        self.chunk_ids: List[str] = []
        self.vector_cache: List[np.ndarray] = []
        self.local_graph: Dict[int, List[int]] = {}  
        self.max_connections = max_connections  
        self.entry_point_idx: Optional[int] = None
        self.deleted_indices: Set[int] = set()

    def _cosine_similarity(self, vec_a: np.ndarray, vec_matrix: np.ndarray) -> np.ndarray:
        norm_a = np.linalg.norm(vec_a) + 1e-9
        norms_matrix = np.linalg.norm(vec_matrix, axis=1) + 1e-9
        return np.dot(vec_matrix, vec_a) / (norm_a * norms_matrix)

    def insert(self, chunk: TextChunk, storage: LocalJSONStorage, pipeline_ref: 'AIGVSPipeline') -> None:
        storage.write_chunk(chunk)
        new_idx = len(self.chunk_ids)
        self.chunk_ids.append(chunk.chunk_id)
        self.vector_cache.append(chunk.embedding)
        self.local_graph[new_idx] = []

        pipeline_ref.node_registry[self.node_id] = self
        pipeline_ref.chunk_location_map[chunk.chunk_id] = self.node_id

        # Centroid running average
        if self.centroid_vector is None:
            self.centroid_vector = chunk.embedding.copy()
            self.entry_point_idx = new_idx
            return
            
        n = len(self.vector_cache)
        self.centroid_vector = ((self.centroid_vector * (n - 1)) + chunk.embedding) / n
        self.centroid_vector /= (np.linalg.norm(self.centroid_vector) + 1e-9)

        # NSW Graph Wiring
        if n <= self.max_connections + 1:
            for i in range(new_idx):
                self.local_graph[new_idx].append(i)
                self.local_graph[i].append(new_idx)
        else:
            neighbors = self._greedy_search(chunk.embedding, top_k=self.max_connections)
            for n_idx in neighbors:
                self.local_graph[new_idx].append(n_idx)
                self.local_graph[n_idx].append(new_idx)
                if len(self.local_graph[n_idx]) > self.max_connections * 2:
                    self._prune_edges(n_idx)

    def _prune_edges(self, target_idx: int) -> None:
        neighbors = self.local_graph[target_idx]
        target_vec = self.vector_cache[target_idx]
        neighbor_vecs = np.array([self.vector_cache[i] for i in neighbors])
        scores = self._cosine_similarity(target_vec, neighbor_vecs)
        best_indices = np.argsort(-scores)[:self.max_connections]
        self.local_graph[target_idx] = [neighbors[i] for i in best_indices]

    def _greedy_search(self, query_vector: np.ndarray, top_k: int, search_depth: int = 15) -> List[int]:
        if self.entry_point_idx is None:
            return []

        candidates = []
        visited: Set[int] = {self.entry_point_idx}
        initial_score = self._cosine_similarity(query_vector, np.array([self.vector_cache[self.entry_point_idx]]))[0]
        heapq.heappush(candidates, (-initial_score, self.entry_point_idx))
        
        best_results = [(-initial_score, self.entry_point_idx)]
        
        for _ in range(search_depth):
            if not candidates: break
            current_neg_score, current_idx = heapq.heappop(candidates)
            
            neighbors = [n for n in self.local_graph[current_idx] if n not in visited]
            if not neighbors: continue
                
            visited.update(neighbors)
            neighbor_vecs = np.array([self.vector_cache[n] for n in neighbors])
            scores = self._cosine_similarity(query_vector, neighbor_vecs)
            
            for score, n_idx in zip(scores, neighbors):
                heapq.heappush(candidates, (-score, n_idx))
                best_results.append((-score, n_idx))
                
        best_results.sort(key=lambda x: x[0])
        return [idx for _, idx in best_results[:top_k]]

    def mark_deleted(self, chunk_id: str, parent_router_node: 'DisjointFamily') -> None:
        """ Tombstoning with Lazy Centroid Evaluation."""
        if chunk_id not in self.chunk_ids: return
        idx = self.chunk_ids.index(chunk_id)
        if idx in self.deleted_indices: return
            
        self.deleted_indices.add(idx)
        deleted_vector = self.vector_cache[idx]
        
        active_count = len(self.chunk_ids) - len(self.deleted_indices)
        if active_count > 0:
            current_sum = self.centroid_vector * (active_count + 1)
            new_sum = current_sum - deleted_vector
            self.centroid_vector = new_sum / active_count
            self.centroid_vector /= (np.linalg.norm(self.centroid_vector) + 1e-9)
            parent_router_node.router.add_or_update_attractor(self.node_id, self.centroid_vector, active_count)
        else:
            parent_router_node.router.remove_attractor(self.node_id)
            parent_router_node.children.pop(self.node_id, None)

    def query(self, query_vector: np.ndarray, top_k: int, storage: LocalJSONStorage) -> List[str]:
        if not self.vector_cache: return []
        raw_indices = self._greedy_search(query_vector, top_k=top_k + len(self.deleted_indices))
        
        clean_indices = [idx for idx in raw_indices if idx not in self.deleted_indices]
        best_indices = clean_indices[:top_k]
        
        if best_indices:
            primary_hit = best_indices[0]
            context_neighbors = self.local_graph[primary_hit][:2]
            for n in context_neighbors:
                if n not in best_indices and n not in self.deleted_indices:
                    best_indices.append(n)
                    
        return [self.chunk_ids[idx] for idx in best_indices]


class DisjointFamily(BaseNode):
    """ Routing Node: Concurrent paths and Spherical Bifurcation."""
    def __init__(self, node_id: str, embedding_dim: int, parent_id: Optional[str] = None):
        super().__init__(node_id, parent_id)
        self.embedding_dim = embedding_dim
        self.children: Dict[str, BaseNode] = {}
        self.router = HopfieldRouter(embedding_dim=embedding_dim)
        self.max_leaf_capacity = 1000  

    def query(self, query_vector: np.ndarray, top_k: int, storage: LocalJSONStorage) -> List[str]:
        target_child_ids = self.router.route_query(query_vector)
        if not target_child_ids: return []

        def fetch_child_results(cid: str):
            return self.children[cid].query(query_vector, top_k, storage)

        aggregated_results: List[str] = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(fetch_child_results, cid) for cid in target_child_ids if cid in self.children]
            for future in futures:
                aggregated_results.extend(future.result())

        return list(set(aggregated_results))

    def insert(self, chunk: TextChunk, storage: LocalJSONStorage, pipeline_ref: 'AIGVSPipeline') -> None:
        norm = np.linalg.norm(chunk.embedding)
        if norm > 0: chunk.embedding = chunk.embedding / norm

        target_child_ids = self.router.route_query(chunk.embedding)
        
        if not target_child_ids or not self.children:
            new_leaf_id = f"leaf_{uuid.uuid4().hex[:6]}"
            new_leaf = JointFamily(node_id=new_leaf_id, parent_id=self.node_id)
            new_leaf.insert(chunk, storage, pipeline_ref)
            
            self.children[new_leaf_id] = new_leaf
            self.router.add_or_update_attractor(new_leaf_id, new_leaf.centroid_vector, len(new_leaf.chunk_ids))
            self.centroid_vector = new_leaf.centroid_vector
            pipeline_ref.node_registry[self.node_id] = self
            return

        primary_child_id = target_child_ids[0]
        target_child = self.children[primary_child_id]
        target_child.insert(chunk, storage, pipeline_ref)
        
        active_count = len(getattr(target_child, 'chunk_ids', [])) - len(getattr(target_child, 'deleted_indices', []))
        self.router.add_or_update_attractor(primary_child_id, target_child.centroid_vector, active_count)
        self.centroid_vector = target_child.centroid_vector

        if isinstance(target_child, JointFamily) and active_count >= self.max_leaf_capacity:
            self.execute_native_hopfield_split(primary_child_id, storage, pipeline_ref)

    def execute_native_hopfield_split(self, child_id: str, storage: LocalJSONStorage, pipeline_ref: 'AIGVSPipeline') -> None:
        bloated_node: JointFamily = self.children[child_id]
        
        # Filter out tombstones before splitting
        active_indices = [i for i in range(len(bloated_node.vector_cache)) if i not in bloated_node.deleted_indices]
        if len(active_indices) < 2: return
        
        X = np.array([bloated_node.vector_cache[i] for i in active_indices], dtype=np.float32)

        avg_vector = np.mean(X, axis=0)
        seed1_idx = np.argmin(np.dot(X, avg_vector / (np.linalg.norm(avg_vector) + 1e-9)))
        xi_1 = X[seed1_idx].copy()
        
        seed2_idx = np.argmin(np.dot(X, xi_1))
        xi_2 = X[seed2_idx].copy()

        beta = 20.0 
        for _ in range(15):
            prev_xi_1, prev_xi_2 = xi_1.copy(), xi_2.copy()

            scores_1 = np.dot(X, xi_1)
            weights_1 = np.exp(beta * (scores_1 - np.max(scores_1))) / (np.sum(np.exp(beta * (scores_1 - np.max(scores_1)))) + 1e-9)
            xi_1 = np.dot(weights_1, X)
            xi_1 /= (np.linalg.norm(xi_1) + 1e-9)

            scores_2 = np.dot(X, xi_2)
            weights_2 = np.exp(beta * (scores_2 - np.max(scores_2))) / (np.sum(np.exp(beta * (scores_2 - np.max(scores_2)))) + 1e-9)
            xi_2 = np.dot(weights_2, X)
            xi_2 /= (np.linalg.norm(xi_2) + 1e-9)

            if np.allclose(xi_1, prev_xi_1, atol=1e-6) and np.allclose(xi_2, prev_xi_2, atol=1e-6): break

        assignments = np.dot(X, xi_1) < np.dot(X, xi_2)

        self.router.remove_attractor(child_id)
        
        leaf_a_id = f"{child_id}_alpha"
        leaf_b_id = f"{child_id}_beta"
        leaf_a = JointFamily(node_id=leaf_a_id, parent_id=self.node_id)
        leaf_b = JointFamily(node_id=leaf_b_id, parent_id=self.node_id)

        for i, original_idx in enumerate(active_indices):
            chunk_id = bloated_node.chunk_ids[original_idx]
            chunk = storage.read_chunk(chunk_id)
            if chunk:
                if assignments[i] == 0: leaf_a.insert(chunk, storage, pipeline_ref)
                else: leaf_b.insert(chunk, storage, pipeline_ref)

        if len(leaf_a.chunk_ids) == 0 or len(leaf_b.chunk_ids) == 0:
            half = len(active_indices) // 2
            leaf_a, leaf_b = JointFamily(node_id=leaf_a_id, parent_id=self.node_id), JointFamily(node_id=leaf_b_id, parent_id=self.node_id)
            for i, original_idx in enumerate(active_indices):
                chunk = storage.read_chunk(bloated_node.chunk_ids[original_idx])
                if chunk:
                    if i < half: leaf_a.insert(chunk, storage, pipeline_ref)
                    else: leaf_b.insert(chunk, storage, pipeline_ref)

        self.children[leaf_a_id] = leaf_a
        self.children[leaf_b_id] = leaf_b
        self.router.add_or_update_attractor(leaf_a_id, leaf_a.centroid_vector, len(leaf_a.chunk_ids))
        self.router.add_or_update_attractor(leaf_b_id, leaf_b.centroid_vector, len(leaf_b.chunk_ids))
        
        self.children.pop(child_id, None)
        pipeline_ref.node_registry.pop(child_id, None) 