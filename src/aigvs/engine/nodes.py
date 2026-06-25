"""
nodes.py — Python tree nodes for the AIGVS hierarchy.

Design contract
---------------
Pure Python bookkeeping — tree topology, centroid vectors, chunk-ID lists.
No embedded Rust objects. Every NSW and router mutation goes through
``pipeline_ref.engine`` so there is one PyO3 boundary crossing per operation.
"""

import uuid
import numpy as np
from typing import Dict, List, Set, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.aigvs.pipeline import AIGVSPipeline


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseNode:
    def __init__(self, node_id: str, parent_id: Optional[str] = None):
        self.node_id = node_id
        self.parent_id = parent_id
        self.centroid_vector: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Leaf  (JointFamily)
# ---------------------------------------------------------------------------

class JointFamily(BaseNode):
    """
    Leaf vessel — a flat collection of semantically cohesive chunks.

    The NSW graph lives inside ``pipeline_ref.engine``; Python only tracks
    chunk_ids, deleted_indices, and the mathematical centroid sum.
    """

    def __init__(
        self,
        node_id: str,
        parent_id: Optional[str] = None,
        pipeline_cfg: Optional[dict] = None,
    ):
        super().__init__(node_id, parent_id)
        cfg = pipeline_cfg or {}
        self.max_connections: int = cfg.get("max_connections", 32) # Bumped to 32 for SOTA recall
        self.ef_construction: int = cfg.get("ef_construction", 200) # Bumped to 200 for SOTA recall
        self.chunk_ids: List[str] = []
        self.deleted_indices: Set[int] = set()
        
        # SOTA Fix: Track the raw unnormalized sum to prevent mathematical drift
        self.centroid_sum: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, chunk, storage, pipeline_ref: "AIGVSPipeline") -> None:
        storage.write_chunk(chunk)
        self.chunk_ids.append(chunk.chunk_id)

        pipeline_ref.node_registry[self.node_id] = self
        pipeline_ref.chunk_location_map[chunk.chunk_id] = self.node_id
        pipeline_ref._cache_payload(chunk)

        # SOTA Fix: Mathematically accurate online spherical mean
        if self.centroid_sum is None:
            self.centroid_sum = chunk.embedding.copy()
        else:
            self.centroid_sum += chunk.embedding

        # Only normalize for the external router vector, never mutate the internal sum
        norm = np.linalg.norm(self.centroid_sum)
        if norm > 0:
            self.centroid_vector = self.centroid_sum / norm
        else:
            self.centroid_vector = self.centroid_sum.copy()

        pipeline_ref.engine.native_insert_to_leaf(
            self.node_id, chunk.chunk_id, chunk.embedding
        )

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def mark_deleted(
        self,
        chunk_id: str,
        parent_router_node: "DisjointFamily",
        engine,
    ) -> None:
        if chunk_id not in self.chunk_ids:
            return
        idx = self.chunk_ids.index(chunk_id)
        if idx in self.deleted_indices:
            return

        self.deleted_indices.add(idx)

        if self.active_count > 0:
            engine.update_attractor(
                parent_router_node.node_id, self.node_id, self.centroid_vector
            )
        else:
            engine.remove_attractor(parent_router_node.node_id, self.node_id)
            parent_router_node.children.pop(self.node_id, None)

    @property
    def active_count(self) -> int:
        return len(self.chunk_ids) - len(self.deleted_indices)

    def pipeline_cfg(self) -> dict:
        return {
            "max_connections": self.max_connections,
            "ef_construction": self.ef_construction,
        }


# ---------------------------------------------------------------------------
# Branch  (DisjointFamily)
# ---------------------------------------------------------------------------

class DisjointFamily(BaseNode):
    """
    Internal routing vessel fanning out to child nodes.
    """

    def __init__(
        self,
        node_id: str,
        embedding_dim: int,
        parent_id: Optional[str] = None,
        pipeline_cfg: Optional[dict] = None,
    ):
        super().__init__(node_id, parent_id)
        self.embedding_dim = embedding_dim
        self.children: Dict[str, BaseNode] = {}
        
        # SOTA Fix: The HNSW Paradox. Leaves must be fat enough to justify graph traversals.
        self.max_leaf_capacity: int = 4096 

        cfg = pipeline_cfg or {}
        self._pipeline_cfg: dict = cfg

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, chunk, storage, pipeline_ref: "AIGVSPipeline") -> None:
        norm = np.linalg.norm(chunk.embedding)
        if norm > 0:
            chunk.embedding = chunk.embedding / norm

        # Softmax beta is stripped from engine backend, keeping signature for compatibility
        target_child_ids: List[str] = pipeline_ref.engine.route_query(
            self.node_id, chunk.embedding, max_beam=1
        )

        if not target_child_ids or not self.children:
            self._create_leaf_and_insert(chunk, storage, pipeline_ref)
            return

        primary_child_id = target_child_ids[0]
        target_child = self.children[primary_child_id]
        target_child.insert(chunk, storage, pipeline_ref)

        if target_child.centroid_vector is not None:
            pipeline_ref.engine.update_attractor(
                self.node_id, primary_child_id, target_child.centroid_vector
            )
        self.centroid_vector = target_child.centroid_vector

        if (
            isinstance(target_child, JointFamily)
            and target_child.active_count >= self.max_leaf_capacity
        ):
            self.execute_native_hopfield_split(
                primary_child_id, storage, pipeline_ref
            )

    def _create_leaf_and_insert(
        self, chunk, storage, pipeline_ref: "AIGVSPipeline"
    ) -> None:
        new_leaf_id = f"leaf_{uuid.uuid4().hex[:6]}"
        new_leaf = JointFamily(
            node_id=new_leaf_id,
            parent_id=self.node_id,
            pipeline_cfg=pipeline_ref._node_cfg(),
        )

        pipeline_ref._register_leaf_in_engine(new_leaf_id, self.node_id)
        new_leaf.insert(chunk, storage, pipeline_ref)

        self.children[new_leaf_id] = new_leaf
        if new_leaf.centroid_vector is not None:
            pipeline_ref.engine.update_attractor(
                self.node_id, new_leaf_id, new_leaf.centroid_vector
            )
        self.centroid_vector = new_leaf.centroid_vector
        pipeline_ref.node_registry[self.node_id] = self

    # ------------------------------------------------------------------
    # Hopfield split
    # ------------------------------------------------------------------

    def execute_native_hopfield_split(
        self,
        child_id: str,
        storage,
        pipeline_ref: "AIGVSPipeline",
        num_clusters: int = 4,
    ) -> None:
        bloated_node = self.children.get(child_id)
        if not bloated_node:
            return

        result = pipeline_ref.engine.compute_disjoint_clusters(
            child_id, num_clusters, 15
        )
        
        if not result:
            self.max_leaf_capacity += 1000
            return
            
        centroids_raw, assignments = result
        num_branches = len(centroids_raw)

        if num_branches == 1:
            self.max_leaf_capacity += 1000
            return

        pipeline_ref.engine.remove_attractor(self.node_id, child_id)
        del self.children[child_id] 
        
        new_leaves: List[JointFamily] = []
        for branch_idx in range(num_branches):
            leaf_id = f"{child_id}_macro_{branch_idx}"
            leaf = JointFamily(
                node_id=leaf_id,
                parent_id=self.node_id,
                pipeline_cfg=pipeline_ref._node_cfg(),
            )
            pipeline_ref._register_leaf_in_engine(leaf_id, self.node_id)
            new_leaves.append(leaf)

        active_chunks = [
            storage.read_chunk(cid) 
            for i, cid in enumerate(bloated_node.chunk_ids) 
            if i not in bloated_node.deleted_indices
        ]
        active_chunks = [c for c in active_chunks if c is not None]

        for chunk, assignment in zip(active_chunks, assignments):
            new_leaves[assignment].insert(chunk, storage, pipeline_ref)

        for i, leaf in enumerate(new_leaves):
            if not leaf.chunk_ids:
                continue
            
            leaf.centroid_vector = np.array(centroids_raw[i], dtype=np.float32)
            self.children[leaf.node_id] = leaf
            
            pipeline_ref.engine.update_attractor(
                self.node_id, leaf.node_id, leaf.centroid_vector
            )
            pipeline_ref.node_registry[leaf.node_id] = leaf