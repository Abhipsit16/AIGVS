"""
nodes.py — Python tree nodes for the AIGVS hierarchy.

Design contract
---------------
Pure Python bookkeeping — tree topology, centroid vectors, chunk-ID lists.
No embedded Rust objects.  Every NSW and router mutation goes through
``pipeline_ref.engine`` so there is one PyO3 boundary crossing per operation.

NSW graph quality parameters (max_connections, ef_construction) are injected
at construction time via ``pipeline_cfg`` and propagated to every leaf and
sub-leaf created during splits, ensuring consistent graph quality throughout
the lifetime of the index regardless of where a leaf is created.
"""

import uuid
import numpy as np
from typing import Dict, List, Set, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aigvs.pipeline import AIGVSPipeline


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
    chunk_ids, deleted_indices, and the running centroid.
    """

    def __init__(
        self,
        node_id: str,
        parent_id: Optional[str] = None,
        pipeline_cfg: Optional[dict] = None,
    ):
        super().__init__(node_id, parent_id)
        cfg = pipeline_cfg or {}
        self.max_connections: int = cfg.get("max_connections", 16)
        self.ef_construction: int = cfg.get("ef_construction", 100)
        self.chunk_ids: List[str] = []
        self.deleted_indices: Set[int] = set()

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, chunk, storage, pipeline_ref: "AIGVSPipeline") -> None:
        storage.write_chunk(chunk)
        self.chunk_ids.append(chunk.chunk_id)

        pipeline_ref.node_registry[self.node_id] = self
        pipeline_ref.chunk_location_map[chunk.chunk_id] = self.node_id

        # Populate the hot-path payload cache so query() never touches storage.
        pipeline_ref._cache_payload(chunk)

        # Online normalised mean centroid.
        if self.centroid_vector is None:
            self.centroid_vector = chunk.embedding.copy()
        else:
            n = len(self.chunk_ids)
            self.centroid_vector = (
                (self.centroid_vector * (n - 1)) + chunk.embedding
            ) / n
            norm = np.linalg.norm(self.centroid_vector)
            if norm > 0:
                self.centroid_vector /= norm

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
        """
        Tombstone chunk_id and update the parent attractor.
        The engine-level tombstone (native_mark_deleted) is applied by the
        pipeline *before* calling this method; here we update Python bookkeeping
        and the router centroid.
        """
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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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

    The Hopfield router lives inside ``pipeline_ref.engine``;
    Python holds ``self.children`` for topology traversal and split logic only.
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
        self.children: Dict[str, Union["JointFamily", "DisjointFamily"]] = {}
        self.chunk_count = 0
        self.max_leaf_capacity: int = 5000

        # Propagated to every leaf this branch creates.
        cfg = pipeline_cfg or {}
        self._pipeline_cfg: dict = cfg

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, chunk, storage, pipeline_ref: "AIGVSPipeline") -> None:
        norm = np.linalg.norm(chunk.embedding)
        if norm > 0:
            chunk.embedding = chunk.embedding / norm

        target_child_ids: List[str] = pipeline_ref.engine.route_query(
            self.node_id, chunk.embedding,
            max_beam=1, target_confidence=0.95, beta=10.0,
        )

        if not target_child_ids or not self.children:
            self._create_leaf_and_insert(chunk, storage, pipeline_ref)
            return

        for child_id in target_child_ids:
            target_child = self.children[child_id]
            target_child.insert(chunk, storage, pipeline_ref)

        primary_child_id = target_child_ids[0]
        target_child = self.children[primary_child_id]

        if target_child.centroid_vector is not None:
            pipeline_ref.engine.update_attractor(
                self.node_id, primary_child_id, target_child.centroid_vector
            )
            
        if self.centroid_vector is None:
            self.centroid_vector = chunk.embedding.copy()
        else:
            self.centroid_vector = (
                (self.centroid_vector * self.chunk_count) + chunk.embedding
            ) / (self.chunk_count + 1)
            norm = np.linalg.norm(self.centroid_vector)
            if norm > 0:
                self.centroid_vector /= norm
        self.chunk_count += 1

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
        """Spin up a new leaf using the pipeline's NSW config."""
        new_leaf_id = f"leaf_{uuid.uuid4().hex[:6]}"
        new_leaf = JointFamily(
            node_id=new_leaf_id,
            parent_id=self.node_id,
            pipeline_cfg=pipeline_ref._node_cfg(),
        )

        # Register engine-side NSW graph with the same M and ef as the pipeline.
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
        cohesion_threshold: float = 0.70,
    ) -> None:
        """
        Dynamically discover macro-communities inside a bloated leaf and
        replace it with a new branch containing sub-community leaves.

        Algorithm
        ---------
        1. Seed selection — maximally separated anchor points (farthest-first).
        2. High-cohesion abort — single community → widen capacity, don't split.
        3. Variance-adaptive Hopfield convergence — soft EM with per-community
           temperature scaling driven by intra-community score variance.
        4. Hierarchical wiring — rebuild engine state for the new sub-tree,
           using the pipeline's NSW config so sub-leaves have the same graph
           quality as any other leaf in the index.
        """
        bloated_node = self.children[child_id]

        # --- Collect active chunks ---
        active_chunks = []
        X_list = []
        for i, cid in enumerate(bloated_node.chunk_ids):
            if i not in bloated_node.deleted_indices:
                chunk = storage.read_chunk(cid)
                if chunk is not None:
                    active_chunks.append(chunk)
                    X_list.append(chunk.embedding)

        if len(X_list) < 2:
            return

        X = np.array(X_list, dtype=np.float32)  # (N, D)

        # --- 1. Farthest-first seed selection ---
        avg = np.mean(X, axis=0)
        avg_norm = np.linalg.norm(avg)
        if avg_norm > 0:
            avg /= avg_norm
        seeds = [X[int(np.argmax(X @ avg))].copy()]

        max_seeds = min(15, max(3, int(np.sqrt(len(X)))))
        for _ in range(1, max_seeds):
            sims = X @ np.array(seeds).T          # (N, K)
            max_sim = np.max(sims, axis=1)         # (N,)
            furthest = int(np.argmin(max_sim))
            if max_sim[furthest] > cohesion_threshold:
                break
            seeds.append(X[furthest].copy())

        centroids = np.array(seeds, dtype=np.float32)
        num_branches = len(centroids)

        # --- 2. High-cohesion abort ---
        if num_branches == 1:
            # Entire vessel is one cohesive community — widen it and keep going.
            bloated_node.max_connections = min(
                bloated_node.max_connections + pipeline_ref.max_connections, 64
            )
            self.max_leaf_capacity += 100
            # Also expand the engine-side graph's effective connection budget
            # by re-registering with higher M. The existing edges are preserved;
            # new inserts will now wire to more neighbours.
            # (Engine doesn't support in-place M upgrade, so we leave it — the
            # wider Python-side capacity check is the effective guard here.)
            return

        # --- 3. Variance-adaptive Hopfield convergence ---
        lambda_scale = 5.0
        for _ in range(15):
            prev = centroids.copy()
            scores = X @ centroids.T                              # (N, K)
            variances = np.var(scores, axis=0)                    # (K,)
            betas = lambda_scale / (variances + 1e-5)
            scaled = scores * betas
            shifted = scaled - np.max(scaled, axis=1, keepdims=True)
            exp_s = np.exp(shifted)
            weights = exp_s / (exp_s.sum(axis=1, keepdims=True) + 1e-9)
            centroids = weights.T @ X                             # (K, D)
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            centroids /= norms + 1e-9
            if np.allclose(centroids, prev, atol=1e-6):
                break

        sims = X @ centroids.T
        max_sims = np.max(sims, axis=1)

        # --- 4. Hierarchical wiring ---

        pipeline_ref.engine.remove_attractor(self.node_id, child_id)

        # Reuse child_id as the new branch's id — no upstream re-wiring needed.
        new_branch = DisjointFamily(
            node_id=child_id,
            embedding_dim=self.embedding_dim,
            parent_id=self.node_id,
            pipeline_cfg=pipeline_ref._node_cfg(),
        )
        pipeline_ref.engine.register_branch(child_id, self.node_id, False)

        # Build sub-community leaves — all using the pipeline's NSW config so
        # graph quality is consistent with every other leaf in the index.
        new_leaves: List[JointFamily] = []
        for branch_idx in range(num_branches):
            leaf_id = f"{child_id}_macro_{branch_idx}"
            leaf = JointFamily(
                node_id=leaf_id,
                parent_id=new_branch.node_id,
                pipeline_cfg=pipeline_ref._node_cfg(),
            )
            # Register engine-side NSW graph before any inserts.
            pipeline_ref._register_leaf_in_engine(leaf_id, child_id)
            new_leaves.append(leaf)
        # Distribute chunks — insert in their original order per community so
        # the NSW graph builds up the same way it would have on live ingestion.
        assignments = np.argmax(sims, axis=1)

        community_chunks: List[List] = [[] for _ in range(num_branches)]
        for i, chunk in enumerate(active_chunks):
            community_chunks[int(assignments[i])].append(chunk)

        for branch_idx in range(num_branches):
            leaf = new_leaves[branch_idx]
            for chunk in community_chunks[branch_idx]:
                leaf.insert(chunk, storage, pipeline_ref)

        # Wire populated leaves into the new branch and engine.
        valid_centroids = []
        for leaf in new_leaves:
            if not leaf.chunk_ids:
                continue
            new_branch.children[leaf.node_id] = leaf
            if leaf.centroid_vector is not None:
                pipeline_ref.engine.update_attractor(
                    child_id, leaf.node_id, leaf.centroid_vector
                )
                valid_centroids.append(leaf.centroid_vector)
            pipeline_ref.node_registry[leaf.node_id] = leaf

        if valid_centroids:
            mean_c = np.mean(valid_centroids, axis=0)
            n = np.linalg.norm(mean_c)
            new_branch.centroid_vector = mean_c / (n + 1e-9)
        else:
            new_branch.centroid_vector = centroids[0]

        self.children[child_id] = new_branch
        pipeline_ref.engine.update_attractor(
            self.node_id, child_id, new_branch.centroid_vector
        )
        pipeline_ref.node_registry[child_id] = new_branch

        # --- Recursive depth building ---
        # If any sub-leaf still exceeds capacity after redistribution,
        # recursively split it so the tree grows deeper than 2 levels.
        for leaf in new_leaves:
            if (
                isinstance(leaf, JointFamily)
                and leaf.active_count >= new_branch.max_leaf_capacity
                and leaf.active_count < 0.9 * len(active_chunks)
            ):
                new_branch.execute_native_hopfield_split(
                    leaf.node_id, storage, pipeline_ref
                )