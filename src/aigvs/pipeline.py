import os
import pickle
import time
import numpy as np
from collections import deque
from typing import Dict, List, Optional
from aigvs.core.models import TextChunk
from aigvs.storage import LocalJSONStorage
from aigvs.engine.nodes import DisjointFamily, BaseNode, JointFamily
import aigvs_rust


class AIGVSPipeline:
    """
    Orchestrates the AIGVS hierarchy.

    Latency architecture
    --------------------
    The query hot path has exactly three steps:
      1. np normalise (Python, ~5µs)
      2. engine.unified_search() — one PyO3 crossing, GIL released, parallel
         Rayon leaf search inside Rust (~1–3ms depending on corpus size)
      3. payload_cache lookup × top_k — O(1) dict lookup, no I/O (~5µs total)

    storage.read_chunk() is NEVER called during query.  All payloads (text +
    metadata) are kept in ``self.payload_cache`` (a plain Python dict keyed by
    chunk_id).  This is the primary reason the old code was slow: JSON file I/O
    or storage deserialization on every result, for every query.

    NSW graph quality parameters
    ----------------------------
    max_connections (M)  : max edges per node.  Rule of thumb: 16 for 768-dim.
    ef_construction      : candidate pool during insert.  Rule of thumb: 4–8× M.
    ef_search            : candidate pool during query.  Raise to trade
                           latency for recall.  Default: max(top_k * 4, 50).
    """

    def __init__(
        self,
        embedding_dim: int,
        storage_path: str = "./aigvs_data",
        max_connections: int = 32,
        ef_construction: int = 200,
    ):
        self.embedding_dim = embedding_dim
        self.storage_path = storage_path
        self.storage = LocalJSONStorage(storage_path)
        self.index_file = os.path.join(storage_path, "aigvs_memory.pkl")

        self.max_connections = max_connections
        self.ef_construction = ef_construction

        self.engine = aigvs_rust.AIGVSEngine(embedding_dim=embedding_dim)

        self.root = DisjointFamily(
            node_id="root_node",
            embedding_dim=embedding_dim,
            pipeline_cfg=self._node_cfg(),
        )
        self.node_registry: Dict[str, BaseNode] = {"root_node": self.root}
        self.chunk_location_map: Dict[str, str] = {}

        # Hot-path payload cache: chunk_id → {"id", "text", "metadata"}.
        # Populated on every insert, never touched during query I/O.
        self.payload_cache: Dict[str, dict] = {}

        self.engine.register_branch("root_node", None, True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _node_cfg(self) -> dict:
        return {
            "max_connections": self.max_connections,
            "ef_construction": self.ef_construction,
        }

    def _register_leaf_in_engine(self, leaf_id: str, parent_id: str) -> None:
        """Single call-site for register_leaf — parameters always consistent."""
        self.engine.register_leaf(
            leaf_id,
            parent_id,
            max_connections=self.max_connections,
            ef_construction=self.ef_construction,
        )

    def _cache_payload(self, chunk: TextChunk) -> None:
        self.payload_cache[chunk.chunk_id] = {
            "id": chunk.chunk_id,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self) -> None:
        state = {
            "root": self.root,
            "node_registry": self.node_registry,
            "chunk_location_map": self.chunk_location_map,
            "payload_cache": self.payload_cache,
            "max_connections": self.max_connections,
            "ef_construction": self.ef_construction,
        }
        os.makedirs(self.storage_path, exist_ok=True)
        with open(self.index_file, "wb") as f:
            pickle.dump(state, f)
        print(f"[SYSTEM] Index state frozen and saved to {self.index_file}")

    def load_index(self) -> bool:
        if not os.path.exists(self.index_file):
            print(f"[SYSTEM] No index found at {self.index_file}. Starting fresh.")
            return False
        with open(self.index_file, "rb") as f:
            state = pickle.load(f)

        self.root = state["root"]
        self.node_registry = state["node_registry"]
        self.chunk_location_map = state["chunk_location_map"]
        self.payload_cache = state.get("payload_cache", {})
        self.max_connections = state.get("max_connections", self.max_connections)
        self.ef_construction = state.get("ef_construction", self.ef_construction)

        # If the cache is missing (old index format), rebuild it from storage.
        if not self.payload_cache:
            print("[SYSTEM] Payload cache missing — rebuilding from storage...")
            self._rebuild_payload_cache_from_storage()

        self.engine = aigvs_rust.AIGVSEngine(embedding_dim=self.embedding_dim)
        self._rebuild_engine_from_registry()
        print(f"[SYSTEM] Index loaded from {self.index_file}")
        return True

    def _rebuild_payload_cache_from_storage(self) -> None:
        """One-time migration: populate cache from storage for old index files."""
        for chunk_id in self.chunk_location_map:
            chunk = self.storage.read_chunk(chunk_id)
            if chunk is not None:
                self._cache_payload(chunk)

    def _rebuild_engine_from_registry(self) -> None:
        """BFS over the Python tree to reconstruct the Rust engine state."""
        queue: deque = deque(["root_node"])
        visited: set = set()

        while queue:
            node_id = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)

            node = self.node_registry.get(node_id)
            if node is None:
                continue

            if isinstance(node, DisjointFamily):
                self.engine.register_branch(node_id, node.parent_id, node_id == "root_node")
                for child_id, child_node in node.children.items():
                    if child_node.centroid_vector is not None:
                        self.engine.update_attractor(node_id, child_id, child_node.centroid_vector)
                    queue.append(child_id)

            elif isinstance(node, JointFamily):
                self._register_leaf_in_engine(node_id, node.parent_id)
                # Re-insert in original order — order matters for NSW graph quality.
                for i, chunk_id in enumerate(node.chunk_ids):
                    if i in node.deleted_indices:
                        continue
                    # Use cached embedding if available, else fall back to storage.
                    payload = self.payload_cache.get(chunk_id)
                    if payload is not None and "embedding" in payload:
                        embedding = np.array(payload["embedding"], dtype=np.float32)
                    else:
                        chunk = self.storage.read_chunk(chunk_id)
                        if chunk is None:
                            continue
                        embedding = chunk.embedding
                    self.engine.native_insert_to_leaf(node_id, chunk_id, embedding)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_hierarchical_json(self, json_data: List[dict]) -> None:
        """Bulk ingestion with hierarchical attractor discovery.

        Three-stage pipeline:
          1. Discover macro semantic basins via farthest-point + spherical k-means.
          2. Assign documents to basins.
          3. Insert through each trunk's normal path so local Hopfield
             bifurcation handles sub-community splits automatically.

        Result: root → K trunks → ~K leaves each, instead of root → 1 trunk → 1000+ leaves.
        """
        if not json_data:
            return

        # --- Build all chunks + embedding matrix ---
        chunks = []
        embeddings = []
        for entry in json_data:
            emb = np.array(entry["embedding"], dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            chunks.append(TextChunk(
                chunk_id=entry["id"],
                text=entry["text"],
                embedding=emb,
                metadata=entry.get("metadata", {}),
            ))
            embeddings.append(emb)

        X = np.array(embeddings, dtype=np.float32)  # (N, D)
        N = len(chunks)
        print(f"[AIGVS] Converted dicts to chunks. N={N}", flush=True)
        
        # --- Small corpus: skip macro discovery, insert directly ---
        num_leaves_est = N / self.root.max_leaf_capacity
        if num_leaves_est < 4:
            for chunk in chunks:
                self.root.insert(chunk, self.storage, self)
            return

        # --- Stage 1: Macro-Attractor Discovery ---
        print(f"[AIGVS] Discovering macro attractors...", flush=True)
        t_start = time.time()
        num_trunks = min(32, max(4, int(np.sqrt(num_leaves_est))))
        print(f"[AIGVS] Discovering {num_trunks} macro semantic basins from {N} documents...", flush=True)
        macro_centroids = self._discover_macro_attractors(X, num_trunks)
        print(f"[AIGVS] Discovered macro attractors: {time.time()-t_start:.2f}s", flush=True)

        # --- Stage 2: Assign docs to trunks (Polysemic) ---
        t_start = time.time()
        sims = X @ macro_centroids.T
        max_sims = np.max(sims, axis=1)

        trunk_chunks: List[List[TextChunk]] = [[] for _ in range(len(macro_centroids))]
        polysemy_count = 0
        for i, chunk in enumerate(chunks):
            threshold = max_sims[i] - 0.08
            assigned = np.where(sims[i] >= threshold)[0]
            for t_idx in assigned:
                trunk_chunks[t_idx].append(chunk)
            if len(assigned) > 1:
                polysemy_count += 1
                
        print(f"[AIGVS] Polysemic assignments: {polysemy_count} chunks span multiple macro trunks")

        # --- Stage 3: Create trunk branches and insert ---
        for trunk_idx, t_chunks in enumerate(trunk_chunks):
            if not t_chunks:
                continue

            trunk_id = f"trunk_{trunk_idx}"
            trunk = DisjointFamily(
                node_id=trunk_id,
                embedding_dim=self.embedding_dim,
                parent_id="root_node",
                pipeline_cfg=self._node_cfg(),
            )
            self.root.children[trunk_id] = trunk
            self.engine.register_branch(trunk_id, "root_node", False)
            self.engine.update_attractor(
                "root_node", trunk_id, macro_centroids[trunk_idx]
            )
            trunk.centroid_vector = macro_centroids[trunk_idx]
            self.node_registry[trunk_id] = trunk

            # BATCH INSERT OPTIMIZATION (Bypass python loop & FFI overhead)
            new_leaf_id = f"{trunk_id}_leaf_0"
            from aigvs.engine.nodes import JointFamily
            new_leaf = JointFamily(
                node_id=new_leaf_id,
                parent_id=trunk_id,
                pipeline_cfg=self._node_cfg(),
            )
            self._register_leaf_in_engine(new_leaf_id, trunk_id)
            trunk.children[new_leaf_id] = new_leaf
            self.engine.update_attractor(trunk_id, new_leaf_id, macro_centroids[trunk_idx])
            
            t_start = time.time()
            c_ids = []
            embs = []
            for chunk in t_chunks:
                self.storage.write_chunk(chunk)
                new_leaf.chunk_ids.append(chunk.chunk_id)
                self.chunk_location_map[chunk.chunk_id] = new_leaf_id
                self._cache_payload(chunk)
                c_ids.append(chunk.chunk_id)
                embs.append(chunk.embedding)
            
            vectors_array = np.vstack(embs).astype(np.float32)
            print(f"[AIGVS] Prepared {len(c_ids)} chunks for trunk {trunk_idx}: {time.time()-t_start:.2f}s", flush=True)
            
            t_start = time.time()
            self.engine.native_batch_insert_to_leaf(new_leaf_id, c_ids, vectors_array)
            print(f"[AIGVS] native_batch_insert_to_leaf ({len(c_ids)}): {time.time()-t_start:.2f}s", flush=True)
            
            new_leaf.centroid_vector = trunk.centroid_vector
            trunk.chunk_count = len(t_chunks)
            self.node_registry[new_leaf_id] = new_leaf

        # --- Final pass: update root attractors to reflect actual content ---
        for trunk_id, trunk in self.root.children.items():
            if trunk.centroid_vector is not None:
                self.engine.update_attractor(
                    "root_node", trunk_id, trunk.centroid_vector
                )

        stats = self.stats()
        print(
            f"[AIGVS] Tree built: {stats['branches']} branches, "
            f"{stats['leaves']} leaves, {stats['indexed_chunks']} chunks"
        )

    def _discover_macro_attractors(
        self, X: np.ndarray, num_trunks: int
    ) -> np.ndarray:
        """Fast macro-attractor discovery: farthest-point seeding + spherical k-means."""
        # --- Farthest-point sampling for maximally separated seeds ---
        avg = np.mean(X, axis=0)
        avg_norm = np.linalg.norm(avg)
        if avg_norm > 0:
            avg /= avg_norm

        seeds = [X[int(np.argmax(X @ avg))].copy()]

        for _ in range(num_trunks - 1):
            seed_matrix = np.array(seeds)
            sims = X @ seed_matrix.T          # (N, K)
            max_sim = np.max(sims, axis=1)    # (N,)
            seeds.append(X[int(np.argmin(max_sim))].copy())

        centroids = np.array(seeds, dtype=np.float32)

        # --- Spherical k-means refinement (5 iterations) ---
        for _ in range(5):
            scores = X @ centroids.T
            assignments = np.argmax(scores, axis=1)
            for k in range(len(centroids)):
                mask = assignments == k
                if mask.any():
                    c = np.mean(X[mask], axis=0)
                    norm = np.linalg.norm(c)
                    if norm > 0:
                        centroids[k] = c / norm

        return centroids

    def insert_chunk(self, chunk: TextChunk) -> None:
        self.root.insert(chunk, self.storage, self)

    # ------------------------------------------------------------------
    # Query  — zero storage I/O on the hot path
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        max_beam: int = 10,
        target_confidence: float = 0.95,
        beta: float = 5.0,
        ef_search: Optional[int] = None,
    ) -> List[dict]:
        """
        Parameters
        ----------
        ef_search : int, optional
            NSW candidate pool per leaf at query time.
            Default: max(top_k * 4, 50).
            This is the cheapest recall lever — raise it before anything else.
        beta : float
            Router softmax temperature.
            Lower (~2) = exploratory, broader beam.
            Higher (~20) = greedy, fastest path to the best leaf.
        """
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        ef = ef_search if ef_search is not None else max(top_k * 8, 100)

        # Single PyO3 crossing — Rust does all routing + parallel leaf search.
        matched_ids: List[str] = self.engine.unified_search(
            query_embedding,
            top_k=top_k,
            max_beam=max_beam,
            target_confidence=target_confidence,
            beta=beta,
            ef_search=ef,
        )

        if not matched_ids:
            return []

        # O(1) dict lookup per result — no I/O, no deserialization.
        return [
            self.payload_cache[cid]
            for cid in matched_ids
            if cid in self.payload_cache
        ]

    # ------------------------------------------------------------------
    # Deletion & Update
    # ------------------------------------------------------------------

    def delete_chunk(self, chunk_id: str) -> bool:
        leaf_id = self.chunk_location_map.get(chunk_id)
        if not leaf_id:
            return False

        self.storage.delete_chunk(chunk_id)
        self.engine.native_mark_deleted(leaf_id, chunk_id)
        self.payload_cache.pop(chunk_id, None)

        leaf_node: Optional[JointFamily] = self.node_registry.get(leaf_id)
        parent_node: Optional[DisjointFamily] = (
            self.node_registry.get(leaf_node.parent_id) if leaf_node else None
        )
        if leaf_node and parent_node:
            leaf_node.mark_deleted(chunk_id, parent_node, self.engine)

        self.chunk_location_map.pop(chunk_id, None)
        return True

    def update_chunk(self, chunk: TextChunk) -> None:
        self.delete_chunk(chunk.chunk_id)
        self.root.insert(chunk, self.storage, self)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "branches": sum(1 for n in self.node_registry.values() if isinstance(n, DisjointFamily)),
            "leaves": sum(1 for n in self.node_registry.values() if isinstance(n, JointFamily)),
            "indexed_chunks": len(self.chunk_location_map),
            "payload_cache_size": len(self.payload_cache),
            "embedding_dim": self.embedding_dim,
            "max_connections": self.max_connections,
            "ef_construction": self.ef_construction,
        }