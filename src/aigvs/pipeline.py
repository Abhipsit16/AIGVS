import os
import pickle
import numpy as np
from collections import deque
from typing import Dict, List, Optional
from src.aigvs.core.models import TextChunk
from src.aigvs.storage import LocalJSONStorage
from src.aigvs.engine.nodes import DisjointFamily, BaseNode, JointFamily
import time
from src.aigvs.profiler import add, PROFILE, COUNTS
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
        max_connections: int = 16,
        ef_construction: int = 100,
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
        for entry in json_data:
            chunk = TextChunk(
                chunk_id=entry["id"],
                text=entry["text"],
                embedding=np.array(entry["embedding"], dtype=np.float32),
                metadata=entry.get("metadata", {}),
            )
            self.root.insert(chunk, self.storage, self)

    def insert_chunk(self, chunk: TextChunk) -> None:
        self.root.insert(chunk, self.storage, self)

    # ------------------------------------------------------------------
    # Query  — zero storage I/O on the hot path
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        max_beam: int = 5,           # Tighten the default beam
        target_confidence: float = 0.99, 
        beta: float = 18.0,          # THE CLIFF
        ef_search: Optional[int] = None,
    ) -> List[dict]:
        
        t_total_start = time.perf_counter()

        # 1. Normalization Profiling
        t_norm_start = time.perf_counter()
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        add("py_01_normalization_ms", (time.perf_counter() - t_norm_start) * 1000)

        ef = ef_search if ef_search is not None else max(top_k * 4, 50)

        # 2. Rust Engine Profiling
        t_rust_call = time.perf_counter()
        matched_ids, rust_telemetry = self.engine.unified_search(
            query_embedding,
            top_k=top_k,
            max_beam=max_beam,
            target_confidence=target_confidence,
            beta=beta,
            ef_search=ef,
        )
        add("rust_unified_search_call", (time.perf_counter() - t_rust_call) * 1000)

        # Ingest Rust telemetry into global profile
        for key, value in rust_telemetry.items():
            if key in ["attractors_visited_per_query", "leaves_visited_per_query", "total_candidates_scored"]:
                PROFILE[key] += value  # Pure counters: Add to sum, don't increment calls
            else:
                add(key, value) # Timings: Add to sum, increment calls
        if not matched_ids:
            return []

        # 3. Payload Cache Lookup Profiling
        t_cache_start = time.perf_counter()
        results = [
            self.payload_cache[cid]
            for cid in matched_ids
            if cid in self.payload_cache
        ]
        add("py_02_payload_cache_ms", (time.perf_counter() - t_cache_start) * 1000)
        
        add("py_total_query_ms", (time.perf_counter() - t_total_start) * 1000)
        
        return results

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