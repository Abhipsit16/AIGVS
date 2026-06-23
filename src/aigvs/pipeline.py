import os
import pickle
import numpy as np
from typing import Dict, List
from src.aigvs.core.models import TextChunk
from src.aigvs.storage import LocalJSONStorage
from src.aigvs.engine.nodes import DisjointFamily, BaseNode

class AIGVSPipeline:
    def __init__(self, embedding_dim: int, storage_path: str = "./aigvs_data"):
        self.embedding_dim = embedding_dim
        self.storage_path = storage_path
        self.storage = LocalJSONStorage(storage_path)
        self.root = DisjointFamily(node_id="root_node", embedding_dim=embedding_dim)
        
        self.node_registry: Dict[str, BaseNode] = {"root_node": self.root}
        self.chunk_location_map: Dict[str, str] = {} 
        
        # The physical file where the mathematical graph will be frozen
        self.index_file = os.path.join(storage_path, "aigvs_memory.pkl")

    def save_index(self) -> None:
        """Serializes the routing matrices and NSW graphs to disk."""
        state = {
            "root": self.root,
            "node_registry": self.node_registry,
            "chunk_location_map": self.chunk_location_map
        }
        with open(self.index_file, "wb") as f:
            pickle.dump(state, f)
        print(f"[SYSTEM] Index state frozen and saved to {self.index_file}")

    def load_index(self) -> bool:
        """Wakes the mathematical index back up from disk into RAM."""
        if not os.path.exists(self.index_file):
            print("[SYSTEM] No existing index found. Starting fresh.")
            return False
            
        with open(self.index_file, "rb") as f:
            state = pickle.load(f)
            self.root = state["root"]
            self.node_registry = state["node_registry"]
            self.chunk_location_map = state["chunk_location_map"]
        print("[SYSTEM] Index successfully loaded into memory.")
        return True


    def ingest_hierarchical_json(self, json_data: List[Dict]) -> None:
        for chunk_entry in json_data:
            chunk = TextChunk(
                chunk_id=chunk_entry["id"],
                text=chunk_entry["text"],
                embedding=np.array(chunk_entry["embedding"], dtype=np.float32),
                metadata=chunk_entry.get("metadata", {})
            )
            self.root.insert(chunk, self.storage, self)

    def query(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        norm = np.linalg.norm(query_embedding)
        if norm > 0: query_embedding = query_embedding / norm
            
        matched_ids = self.root.query(query_embedding, top_k, self.storage)
        
        payloads = []
        for cid in matched_ids:
            chunk = self.storage.read_chunk(cid)
            if chunk:
                payloads.append({"id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata})
        return payloads

    def delete_chunk(self, chunk_id: str) -> bool:
        leaf_id = self.chunk_location_map.get(chunk_id)
        if not leaf_id: return False 
            
        self.storage.delete_chunk(chunk_id)
        
        leaf_node = self.node_registry.get(leaf_id)
        parent_node = self.node_registry.get(leaf_node.parent_id) if leaf_node else None
        
        if leaf_node and parent_node:
            leaf_node.mark_deleted(chunk_id, parent_node)
            
        self.chunk_location_map.pop(chunk_id, None)
        return True

    def update_chunk(self, chunk: TextChunk) -> None:
        self.delete_chunk(chunk.chunk_id)
        self.root.insert(chunk, self.storage, self)