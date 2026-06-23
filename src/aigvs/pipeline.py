import numpy as np
from typing import Dict, List
from src.aigvs.core.models import TextChunk
from src.aigvs.storage.local_json import LocalJSONStorage
from src.aigvs.engine.nodes import DisjointFamily, BaseNode

class AIGVSPipeline:
    """The master framework manager handling clear write pipelines and decoupled query paths."""
    def __init__(self, embedding_dim: int, storage_path: str = "./aigvs_data"):
        self.embedding_dim = embedding_dim
        self.storage = LocalJSONStorage(storage_path)
        self.root = DisjointFamily(node_id="root_node", embedding_dim=embedding_dim)
        
        # O(1) Global Registries for fast CRUD routing
        self.node_registry: Dict[str, BaseNode] = {"root_node": self.root}
        self.chunk_location_map: Dict[str, str] = {} 

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