from typing import Dict, Optional
from src.aigvs.core.models import TextChunk

class InMemoryStorage:
    def __init__(self):
        # The entire "hard drive" is just a high-speed hash map in RAM
        self.memory_store: Dict[str, dict] = {}

    def write_chunk(self, chunk: TextChunk) -> None:
        # Instantly dump the chunk data to RAM (Now including the embedding!)
        self.memory_store[chunk.chunk_id] = {
            "id": chunk.chunk_id,
            "text": chunk.text,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding  # <--- FIX 1
        }

    def read_chunk(self, chunk_id: str) -> Optional[TextChunk]:
        data = self.memory_store.get(chunk_id)
        if data:
            return TextChunk(
                chunk_id=data["id"],
                text=data["text"],
                metadata=data["metadata"],
                embedding=data["embedding"]  # <--- FIX 2
            )
        return None

    def delete_chunk(self, chunk_id: str) -> None:
        if chunk_id in self.memory_store:
            del self.memory_store[chunk_id]
