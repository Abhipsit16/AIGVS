import numpy as np
from typing import Dict, Optional

class TextChunk:
    """The granular unit of text and vector data stored inside the AIGVS ecosystem."""
    def __init__(self, chunk_id: str, text: str, embedding: np.ndarray, metadata: Optional[Dict] = None):
        self.chunk_id = chunk_id
        self.text = text
        # Enforce float32 for fast matrix operations
        self.embedding = embedding.astype(np.float32)
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "embedding": self.embedding.tolist(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TextChunk':
        return cls(
            chunk_id=data["chunk_id"],
            text=data["text"],
            embedding=np.array(data["embedding"], dtype=np.float32),
            metadata=data["metadata"]
        )