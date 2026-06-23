import os
import json
import re 
from typing import Optional
from src.aigvs.core.models import TextChunk

class LocalJSONStorage:
    """Manages the raw text payload records on disk, keeping the memory index thin."""
    def __init__(self, base_dir: str = "./aigvs_data"):
        self.base_dir = base_dir
        self.chunks_dir = os.path.join(base_dir, "chunks")
        os.makedirs(self.chunks_dir, exist_ok=True)

    def _get_path(self, chunk_id: str) -> str:
        # Replaces illegal OS filename characters (\ / * ? : " < > |) with an underscore
        safe_filename = re.sub(r'[\\/*?:"<>|]', '_', chunk_id)
        return os.path.join(self.base_dir, f"{safe_filename}.json")

    def write_chunk(self, chunk: TextChunk) -> None:
        with open(self._get_path(chunk.chunk_id), "w", encoding="utf-8") as f:
            json.dump(chunk.to_dict(), f, indent=2)

    def read_chunk(self, chunk_id: str) -> Optional[TextChunk]:
        path = self._get_path(chunk_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return TextChunk.from_dict(json.load(f))

    def delete_chunk(self, chunk_id: str) -> bool:
        path = self._get_path(chunk_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False