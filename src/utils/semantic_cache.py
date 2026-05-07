import json
import hashlib
import math
import time
import os
from typing import Optional

try:
    import redis as redis_lib
except ImportError:
    redis_lib = None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class SemanticCache:
    def __init__(
        self,
        embedding_func,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        similarity_threshold: float = 0.90,
        ttl: int = 3600,
    ):
        if redis_lib is None:
            raise ImportError("redis package is required. Run: pip install redis")
        self.embedding_func = embedding_func
        self.client = redis_lib.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.threshold = similarity_threshold
        self.ttl = ttl
        self._index_key = "semantic_cache:index"

    def lookup(self, query: str) -> Optional[str]:
        query_embedding = self.embedding_func.embed_query(query)
        entry_ids = self.client.smembers(self._index_key)
        best_score = 0.0
        best_response = None
        for entry_id in entry_ids:
            entry_json = self.client.get(f"semantic_cache:entry:{entry_id}")
            if not entry_json:
                continue
            entry = json.loads(entry_json)
            score = _cosine_similarity(query_embedding, entry["embedding"])
            if score > best_score:
                best_score = score
                best_response = entry["response"]
        if best_score >= self.threshold:
            return best_response
        return None

    def store(self, query: str, response: str):
        query_embedding = self.embedding_func.embed_query(query)
        entry_id = hashlib.md5(query.encode()).hexdigest()
        entry = json.dumps({
            "embedding": query_embedding,
            "query": query,
            "response": response,
            "timestamp": time.time(),
        })
        self.client.setex(f"semantic_cache:entry:{entry_id}", self.ttl, entry)
        self.client.sadd(self._index_key, entry_id)

    def clear(self):
        entry_ids = self.client.smembers(self._index_key)
        for entry_id in entry_ids:
            self.client.delete(f"semantic_cache:entry:{entry_id}")
        self.client.delete(self._index_key)
