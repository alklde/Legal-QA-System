"""ChromaDB向量检索层（案例语义检索）"""

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _fake_embedding(text: str, dim: int = 384) -> list:
    import hashlib

    h = hashlib.sha512(text.encode("utf-8")).digest()
    vec = []
    for i in range(0, len(h) - 1 and len(h) or 1, 2):
        if len(vec) >= dim:
            break
        vec.append(((h[i] % 255) - 128) / 128.0)
    while len(vec) < dim:
        vec.append(0.0)
    return vec[:dim]


class FakeEmbeddingFunction:
    @staticmethod
    def name() -> str:
        return "fake_hash_embedding"

    def __call__(self, input: list) -> list:
        return [_fake_embedding(t) for t in input]

    def embed_query(self, input: list) -> list:
        return self.__call__(input)


class ChromaCaseStore:
    """基于ChromaDB的案例向量检索"""

    def __init__(
        self,
        collection_name: str = "legal_cases",
        persist_directory: str = "data/vector_db",
        embedding_fn=None,
        use_openai: bool = False,
        openai_api_key: str = None,
    ):
        import chromadb
        from chromadb.config import Settings

        self.persist_directory = persist_directory
        self.collection_name = collection_name

        if use_openai and openai_api_key:
            self._embed_fn = self._build_openai_embed(openai_api_key)
        elif embedding_fn is not None:
            self._embed_fn = embedding_fn
        else:
            self._embed_fn = FakeEmbeddingFunction()

        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
        )

    def _build_openai_embed(self, api_key: str):
        try:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

            return OpenAIEmbeddingFunction(
                api_key=api_key, model_name="text-embedding-3-small"
            )
        except Exception:
            return FakeEmbeddingFunction()

    def add_case(self, case_id: str, fact: str, metadata: dict = None):
        if not fact or len(fact) < 5:
            return
        try:
            self._collection.upsert(
                ids=[case_id],
                documents=[fact[:2000]],
                metadatas=([metadata] if metadata else None),
            )
        except Exception as e:
            logger.warning(f"Chroma add_case failed: {e}")

    def add_cases_batch(self, records: List[dict]):
        if not records:
            return
        batch_ids, docs, metas = [], [], []
        for rec in records:
            cid = rec.get("case_id")
            fact = rec.get("fact", "")
            if not cid or not fact:
                continue
            batch_ids.append(cid)
            docs.append(fact[:2000])
            meta = rec.get("metadata")
            metas.append(meta if meta else None)
        try:
            self._collection.upsert(ids=batch_ids, documents=docs, metadatas=metas)
        except Exception as e:
            logger.warning(f"Chroma batch upsert failed: {e}")

    def search(self, query: str, n_results: int = 5) -> List[dict]:
        if not query or n_results <= 0:
            return []
        try:
            res = self._collection.query(query_texts=[query], n_results=n_results)
        except Exception as e:
            logger.warning(f"Chroma query failed: {e}")
            return []
        results = []
        docs = res.get("documents", [[]])[0]
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        for i, doc in enumerate(docs):
            dist = distances[i] if i < len(distances) else 1.0
            score = max(0.0, 1.0 - dist / 2.0) if dist is not None else 0.0
            meta = metadatas[i] if i < len(metadatas) else {}
            results.append(
                {
                    "case_id": ids[i] if i < len(ids) else "",
                    "fact": doc[:500],
                    "relevance": round(score, 4),
                    "metadata": meta,
                }
            )
        return results

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:
            return 0

    def delete(self, case_id: str):
        try:
            self._collection.delete(ids=[case_id])
        except Exception:
            pass