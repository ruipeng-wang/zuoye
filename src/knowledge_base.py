"""
知识库构建模块：
- 文本分块（滑动窗口）
- 向量嵌入 + ChromaDB 存储
- BM25 关键词检索
- 混合检索
"""
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import jieba

from .config import get_config

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """知识库：chunking + embedding + 混合检索"""

    def __init__(self):
        cfg = get_config()
        self.chunk_size = cfg.chunk_size
        self.chunk_overlap = cfg.chunk_overlap
        self.top_k = cfg.retrieval_top_k
        self.bm25_weight = cfg.bm25_weight

        # ChromaDB
        persist_dir = str(cfg.project_root / cfg.chroma_persist_dir)
        self._client = chromadb.PersistentClient(path=persist_dir)
        try:
            self._collection = self._client.get_collection("pdf_docs")
            logger.info("加载已存在的 ChromaDB 集合")
        except Exception:
            self._collection = None

        # Embedding function
        api_key = cfg.openai_api_key
        base_url = cfg.openai_base_url
        if api_key:
            ef_kwargs = {"api_key": api_key, "model_name": cfg.embedding_model}
            if base_url:
                ef_kwargs["api_base"] = base_url
            self._ef = embedding_functions.OpenAIEmbeddingFunction(**ef_kwargs)
        else:
            # fallback: 使用本地 sentence-transformers
            self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="paraphrase-multilingual-MiniLM-L12-v2"
            )

        # BM25 索引
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: List[Dict[str, Any]] = []

    def _tokenize(self, text: str) -> List[str]:
        return list(jieba.cut(text))

    @staticmethod
    def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
        """滑动窗口分块"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def build(self, records: List[Dict[str, Any]], force_rebuild: bool = False):
        """
        构建知识库：分块 → 向量化 → 存储
        """
        if self._collection is not None and not force_rebuild:
            logger.info("知识库已存在，跳过构建。使用 force_rebuild=True 强制重建")
            return

        cfg = get_config()
        persist_dir = str(cfg.project_root / cfg.chroma_persist_dir)

        # 删除旧集合
        try:
            self._client.delete_collection("pdf_docs")
        except Exception:
            pass

        self._collection = self._client.create_collection(
            "pdf_docs",
            embedding_function=self._ef,
        )

        all_chunks = []
        all_meta = []
        all_ids = []
        bm25_corpus = []
        chunk_idx = 0

        for record in records:
            chunks = self._chunk_text(
                record["text"], self.chunk_size, self.chunk_overlap
            )
            for chunk in chunks:
                chunk_id = f"chunk_{chunk_idx}"
                meta = {
                    "page": record["page"],
                    "source": record["source"],
                    "chunk_type": record.get("chunk_type", "text"),
                }
                all_ids.append(chunk_id)
                all_chunks.append(chunk)
                all_meta.append(meta)
                bm25_corpus.append(chunk)
                chunk_idx += 1

        if all_chunks:
            # 百炼/某些 API 对 Embedding 有 batch size 限制（通常 ≤10）
            # 分批添加，每批 10 条
            BATCH_SIZE = 10
            for i in range(0, len(all_chunks), BATCH_SIZE):
                batch_ids = all_ids[i:i + BATCH_SIZE]
                batch_chunks = all_chunks[i:i + BATCH_SIZE]
                batch_meta = all_meta[i:i + BATCH_SIZE]
                self._collection.add(
                    ids=batch_ids,
                    documents=batch_chunks,
                    metadatas=batch_meta,
                )
                logger.info(f"已添加批次 {i // BATCH_SIZE + 1}: {len(batch_ids)} 条 chunks")
            logger.info(f"已添加 {len(all_chunks)} 条 chunks 到 ChromaDB")

        # 构建 BM25
        tokenized_corpus = [self._tokenize(doc) for doc in bm25_corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._bm25_docs = [
            {"id": all_ids[i], "doc": bm25_corpus[i], "meta": all_meta[i]}
            for i in range(len(bm25_corpus))
        ]

    def _load_bm25_if_needed(self):
        if self._bm25 is not None:
            return
        if self._collection is None:
            cfg = get_config()
            persist_dir = str(cfg.project_root / cfg.chroma_persist_dir)
            self._client = chromadb.PersistentClient(path=persist_dir)
            try:
                self._collection = self._client.get_collection("pdf_docs")
            except Exception:
                raise RuntimeError("知识库未构建，请先调用 build()")

        results = self._collection.get()
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        ids = results.get("ids", [])
        if docs:
            tokenized_corpus = [self._tokenize(doc) for doc in docs]
            self._bm25 = BM25Okapi(tokenized_corpus)
            self._bm25_docs = [
                {"id": ids[i], "doc": docs[i], "meta": metas[i]}
                for i in range(len(docs))
            ]

    def search(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        混合检索：语义向量 + BM25 关键词
        """
        self._load_bm25_if_needed()
        k = top_k or self.top_k

        # 语义检索
        semantic_results = self._collection.query(
            query_texts=[query], n_results=k * 2
        )

        # BM25 检索
        tokenized_query = self._tokenize(query)
        bm25_scores = self._bm25.get_scores(tokenized_query)
        bm25_ranked = sorted(
            enumerate(bm25_scores), key=lambda x: x[1], reverse=True
        )[: k * 2]

        # 混合排序
        combined = {}
        # 语义分数归一化
        sem_docs = semantic_results["documents"][0]
        sem_metas = semantic_results["metadatas"][0]
        sem_dists = semantic_results.get("distances", [[1.0] * len(sem_docs)])[0]
        max_dist = max(sem_dists) if sem_dists else 1.0

        for i, doc in enumerate(sem_docs):
            norm_score = 1.0 - (sem_dists[i] / max_dist if max_dist > 0 else 0)
            combined[doc[:80]] = {
                "doc": doc,
                "meta": sem_metas[i] if i < len(sem_metas) else {},
                "sem_score": norm_score,
                "bm25_score": 0.0,
            }

        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1.0
        for idx, score in bm25_ranked:
            doc = self._bm25_docs[idx]
            key = doc["doc"][:80]
            norm_bm25 = score / max_bm25 if max_bm25 > 0 else 0
            if key in combined:
                combined[key]["bm25_score"] = norm_bm25
            else:
                combined[key] = {
                    "doc": doc["doc"],
                    "meta": doc["meta"],
                    "sem_score": 0.0,
                    "bm25_score": norm_bm25,
                }

        # 加权融合
        for item in combined.values():
            item["score"] = (
                (1 - self.bm25_weight) * item["sem_score"]
                + self.bm25_weight * item["bm25_score"]
            )

        sorted_results = sorted(
            combined.values(), key=lambda x: x["score"], reverse=True
        )
        return sorted_results[:k]