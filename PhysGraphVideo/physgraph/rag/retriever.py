# File: physgraph/rag/retriever.py
# GraphRAG that uses the safe get_embedder() with automatic fallback to hashing.
# No numpy / transformers hard dependency at import time.

from __future__ import annotations
import json
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F

from .embedder import get_embedder


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except Exception:
                # tolerate minor json issues
                continue
    return docs


def _doc_text(d: Dict[str, Any]) -> str:
    # flexible field picking
    fields = []
    for k in ("title", "name", "text", "content", "desc", "description"):
        v = d.get(k, None)
        if v:
            fields.append(str(v))
    if not fields and d:
        # last resort: dump selected kvs
        fields.append(" ".join(f"{k}:{v}" for k, v in d.items() if isinstance(v, (str, int, float))))
    return " | ".join(fields) if fields else ""


class GraphRAG:
    """
    Minimal Graph RAG for small knowledge graphs exported as JSONL.
    - Embeds node texts once.
    - On query(), returns top-k texts as conditioning snippets.
    - Embedding backend chosen by embedder.get_embedder() with graceful fallback.
    """

    def __init__(
        self,
        jsonl_path: str,
        topk: int = 4,
        embedder_prefer: str = "auto",  # "auto" | "st" | "hash"
        device: Optional[str] = None,
        hash_dim: int = 2048,
        hash_ngram: int = 2,
    ):
        self.topk = int(topk)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.docs = _load_jsonl(jsonl_path)

        # Prepare texts
        self.texts: List[str] = [_doc_text(d) for d in self.docs]
        # Build embedder (auto -> try ST, else fallback hashing)
        self.embedder = get_embedder(
            prefer=embedder_prefer, device="cpu",  # keep on CPU; tiny graphs are cheap
            hash_dim=hash_dim, hash_ngram=hash_ngram
        )
        with torch.no_grad():
            self.doc_emb = self.embedder.encode(self.texts)  # [N, D], L2-normalized
            if self.doc_emb.ndim != 2:
                raise RuntimeError("Embedder returned invalid shape for doc embeddings.")

    @torch.no_grad()
    def query(self, q: str, topk: Optional[int] = None) -> List[str]:
        k = int(topk or self.topk)
        qv = self.embedder.encode([q])  # [1, D]
        if qv.ndim != 2:
            raise RuntimeError("Embedder returned invalid shape for query embedding.")
        # Cosine similarity: embeddings are already L2-normalized
        sims = torch.matmul(qv, self.doc_emb.T).squeeze(0)  # [N]
        if sims.numel() == 0:
            return []
        k = min(k, sims.numel())
        vals, idx = torch.topk(sims, k=k, largest=True, sorted=True)
        return [self.texts[int(i)] for i in idx.tolist()]

    @torch.no_grad()
    def query_many(self, queries: List[str], topk: Optional[int] = None) -> List[List[str]]:
        return [self.query(q, topk=topk) for q in queries]
