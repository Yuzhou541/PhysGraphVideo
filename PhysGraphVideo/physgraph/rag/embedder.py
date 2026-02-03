# File: physgraph/rag/embedder.py
# Safe embedder with graceful fallback:
# 1) Try to use SentenceTransformer ("all-MiniLM-L6-v2")
# 2) If any import/init error occurs (e.g., numpy binary issue), fall back to a
#    lightweight hashing-based text embedder implemented in pure Python + PyTorch.

from __future__ import annotations
import os
import re
import hashlib
from typing import List, Optional

import torch
import torch.nn.functional as F


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _stable_hash_to_index(s: str, dim: int) -> int:
    # Stable across runs (unlike built-in hash). Use blake2b for speed & stability.
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()
    return int(h, 16) % dim


class _HashingEmbedder:
    """
    Lightweight fallback embedder:
    - Character/word hashing (unigram + bigram over tokens) into fixed dim.
    - L2-normalized output. No external deps (only torch).
    """
    def __init__(self, dim: int = 2048, ngram: int = 2, device: Optional[str] = None):
        self.dim = int(dim)
        self.ngram = int(ngram)
        self.device = device or "cpu"

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        vecs = torch.zeros((len(texts), self.dim), dtype=torch.float32, device=self.device)
        for i, t in enumerate(texts):
            toks = _tokenize(t)
            # unigrams
            for tok in toks:
                j = _stable_hash_to_index("uni:" + tok, self.dim)
                vecs[i, j] += 1.0
            # bigrams (if enabled)
            if self.ngram >= 2 and len(toks) >= 2:
                for a, b in zip(toks[:-1], toks[1:]):
                    j = _stable_hash_to_index("bi:" + a + "_" + b, self.dim)
                    vecs[i, j] += 1.0
        vecs = F.normalize(vecs, dim=1, eps=1e-12)
        return vecs


class _STEmbedder:
    """
    SentenceTransformer wrapper. Imported lazily inside __init__ to avoid
    top-level import failures when numpy/transformers are misinstalled.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError(f"SentenceTransformer unavailable: {e}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SentenceTransformer: {e}")

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        # ST returns numpy by default; request torch tensors if possible.
        try:
            emb = self.model.encode(
                texts, convert_to_tensor=True, device=self.device, normalize_embeddings=True
            )
            return emb.float()
        except TypeError:
            # Older versions may not support convert_to_tensor; fall back & convert
            vec = self.model.encode(texts, normalize_embeddings=True)
            if hasattr(vec, "to"):
                return vec.to(self.device).float()
            # numpy fallback (should not happen in our guarded path)
            t = torch.tensor(vec, device=self.device, dtype=torch.float32)
            t = F.normalize(t, dim=1, eps=1e-12)
            return t


def get_embedder(
    prefer: str = "auto",
    device: Optional[str] = None,
    hash_dim: int = 2048,
    hash_ngram: int = 2,
):
    """
    Create an embedder.

    Args:
        prefer: "st" to force SentenceTransformer, "hash" to force fallback,
                "auto" to try ST then fallback on failure. Can also be set via
                env PHYSGRAPH_RAG_EMBEDDER.
        device: "cuda" or "cpu". If None, auto-detect.
        hash_dim: dimension for hashing embedder fallback.
        hash_ngram: max n-gram (1=unigram, 2=uni+bi).

    Returns:
        An object with encode(List[str]) -> torch.Tensor [N, D], L2-normalized.
    """
    prefer = (os.getenv("PHYSGRAPH_RAG_EMBEDDER", prefer) or "auto").lower()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if prefer == "hash":
        return _HashingEmbedder(dim=hash_dim, ngram=hash_ngram, device=device)

    if prefer in ("st", "auto"):
        try:
            return _STEmbedder(device=device)
        except Exception:
            if prefer == "st":
                # explicit 'st' should propagate the error
                raise

    # fallback
    return _HashingEmbedder(dim=hash_dim, ngram=hash_ngram, device=device)
