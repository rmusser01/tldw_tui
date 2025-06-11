# Embeddings_Lib.py
#
from __future__ import annotations
"""Adaptive embedding factory for long‑running desktop or TUI apps.

Changes vs. the first iteration
────────────────────────────────
• **Idle eviction + LRU**: keep at most ``max_cached`` models in RAM and
  automatically unload any Hugging‑Face model that hasn’t been used for
  ``idle_seconds``.
• Hugging‑Face models are wrapped in a lightweight class that exposes
  ``embed()`` and ``close()`` so GPU / CPU memory is released.
• OpenAI provider remains stateless and is never cached twice.

Integrating with your Textual app
─────────────────────────────────
```python
factory = EmbeddingFactory(cfg, max_cached=1, idle_seconds=900)  # 15 min
mgr = ChromaDBManager(Path("~/.myapp/chroma"), cfg)

vecs = factory.embed("gte-large", ["hello"])
results = mgr.vector_search(
    query="hello", collection="docs", k=5,
    model_id_override="gte-large",
    create_embedding_fn=lambda txt, mid: factory.embed_one(mid, txt),
)
```
The factory runs an eviction check on every call – no background threads and
no implicit timers.
"""
#
# Imports
from typing import Dict, List, Any, Callable
import time
import requests
#
# Third-Party Libraries
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from pydantic import BaseModel
#
# Local Imports
#
########################################################################################################################
#
__all__ = [

    "EmbeddingFactory",
    "EmbeddingConfigSchema",
]
#
# --------------------------------------------------------------------------- cfg
class HFModelCfg(BaseModel):
    provider: str = "huggingface"
    model_name_or_path: str
    trust_remote_code: bool = False
    max_length: int = 512


class OpenAICfg(BaseModel):
    provider: str = "openai"
    model_name_or_path: str = "text-embedding-3-small"
    api_key: str


ModelCfg = HFModelCfg | OpenAICfg


class EmbeddingConfigSchema(BaseModel):
    default_model_id: str | None = None
    models: Dict[str, ModelCfg]


# ------------------------------------------------------------------- providers
class _HuggingFaceEmbedder:
    """Small wrapper that can unload its weights on demand."""

    def __init__(self, cfg: HFModelCfg):
        self._tok = AutoTokenizer.from_pretrained(
            cfg.model_name_or_path, trust_remote_code=cfg.trust_remote_code
        )
        self._model = AutoModel.from_pretrained(
            cfg.model_name_or_path, trust_remote_code=cfg.trust_remote_code
        )
        self._model.eval()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._max_len = cfg.max_length

    # ----------------------------------------------------------------- public
    def embed(self, texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            inp = self._tok(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._max_len,
            )
            inp = {k: v.to(self._device) for k, v in inp.items()}
            out = self._model(**inp).last_hidden_state.mean(dim=1).cpu().float().numpy()
        return out.tolist()

    def close(self) -> None:
        del self._model, self._tok  # type: ignore[attr-defined]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _openai_embedder(model: str, api_key: str) -> Callable[[List[str]], List[List[float]]]:
    sess = requests.Session()
    sess.headers["Authorization"] = f"Bearer {api_key}"

    def _embed(texts: List[str]) -> List[List[float]]:
        resp = sess.post(
            "https://api.openai.com/v1/embeddings",
            json={"input": texts, "model": model},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [d["embedding"] for d in data]

    return _embed


# -------------------------------------------------------------------- factory
class EmbeddingFactory:
    def __init__(
        self,
        cfg: Dict[str, Any] | EmbeddingConfigSchema,
        *,
        max_cached: int = 2,
        idle_seconds: int = 900,  # 15 minutes
    ) -> None:
        self._cfg = cfg if isinstance(cfg, EmbeddingConfigSchema) else EmbeddingConfigSchema(**cfg)
        self._max_cached = max_cached
        self._idle = idle_seconds
        # _cache maps model_id -> {"embed": Callable, "close": Callable | None, "last": float}
        self._cache: Dict[str, Dict[str, Any]] = {}

    # ---------------------------------------------------------------- private
    def _build(self, model_id: str) -> Dict[str, Any]:
        spec = self._cfg.models[model_id]
        if spec.provider == "huggingface":
            hf = _HuggingFaceEmbedder(spec)
            return {"embed": hf.embed, "close": hf.close, "last": time.time()}
        if spec.provider == "openai":
            fn = _openai_embedder(spec.model_name_or_path, spec.api_key)
            return {"embed": fn, "close": None, "last": time.time()}
        raise ValueError(f"Unsupported provider: {spec.provider}")

    def _evict_if_needed(self) -> None:
        # time‑based eviction first
        now = time.time()
        to_delete = [mid for mid, rec in self._cache.items() if now - rec["last"] > self._idle]
        for mid in to_delete:
            rec = self._cache.pop(mid)
            if rec["close"]:
                rec["close"]()
        # size‑based eviction (LRU)
        while len(self._cache) > self._max_cached:
            lru_mid = min(self._cache.items(), key=lambda kv: kv[1]["last"])[0]
            rec = self._cache.pop(lru_mid)
            if rec["close"]:
                rec["close"]()

    # ---------------------------------------------------------------- public
    def embed(self, model_id: str, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        rec = self._cache.get(model_id)
        if rec is None:
            rec = self._build(model_id)
            self._cache[model_id] = rec
        rec["last"] = time.time()
        self._evict_if_needed()
        return rec["embed"](texts)

    def embed_one(self, model_id: str, text: str) -> List[float]:
        return self.embed(model_id, [text])[0]

#
# End of Embeddings_Lib.py
########################################################################################################################
