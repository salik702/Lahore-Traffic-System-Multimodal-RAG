"""Adapter for retrieving persisted LightRAG chunks without changing production code."""
from __future__ import annotations

import asyncio
from typing import Any

from app.core.config import get_settings
from app.rag.engine import RAGAnythingEngine


class EvaluationRetriever:
    def __init__(self, top_k: int = 10):
        self.settings = get_settings()
        self.engine = RAGAnythingEngine(self.settings)
        self.top_k = top_k
        self._initialized = False

    async def initialize(self) -> None:
        if not self._initialized:
            await self.engine.initialize()
            self._initialized = True

    async def retrieve(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        await self.initialize()
        result = await self.engine.retrieve(question, top_k=top_k or self.top_k, intent="evaluation")
        if not result:
            return []
        chunks = result.get("data", {}).get("chunks", [])
        normalized = []
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id") or chunk.get("__id__") or chunk.get("id")
            content = chunk.get("content", "")
            if chunk_id:
                normalized.append({"id": str(chunk_id), "content": str(content), "score": chunk.get("distance")})
        return normalized

    def retrieve_sync(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return asyncio.run(self.retrieve(question, top_k))
