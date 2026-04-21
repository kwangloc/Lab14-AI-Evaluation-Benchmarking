import asyncio
import sys
import os
from pathlib import Path
from typing import Dict

# Make data_builder importable regardless of working directory
_DATA_BUILDER_DIR = Path(__file__).parent.parent / "data_builder"
if str(_DATA_BUILDER_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_BUILDER_DIR))

from rag_answer import rag_answer  # noqa: E402


class MainAgent:
    """
    RAG Agent backed by the real pipeline in data_builder/rag_answer.py.
    Uses hybrid retrieval + cross-encoder reranking by default.
    """

    def __init__(
        self,
        retrieval_mode: str = "hybrid",
        use_rerank: bool = True,
        top_k_search: int = 10,
        top_k_select: int = 3,
    ):
        self.name = "SupportAgent-v1"
        self.retrieval_mode = retrieval_mode
        self.use_rerank = use_rerank
        self.top_k_search = top_k_search
        self.top_k_select = top_k_select

    async def query(self, question: str) -> Dict:
        """
        Full RAG pipeline:
        1. Retrieve relevant chunks from ChromaDB.
        2. (Optional) Rerank with cross-encoder.
        3. Generate grounded answer via LLM.

        Returns:
            {
                "answer": str,
                "contexts": List[str],       # chunk texts used
                "retrieved_ids": List[str],  # chunk_ids for retrieval eval
                "metadata": {
                    "model": str,
                    "sources": List[str],
                    "retrieval_mode": str,
                }
            }
        """
        result = await asyncio.to_thread(
            rag_answer,
            question,
            retrieval_mode=self.retrieval_mode,
            use_rerank=self.use_rerank,
            top_k_search=self.top_k_search,
            top_k_select=self.top_k_select,
        )

        chunks_used = result.get("chunks_used", [])

        # Build retrieved_ids from chunk metadata for retrieval eval (Hit Rate / MRR)
        retrieved_ids = [
            c.get("metadata", {}).get("source", "unknown")
            for c in chunks_used
        ]

        return {
            "answer": result["answer"],
            "contexts": [c["text"] for c in chunks_used],
            "retrieved_ids": retrieved_ids,
            "metadata": {
                "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
                "sources": result.get("sources", []),
                "retrieval_mode": self.retrieval_mode,
            },
        }


if __name__ == "__main__":
    agent = MainAgent()

    async def test():
        resp = await agent.query("Làm thế nào để đổi mật khẩu?")
        print("Answer:", resp["answer"])
        print("Sources:", resp["metadata"]["sources"])

    asyncio.run(test())
