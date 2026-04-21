from typing import List, Dict


class RetrievalEvaluator:
    def __init__(self):
        pass

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """1.0 if any expected_id appears in the top_k retrieved_ids, else 0.0."""
        top_retrieved = retrieved_ids[:top_k]
        return 1.0 if any(doc_id in top_retrieved for doc_id in expected_ids) else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """Mean Reciprocal Rank: 1 / (1-indexed position of first hit), or 0.0."""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(self, results: List[Dict]) -> Dict:
        """
        Compute Hit Rate and MRR over a list of benchmark result dicts.

        Each result dict must have:
          - "test_case_meta": the original golden-set case (with "expected_retrieval_ids")
          - "agent_response": the agent response dict (with "retrieved_ids")

        Cases with no expected_retrieval_ids (e.g. adversarial out-of-scope) are skipped
        for retrieval metrics but counted in total.
        """
        hit_rates = []
        mrrs = []

        for r in results:
            case_meta = r.get("test_case_meta", {})
            expected_ids: List[str] = case_meta.get("expected_retrieval_ids") or []

            # Skip out-of-scope cases that have no ground-truth retrieval target
            if not expected_ids:
                continue

            retrieved_ids: List[str] = (r.get("agent_response") or {}).get("retrieved_ids") or []

            hit_rates.append(self.calculate_hit_rate(expected_ids, retrieved_ids))
            mrrs.append(self.calculate_mrr(expected_ids, retrieved_ids))

        n = len(hit_rates)
        return {
            "avg_hit_rate": round(sum(hit_rates) / n, 4) if n else 0.0,
            "avg_mrr": round(sum(mrrs) / n, 4) if n else 0.0,
            "evaluated_cases": n,
        }

