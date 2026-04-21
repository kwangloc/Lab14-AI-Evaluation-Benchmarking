import asyncio
import time
from typing import List, Dict


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()

        # 1. Call Agent
        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start_time

        # 2. Faithfulness / relevancy scores (from evaluator)
        ragas_scores = await self.evaluator.score(test_case, response)

        # 3. Multi-Judge (2 OpenAI models)
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case["expected_answer"],
        )

        return {
            "test_case": test_case["question"],
            "test_case_meta": test_case,          # full case — needed for retrieval eval
            "agent_response": response,            # full dict — has retrieved_ids
            "latency": round(latency, 3),
            "ragas": ragas_scores,
            "judge": judge_result,
            "tokens": judge_result.get("total_tokens", 0),
            "status": "fail" if judge_result["final_score"] < 3 else "pass",
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        """Run in parallel batches to respect rate limits."""
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            print(f"  Batch {i // batch_size + 1}: {len(results)}/{len(dataset)} done")
        return results


    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        """
        Chạy song song bằng asyncio.gather với giới hạn batch_size để không bị Rate Limit.
        """
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results
