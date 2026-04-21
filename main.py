import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from agent.main_agent import MainAgent

# Giả lập các components Expert
class ExpertEvaluator:
    def __init__(self):
        self._ret_eval = RetrievalEvaluator()

    async def score(self, case, resp): 
        # Giả lập tính toán Hit Rate và MRR
        expected_ids = case.get("expected_retrieval_ids") or []
        retrieved_ids = (resp or {}).get("retrieved_ids") or []
        hit_rate = self._ret_eval.calculate_hit_rate(expected_ids, retrieved_ids)
        mrr = self._ret_eval.calculate_mrr(expected_ids, retrieved_ids)
        return {
            "retrieval": {"hit_rate": hit_rate, "mrr": mrr}
        }

class MultiModelJudge:
    def __init__(self):
        self._judge = LLMJudge()

    async def evaluate_multi_judge(self, q, a, gt): 
        return await self._judge.evaluate_multi_judge(q, a, gt)

async def run_benchmark_with_results(agent_version: str, agent=None):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    if agent is None:
        agent = MainAgent()
    runner = BenchmarkRunner(agent, ExpertEvaluator(), MultiModelJudge())
    results = await runner.run_all(dataset)

    total = len(results)
    summary = {
        "metadata": {"version": agent_version, "total": total, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total
        }
    }
    return results, summary

def format_result(r: dict) -> dict:
    """Flatten a raw runner result into the required benchmark_results.json format."""
    ragas = r.get("ragas", {})
    retrieval = ragas.get("retrieval", {})
    judge = r.get("judge", {})
    individual_scores = judge.get("individual_scores", {})
    reasoning = judge.get("reasoning", {})
    individual_results = {
        model: {"score": individual_scores.get(model, 0), "reasoning": reasoning.get(model, "")}
        for model in individual_scores
    }
    agent_response = r.get("agent_response", {})
    return {
        "test_case": r.get("test_case", ""),
        "agent_response": agent_response.get("answer", "") if isinstance(agent_response, dict) else str(agent_response),
        "latency": r.get("latency", 0),
        "ragas": {
            "hit_rate": retrieval.get("hit_rate", 0),
            "mrr": retrieval.get("mrr", 0),
            # faithfulness: is the answer grounded in retrieved context?
            # Uses normalized judge score (judges explicitly check grounding): score/5 → [0.0, 1.0]
            "faithfulness": round(judge.get("final_score", 0) / 5, 2),
            # relevancy: are retrieved docs relevant to the question?
            # Average of hit_rate and MRR — both measure retrieval quality from different angles
            "relevancy": round((retrieval.get("hit_rate", 0) + retrieval.get("mrr", 0)) / 2, 2),
        },
        "judge": {
            "final_score": judge.get("final_score", 0),
            "agreement_rate": judge.get("agreement_rate", 0),
            "individual_results": individual_results,
            "status": "conflict" if judge.get("conflict") else "consensus",
        },
        "status": r.get("status", "fail"),
    }

async def run_benchmark(version, agent=None):
    results, summary = await run_benchmark_with_results(version, agent=agent)
    return results, summary

async def main():
    v1_results, v1_summary = await run_benchmark("Agent_V1_Base", agent=MainAgent(retrieval_mode="dense", use_rerank=False, top_k_search=3, top_k_select=1))
    
    # Giả lập V2 có cải tiến (để test logic)
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized", agent=MainAgent(retrieval_mode="hybrid", use_rerank=False, top_k_search=10, top_k_select=3))
    
    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "v1": [format_result(r) for r in v1_results],
                "v2": [format_result(r) for r in v2_results],
            },
            f, ensure_ascii=False, indent=2
        )

    if delta > 0:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")

if __name__ == "__main__":
    asyncio.run(main())
