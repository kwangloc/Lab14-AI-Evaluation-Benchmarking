import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator
from agent.main_agent import MainAgent

# ---------------------------------------------------------------------------
# Cost pricing table (USD per 1M tokens, as of 2025)
# Source: https://openai.com/pricing
# ---------------------------------------------------------------------------
_COST_PER_1M = {
    "gpt-4o-mini":  {"input": 0.15,  "output": 0.60},
    "gpt-4.1-nano": {"input": 0.10,  "output": 0.40},
    "gpt-4o":       {"input": 2.50,  "output": 10.00},
    "gpt-4-turbo":  {"input": 10.00, "output": 30.00},
}
_DEFAULT_COST = {"input": 2.50, "output": 10.00}  # fallback


def _token_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = _COST_PER_1M.get(model, _DEFAULT_COST)
    return (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1_000_000


def compute_cost_report(v1_results: list, v2_results: list) -> dict:
    """
    Aggregate token usage and estimated USD cost broken down by:
      - role: "agent" (answer generation) vs "judges" (evaluation)
      - model name
    Produces reports/cost_report.json.
    """
    def _empty_stats():
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}

    def _aggregate(results: list) -> dict:
        # Track agent and judge usage independently, even if they share the same model
        agent_by_model: dict = {}
        judge_by_model: dict = {}

        def _add(bucket: dict, model: str, prompt: int, completion: int):
            if not model:
                return
            if model not in bucket:
                bucket[model] = _empty_stats()
            bucket[model]["prompt_tokens"] += prompt
            bucket[model]["completion_tokens"] += completion
            bucket[model]["total_tokens"] += prompt + completion
            bucket[model]["estimated_cost_usd"] += _token_cost(model, prompt, completion)

        for r in results:
            # ---- Agent LLM (answer generation) ----
            agent_tok = r.get("agent_tokens", {})
            _add(agent_by_model, agent_tok.get("model"), agent_tok.get("prompt_tokens", 0), agent_tok.get("completion_tokens", 0))

            # ---- Judge LLMs (evaluation) ----
            for model, tok in r.get("judge_tokens_by_model", {}).items():
                _add(judge_by_model, model, tok.get("prompt_tokens", 0), tok.get("completion_tokens", 0))

        def _round_stats(bucket: dict) -> dict:
            return {m: {**s, "estimated_cost_usd": round(s["estimated_cost_usd"], 6)} for m, s in bucket.items()}

        def _subtotal(bucket: dict) -> dict:
            return {
                "total_tokens": sum(s["total_tokens"] for s in bucket.values()),
                "total_cost_usd": round(sum(s["estimated_cost_usd"] for s in bucket.values()), 6),
                "per_model": _round_stats(bucket),
            }

        agent_sub = _subtotal(agent_by_model)
        judge_sub = _subtotal(judge_by_model)
        grand_total_tokens = agent_sub["total_tokens"] + judge_sub["total_tokens"]
        grand_total_cost = round(agent_sub["total_cost_usd"] + judge_sub["total_cost_usd"], 6)

        return {
            "total_tokens": grand_total_tokens,
            "total_cost_usd": grand_total_cost,
            "agent": agent_sub,
            "judges": judge_sub,
        }

    v1_agg = _aggregate(v1_results)
    v2_agg = _aggregate(v2_results)
    combined = _aggregate(v1_results + v2_results)

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pricing_source": "https://openai.com/pricing (USD per 1M tokens)",
        "v1": v1_agg,
        "v2": v2_agg,
        "combined": combined,
    }

def compute_cohens_kappa(results: list) -> float:
    """
    Cohen's Kappa for inter-rater agreement between Judge A and Judge B.
    κ = (Po - Pe) / (1 - Pe)
    Po = observed proportion of exact-score agreements
    Pe = expected chance agreement (product of marginal frequencies per score level)
    """
    from collections import Counter

    scores_a, scores_b = [], []
    for r in results:
        individual = r.get("judge", {}).get("individual_scores", {})
        models = list(individual.keys())
        if len(models) < 2:
            continue
        scores_a.append(int(round(individual[models[0]])))
        scores_b.append(int(round(individual[models[1]])))

    n = len(scores_a)
    if n == 0:
        return 0.0

    Po = sum(1 for a, b in zip(scores_a, scores_b) if a == b) / n

    count_a = Counter(scores_a)
    count_b = Counter(scores_b)
    all_categories = set(scores_a) | set(scores_b)
    Pe = sum((count_a.get(k, 0) / n) * (count_b.get(k, 0) / n) for k in all_categories)

    if Pe >= 1.0:
        return 1.0

    kappa = (Po - Pe) / (1 - Pe)
    return round(kappa, 4)


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
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total,
            "cohens_kappa": compute_cohens_kappa(results),
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
    v1_results, v1_summary = await run_benchmark("Agent_V1_Base", agent=MainAgent(retrieval_mode="dense", use_rerank=True, top_k_search=1, top_k_select=1))
    
    # Giả lập V2 có cải tiến (để test logic)
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized", agent=MainAgent(retrieval_mode="dense", use_rerank=False, top_k_search=10, top_k_select=3))
    
    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")
    print(f"V1 Cohen's Kappa: {v1_summary['metrics']['cohens_kappa']}")
    print(f"V2 Cohen's Kappa: {v2_summary['metrics']['cohens_kappa']}")

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
    cost_report = compute_cost_report(v1_results, v2_results)
    with open("reports/cost_report.json", "w", encoding="utf-8") as f:
        json.dump(cost_report, f, ensure_ascii=False, indent=2)
    print(f"\n💰 Cost Report: {cost_report['combined']['total_tokens']:,} tokens total | ~${cost_report['combined']['total_cost_usd']:.4f} USD")

    if delta > 0:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")

if __name__ == "__main__":
    asyncio.run(main())
