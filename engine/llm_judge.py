import asyncio
import json
import os
import re
from typing import Dict, Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Two distinct OpenAI models act as independent judges.
# Using different model families gives genuine score diversity.
JUDGE_A_MODEL = os.getenv("JUDGE_A_MODEL", "gpt-4o-mini")
JUDGE_B_MODEL = os.getenv("JUDGE_B_MODEL", "gpt-4o")

# Threshold: if |score_a - score_b| > CONFLICT_THRESHOLD → conflict handling kicks in
CONFLICT_THRESHOLD = 1.0

JUDGE_PROMPT = """\
You are a strict evaluation judge for an AI support assistant.
Score the answer on a scale from 1 to 5 based on the rubric below.

RUBRIC:
- 5: Completely accurate, fully grounded in ground truth, professional tone.
- 4: Mostly accurate with minor omissions, grounded, good tone.
- 3: Partially correct — some key facts missing or slightly off.
- 2: Significant inaccuracies or hallucination, poor grounding.
- 1: Wrong answer, irrelevant, or harmful.

QUESTION: {question}

GROUND TRUTH: {ground_truth}

ANSWER TO EVALUATE: {answer}

Respond with ONLY a JSON object in this exact format (no markdown, no explanation):
{{"score": <integer 1-5>, "reasoning": "<one sentence>"}}
"""


async def _call_judge(model: str, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
    """Call one OpenAI model as a judge and parse its score."""
    prompt = JUDGE_PROMPT.format(question=question, answer=answer, ground_truth=ground_truth)
    try:
        resp = await _client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        usage = resp.usage
        parsed = json.loads(raw)
        score = int(parsed.get("score", 3))
        score = max(1, min(5, score))  # clamp to [1, 5]
        return {
            "score": score,
            "reasoning": parsed.get("reasoning", ""),
            "tokens": usage.total_tokens if usage else 0,
            "model": model,
        }
    except Exception as e:
        return {"score": 3, "reasoning": f"Judge error: {e}", "tokens": 0, "model": model}


class LLMJudge:
    def __init__(self):
        self.judge_a_model = JUDGE_A_MODEL
        self.judge_b_model = JUDGE_B_MODEL

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        Call 2 OpenAI models (GPT-4o and GPT-4o-mini) independently.
        Agreement rate = 1.0 if |score_a - score_b| <= CONFLICT_THRESHOLD else 0.5.
        Conflict resolution: if scores diverge > CONFLICT_THRESHOLD, use the lower score
        (conservative — prefer not to overrate).
        """
        result_a, result_b = await asyncio.gather(
            _call_judge(self.judge_a_model, question, answer, ground_truth),
            _call_judge(self.judge_b_model, question, answer, ground_truth),
        )

        score_a = result_a["score"]
        score_b = result_b["score"]
        diff = abs(score_a - score_b)

        if diff <= CONFLICT_THRESHOLD:
            final_score = (score_a + score_b) / 2
            agreement_rate = 1.0
            conflict = False
        else:
            # Conflict: use conservative (lower) score
            final_score = float(min(score_a, score_b))
            agreement_rate = round(1.0 - diff / 4.0, 2)  # scale: max diff=4 → 0.0
            conflict = True

        return {
            "final_score": round(final_score, 2),
            "agreement_rate": agreement_rate,
            "conflict": conflict,
            "individual_scores": {
                self.judge_a_model: score_a,
                self.judge_b_model: score_b,
            },
            "reasoning": {
                self.judge_a_model: result_a["reasoning"],
                self.judge_b_model: result_b["reasoning"],
            },
            "total_tokens": result_a["tokens"] + result_b["tokens"],
        }

    async def check_position_bias(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        Detect position bias: swap answer and ground_truth, re-judge.
        If score changes significantly, the judge has position bias.
        """
        normal = await _call_judge(self.judge_a_model, question, answer, ground_truth)
        swapped = await _call_judge(self.judge_a_model, question, ground_truth, answer)
        bias_delta = abs(normal["score"] - swapped["score"])
        return {
            "normal_score": normal["score"],
            "swapped_score": swapped["score"],
            "bias_delta": bias_delta,
            "has_position_bias": bias_delta > 1,
        }

