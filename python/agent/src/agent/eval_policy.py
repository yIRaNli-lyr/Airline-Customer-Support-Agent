"""
Eval: Naive vs RAG pipeline for policy QA. Computes hallucination rate and reduction.

- Naive: LLM answers with no policy context.
- RAG: answer_policy_rag_only (retrieve + generate).
- Judge: LLM judges whether each answer is fully supported by reference policy (no invented facts).
- Metric: hallucination_rate = fraction of answers judged "NO". reduction = (naive_rate - rag_rate) / naive_rate.

Usage:
    agent-eval-policy
    agent-eval-policy --limit 5
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from litellm import completion

from .config import TAU2_DOMAIN_DATA_PATH
from .rag import answer_policy_rag_only, retrieve

_MODEL = "gemini/gemini-2.0-flash"
_AGENT_DIR = Path(__file__).resolve().parent
_DATA_DIR = (_AGENT_DIR / TAU2_DOMAIN_DATA_PATH).resolve()
_EVAL_QUESTIONS_PATH = _DATA_DIR / "eval_policy_questions.json"
_POLICY_PATH = _DATA_DIR / "policy.md"


def _load_json(path: Path) -> list | dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _eval_llm(messages: list, model: str = _MODEL):
    """Single completion, no tools. Optional small delay to avoid throttling."""
    time.sleep(0.5)
    r = completion(model=model, messages=messages, temperature=0.0)
    return (r.choices[0].message.content or "").strip()


def _answer_naive(question: str, model: str = _MODEL) -> str:
    sys = (
        "Answer the following question about airline customer service policy. "
        "If you are not sure, say so. Do not invent specific rules or numbers."
    )
    return _eval_llm(
        [{"role": "system", "content": sys}, {"role": "user", "content": question}],
        model=model,
    )


def _judge(question: str, answer: str, reference: str, model: str = _MODEL) -> bool:
    """True = supported (no hallucination), False = hallucination."""
    sys = (
        "You are a judge. Given a reference policy, a question, and a model answer, "
        "determine if the answer is FULLY supported by the reference—no invented facts, "
        "numbers, or rules. Reply with exactly YES or NO."
    )
    user = (
        f"Reference policy:\n{reference}\n\n"
        f"Question: {question}\n\n"
        f"Model answer: {answer}\n\n"
        "Is the answer fully supported by the reference? YES or NO:"
    )
    out = _eval_llm(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        model=model,
    )
    return bool(re.search(r"\bYES\b", out, re.I))


def run_eval(limit: int | None = None, model: str = _MODEL) -> dict:
    questions = _load_json(_EVAL_QUESTIONS_PATH)
    if limit is not None:
        questions = questions[:limit]

    reference = _load_text(_POLICY_PATH)
    results = []

    for i, item in enumerate(questions):
        qid = item["id"]
        q = item["question"]
        print(f"  [{i+1}/{len(questions)}] {qid} ...", end=" ", flush=True)

        naive_ans = _answer_naive(q, model=model)
        rag_ans = answer_policy_rag_only(q, model=model)

        naive_ok = _judge(q, naive_ans, reference, model=model)
        rag_ok = _judge(q, rag_ans, reference, model=model)

        results.append(
            {
                "id": qid,
                "question": q,
                "naive_answer": naive_ans,
                "rag_answer": rag_ans,
                "naive_supported": naive_ok,
                "rag_supported": rag_ok,
            }
        )
        print(f"naive={'✓' if naive_ok else '✗'} rag={'✓' if rag_ok else '✗'}")

    n = len(results)
    naive_h = sum(1 for r in results if not r["naive_supported"])
    rag_h = sum(1 for r in results if not r["rag_supported"])
    naive_rate = naive_h / n if n else 0.0
    rag_rate = rag_h / n if n else 0.0
    reduction = (naive_rate - rag_rate) / naive_rate if naive_rate > 0 else 0.0

    return {
        "n": n,
        "naive_hallucinations": naive_h,
        "rag_hallucinations": rag_h,
        "naive_hallucination_rate": naive_rate,
        "rag_hallucination_rate": rag_rate,
        "reduction": reduction,
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser(description="Eval naive vs RAG policy QA.")
    ap.add_argument("--limit", type=int, default=None, help="Max number of questions")
    ap.add_argument("--model", type=str, default=_MODEL, help="LLM model")
    ap.add_argument("--out", type=str, default=None, help="Write results JSON here")
    args = ap.parse_args()

    if not _EVAL_QUESTIONS_PATH.exists():
        raise SystemExit(f"Eval questions not found: {_EVAL_QUESTIONS_PATH}")
    if not _POLICY_PATH.exists():
        raise SystemExit(f"Policy not found: {_POLICY_PATH}")

    print("Running policy eval (naive vs RAG pipeline)...\n")
    out = run_eval(limit=args.limit, model=args.model)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  N = {out['n']}")
    print(f"  Naive hallucination rate: {out['naive_hallucination_rate']:.1%} ({out['naive_hallucinations']}/{out['n']})")
    print(f"  RAG   hallucination rate: {out['rag_hallucination_rate']:.1%} ({out['rag_hallucinations']}/{out['n']})")
    print(f"  Reduction: {out['reduction']:.1%}")
    print("=" * 60)

    if args.out:
        p = Path(args.out)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in out.items() if k != "results"},
                f,
                indent=2,
            )
        details_path = p.parent / f"{p.stem}_details{p.suffix}"
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {p} and {details_path}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
