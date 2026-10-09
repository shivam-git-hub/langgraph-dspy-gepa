"""The deliverable: run the *production* LangGraph app (LangChain backend, no DSPy in the loop) on the
held-out test set with baseline vs optimized prompts, and compare.

python -m optimize.evaluate [--split test] [--limit N] [--threads 8]
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy  # noqa: F401
import dspy
import pandas as pd

from app.config import BASELINE_PROMPTS, OPTIMIZED_PROMPTS, RUNS_DIR
from app.graph import build_graph
from app.prompts import load_prompts
from optimize.data import load_split
from optimize.metric import evaluate_one, exact_match


def run_variant(graph, prompts: dict[str, str], examples, threads: int) -> list[dict]:
    def one(ex):
        try:
            state = graph.invoke({"question": ex.question}, config={"configurable": {"prompts": prompts}})
        except Exception as e:  # report failures separately; metrics are averaged over successful runs
            return {"id": ex.id, "error": repr(e)}
        pred = dspy.Prediction(answer=state["answer"], chunks=state["chunks"])
        gold = evaluate_one(ex, pred)
        s = state["scores"]
        return {
            "id": ex.id, "question": ex.question, "gold": ex.answer, "answer": state["answer"],
            "search_query": state["search_query"],
            "em": exact_match(state["answer"], ex.answer),
            **{k: gold[k] for k in ("score", "correctness", "f1", "faithfulness", "retrieval_recall")},
            "judge_faithfulness": s["faithfulness_score"],
            "judge_context_relevance": s["context_relevance_score"],
            "judge_completeness": s["completeness_score"],
        }

    with ThreadPoolExecutor(threads) as pool:
        return list(pool.map(one, examples))


METRIC_COLS = ["score", "correctness", "em", "f1", "faithfulness", "retrieval_recall",
               "judge_faithfulness", "judge_context_relevance", "judge_completeness"]


def evaluate_variants(variants: dict[str, dict[str, str]], examples, threads: int, out_dir) -> pd.DataFrame:
    """Run each prompt variant through the production graph; returns metric means (rows) x variants (cols)."""
    graph = build_graph()
    out_dir.mkdir(parents=True, exist_ok=True)
    table = {}
    for name, prompts in variants.items():
        print(f"Running {name} on {len(examples)} examples...")
        rows = run_variant(graph, prompts, examples, threads)
        (out_dir / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        df = pd.DataFrame(rows)
        table[name] = df[[c for c in METRIC_COLS if c in df]].mean().round(3)
        table[name]["errors"] = int(df["error"].notna().sum()) if "error" in df else 0
    result = pd.DataFrame(table)
    result.to_csv(out_dir / "summary.csv")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--optimized", default=str(OPTIMIZED_PROMPTS))
    args = ap.parse_args()

    variants = {"baseline": load_prompts(BASELINE_PROMPTS), "optimized": load_prompts(args.optimized)}
    out_dir = RUNS_DIR / f"eval_{time.strftime('%Y%m%d_%H%M%S')}"
    result = evaluate_variants(variants, load_split(args.split, args.limit), args.threads, out_dir)
    result["delta"] = (result["optimized"] - result["baseline"]).round(3)
    print("\n" + result.to_string())
    print(f"\nPer-example results -> {out_dir}")


if __name__ == "__main__":
    main()
