"""Optimize the LangGraph app's prompts with GEPA, then export them back to the prompt registry.

python -m optimize.run_gepa [--auto light|medium|heavy | --max-metric-calls N] [--threads 8]
"""
import argparse
import json
import time

import numpy  # noqa: F401
import dspy

from app.config import OPTIMIZED_PROMPTS, REFLECTION_MODEL, RUNS_DIR, TASK_MODEL
from app.graph import build_graph
from app.prompts import load_prompts, save_prompts
from optimize.bridge import langgraph_to_dspy
from optimize.data import load_split
from optimize.metric import gepa_metric

OPTIMIZABLE = ["query_rewrite", "answer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", choices=["light", "medium", "heavy"], default=None)
    ap.add_argument("--max-metric-calls", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--train", type=int, default=None, help="limit train examples")
    ap.add_argument("--val", type=int, default=None, help="limit val examples")
    args = ap.parse_args()
    if not args.auto and not args.max_metric_calls:
        args.auto = "light"

    run_dir = RUNS_DIR / f"gepa_{time.strftime('%Y%m%d_%H%M%S')}"
    dspy.configure(lm=dspy.LM(f"gemini/{TASK_MODEL}", temperature=0, max_tokens=4000, num_retries=8))
    reflection_lm = dspy.LM(f"gemini/{REFLECTION_MODEL}", temperature=1.0, max_tokens=32000, num_retries=8)

    program = langgraph_to_dspy(
        build_graph(), load_prompts(), OPTIMIZABLE,
        output_keys=("answer", "search_query", "chunks"),
        extra_config={"skip_judge": True},  # the in-graph judge isn't needed during optimization
    )
    trainset, valset = load_split("train", args.train), load_split("val", args.val)

    gepa = dspy.GEPA(
        metric=gepa_metric,
        auto=args.auto,
        max_metric_calls=args.max_metric_calls,
        reflection_lm=reflection_lm,
        num_threads=args.threads,
        track_stats=True,
        log_dir=str(run_dir),
        seed=0,
    )
    optimized = gepa.compile(program, trainset=trainset, valset=valset)

    prompts = optimized.export_prompts()
    save_prompts(prompts, OPTIMIZED_PROMPTS)
    save_prompts(prompts, run_dir / "optimized_prompts.json")

    res = optimized.detailed_results
    best = res.best_idx
    summary = {
        "val_score_baseline": res.val_aggregate_scores[0],
        "val_score_best": res.val_aggregate_scores[best],
        "num_candidates": len(res.candidates),
        "total_metric_calls": res.total_metric_calls,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    for name, text in prompts.items():
        print(f"\n=== {name} ===\n{text}")
    print(f"\nSaved -> {OPTIMIZED_PROMPTS}")


if __name__ == "__main__":
    main()
