"""Turn a GEPA run into presentation data.

Picks the lineage from the baseline to the best candidate (plus intermediate steps), evaluates each
selected candidate on the held-out test set through the *production* LangGraph path, and writes
runs/<gepa_run>/report/report.json with prompts, val/test metrics and per-question answers.

python -m optimize.report [--run runs/gepa_...] [--max-steps 5] [--limit N] [--threads 12]
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy  # noqa: F401
import pandas as pd

from app.config import RUNS_DIR
from app.prompts import load_prompts
from optimize.data import load_split
from optimize.evaluate import evaluate_variants


def latest_run() -> Path:
    runs = sorted(p for p in RUNS_DIR.glob("gepa_2*") if (p / "gepa_state.bin").exists())
    return runs[-1]


def lineage(parents: list[list[int | None]], idx: int) -> list[int]:
    chain = [idx]
    while parents[chain[-1]] and parents[chain[-1]][0] is not None:
        chain.append(parents[chain[-1]][0])
    return chain[::-1]


def pick_steps(chain: list[int], k: int) -> list[int]:
    """Baseline, final, and evenly spaced intermediates along the lineage (at most k total)."""
    if len(chain) <= k:
        return chain
    pos = sorted({round(i * (len(chain) - 1) / (k - 1)) for i in range(k)})
    return [chain[p] for p in pos]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    run = Path(args.run) if args.run else latest_run()
    state = pickle.load(open(run / "gepa_state.bin", "rb"))
    cands = state["program_candidates"]
    val = [sum(s.values()) / len(s) for s in state["prog_candidate_val_subscores"]]
    best = max(range(len(cands)), key=lambda i: val[i])
    chain = lineage(state["parent_program_for_candidate"], best)
    steps = pick_steps(chain, args.max_steps)
    print(f"{len(cands)} candidates; best={best} (val {val[best]:.3f}); lineage {chain}; presenting {steps}")

    variants = {f"cand{i}": load_prompts() | cands[i] for i in steps}
    out_dir = run / "report"
    summary = evaluate_variants(variants, load_split("test", args.limit), args.threads, out_dir)
    print("\n" + summary.to_string())

    per_q = {name: {r["id"]: r for r in map(json.loads, open(out_dir / f"{name}.jsonl"))} for name in variants}
    report = {
        "run": str(run),
        "best": best,
        "lineage": chain,
        "steps": steps,
        "candidates": [
            {
                "idx": i, "parent": state["parent_program_for_candidate"][i][0], "val_score": val[i],
                "found_at_metric_calls": state["num_metric_calls_by_discovery"][i],
                "updated_component": None if i == 0 else
                    next((k for k in cands[i] if cands[i][k] != cands[state["parent_program_for_candidate"][i][0]][k]), None),
                "prompts": cands[i],
            }
            for i in range(len(cands))
        ],
        "test_summary": json.loads(summary.to_json()),
        "per_question": {name: list(rows.values()) for name, rows in per_q.items()},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(
        [{"idx": c["idx"], "parent": c["parent"], "val": round(c["val_score"], 3),
          "calls": c["found_at_metric_calls"], "changed": c["updated_component"]} for c in report["candidates"]]
    ).to_csv(out_dir / "candidates.csv", index=False)
    print(f"\nReport -> {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
