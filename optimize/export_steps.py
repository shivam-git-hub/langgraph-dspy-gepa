"""Write one markdown file per presented GEPA step (full prompts + worked examples with metric feedback),
for showing alongside the slides.

python -m optimize.export_steps [--run runs/gepa_...] [--ids <test id> ...]
"""
import argparse
import json
from pathlib import Path

import numpy  # noqa: F401
import dspy

from app.config import PROMPTS_DIR
from app.retriever import retrieve
from optimize.data import load_split
from optimize.metric import gepa_metric
from optimize.report import latest_run

DEFAULT_IDS = ["5ae320f855429928c4239620", "5ac083b0554299294b21900d"]  # Pittsburgh Panthers, Planter's Punch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--ids", nargs="*", default=None)
    args = ap.parse_args()

    run = Path(args.run) if args.run else latest_run()
    report = json.loads((run / "report" / "report.json").read_text())
    cands = {c["idx"]: c for c in report["candidates"]}
    test = {ex.id: ex for ex in load_split("test")}
    ids = args.ids or DEFAULT_IDS
    out_dir = PROMPTS_DIR / "steps"
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = {}
    for step, idx in enumerate(report["steps"]):
        c = cands[idx]
        rows = {r["id"]: r for r in report["per_question"][f"cand{idx}"]}
        label = "baseline" if step == 0 else ("final" if idx == report["best"] else f"step{step}")
        lines = [
            f"# Step {step} ({label}): GEPA candidate {idx}",
            "",
            f"- Parent candidate: {c['parent']}",
            f"- Prompt GEPA changed at this step: {c['updated_component'] or '-'}",
            f"- Found after {c['found_at_metric_calls']} rollouts; GEPA val score {c['val_score']:.3f}",
            "",
        ]
        for name, text in c["prompts"].items():
            lines += [f"## Prompt: {name}" + (" (changed at this step)" if name == c["updated_component"] else ""), "", "```text", text, "```", ""]
        lines += ["## Examples (test set, production LangGraph path)", ""]
        for qid in ids:
            ex, r = test[qid], rows[qid]
            # Retrieval is deterministic, so re-running it for this candidate's query reproduces its chunks.
            pred = dspy.Prediction(answer=r["answer"], search_query=r["search_query"],
                                   chunks=retrieve(r["search_query"]))
            fb_q = gepa_metric(ex, pred, pred_name="query_rewrite").feedback
            fb_a = gepa_metric(ex, pred, pred_name="answer").feedback
            examples.setdefault(qid, []).append({
                "step": step, "cand": idx, "query": r["search_query"], "answer": r["answer"],
                "retrieved": [ch["title"] for ch in pred["chunks"]], "recall": r["retrieval_recall"],
                "correctness": r["correctness"], "feedback_query_rewrite": fb_q, "feedback_answer": fb_a,
            })
            lines += [
                f"### Q: {ex.question}",
                f"- Gold answer: {ex.answer}",
                f"- Gold paragraphs: {ex.supporting_titles}",
                f"- Search query: {r['search_query']}",
                f"- Retrieved: {[ch['title'] for ch in pred['chunks']]}",
                f"- Our answer: {r['answer']}",
                f"- Feedback to query_rewrite: {fb_q}",
                f"- Feedback to answer: {fb_a}",
                "",
            ]
        path = out_dir / (f"step{step}_{label}_cand{idx}.md" if label in ("baseline", "final") else f"step{step}_cand{idx}.md")
        path.write_text("\n".join(lines))
        print("wrote", path)
    (run / "report" / "examples.json").write_text(json.dumps(examples, indent=2))


if __name__ == "__main__":
    main()
