"""Gate check: do dspy.Predict calls made *inside LangGraph nodes* land in DSPy's trace,
and does GEPA get non-empty reflective datasets from them? Uses DummyLM: no API calls.

python -m spikes.trace_spike
"""
import json

import numpy  # noqa: F401
import dspy
from dspy.utils import DummyLM

from app.graph import build_graph
from app.prompts import load_prompts
from optimize.bridge import langgraph_to_dspy

task_lm = DummyLM([{"search_query": "Horizon Zero Dawn release", "reasoning": "Aloy is in Horizon Zero Dawn (2017).", "answer": "2017"}] * 1000)
reflect_lm = DummyLM([{"new_instruction": "IMPROVED: be precise and terse."}] * 1000)
dspy.configure(lm=task_lm)

program = langgraph_to_dspy(
    build_graph(), load_prompts(), optimizable=["query_rewrite", "answer"],
    output_keys=("answer", "search_query", "chunks"), extra_config={"skip_judge": True},
)
print("named predictors:", [n for n, _ in program.named_predictors()])

# 1) trace capture
with dspy.context(trace=[]):
    pred = program(question="Ashly Burch voiced Aloy in a game released in what year?")
    trace = dspy.settings.trace.copy()
print("prediction:", pred.answer, "| trace entries:", len(trace),
      [type(p).__name__ + ":" + p.signature.__name__ for p, _, _ in trace])
assert len(trace) == 2, "predictor calls inside LangGraph nodes were NOT traced"

# 2) GEPA end to end with a trivial metric; collect reflective datasets
rows = [json.loads(l) for l in open("data/train.jsonl")][:6]
trainset = [dspy.Example(question=r["question"], answer=r["answer"]).with_inputs("question") for r in rows]
seen = {}


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    score = float(gold.answer.lower() in (pred.answer or "").lower())
    if pred_name:
        seen[pred_name] = seen.get(pred_name, 0) + 1
        return dspy.Prediction(score=score, feedback=f"[{pred_name}] gold={gold.answer}")
    return score


opt = dspy.GEPA(metric=metric, max_metric_calls=30, reflection_lm=reflect_lm, num_threads=2,
                reflection_minibatch_size=2, use_merge=False)
compiled = opt.compile(program, trainset=trainset[:3], valset=trainset[3:])
print("per-predictor feedback calls:", seen)
print("exported:", compiled.export_prompts())
assert seen, "GEPA never requested per-predictor feedback (empty reflective dataset)"
print("SPIKE PASSED")
