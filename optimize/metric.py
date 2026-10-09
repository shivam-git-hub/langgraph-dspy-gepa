"""GEPA metric: gold-aware LLM judge + deterministic signals, with feedback routed per predictor.

Score (0-1) = 0.5 correctness (judge vs gold) + 0.1 token-F1 + 0.2 faithfulness (judge) + 0.2 retrieval
recall of gold supporting paragraphs. This judge is separate from the in-graph judge and is never
optimized; it sees the gold answer, which makes it much harder to game than a reference-free judge.
"""
import re
import string
import threading
from collections import Counter
from functools import lru_cache

import numpy  # noqa: F401
import dspy

from app.config import JUDGE_MODEL
from app.retriever import format_context

WEIGHTS = {"correctness": 0.5, "f1": 0.1, "faithfulness": 0.2, "retrieval_recall": 0.2}


# ---------- deterministic signals (HotpotQA-style normalization) ----------
def normalize(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if not p or not g or overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def retrieval_recall(chunks: list[dict], supporting: list[str]) -> float:
    titles = {c["title"] for c in chunks or []}
    return len(titles & set(supporting)) / len(supporting) if supporting else 0.0


# ---------- gold-aware judge ----------
class GoldJudge(dspy.Signature):
    """You grade a question-answering system against a reference answer.
    correctness_score (1-5): 5 = semantically equivalent to the gold answer; 3 = partially correct;
    1 = wrong or no answer. Extra correct detail is fine; wrong extra claims lower the score.
    faithfulness_score (1-5): 5 = every claim in the answer is supported by the retrieved context;
    1 = mostly unsupported. Explain briefly what (if anything) is wrong and why."""

    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField()
    context: str = dspy.InputField(desc="Passages the system retrieved")
    answer: str = dspy.InputField(desc="The system's answer")
    rationale: str = dspy.OutputField()
    correctness_score: int = dspy.OutputField()
    faithfulness_score: int = dspy.OutputField()


@lru_cache(maxsize=1)
def judge_lm() -> dspy.LM:
    return dspy.LM(f"gemini/{JUDGE_MODEL}", temperature=0, max_tokens=4000, num_retries=8)


_judge = dspy.Predict(GoldJudge)
_cache: dict[tuple, dspy.Prediction] = {}
_lock = threading.Lock()


def gold_judge(question: str, gold: str, context: str, answer: str) -> dspy.Prediction:
    """Cached: GEPA asks for feedback per predictor on the same rollout; judge each rollout once."""
    key = (question, gold, context, answer)
    with _lock:
        if key in _cache:
            return _cache[key]
    with dspy.context(lm=judge_lm(), trace=None):
        out = _judge(question=question, gold_answer=gold, context=context, answer=answer or "")
    with _lock:
        _cache[key] = out
    return out


def _clip(x) -> float:
    try:
        return (min(5, max(1, int(x))) - 1) / 4
    except (TypeError, ValueError):
        return 0.0


def evaluate_one(gold: dspy.Example, pred: dspy.Prediction) -> dict:
    answer = pred.get("answer") or ""
    chunks = pred.get("chunks") or []
    j = gold_judge(gold.question, gold.answer, format_context(chunks), answer)
    parts = {
        "correctness": _clip(j.correctness_score),
        "f1": token_f1(answer, gold.answer),
        "faithfulness": _clip(j.faithfulness_score),
        "retrieval_recall": retrieval_recall(chunks, gold.supporting_titles),
    }
    parts["score"] = sum(WEIGHTS[k] * parts[k] for k in WEIGHTS)
    parts["rationale"] = j.rationale
    return parts


# ---------- the GEPA metric ----------
def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    r = evaluate_one(gold, pred)
    if pred_name is None:
        return r["score"]

    chunks = pred.get("chunks") or []
    retrieved = [c["title"] for c in chunks]
    missed = [t for t in gold.supporting_titles if t not in retrieved]

    if pred_name == "query_rewrite":
        fb = (
            f"The search query '{pred.get('search_query')}' retrieved: {retrieved}. "
            f"Paragraphs needed to answer: {gold.supporting_titles}. Retrieval recall = {r['retrieval_recall']:.2f}. "
        )
        fb += (
            f"MISSED: {missed}. The query must surface these; multi-hop questions often need the names of "
            f"BOTH entities involved (the gold answer was '{gold.answer}')."
            if missed else "All needed paragraphs were retrieved - good query."
        )
    else:  # answer
        fb = (
            f"Gold answer: '{gold.answer}'. System answer: '{pred.get('answer')}'. "
            f"Correctness {r['correctness']:.2f}, token-F1 {r['f1']:.2f}, faithfulness {r['faithfulness']:.2f}. "
            f"Judge: {r['rationale']} "
        )
        if r["f1"] < 0.5 and r["correctness"] >= 0.75:
            fb += ("The answer is right but wordy: keep the reasoning in the reasoning field and make the final "
                   "answer field a short, direct span (entity, date, number, yes/no). ")
        if missed:
            fb += f"Note: retrieval missed {missed}, so the context may be insufficient; reason only from given passages."
    return dspy.Prediction(score=r["score"], feedback=fb)
