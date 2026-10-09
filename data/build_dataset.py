"""Sample HotpotQA (distractor) and build:
  - one shared Chroma collection with every paragraph of every sampled question (gold + distractors),
  - train/val/test JSONL splits of {question, answer, supporting_titles}.

python -m data.build_dataset [--n 200] [--seed 0]
"""
import argparse
import json
import random

import numpy  # noqa: F401
from datasets import load_dataset

from app.config import DATA_DIR
from app.retriever import get_collection

SPLITS = {"train": 50, "val": 50, "test": 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=sum(SPLITS.values()))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    idx = random.Random(args.seed).sample(range(len(ds)), args.n)
    rows = [ds[i] for i in idx]

    paragraphs: dict[str, str] = {}
    for r in rows:
        for title, sents in zip(r["context"]["title"], r["context"]["sentences"]):
            paragraphs.setdefault(title, "".join(sents).strip())

    col = get_collection()
    titles = list(paragraphs)
    for i in range(0, len(titles), 500):
        batch = titles[i : i + 500]
        col.upsert(
            ids=batch,
            documents=[paragraphs[t] for t in batch],
            metadatas=[{"title": t} for t in batch],
        )
    print(f"Indexed {col.count()} paragraphs")

    start = 0
    for name, size in SPLITS.items():
        with open(DATA_DIR / f"{name}.jsonl", "w") as f:
            for r in rows[start : start + size]:
                f.write(json.dumps({
                    "id": r["id"],
                    "question": r["question"],
                    "answer": r["answer"],
                    "supporting_titles": sorted(set(r["supporting_facts"]["title"])),
                    "type": r["type"],
                }) + "\n")
        start += size
        print(f"{name}: {size}")


if __name__ == "__main__":
    main()
