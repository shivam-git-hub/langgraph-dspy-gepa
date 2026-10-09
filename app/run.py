"""CLI: python -m app.run "question" [--prompts prompts/optimized.json]"""
import argparse
import json

from app.graph import build_graph
from app.prompts import load_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--prompts", default=None, help="prompt JSON layered over baseline")
    args = ap.parse_args()

    graph = build_graph()
    state = graph.invoke(
        {"question": args.question},
        config={"configurable": {"prompts": load_prompts(args.prompts)}},
    )
    print(f"Search query: {state['search_query']}\n")
    print("Retrieved:", ", ".join(c["title"] for c in state["chunks"]), "\n")
    print(f"Answer: {state['answer']}\n")
    print(json.dumps(state["scores"], indent=2))


if __name__ == "__main__":
    main()
