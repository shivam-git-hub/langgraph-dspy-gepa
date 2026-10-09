"""Prompt registry.

Every LLM call in the graph is described by a PromptSpec: the instruction text (stored in JSON so it
can be swapped for an optimized version) plus typed input fields and a Pydantic output schema.
This is the only contract the DSPy bridge relies on.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import BASELINE_PROMPTS


class SearchQuery(BaseModel):
    search_query: str = Field(description="A concise search query for the document retriever")


class Answer(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning over the context before answering")
    answer: str = Field(description="The answer to the question")


class JudgeScores(BaseModel):
    rationale: str = Field(description="Brief reasoning behind the scores")
    faithfulness_score: int = Field(ge=1, le=5, description="1-5: is every claim in the answer supported by the context?")
    context_relevance_score: int = Field(ge=1, le=5, description="1-5: is the retrieved context relevant to the question?")
    completeness_score: int = Field(ge=1, le=5, description="1-5: does the answer fully address the question?")


@dataclass(frozen=True)
class PromptSpec:
    name: str
    inputs: dict[str, str]  # input field name -> description
    output: type[BaseModel]


SPECS: dict[str, PromptSpec] = {
    "query_rewrite": PromptSpec(
        name="query_rewrite",
        inputs={"question": "The user's question"},
        output=SearchQuery,
    ),
    "answer": PromptSpec(
        name="answer",
        inputs={"question": "The user's question", "context": "Retrieved passages"},
        output=Answer,
    ),
    "judge": PromptSpec(
        name="judge",
        inputs={
            "question": "The user's question",
            "context": "Retrieved passages",
            "answer": "The answer to evaluate",
        },
        output=JudgeScores,
    ),
}


def load_prompts(path: Path | str | None = None) -> dict[str, str]:
    """Load instruction texts. A partial file (e.g. optimized.json) is layered over the baseline."""
    prompts = json.loads(Path(BASELINE_PROMPTS).read_text())
    if path is not None and Path(path) != Path(BASELINE_PROMPTS):
        prompts.update(json.loads(Path(path).read_text()))
    return prompts


def save_prompts(prompts: dict[str, str], path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(prompts, indent=2, ensure_ascii=False) + "\n")
