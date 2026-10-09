"""Single entry point for every LLM call in the graph.

Production path: LangChain + Gemini with structured output, instructions from the prompt registry.
Optimization path: if the run config carries a DSPy predictor for this prompt name
(`configurable.dspy_predictors[name]`), the call is routed to it instead, so DSPy can trace and
optimize it. Nodes never know which backend served them.
"""
from functools import lru_cache

from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from app.config import GEMINI_API_KEY, JUDGE_MODEL, TASK_MODEL
from app.prompts import SPECS, load_prompts


@lru_cache(maxsize=None)
def chat_model(model: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=model, google_api_key=GEMINI_API_KEY, temperature=0, max_retries=6)


@lru_cache(maxsize=None)
def _default_prompts() -> dict[str, str]:
    return load_prompts()


def render_inputs(inputs: dict[str, str]) -> str:
    return "\n\n".join(f"## {k}\n{v}" for k, v in inputs.items())


def call_llm(name: str, inputs: dict[str, str], config: RunnableConfig | None = None) -> BaseModel:
    spec = SPECS[name]
    cfg = (config or {}).get("configurable", {})

    predictor = (cfg.get("dspy_predictors") or {}).get(name)
    if predictor is not None:
        pred = predictor(**inputs)
        return spec.output(**{f: pred[f] for f in spec.output.model_fields})

    instructions = (cfg.get("prompts") or {}).get(name) or _default_prompts()[name]
    model = JUDGE_MODEL if name == "judge" else TASK_MODEL
    llm = chat_model(model).with_structured_output(spec.output)
    return llm.invoke([("system", instructions), ("human", render_inputs(inputs))])
