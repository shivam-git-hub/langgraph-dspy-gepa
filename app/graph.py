"""The RAG QA agent as a plain LangGraph program: rewrite_query -> retrieve -> generate_answer -> judge."""
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.llm import call_llm
from app.retriever import format_context, retrieve


class RAGState(TypedDict, total=False):
    question: str
    search_query: str
    chunks: list[dict]
    context: str
    answer: str
    scores: dict


def rewrite_query(state: RAGState, config: RunnableConfig) -> RAGState:
    out = call_llm("query_rewrite", {"question": state["question"]}, config)
    return {"search_query": out.search_query}


def retrieve_node(state: RAGState) -> RAGState:
    chunks = retrieve(state["search_query"])
    return {"chunks": chunks, "context": format_context(chunks)}


def generate_answer(state: RAGState, config: RunnableConfig) -> RAGState:
    out = call_llm("answer", {"question": state["question"], "context": state["context"]}, config)
    return {"answer": out.answer}


def judge(state: RAGState, config: RunnableConfig) -> RAGState:
    out = call_llm(
        "judge",
        {"question": state["question"], "context": state["context"], "answer": state["answer"]},
        config,
    )
    return {"scores": out.model_dump()}


def _route_after_answer(state: RAGState, config: RunnableConfig) -> str:
    return END if config.get("configurable", {}).get("skip_judge") else "judge"


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate_answer", generate_answer)
    g.add_node("judge", judge)
    g.add_edge(START, "rewrite_query")
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("retrieve", "generate_answer")
    g.add_conditional_edges("generate_answer", _route_after_answer, ["judge", END])
    g.add_edge("judge", END)
    return g.compile()
