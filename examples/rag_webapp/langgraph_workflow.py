"""LangGraph workflow for RAG queries."""

from __future__ import annotations

from langgraph.graph import EndNode, StateGraph
from langchain.chains import RetrievalQA


def create_workflow(retriever, llm):
    """Create a minimal LangGraph workflow."""
    graph = StateGraph()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    graph.add_node("qa", qa)
    graph.set_entry_point("qa")
    graph.set_finish_node(EndNode())
    return graph.compile()
