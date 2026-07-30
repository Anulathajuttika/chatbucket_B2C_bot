"""
LangGraph pipeline:

    START -> retrieve_context -> generate_answer -> END
"""
import logging
from typing import TypedDict, List

from langgraph.graph import StateGraph, START, END
from openai import OpenAI

import config
from retriever import retrieve_context
from prompt import build_messages

logger = logging.getLogger("graph")
client = OpenAI(api_key=config.OPENAI_API_KEY)


class ChatState(TypedDict):
    session_id: str
    question: str
    history: List[dict]
    context: str
    answer: str


def _translate_query_for_search(question: str) -> str:
    """Translate a user question into English for vector search only.

    knowledge.md is entirely in English, and cross-lingual embedding
    similarity (e.g. a Kannada or Gujarati query against English chunks) is
    much weaker than same-language matching -- this was measured to cause
    real grounding failures for non-English questions (relevant chunks
    scoring below MIN_RELEVANCE_SCORE purely because of the language
    mismatch, not because the content wasn't relevant). The original-
    language question is still what gets sent to the chat model for the
    actual reply -- this translation exists solely to make retrieval work.
    """
    try:
        response = client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[
                {"role": "system", "content": (
                    "Translate the user's message into English for use as a "
                    "search query. If it's already in English, return it "
                    "unchanged. Output ONLY the translation -- no quotes, "
                    "no explanation, no extra text."
                )},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        translated = (response.choices[0].message.content or "").strip()
        return translated or question
    except Exception as e:
        logger.error(f"Query translation for retrieval failed, falling back to original text: {e}")
        return question


def retrieve_context_node(state: ChatState) -> ChatState:
    """Node 1: retrieve relevant chunks from ChromaDB for the user's question."""
    search_query = _translate_query_for_search(state["question"])
    chunks = retrieve_context(search_query)
    state["context"] = "\n\n---\n\n".join(chunks) if chunks else ""
    logger.info(f"Retrieved {len(chunks)} chunk(s) for session {state['session_id']}")
    return state


def generate_answer_node(state: ChatState) -> ChatState:
    """Node 2: call OpenAI to generate an answer grounded in the retrieved context."""
    try:
        messages = build_messages(state["context"], state["question"], state.get("history", []))
        response = client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=messages,
            temperature=config.TEMPERATURE,
        )
        state["answer"] = (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.error(f"OpenAI generation failed: {e}")
        state["answer"] = "Sorry, something went wrong while generating a response. Please try again."
    return state


def build_graph():
    graph = StateGraph(ChatState)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


# compiled once at import time and reused across requests
chat_graph = build_graph()


def run_chat(session_id: str, question: str, history: list) -> str:
    """Invoke the compiled graph and return the final answer string."""
    result = chat_graph.invoke({
        "session_id": session_id,
        "question": question,
        "history": history,
        "context": "",
        "answer": "",
    })
    return result["answer"]