"""
Handles building/loading the ChromaDB vector store from knowledge.md
and retrieving relevant chunks for a given user query.
"""
import os
import logging

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownTextSplitter

import config

logger = logging.getLogger("retriever")

_embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)
_vectorstore = None  # cached in-process singleton


def _build_vectorstore() -> Chroma:
    """Split knowledge.md into chunks, embed them, and persist to disk."""
    logger.info("No existing vector DB found. Building a new one from knowledge.md ...")

    with open(config.KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    # chunk_size=800 used to split several ToDoZee feature categories
    # mid-list (e.g. "AI Tools" is ~1150 chars for its 7 features), silently
    # dropping the tail items (Voice Call Summary, Homework Help) from any
    # retrieval that only pulled in the first half. Measured empirically:
    # 1300/150 is enough for every knowledge.md category/section (the longest,
    # "AI Tools", is ~1150 chars) to survive as one self-contained chunk while
    # still splitting the document into many distinct chunks overall.
    splitter = MarkdownTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_text(text)

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=_embeddings,
        persist_directory=config.VECTOR_DB_DIR,
        collection_name="chatbucket_knowledge",
    )
    logger.info(f"Vector database built successfully with {len(chunks)} chunks.")
    return vectorstore


def get_vectorstore() -> Chroma:
    """Return a cached vector store, loading it from disk or building it if missing."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    os.makedirs(config.VECTOR_DB_DIR, exist_ok=True)
    # Chroma persists a chroma.sqlite3 file once a collection has been written.
    db_exists = os.path.exists(os.path.join(config.VECTOR_DB_DIR, "chroma.sqlite3"))

    if db_exists:
        logger.info("Loading existing vector database from disk ...")
        _vectorstore = Chroma(
            persist_directory=config.VECTOR_DB_DIR,
            embedding_function=_embeddings,
            collection_name="chatbucket_knowledge",
        )
        # Guard against a stale/corrupted persisted DB (e.g. a prior build that
        # failed partway through, such as an embeddings-API error after the
        # sqlite file was already created). Without this check, an empty
        # collection loads "successfully" and silently returns zero chunks for
        # every query forever, starving every grounded question of context.
        if _vectorstore._collection.count() == 0:
            logger.warning(
                "Existing vector DB has 0 vectors — treating as corrupt/incomplete "
                "and rebuilding from knowledge.md ..."
            )
            _vectorstore = _build_vectorstore()
    else:
        _vectorstore = _build_vectorstore()

    return _vectorstore

def retrieve_context(query: str, k: int = config.TOP_K) -> list[str]:
    try:
        vectorstore = get_vectorstore()
        # similarity_search() alone returns the top-k regardless of score --
        # MIN_RELEVANCE_SCORE only actually filters anything if we use the
        # score-returning variant and apply the cutoff ourselves.
        docs_with_scores = vectorstore.similarity_search_with_relevance_scores(query, k=k)
        docs = [doc for doc, score in docs_with_scores if score >= config.MIN_RELEVANCE_SCORE]

        print("\n========== RETRIEVED CHUNKS ==========")
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            kept = "KEPT" if score >= config.MIN_RELEVANCE_SCORE else "dropped (below MIN_RELEVANCE_SCORE)"
            print(f"\n----- Chunk {i} (score={score:.4f}, {kept}) -----")
            print(doc.page_content)
        print("======================================\n")

        return [doc.page_content for doc in docs]
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return []