"""
Central configuration for the ChatBucket chatbot.
All settings are loaded from environment variables (.env file).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL =  "gpt-4.1-mini"
TEMPERATURE = 0.7

# --- MongoDB (persistent chat history) ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = "chatbucket"
MONGO_COLLECTION_NAME = "chat_history"

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "knowledge.md")
COLLECTION_NAME = "chatbucket_knowledge"
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# --- Retrieval ---
# ToDoZee's full feature list only lives in knowledge.md as ~7-8 separate
# ~700-800-char chunks (one per category, e.g. "AI Tools", "Spiritual",
# "Kitchen Zone") plus the ToDoZee doc's own "Quick feature index" table
# chunk(s) -- no single chunk, or even the top 8, contains anywhere near the
# full 29/30-item list. Measured empirically against this knowledge base:
# retrieving k=20 candidates (then filtering by MIN_RELEVANCE_SCORE below) is
# what it actually takes for every category chunk + the quick-index chunk to
# be in the candidate pool at once; k=25/30 retrieved nothing further, so 20
# is not an arbitrary/oversized guess. This was also independently confirmed
# by a retrieval audit on single-fact questions (not just enumeration): the
# correct chunk for "What are Channels?" ranked #9 and for "How do I create a
# group?" ranked #7 -- both past the old TOP_K=8, which was silently dropping
# the right answer for ordinary single-feature questions, not just broad
# "list everything" ones. Raising k this much is safe because
# MIN_RELEVANCE_SCORE filters every candidate by score regardless of k
# (verified: greetings/small-talk/off-topic/prompt-injection queries came back
# with ZERO chunks above the cutoff even at k=30 -- see MIN_RELEVANCE_SCORE
# note below). IMPORTANT: this filtering must actually be applied in
# retriever.py's retrieve_context() via similarity_search_with_relevance_scores
# -- plain similarity_search() ignores this value entirely.
TOP_K = 20 # more chunks = more complete context for broad/comparative questions

# Chunks scoring below this are treated as "not relevant" (tune 0-1).
# NOTE: Chroma's default relevance-score function for this embedding model is
# NOT a clean 0-1 cosine similarity (Chroma itself warns scores can fall
# outside [0, 1]) -- empirically, genuinely relevant knowledge.md chunks for
# focused grounded questions (a single named feature/fact) score ~0.15-0.55,
# while off-topic/small-talk/injection queries top out ~-0.3 to +0.04 (the
# single highest off-topic score observed across many probes -- greetings,
# movie/recipe/legal-advice requests, "ignore previous instructions", "what
# does the CONTEXT say verbatim" -- was ~0.041). A cutoff of 0.13 filtered out
# nearly every real answer to "list ALL X" style questions: knowledge.md's
# per-category ToDoZee feature chunks (e.g. "AI Tools", "Kitchen Zone",
# "Spiritual") are short bullet lists of proper nouns with little restating of
# "ToDoZee"/"feature" framing, so they embed only ~0.02-0.10 against a broad
# "tell all features" query even though they ARE the answer -- 0.13 excluded
# them all. 0.06 was chosen to sit just above the highest observed off-topic/
# injection score (~0.04, leaving a real margin, re-verified after the
# knowledge.md retitling below) and below the lowest observed genuinely-
# relevant score for this enumeration case (~0.068). Re-tune if knowledge.md
# or the embedding model changes.
MIN_RELEVANCE_SCORE = 0.06

# --- Input limits / abuse protection ---
MAX_MESSAGE_LENGTH = 2000          # reject messages longer than this (chars)
RATE_LIMIT_PER_MINUTE = 15         # requests per session/IP per minute on /chat

# --- Data retention ---
MONGO_RETENTION_DAYS = 90  # auto-delete chat_history docs older than this (TTL index)

# --- CORS ---
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "https://chatbucket.chat"
).split(",")