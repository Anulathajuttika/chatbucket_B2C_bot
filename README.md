# ChatBucket Bot

A simple, production-ready RAG chatbot that answers questions **only** about
**ChatBucket** and **ToDoZee** (its built-in AI assistant), grounded strictly
in `knowledge.md`.

## Architecture

```
User → FastAPI (/chat) → LangGraph
                              ├─ retrieve_context  (ChromaDB similarity search)
                              └─ generate_answer    (OpenAI gpt-4.1-mini)
                          → Redis (session cache, 30-min TTL, last 10 turns)
                          → MongoDB (permanent chat_history collection)
                          → JSON response
```

- **Vector DB**: built automatically from `knowledge.md` on first run using
  `text-embedding-3-small`, and persisted to `vector_db/`. Subsequent runs
  reuse the existing store — no re-embedding.
- **LangGraph**: a 2-node graph — `retrieve_context → generate_answer`.
- **Guardrails**: if the answer isn't in the retrieved context, the bot replies
  *"I couldn't find that information in the ChatBucket knowledge base."*
  If the question is unrelated to ChatBucket/ToDoZee, it replies
  *"I'm designed to answer only questions related to ChatBucket and ToDoZee."*

## Project Structure

```
chatbucket_bot/
│
├── main.py            # FastAPI app, routes, startup vector-db warmup
├── graph.py            # LangGraph 2-node pipeline
├── retriever.py        # ChromaDB build/load + similarity search
├── database.py         # Redis (cache) + MongoDB (history)
├── config.py            # All settings, env vars, paths
├── prompt.py           # System prompt + message builder
├── knowledge.md        # ONLY knowledge source (ChatBucket + ToDoZee)
├── requirements.txt
├── .env.example
│
├── templates/
│   └── index.html      # Chat UI shell
├── static/
│   ├── style.css        # Dark, ChatGPT-style theme
│   └── script.js        # Chat logic (fetch, markdown render, typing dots)
│
├── vector_db/           # Persisted Chroma collection (auto-created)
└── logs/                # app.log
```

## Prerequisites

- Python 3.12
- A running **Redis** server
- A running **MongoDB** server
- An **OpenAI API key**
- [`uv`](https://github.com/astral-sh/uv) for dependency management

## Installation

```bash
cd chatbucket_bot

# create and activate a virtual environment
uv venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# install dependencies
uv pip install -r requirements.txt

# configure environment
cp .env.example .env
# then edit .env and set OPENAI_API_KEY (and REDIS_URL / MONGO_URI if not local)
```

## Running

Make sure Redis and MongoDB are running locally (or update `.env` to point to
your instances), then:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Web UI: **http://localhost:8000/ui**
- Health check: **http://localhost:8000/**
- Chat API: **POST http://localhost:8000/chat**

On the very first run, the app builds the Chroma vector store from
`knowledge.md` and persists it under `vector_db/`. Every run after that loads
the existing store directly — delete `vector_db/` if you edit `knowledge.md`
and want it re-indexed.

## API Reference

### `GET /`
```json
{ "status": "running" }
```

### `POST /chat`
Request:
```json
{ "session_id": "123", "message": "What is ChatBucket?" }
```
Response:
```json
{ "response": "ChatBucket is a next-generation chatting application..." }
```

## Notes

- Temperature is fixed at `0` for deterministic, grounded answers.
- Redis stores only the last 10 turns per `session_id`, expiring after 30
  minutes of inactivity.
- MongoDB (`chatbucket.chat_history`) stores the full conversation log with
  `session_id`, `user_message`, `bot_response`, and `timestamp`. A TTL index
  auto-deletes records older than `MONGO_RETENTION_DAYS` (default 90).
- All errors (OpenAI failures, Redis/Mongo outages, retrieval issues) are
  caught and logged to `logs/app.log`; the API still returns a graceful
  fallback message instead of crashing.

## Recent fixes (security + conversational quality)

- **Prompt behavior fixed**: the bot used to refuse *everything* not found
  verbatim in `knowledge.md`, including greetings ("hi", "thanks"). The
  system prompt in `prompt.py` now has explicit rules for small talk vs.
  product questions vs. off-topic requests, and off-topic/no-answer replies
  are natural instead of a robotic fixed string.
- **Retrieval no longer force-feeds irrelevant chunks**: `retriever.py` now
  applies a relevance-score cutoff (`config.MIN_RELEVANCE_SCORE`) so a
  greeting doesn't get the "nearest anyway" knowledge chunks as if they were
  relevant context.
- **Session ids are now server-issued** (`GET /session/new`) instead of
  generated client-side, closing an easy session-guessing/hijack path.
  `script.js` and `main.py` both updated; old-format ids are rejected (422).
- **Input validation**: `/chat` now rejects empty/whitespace messages and
  messages over `MAX_MESSAGE_LENGTH` (2000 chars) with a 422, instead of
  silently accepting arbitrarily large payloads.
- **Rate limiting**: `slowapi` limits `/chat` to `RATE_LIMIT_PER_MINUTE`
  (default 15) requests per client per minute, returning 429 past that.
- **CORS locked down**: only origins in `ALLOWED_ORIGINS` (set via `.env`)
  can call the API from a browser.
- **XSS fix**: `script.js` now sanitizes the bot's rendered Markdown/HTML
  with DOMPurify before inserting it into the page, since Markdown rendering
  otherwise passes raw HTML through unsanitized.
- **System-prompt leak guard**: `main.py` checks the model's answer for
  signs it echoed the system prompt (e.g. from a prompt-injection attempt)
  and swaps in a safe fallback if so.
- **Data retention**: MongoDB chat history now auto-expires after
  `MONGO_RETENTION_DAYS` instead of being kept forever.

All of the above were verified by running the actual app code (mocking
Redis/MongoDB/OpenAI) through a battery of test scenarios — not just
inspected by reading.