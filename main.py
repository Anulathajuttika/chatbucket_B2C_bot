"""
FastAPI entrypoint for the ChatBucket chatbot.

Workflow per request:
    User -> FastAPI -> LangGraph -> retrieve_context -> OpenAI -> answer
          -> save to Redis + MongoDB -> response
"""
import logging
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import config
from graph import run_chat
from database import get_session_history, save_to_mongo

# --- logging setup ---
os.makedirs(config.LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(config.LOGS_DIR, "app.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("main")

# --- rate limiting (per client IP) ---
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the vector DB on startup so the first user request isn't slow."""
    try:
        from retriever import get_vectorstore
        get_vectorstore()
        logger.info("Vector database is ready.")
    except Exception as e:
        logger.error(f"Failed to initialize vector database on startup: {e}")
    yield


app = FastAPI(title="ChatBucket Chatbot", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS: only allow the real ChatBucket site (and any others from .env) to call this from a browser ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
INDEX_HTML_PATH = os.path.join(BASE_DIR, "templates", "index.html")

# Session ids we hand out ourselves look like "sess-<32 hex chars>".
# Anything else is rejected so clients can't set arbitrary/guessable ids
# for another user's session.
_SESSION_ID_RE = re.compile(r"^sess-[0-9a-f]{32}$")


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1, max_length=config.MAX_MESSAGE_LENGTH)

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message cannot be empty/whitespace-only")
        return v

    @field_validator("session_id")
    @classmethod
    def valid_session_id(cls, v: str) -> str:
        v = v.strip()
        if not _SESSION_ID_RE.match(v):
            raise ValueError("invalid session_id format")
        return v


@app.get("/")
async def root():
    """Health check endpoint, as required by the API spec."""
    return {"status": "running"}


@app.get("/session/new")
async def new_session():
    """Issue a fresh, unguessable session id for the frontend to use.

    Frontend should call this once per new conversation instead of
    generating its own id, so session ids can't be forged/guessed.
    """
    import secrets
    return {"session_id": f"sess-{secrets.token_hex(16)}"}


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    """Serve the ChatGPT-style web interface."""
    with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/chat")
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
async def chat(request: Request, payload: ChatRequest):
    """Main chat endpoint: retrieve context, generate an answer, persist the turn."""
    session_id = payload.session_id
    message = payload.message

    try:
        history = get_session_history(session_id)
        answer = run_chat(session_id, message, history)

        # Defense-in-depth: never let a leaked system prompt reach the user,
        # even if the model was tricked into echoing it.
        if "official ChatBucket support assistant" in answer and "CONTEXT:" in answer:
            logger.warning(f"Blocked probable system-prompt leak for session {session_id}")
            answer = "I can't share that, but I'm happy to help with ChatBucket or ToDoZee questions!"

        save_to_mongo(session_id, message, answer)

        return {"response": answer}
    except Exception as e:
        logger.error(f"/chat endpoint error: {e}")
        return {"response": "Sorry, something went wrong. Please try again."}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return a clean 400 for bad input instead of FastAPI's default 422,
    which echoes the offending input and internal validation details back
    to the client."""
    logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(status_code=400, content={"response": "Invalid request."})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Last-resort safety net: return a non-leaky 500 for unexpected errors
    rather than a stack trace."""
    logger.error(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"response": "Something went wrong."})