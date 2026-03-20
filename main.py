import os
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

from database import init_db
from agent import configure_client, chat, reset_session

app = FastAPI(title="Memory Agent", version="0.1.0")

db_conn = None


@app.on_event("startup")
def startup():
    global db_conn
    configure_client(api_key=os.environ["GOOGLE_API_KEY"])
    db_conn = init_db()


# --- Request / Response models ---

class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class ResetRequest(BaseModel):
    session_id: str


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    session_id = req.session_id or f"{req.user_id}-{uuid4().hex[:8]}"

    try:
        response_text = chat(req.user_id, req.message, session_id, db_conn)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(response=response_text, session_id=session_id)


@app.post("/chat/reset")
def reset_endpoint(req: ResetRequest):
    reset_session(req.session_id)
    return {"status": "session reset"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
