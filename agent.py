import os
import sqlite3
import time

from google import genai
from google.genai import types
from google.genai.errors import APIError

from tools import MEMORY_TOOLS, execute_memory_tool

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

SYSTEM_PROMPT = """You are a helpful assistant with persistent memory that works across conversations.

CRITICAL RULES:
1. ALWAYS call memory_search FIRST before responding to ANY message. Search using keywords from the user's message to find relevant past context. This is mandatory — never skip this step.
2. When the user shares important information (preferences, facts about themselves, decisions, project context), ALWAYS call memory_save to store it.
3. When answering, use the context from memory search results to personalize your response. Reference what you know about the user from memory.
4. Do NOT save trivial greetings or redundant information already found in memory.
5. Be natural about memory usage — do not announce every search or save operation to the user.
6. Use appropriate categories when saving: preference, fact, decision, context, general.
"""

MAX_TOOL_ITERATIONS = 10

CHAT_CONFIG = types.GenerateContentConfig(
    tools=[MEMORY_TOOLS],
    system_instruction=SYSTEM_PROMPT,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

# Session storage: session_id -> Chat
_sessions: dict = {}

# Client (set after configure)
_client: genai.Client | None = None


def configure_client(api_key: str):
    global _client
    _client = genai.Client(api_key=api_key)


def _get_or_create_session(session_id: str):
    if session_id not in _sessions:
        _sessions[session_id] = _client.chats.create(model=GEMINI_MODEL)
    return _sessions[session_id]


def reset_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


def _send_with_retry(chat, message, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return chat.send_message(message=message, config=CHAT_CONFIG)
        except APIError:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt * 10
            print(f"Rate limited, retrying in {wait}s...")
            time.sleep(wait)


def chat(user_id: str, user_message: str, session_id: str, conn: sqlite3.Connection) -> str:
    session = _get_or_create_session(session_id)
    response = _send_with_retry(session, user_message)

    for _ in range(MAX_TOOL_ITERATIONS):
        if not response.function_calls:
            return response.text

        function_responses = []
        for fc in response.function_calls:
            tool_result = execute_memory_tool(
                fc.name, dict(fc.args), user_id, conn
            )
            function_responses.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": tool_result},
                )
            )

        response = _send_with_retry(session, function_responses)

    return response.text if response.text else "I'm sorry, I couldn't complete that request."
