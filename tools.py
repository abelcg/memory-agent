from google.genai import types

from database import hybrid_search, save_memory
from embeddings import get_embedding

# --- Tool declarations for Gemini function calling ---

MEMORY_TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="memory_search",
            description=(
                "Search long-term memory for relevant past context. "
                "Use this when the user references past conversations, asks about "
                "their preferences, or when prior context would help you give a better answer."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="Natural language search query",
                    ),
                    "limit": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of results to return (default: 5)",
                    ),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="memory_save",
            description=(
                "Save important information to long-term memory. "
                "Use this when the user shares preferences, key facts about themselves, "
                "decisions, or project context that should be remembered across sessions. "
                "Do NOT save trivial greetings or redundant information."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "content": types.Schema(
                        type=types.Type.STRING,
                        description="The fact or information to remember",
                    ),
                    "category": types.Schema(
                        type=types.Type.STRING,
                        description="Category: preference, fact, decision, context, or general",
                    ),
                },
                required=["content", "category"],
            ),
        ),
    ]
)

VALID_CATEGORIES = {"preference", "fact", "decision", "context", "general"}


def execute_memory_tool(name: str, input_data: dict, user_id: str, conn) -> str:
    if name == "memory_search":
        query = input_data["query"]
        limit = int(input_data.get("limit", 5))
        query_embedding = get_embedding(query)
        results = hybrid_search(conn, user_id, query, query_embedding, limit)

        if not results:
            return "No memories found."

        lines = []
        for r in results:
            lines.append(
                f"[Score: {r['hybrid_score']:.2f}] ({r['category']}) "
                f"{r['content']} (saved: {r['created_at']})"
            )
        return "\n".join(lines)

    elif name == "memory_save":
        content = input_data["content"]
        category = input_data.get("category", "general")
        if category not in VALID_CATEGORIES:
            category = "general"

        embedding = get_embedding(content)
        result = save_memory(conn, user_id, content, category, embedding)
        return f"Memory saved (id={result['id']}, category={result['category']}): {result['content']}"

    else:
        return f"Error: unknown tool '{name}'"
