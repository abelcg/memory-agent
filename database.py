import sqlite3
import struct
import math
from datetime import datetime


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def serialize_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def deserialize_embedding(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def init_db(db_path: str = "memory.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT NOT NULL,
            embedding BLOB,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            access_count INTEGER DEFAULT 0,
            last_accessed DATETIME
        )
    """)

    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content, category,
            content='memories',
            content_rowid='id',
            tokenize='porter'
        )
    """)

    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, category)
            VALUES (new.id, new.content, new.category);
        END
    """)

    conn.commit()
    return conn


def save_memory(
    conn: sqlite3.Connection,
    user_id: str,
    content: str,
    category: str,
    embedding: list[float],
) -> dict:
    embedding_blob = serialize_embedding(embedding)
    cursor = conn.execute(
        """
        INSERT INTO memories (user_id, content, category, embedding)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, content, category, embedding_blob),
    )
    conn.commit()

    row = conn.execute(
        "SELECT id, content, category, created_at FROM memories WHERE id = ?",
        (cursor.lastrowid,),
    ).fetchone()

    return {
        "id": row[0],
        "content": row[1],
        "category": row[2],
        "created_at": row[3],
    }


def search_by_vector(
    conn: sqlite3.Connection,
    user_id: str,
    query_embedding: list[float],
    limit: int = 5,
) -> list[dict]:
    rows = conn.execute(
        "SELECT id, content, category, embedding, created_at FROM memories WHERE user_id = ? AND embedding IS NOT NULL",
        (user_id,),
    ).fetchall()

    scored = []
    for row_id, content, category, emb_blob, created_at in rows:
        emb = deserialize_embedding(emb_blob)
        score = cosine_similarity(query_embedding, emb)
        scored.append({
            "id": row_id,
            "content": content,
            "category": category,
            "created_at": created_at,
            "vector_score": score,
        })

    scored.sort(key=lambda x: x["vector_score"], reverse=True)

    top = scored[:limit]
    now = datetime.utcnow().isoformat()
    for item in top:
        conn.execute(
            "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (now, item["id"]),
        )
    conn.commit()

    return top


def search_by_bm25(
    conn: sqlite3.Connection,
    user_id: str,
    query: str,
    limit: int = 5,
) -> list[dict]:
    try:
        rows = conn.execute(
            """
            SELECT m.id, m.content, m.category, m.created_at, -fts.rank AS bm25_score
            FROM memories_fts fts
            JOIN memories m ON m.id = fts.rowid
            WHERE memories_fts MATCH ? AND m.user_id = ?
            ORDER BY fts.rank
            LIMIT ?
            """,
            (query, user_id, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    return [
        {
            "id": row[0],
            "content": row[1],
            "category": row[2],
            "created_at": row[3],
            "bm25_score": row[4],
        }
        for row in rows
    ]


def hybrid_search(
    conn: sqlite3.Connection,
    user_id: str,
    query: str,
    query_embedding: list[float],
    limit: int = 5,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> list[dict]:
    wider_limit = limit * 3

    vector_results = search_by_vector(conn, user_id, query_embedding, wider_limit)
    bm25_results = search_by_bm25(conn, user_id, query, wider_limit)

    # Min-max normalize vector scores
    if vector_results:
        v_scores = [r["vector_score"] for r in vector_results]
        v_min, v_max = min(v_scores), max(v_scores)
        v_range = v_max - v_min if v_max != v_min else 1.0
        for r in vector_results:
            r["vector_score_norm"] = (r["vector_score"] - v_min) / v_range
    vector_map = {r["id"]: r for r in vector_results}

    # Min-max normalize BM25 scores
    if bm25_results:
        b_scores = [r["bm25_score"] for r in bm25_results]
        b_min, b_max = min(b_scores), max(b_scores)
        b_range = b_max - b_min if b_max != b_min else 1.0
        for r in bm25_results:
            r["bm25_score_norm"] = (r["bm25_score"] - b_min) / b_range
    bm25_map = {r["id"]: r for r in bm25_results}

    # Merge candidates
    all_ids = set(vector_map.keys()) | set(bm25_map.keys())
    merged = []
    for mem_id in all_ids:
        v = vector_map.get(mem_id)
        b = bm25_map.get(mem_id)

        v_norm = v["vector_score_norm"] if v else 0.0
        b_norm = b["bm25_score_norm"] if b else 0.0
        hybrid_score = vector_weight * v_norm + bm25_weight * b_norm

        base = v or b
        merged.append({
            "id": base["id"],
            "content": base["content"],
            "category": base["category"],
            "created_at": base["created_at"],
            "hybrid_score": round(hybrid_score, 4),
            "vector_score": round(v["vector_score"], 4) if v else 0.0,
            "bm25_score": round(b["bm25_score"], 4) if b else 0.0,
        })

    merged.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return merged[:limit]
