"""SqliteStore：WAL + 线程局部连接（spec §8.1 grill-me 决议 8）。

sqlite3 为同步阻塞 API：所有方法经 asyncio.to_thread 桥接；
连接存于 threading.local，绝不跨线程复用。
"""
import asyncio
import sqlite3
import threading
from datetime import datetime, timezone

import numpy as np

from app.domain import ErrorCode, IndexStatus, PostRecord, SimilarityResult, utcnow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    post_id     TEXT PRIMARY KEY,
    image_url   TEXT,
    image_base64 TEXT,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    enqueued_at TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    note        TEXT,
    image_vec   BLOB,
    text_vec    BLOB
);
CREATE TABLE IF NOT EXISTS results (
    post_id         TEXT PRIMARY KEY REFERENCES posts(post_id),
    max_sim         REAL NOT NULL,
    sim_image       REAL NOT NULL,
    sim_text        REAL NOT NULL,
    matched_post_id TEXT,
    computed_at     TEXT NOT NULL,
    note            TEXT
);
CREATE INDEX IF NOT EXISTS idx_posts_status ON posts(status, enqueued_at);
"""


def encode_vector(v: np.ndarray) -> bytes:
    return np.ascontiguousarray(v, dtype=np.float32).tobytes()


def decode_vector(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32).copy()


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse(s: str) -> datetime:
    return datetime.fromisoformat(s)


class SqliteStore:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self._gen = 0  # 连接代际：close() 后递增，旧线程局部连接作废

    # ---- 线程局部连接 ----
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        gen = getattr(self._local, "gen", -1)
        if conn is None or gen != self._gen:
            # check_same_thread=False：连接仍只在创建它的线程内执行语句，
            # 仅 close() 从事件循环线程跨线程关闭（Py3.10+ 安全）
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
            self._local.gen = self._gen
            with self._lock:
                self._conns.append(conn)
        return conn

    def _execute(self, sql: str, params=()):
        conn = self._conn()
        with conn:  # 事务边界
            cur = conn.execute(sql, params)
        return cur

    def _query(self, sql: str, params=()):
        return self._conn().execute(sql, params).fetchall()

    # ---- 公开 async API（全部 to_thread 桥接）----
    async def init(self):
        # 惰性建连：连接在 to_thread 的 worker 线程内创建，避免跨线程复用
        def _do():
            self._conn().executescript(_SCHEMA)
        await asyncio.to_thread(_do)

    async def close(self):
        # 关闭全部 worker 线程创建的连接（登记簿）；Py3.10+ 允许跨线程 close；
        # 代际递增使各 worker 线程残留的已关闭连接引用自动失效、按需重建
        with self._lock:
            conns, self._conns = self._conns, []
            self._gen += 1
        for c in conns:
            c.close()
        self._local.conn = None

    async def upsert_post(self, rec: PostRecord) -> bool:
        def _do():
            cur = self._execute(
                """INSERT INTO posts(post_id,image_url,image_base64,text,created_at,
                                     enqueued_at,status,note)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(post_id) DO NOTHING""",
                (rec.post_id, rec.image_url, rec.image_base64, rec.text,
                 _iso(rec.created_at), _iso(rec.enqueued_at),
                 rec.status.value, rec.note))
            return cur.rowcount == 1
        return await asyncio.to_thread(_do)

    async def get_post(self, post_id: str) -> PostRecord | None:
        rows = await asyncio.to_thread(
            self._query,
            "SELECT post_id,image_url,image_base64,text,created_at,enqueued_at,status,note "
            "FROM posts WHERE post_id=?", (post_id,))
        if not rows:
            return None
        r = rows[0]
        return PostRecord(post_id=r[0], image_url=r[1], image_base64=r[2], text=r[3],
                          created_at=_parse(r[4]), enqueued_at=_parse(r[5]),
                          status=IndexStatus(r[6]), note=r[7])

    async def persist_vectors(self, post_id, image_vec: bytes, text_vec: bytes):
        await asyncio.to_thread(
            self._execute,
            "UPDATE posts SET image_vec=?, text_vec=? WHERE post_id=?",
            (image_vec, text_vec, post_id))

    async def mark_indexed(self, post_id, result: SimilarityResult):
        def _do():
            conn = self._conn()
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO results VALUES(?,?,?,?,?,?,?)",
                    (post_id, result.max_sim, result.sim_image, result.sim_text,
                     result.matched_post_id, _iso(result.computed_at), result.note))
                conn.execute("UPDATE posts SET status='indexed' WHERE post_id=?", (post_id,))
        await asyncio.to_thread(_do)

    async def mark_failed(self, post_id, code: ErrorCode):
        await asyncio.to_thread(
            self._execute,
            "UPDATE posts SET status='failed', note=? WHERE post_id=?",
            (code.value, post_id))

    async def get_result(self, post_id) -> SimilarityResult | None:
        rows = await asyncio.to_thread(
            self._query, "SELECT * FROM results WHERE post_id=?", (post_id,))
        if not rows:
            return None
        r = rows[0]
        return SimilarityResult(post_id=r[0], max_sim=r[1], sim_image=r[2], sim_text=r[3],
                                matched_post_id=r[4], computed_at=_parse(r[5]), note=r[6])

    async def list_stale_pending(self, older_than: datetime) -> list[str]:
        rows = await asyncio.to_thread(
            self._query,
            "SELECT post_id FROM posts WHERE status='pending' AND enqueued_at<?",
            (_iso(older_than),))
        return [r[0] for r in rows]

    async def list_indexed_within(self, cutoff: datetime):
        rows = await asyncio.to_thread(
            self._query,
            "SELECT post_id,image_vec,text_vec,created_at FROM posts "
            "WHERE status='indexed' AND image_vec IS NOT NULL AND created_at>=?",
            (_iso(cutoff),))
        return [(r[0], r[1], r[2], _parse(r[3])) for r in rows]

    async def reset_failed_to_pending(self, post_id) -> bool:
        cur = await asyncio.to_thread(
            self._execute,
            "UPDATE posts SET status='pending', note=NULL WHERE post_id=? AND status='failed'",
            (post_id,))
        return cur.rowcount == 1

    async def counts(self) -> dict:
        rows = await asyncio.to_thread(
            self._query, "SELECT status, COUNT(*) FROM posts GROUP BY status")
        d = {s: n for s, n in rows}
        return {"index_count": d.get("indexed", 0), "failed_count": d.get("failed", 0)}

    async def oldest_pending_age(self) -> float:
        """最早 pending 帖的入队滞留秒数（spec §6.3/§7.4 /health 指标）；无 pending 返回 0。"""
        rows = await asyncio.to_thread(
            self._query, "SELECT MIN(enqueued_at) FROM posts WHERE status='pending'")
        oldest = rows[0][0]
        if oldest is None:
            return 0.0
        return (utcnow() - _parse(oldest)).total_seconds()
