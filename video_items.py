"""
Curated, human-approved content pool for AutoTube (short-form video narration).

The RAG store (`sources`) holds book *fragments* embedded for retrieval — good for
grounding an answer, useless as a citable unit of content: a chunk may hold half a
hadith, three masail, or a hadith whose reference sits in the next chunk.

This module adds a layer ON TOP of those chunks (it never modifies them):

    sources (RAG chunks)  --extract_video_items.py-->  video_items (draft)
                                  --/admin/video-items (a human)-->  approved
                                  --/api/autotube/next-->  AutoTube

Religious-accuracy invariants, enforced here rather than trusted to callers:

  * Nothing is ever inserted as anything but `draft`. There is no code path in this
    repo that sets status='approved' without an authenticated human doing it.
  * A row can only be BOTH approved and video-enabled when its citation is complete,
    a hadith carries an explicit sahih/hasan grading, it is not fiqh, and it is short
    enough for a 30-60s Short. That is a DB CHECK constraint, not an app-level `if`:
    the review UI can have a bug, the constraint cannot be bypassed.
"""
import hashlib
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import database as db

CATEGORIES = ("hadith", "tafsir", "hikayat", "fiqh")
DARJAT = ("sahih", "hasan", "zaeef", "na_maloom")
STATUSES = ("draft", "approved", "rejected")

# A 30-60s Short at Urdu narration pace. Also the outer bound of what a reviewer can
# verify against the source at a glance.
MIN_WORDS = 25
MAX_WORDS = 160

# How long an AutoTube run holds an item before it returns to the tenant's pool.
RESERVE_HOURS = int(os.getenv("AUTOTUBE_RESERVE_HOURS", "6"))

# Requests per hour per tenant. A tenant makes roughly one call a day; this only
# exists so a looping client cannot walk the whole pool in a minute.
RATE_LIMIT_PER_HOUR = int(os.getenv("AUTOTUBE_RATE_LIMIT", "20"))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_video_items():
    """Create the curation/delivery tables (idempotent, same style as init_db)."""
    with db.get_cursor(commit=True) as cur:
        cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
        cur.execute("""
            CREATE TABLE IF NOT EXISTS video_items (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                category TEXT NOT NULL
                    CHECK (category IN ('hadith', 'tafsir', 'hikayat', 'fiqh')),
                text_ur TEXT NOT NULL,
                text_roman TEXT,
                citation JSONB NOT NULL DEFAULT '{}'::jsonb,
                citation_display TEXT NOT NULL DEFAULT '',
                citation_incomplete BOOLEAN NOT NULL DEFAULT TRUE,
                darja TEXT CHECK (darja IS NULL OR darja IN
                    ('sahih', 'hasan', 'zaeef', 'na_maloom')),
                maslak_tag TEXT,
                source_chunk_ids TEXT[] NOT NULL DEFAULT '{}',
                content_hash TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'draft'
                    CHECK (status IN ('draft', 'approved', 'rejected')),
                allow_video BOOLEAN NOT NULL DEFAULT FALSE,
                word_count INTEGER NOT NULL DEFAULT 0,
                extractor_version TEXT,
                review_note TEXT,
                approved_at TIMESTAMP WITH TIME ZONE,
                approved_by TEXT,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
            );
        """)
        # The last line of defence. Written as a named constraint added separately so
        # a database created before this rule existed also gets it.
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'video_items_publishable'
                ) THEN
                    ALTER TABLE video_items ADD CONSTRAINT video_items_publishable CHECK (
                        NOT (status = 'approved' AND allow_video = TRUE)
                        OR (
                            citation_incomplete = FALSE
                            AND (category <> 'hadith' OR darja IN ('sahih', 'hasan'))
                            AND category <> 'fiqh'
                            AND word_count BETWEEN 25 AND 160
                        )
                    );
                END IF;
            END $$;
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_video_items_serveable "
            "ON video_items(status, allow_video, category);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_video_items_queue "
            "ON video_items(status, created_at);"
        )

        cur.execute("""
            CREATE TABLE IF NOT EXISTS video_item_serves (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                tenant_id TEXT NOT NULL,
                item_id UUID NOT NULL REFERENCES video_items(id) ON DELETE CASCADE,
                status TEXT NOT NULL DEFAULT 'reserved'
                    CHECK (status IN ('reserved', 'used', 'released')),
                run_id TEXT,
                reserved_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                acked_at TIMESTAMP WITH TIME ZONE
            );
        """)
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_serves_tenant_item "
            "ON video_item_serves(tenant_id, item_id);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_serves_tenant_status "
            "ON video_item_serves(tenant_id, status);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_serves_sweep "
            "ON video_item_serves(status, expires_at);"
        )

        cur.execute("""
            CREATE TABLE IF NOT EXISTS autotube_rate_limit (
                tenant_id TEXT NOT NULL,
                window_start TIMESTAMP WITH TIME ZONE NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (tenant_id, window_start)
            );
        """)
    print("Video item store initialized")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

# Urdu/Arabic combining marks, tatweel and the zero-width joiners. Stripped only for
# COMPARISON (hashing, verbatim checking, highlighting) — never from stored text.
_HARAKAT_RE = re.compile("[ً-ْٰۖ-ۭـ‌‍]")
_WS_RE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """NFC + strip diacritics/tatweel + collapse whitespace. Comparison form only."""
    s = unicodedata.normalize("NFC", text or "")
    s = _HARAKAT_RE.sub("", s)
    return _WS_RE.sub(" ", s).strip()


def content_hash(text_ur: str) -> str:
    """Idempotency key for re-runs: same passage extracted twice hashes the same."""
    return hashlib.sha256(normalise(text_ur).encode("utf-8")).hexdigest()


def word_count(text_ur: str) -> int:
    return len(_WS_RE.sub(" ", (text_ur or "").strip()).split()) if (text_ur or "").strip() else 0


def is_verbatim(candidate: str, window_text: str) -> bool:
    """True when `candidate` physically occurs in the source window.

    The extraction agent is only permitted to RESTRUCTURE text that is present in its
    input. This is the mechanical check of that rule: if the model paraphrased,
    completed, or 'corrected' a passage, the result is not in the window and the item
    is dropped. Comparison ignores diacritics and whitespace only.
    """
    return bool(candidate) and normalise(candidate) in normalise(window_text)


# ---------------------------------------------------------------------------
# Publishability
# ---------------------------------------------------------------------------

def publishable_failures(item: Dict[str, Any]) -> List[str]:
    """Which §3a conditions block 'approve & allow video' — mirrors the CHECK.

    The DB constraint is the enforcement; this exists so the UI can grey the button
    out and SAY WHY instead of surfacing a raw constraint violation.
    """
    fails = []
    if item.get("citation_incomplete"):
        fails.append("citation is incomplete")
    if item.get("category") == "hadith" and item.get("darja") not in ("sahih", "hasan"):
        fails.append("hadith grading must be sahih or hasan")
    if item.get("category") == "fiqh":
        fails.append("fiqh is not served to video in v1")
    wc = int(item.get("word_count") or 0)
    if wc < MIN_WORDS or wc > MAX_WORDS:
        fails.append(f"length {wc} words is outside {MIN_WORDS}-{MAX_WORDS}")
    return fails


# ---------------------------------------------------------------------------
# Reading the RAG chunks (read-only; the ingestion pipeline is never touched)
# ---------------------------------------------------------------------------

# Chunk order inside a book is not stored as a column, but it IS recoverable:
# tags = [book_slug, 'jild-N', 'page_NNNN.txt'] and rows are inserted one page at a
# time, one row per transaction, so created_at strictly increases within a page.
# id is the final tiebreaker so the order is total and stable across runs.
_CHUNK_ORDER = (
    "ORDER BY NULLIF(regexp_replace(tags[2], '\\D', '', 'g'), '')::int NULLS FIRST, "
    "NULLIF(regexp_replace(tags[3], '\\D', '', 'g'), '')::int NULLS FIRST, "
    "created_at, id"
)


def ordered_chunks(book_slug: str, jild: Optional[int] = None) -> List[Dict[str, Any]]:
    """Every chunk of one book in reading order (see _CHUNK_ORDER)."""
    sql = (
        "SELECT id::text AS id, title, reference, content, tags "
        "FROM sources WHERE tags[1] = %s "
    )
    params: List[Any] = [book_slug]
    if jild is not None:
        sql += "AND tags[2] = %s "
        params.append(f"jild-{jild}")
    with db.get_cursor() as cur:
        cur.execute(sql + _CHUNK_ORDER + ";", params)
        return [dict(r) for r in cur.fetchall()]


def chunks_by_ids(ids: List[str]) -> List[Dict[str, Any]]:
    """Source chunks behind an item, in reading order — the reviewer's right pane."""
    if not ids:
        return []
    with db.get_cursor() as cur:
        cur.execute(
            "SELECT id::text AS id, title, reference, content, tags "
            "FROM sources WHERE id = ANY(%s::uuid[]) " + _CHUNK_ORDER + ";",
            (ids,),
        )
        return [dict(r) for r in cur.fetchall()]


def book_slugs() -> List[Dict[str, Any]]:
    with db.get_cursor() as cur:
        cur.execute(
            "SELECT tags[1] AS slug, COUNT(*) AS chunks FROM sources "
            "GROUP BY tags[1] ORDER BY chunks DESC;"
        )
        return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Writing / reviewing
# ---------------------------------------------------------------------------

_ITEM_COLUMNS = (
    "id::text AS id, category, text_ur, text_roman, citation, citation_display, "
    "citation_incomplete, darja, maslak_tag, source_chunk_ids, content_hash, status, "
    "allow_video, word_count, extractor_version, review_note, approved_at, approved_by, "
    "created_at, updated_at"
)


def insert_draft(
    *,
    category: str,
    text_ur: str,
    citation: Dict[str, Any],
    citation_display: str,
    citation_incomplete: bool,
    darja: Optional[str],
    maslak_tag: Optional[str],
    source_chunk_ids: List[str],
    extractor_version: str,
    text_roman: Optional[str] = None,
) -> Optional[str]:
    """Insert one proposal as a DRAFT. Returns the new id, or None if it already
    exists (same content_hash) — that is what makes re-extraction idempotent.

    status/allow_video are hardcoded here on purpose: extraction must not be able to
    publish, even by passing an argument.
    """
    from psycopg2.extras import Json

    with db.get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO video_items
                (category, text_ur, text_roman, citation, citation_display,
                 citation_incomplete, darja, maslak_tag, source_chunk_ids,
                 content_hash, status, allow_video, word_count, extractor_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'draft', FALSE, %s, %s)
            ON CONFLICT (content_hash) DO NOTHING
            RETURNING id::text AS id;
            """,
            (
                category, text_ur, text_roman, Json(citation or {}), citation_display,
                bool(citation_incomplete), darja, maslak_tag, list(source_chunk_ids),
                content_hash(text_ur), word_count(text_ur), extractor_version,
            ),
        )
        row = cur.fetchone()
        return row["id"] if row else None


def list_items(
    *,
    status: str = "draft",
    category: Optional[str] = None,
    citation_incomplete: Optional[bool] = None,
    limit: int = 20,
    offset: int = 0,
    with_source: bool = True,
) -> List[Dict[str, Any]]:
    """Review queue, oldest first, each item carrying its raw source chunks."""
    where = ["status = %s"]
    params: List[Any] = [status]
    if category:
        where.append("category = %s")
        params.append(category)
    if citation_incomplete is not None:
        where.append("citation_incomplete = %s")
        params.append(citation_incomplete)
    params += [limit, offset]
    with db.get_cursor() as cur:
        cur.execute(
            f"SELECT {_ITEM_COLUMNS} FROM video_items WHERE {' AND '.join(where)} "
            f"ORDER BY created_at ASC, id LIMIT %s OFFSET %s;",
            params,
        )
        items = [dict(r) for r in cur.fetchall()]
    for it in items:
        it["publishable_failures"] = publishable_failures(it)
        if with_source:
            it["source_chunks"] = chunks_by_ids(it.get("source_chunk_ids") or [])
    return items


def get_item(item_id: str, with_source: bool = True) -> Optional[Dict[str, Any]]:
    with db.get_cursor() as cur:
        cur.execute(f"SELECT {_ITEM_COLUMNS} FROM video_items WHERE id = %s;", (item_id,))
        row = cur.fetchone()
    if not row:
        return None
    item = dict(row)
    item["publishable_failures"] = publishable_failures(item)
    if with_source:
        item["source_chunks"] = chunks_by_ids(item.get("source_chunk_ids") or [])
    return item


_EDITABLE = ("text_ur", "text_roman", "citation", "citation_display",
             "citation_incomplete", "darja", "category", "maslak_tag")


def update_item(item_id: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Apply a reviewer's edits. Only the fields in _EDITABLE can be changed —
    status/allow_video move exclusively through approve()/reject()."""
    from psycopg2.extras import Json

    sets, params = [], []
    for key in _EDITABLE:
        if key not in fields:
            continue
        value = fields[key]
        if key == "citation":
            sets.append("citation = %s")
            params.append(Json(value or {}))
        else:
            sets.append(f"{key} = %s")
            params.append(value)
    if "text_ur" in fields:
        # An edited text is a different passage: its hash and length must follow it,
        # or the idempotency key would point at the pre-edit text.
        sets.append("content_hash = %s")
        params.append(content_hash(fields["text_ur"]))
        sets.append("word_count = %s")
        params.append(word_count(fields["text_ur"]))
    if not sets:
        return get_item(item_id)
    sets.append("updated_at = NOW()")
    params.append(item_id)
    with db.get_cursor(commit=True) as cur:
        cur.execute(
            f"UPDATE video_items SET {', '.join(sets)} WHERE id = %s "
            f"RETURNING {_ITEM_COLUMNS};",
            params,
        )
        row = cur.fetchone()
    return get_item(item_id) if row else None


def approve(item_id: str, allow_video: bool, approved_by: str) -> Dict[str, Any]:
    """Human approval — the only path to status='approved' in this codebase.

    A rejected constraint is returned as a structured refusal, not a 500: the UI
    already greys the button out, so hitting this means the two disagreed and the
    reviewer needs to be told which condition failed.
    """
    item = get_item(item_id, with_source=False)
    if not item:
        return {"ok": False, "error": "not_found"}
    if allow_video:
        fails = publishable_failures(item)
        if fails:
            return {"ok": False, "error": "not_publishable", "failures": fails}
    with db.get_cursor(commit=True) as cur:
        cur.execute(
            """
            UPDATE video_items
            SET status = 'approved', allow_video = %s,
                approved_at = NOW(), approved_by = %s, updated_at = NOW()
            WHERE id = %s
            RETURNING id::text AS id;
            """,
            (bool(allow_video), approved_by, item_id),
        )
        if not cur.fetchone():
            return {"ok": False, "error": "not_found"}
    return {"ok": True, "item": get_item(item_id, with_source=False)}


def reject(item_id: str, review_note: str, reviewed_by: str) -> Dict[str, Any]:
    if not (review_note or "").strip():
        return {"ok": False, "error": "review_note_required"}
    with db.get_cursor(commit=True) as cur:
        cur.execute(
            """
            UPDATE video_items
            SET status = 'rejected', allow_video = FALSE, review_note = %s,
                approved_by = %s, updated_at = NOW()
            WHERE id = %s
            RETURNING id::text AS id;
            """,
            (review_note.strip(), reviewed_by, item_id),
        )
        if not cur.fetchone():
            return {"ok": False, "error": "not_found"}
    return {"ok": True, "item": get_item(item_id, with_source=False)}


def queue_counts() -> Dict[str, Any]:
    """Header counters: what is left to review, and what is video-ready per category."""
    with db.get_cursor() as cur:
        cur.execute(
            "SELECT status, COUNT(*) AS n FROM video_items GROUP BY status;"
        )
        by_status = {r["status"]: int(r["n"]) for r in cur.fetchall()}
        cur.execute(
            "SELECT category, COUNT(*) AS n FROM video_items "
            "WHERE status = 'approved' AND allow_video = TRUE GROUP BY category;"
        )
        ready = {r["category"]: int(r["n"]) for r in cur.fetchall()}
        cur.execute(
            "SELECT category, COUNT(*) AS n FROM video_items "
            "WHERE status = 'draft' GROUP BY category;"
        )
        drafts = {r["category"]: int(r["n"]) for r in cur.fetchall()}
        cur.execute(
            "SELECT COUNT(*) AS n FROM video_items "
            "WHERE status = 'draft' AND citation_incomplete = TRUE;"
        )
        incomplete = int(cur.fetchone()["n"])
    return {
        "by_status": by_status,
        "drafts_by_category": drafts,
        "video_ready_by_category": ready,
        "video_ready_total": sum(ready.values()),
        "drafts_citation_incomplete": incomplete,
    }


# ---------------------------------------------------------------------------
# Serving (AutoTube runtime — plain SQL, no model call anywhere in this path)
# ---------------------------------------------------------------------------

def sweep_expired(tenant_id: Optional[str] = None) -> int:
    """Return timed-out reservations to the pool so a crashed run doesn't burn an item."""
    sql = ("UPDATE video_item_serves SET status = 'released' "
           "WHERE status = 'reserved' AND expires_at < NOW()")
    params: Tuple[Any, ...] = ()
    if tenant_id:
        sql += " AND tenant_id = %s"
        params = (tenant_id,)
    with db.get_cursor(commit=True) as cur:
        cur.execute(sql + ";", params)
        return cur.rowcount or 0


def reserve_next(tenant_id: str, category: str) -> Optional[Dict[str, Any]]:
    """Reserve this tenant's next unseen approved item, or None if the pool is dry.

    Ordering is `md5(id || tenant)`: deterministic, so a tenant always walks the pool
    in the same order and can resume, yet different per tenant, so two channels don't
    publish the same hadith on the same day.
    """
    for _attempt in range(5):
        with db.get_cursor() as cur:
            cur.execute(
                """
                SELECT id::text AS id
                FROM video_items vi
                WHERE vi.status = 'approved'
                  AND vi.allow_video = TRUE
                  AND vi.category = %s
                  AND NOT EXISTS (
                      SELECT 1 FROM video_item_serves s
                      WHERE s.tenant_id = %s AND s.item_id = vi.id
                        AND s.status IN ('reserved', 'used')
                  )
                ORDER BY md5(vi.id::text || %s)
                LIMIT 1;
                """,
                (category, tenant_id, tenant_id),
            )
            row = cur.fetchone()
        if not row:
            return None
        item_id = row["id"]

        # Two concurrent calls from one tenant can pick the same row; the unique
        # (tenant, item) index decides, and the loser simply picks the next one.
        with db.get_cursor(commit=True) as cur:
            cur.execute(
                f"""
                INSERT INTO video_item_serves
                    (tenant_id, item_id, status, reserved_at, expires_at)
                VALUES (%s, %s, 'reserved', NOW(), NOW() + INTERVAL '{RESERVE_HOURS} hours')
                ON CONFLICT (tenant_id, item_id) DO UPDATE
                    SET status = 'reserved', reserved_at = NOW(),
                        expires_at = NOW() + INTERVAL '{RESERVE_HOURS} hours',
                        run_id = NULL, acked_at = NULL
                    WHERE video_item_serves.status = 'released'
                RETURNING expires_at;
                """,
                (tenant_id, item_id),
            )
            serve = cur.fetchone()
        if not serve:
            continue  # someone else took it between SELECT and INSERT
        with db.get_cursor() as cur:
            cur.execute(
                "SELECT id::text AS id, category, text_ur, text_roman, citation, "
                "citation_display, darja, word_count FROM video_items WHERE id = %s;",
                (item_id,),
            )
            item = cur.fetchone()
        if not item:
            continue
        out = dict(item)
        out["expires_at"] = serve["expires_at"]
        return out
    return None


def ack(tenant_id: str, item_id: str, status: str, run_id: Optional[str]) -> Dict[str, Any]:
    """`used` retires the item for this tenant; `failed` releases it back to the pool.
    Idempotent — re-acking an already-acked row is a no-op success."""
    new_status = "used" if status == "used" else "released"
    with db.get_cursor(commit=True) as cur:
        cur.execute(
            """
            UPDATE video_item_serves
            SET status = %s,
                run_id = COALESCE(%s, run_id),
                acked_at = COALESCE(acked_at, NOW())
            WHERE tenant_id = %s AND item_id = %s
            RETURNING status;
            """,
            (new_status, run_id, tenant_id, item_id),
        )
        row = cur.fetchone()
    if not row:
        return {"ok": False, "error": "no_such_reservation"}
    return {"ok": True, "status": row["status"]}


def tenant_stats(tenant_id: str) -> Dict[str, Any]:
    """Remaining un-served items per category for this tenant + total pool size.
    `fiqh` is deliberately absent: it is never served in v1."""
    served_categories = ("hadith", "tafsir", "hikayat")
    with db.get_cursor() as cur:
        cur.execute(
            """
            SELECT vi.category,
                   COUNT(*) AS pool,
                   COUNT(*) FILTER (WHERE s.id IS NULL) AS remaining
            FROM video_items vi
            LEFT JOIN video_item_serves s
              ON s.item_id = vi.id AND s.tenant_id = %s
             AND s.status IN ('reserved', 'used')
            WHERE vi.status = 'approved' AND vi.allow_video = TRUE
              AND vi.category = ANY(%s)
            GROUP BY vi.category;
            """,
            (tenant_id, list(served_categories)),
        )
        rows = {r["category"]: r for r in cur.fetchall()}
    return {
        cat: {
            "remaining": int(rows[cat]["remaining"]) if cat in rows else 0,
            "pool": int(rows[cat]["pool"]) if cat in rows else 0,
        }
        for cat in served_categories
    }


def rate_limit_hit(tenant_id: str) -> Tuple[bool, int]:
    """Fixed one-hour window counter. Returns (allowed, retry_after_seconds)."""
    with db.get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO autotube_rate_limit (tenant_id, window_start, count)
            VALUES (%s, date_trunc('hour', NOW()), 1)
            ON CONFLICT (tenant_id, window_start)
                DO UPDATE SET count = autotube_rate_limit.count + 1
            RETURNING count,
                      EXTRACT(EPOCH FROM (date_trunc('hour', NOW())
                              + INTERVAL '1 hour' - NOW()))::int AS retry_after;
            """,
            (tenant_id,),
        )
        row = cur.fetchone()
        # Old windows are useless once passed; clearing them here keeps the table
        # from growing without a separate cron.
        cur.execute(
            "DELETE FROM autotube_rate_limit "
            "WHERE window_start < NOW() - INTERVAL '2 hours';"
        )
    return int(row["count"]) <= RATE_LIMIT_PER_HOUR, int(row["retry_after"])
