---
title: AI Mufti Backend
emoji: 🕌
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# AI Mufti Backend

An Islamic Q&A chatbot backend powered by Google Gemini AI, built with FastAPI.

## Features

- Answers Islamic questions based on Hanafi Fiqh
- Streaming responses for real-time interaction
- Chat history persistence with database support
- RESTful API endpoints for chat management
- CORS enabled for frontend integration

## API Endpoints

- `GET /` - Health check
- `GET /health` - API health status
- `POST /chat` - Send a message and get streaming response
- `GET /api/chats` - Get all chats for a user
- `POST /api/chats` - Create a new chat
- `GET /api/chats/{chat_id}` - Get a specific chat
- `DELETE /api/chats/{chat_id}` - Delete a chat
- `GET /api/chats/{chat_id}/messages` - Get messages for a chat

## Environment Variables

- `GEMINI_API_KEY` - Your Google Gemini API key
- `DATABASE_URL` - Optional database URL for chat persistence

## RAG: ingesting books

Answers are grounded in real book text stored in Postgres (pgvector). Citations
(jild, bab, masla number) are machine-generated at ingestion — the model is
instructed to only quote reference numbers that come from retrieved excerpts.

Pipeline (run from `backend/`):

```bash
# 1. Scrape a book's Unicode text from the Dawat-e-Islami online reader
#    (the PDFs use non-Unicode fonts and are NOT extractable)
python scrape_book.py bahar-e-shariat-jild-1 ../data/bahar_e_shariat/jild_1

# 2. Chunk (masla-wise) + embed + store in Neon. Resume-able: free-tier Gemini
#    embedding quota (requests/day) may stop it mid-run — just re-run the same
#    command later; ingest_checkpoint.json skips completed pages.
python -u ingest_book.py ../data/bahar_e_shariat/jild_1 --jild 1

# 3. Smoke-test retrieval quality (Urdu / Roman Urdu / English questions)
python verify_rag.py
```

Both scripts read `GEMINI_API_KEY` and `DATABASE_URL` from `.env`.

## AutoTube content bridge

AutoTube generates short-form Islamic videos on end users' own YouTube channels. It
never invents Islamic content: at video time it asks this backend for a specific,
**human-approved** passage with its verbatim citation, or it gets nothing.

The RAG chunks in `sources` cannot serve that purpose — a chunk may hold half a hadith
or three masail, and its citation may sit in the neighbouring chunk. So a curation
layer sits **on top of** them (the chunks and the Q&A behaviour are never modified):

```
sources ──extract_video_items.py──▶ video_items (draft)
                                        │
                              /admin/video-items  ← a human reads it against the source
                                        │
                                  approved + allow_video
                                        │
                             GET /api/autotube/next ──▶ AutoTube
```

Four rules hold this together, and none of them is optional:

1. The extraction model **never authors content**. It may only point at text already
   present in its input; every returned passage is then checked to be an exact
   substring of the source window before it is written.
2. **No invented citations.** A citation field that does not physically appear in the
   window is nulled and the item is marked `citation_incomplete`.
3. **No invented grading.** `darja` survives only if the page literally says
   صحیح / حسن / ضعیف about that narration; otherwise `na_maloom`.
4. **Nothing is served without human approval.** Extraction writes `status='draft'`
   only, and `approved + allow_video` is guarded by a DB CHECK constraint, not app code.

### 1. Extract proposals (offline, batched)

```bash
# See what it would propose, without writing anything
python -u extract_video_items.py --book anwaar-ul-hadees --category hadith --limit 50 --dry-run

# For real. Inserts drafts only; safe to re-run (content_hash makes it idempotent)
python -u extract_video_items.py --book anwaar-ul-hadees --category hadith --limit 200
```

`--book` is the slug in `sources.tags[1]` (run the script with a wrong slug to list
them). It reads a sliding 3-chunk window — previous, current, next — so a reference
printed beside the text is visible; the window steps one chunk at a time. Progress is
checkpointed per window in `.extract_checkpoints/`, so a run stopped by quota or a
Neon connection drop resumes where it left off — just re-run the same command.

The summary reports windows processed, candidates proposed, inserted, duplicates,
what was dropped and why, and the `citation_incomplete` rate. **If that rate is above
30% the script says so loudly** — review those items and you will mostly be looking at
cards that can never be approved for video. Fix the source metadata or widen the
window first.

Most windows correctly yield nothing. An empty result is the expected answer for a
page of front matter, an index, or half a narration.

### 2. Review them (`/admin/video-items`)

Sign in to the Next app with an account listed in `ADMIN_EMAILS`, then open
`/admin/video-items`. The proposal is on the left, the raw source chunks it came from
on the right, with the proposed text highlighted where it appears in the book page.
Approval means you verified it there — that is the entire security model of the feed.

* **Approve & allow video** (`a`) — the item enters the AutoTube pool. Disabled, with
  the reason shown, whenever the DB constraint would reject it.
* **Approve (no video)** (`n`) — good text, unsuitable for a Short (too long, fiqh, zaeef).
* **Reject** (`r`) — requires a note.
* `↑`/`↓` (or `j`/`k`) move through the queue. Shortcuts are off while typing.

Target for v1 launch is 200–300 video-ready items across hadith + tafsir + hikayat;
the header counts them per category.

An item can only be `approved + allow_video` when its citation is complete, a hadith
is graded sahih/hasan, it is not fiqh, and it is 25–160 words. That is enforced by the
`video_items_publishable` CHECK constraint. Prove it any time:

```bash
python test_video_items_constraint.py   # writes raw SQL to try to bypass the rule
python test_autotube_api.py             # boots the app and exercises the whole feed
```

### 3. The feed API

All routes are under `/api/autotube/`, authenticated with a single service key sent as
`Authorization: Bearer $AUTOTUBE_SERVICE_KEY` and compared in constant time.
`tenant` is an opaque string (an HMAC of the end user's id, computed on AutoTube's
side) — it is never logged in plaintext and never resolved to a person.

| route | behaviour |
|---|---|
| `GET /health` | `{"ok": true}`, unauthenticated |
| `GET /next?category=&tenant=&lang=` | reserves this tenant's next approved item for 6h; **204** when their pool is exhausted |
| `POST /ack` | `{tenant, item_id, run_id, status}` — `used` retires it, `failed` returns it to the pool; idempotent |
| `GET /stats?tenant=` | remaining vs total pool size per category |

Ordering is `md5(id || tenant)`: deterministic per tenant, different between tenants,
so two channels do not publish the same hadith on the same day. A crashed run's
reservation expires after 6 hours and is swept back into the pool. `fiqh` is never
served in v1 — maslak differences make it unsafe to auto-publish on someone else's
channel.

**These are server-to-server endpoints with no CORS.** Any request carrying an
`Origin` header is refused with 403.

```bash
curl -i "$API/api/autotube/next?category=hadith&tenant=abc123"                    # 401
curl -i -H "Authorization: Bearer $AUTOTUBE_SERVICE_KEY" \
     "$API/api/autotube/next?category=hadith&tenant=abc123"                       # 200 or 204
curl -i -H "Authorization: Bearer $AUTOTUBE_SERVICE_KEY" -H 'Content-Type: application/json' \
     -d '{"tenant":"abc123","item_id":"<uuid>","run_id":"r1","status":"used"}' \
     "$API/api/autotube/ack"
```

### Rotating the service key

1. Generate one: `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
2. Set `AUTOTUBE_SERVICE_KEY` to the new value on the backend host (HF Space →
   Settings → Variables and secrets) and restart.
3. Give the new value to AutoTube. There is deliberately no grace period with two
   valid keys: the window in which an old key still works is the window in which a
   leaked one still works. Coordinate the swap instead.

### Killing the feed

Set `AUTOTUBE_API_ENABLED=false` and restart. `/next` and `/ack` return 503
immediately; nothing else in the app is affected, and no code or routing changes.
Clearing `AUTOTUBE_SERVICE_KEY` has the same effect and also locks out `/stats`.

## Running Locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Docker

This space runs using Docker. Build and run:

```bash
docker build -t ai-mufti-backend .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key ai-mufti-backend
```
