"""
AutoTube-facing feed API.

AutoTube generates short-form Islamic videos on end users' own YouTube channels. It
must never invent Islamic content, so it does not ask a model for a hadith — it asks
THIS endpoint, which hands back a specific, human-approved passage with its verbatim
citation, or nothing at all.

Design constraints that follow from that:

  * No LLM in the request path. Every route here is a plain SQL lookup.
  * Nothing unapproved is ever served. There is no fallback, no "close enough" item,
    no degraded mode: an exhausted pool returns 204 and AutoTube skips the day.
  * `fiqh` is never served in v1 — fiqhi masail carry maslak differences that are
    unsafe to auto-publish on somebody else's channel.
  * `tenant` is opaque (an HMAC of the end user's id, computed on AutoTube's side).
    It is never logged in plaintext and never resolved to a person.

INERT until AUTOTUBE_SERVICE_KEY is set: every authenticated route 503s without it,
so deploying this file changes nothing on its own.

    AUTOTUBE_SERVICE_KEY   shared secret, sent as `Authorization: Bearer <key>`
    AUTOTUBE_API_ENABLED   kill switch; "false" makes /next and /ack return 503
"""
import hashlib
import os
import secrets
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

import video_items as vi

router = APIRouter(prefix="/api/autotube", tags=["autotube"])

# v1 serves these three only. `fiqh` is extracted and reviewable but never delivered.
SERVED_CATEGORIES = ("hadith", "tafsir", "hikayat")
MAX_TENANT_LEN = 128
ATTRIBUTION = os.getenv(
    "AUTOTUBE_ATTRIBUTION", "Source: Digital Mufti — digitalmufti.vercel.app"
)


def _service_key() -> str:
    return (os.getenv("AUTOTUBE_SERVICE_KEY") or "").strip()


def _api_enabled() -> bool:
    """Read at request time, not import time: flipping the env var must take effect
    without a code change or a redeploy path change."""
    return (os.getenv("AUTOTUBE_API_ENABLED", "true") or "").strip().lower() not in (
        "false", "0", "no", "off",
    )


def tenant_ref(tenant_id: str) -> str:
    """A short digest of the tenant, safe to put in a log line. The raw value never is."""
    return hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:10]


def require_service_key(authorization: Optional[str]) -> None:
    """Bearer check in constant time. 401 with no detail on any mismatch — a caller
    must not be able to learn whether the key was absent, malformed or simply wrong."""
    key = _service_key()
    if not key:
        # Not configured: the feed does not exist yet, and saying "unauthorized"
        # would imply a key would work.
        raise HTTPException(status_code=503, detail="AutoTube API not configured")
    supplied = ""
    if authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            supplied = parts[1].strip()
    if not secrets.compare_digest(supplied, key):
        raise HTTPException(status_code=401, detail="Unauthorized")


def reject_browser(request: Request) -> None:
    """These are server-to-server endpoints; there is deliberately no CORS for them.

    The app's global CORSMiddleware would otherwise hand the site's own origin a
    working cross-origin route to the feed, which would put the service key in a
    browser. A request carrying an Origin header is browser-initiated: refuse it.
    """
    if request.headers.get("origin"):
        raise HTTPException(status_code=403, detail="Forbidden")


def clean_tenant(tenant: str) -> str:
    t = (tenant or "").strip()
    if not t or len(t) > MAX_TENANT_LEN:
        raise HTTPException(status_code=400, detail="Invalid tenant")
    return t


class AckRequest(BaseModel):
    tenant: str = Field(..., max_length=MAX_TENANT_LEN)
    item_id: str
    run_id: Optional[str] = Field(default=None, max_length=200)
    status: str  # "used" | "failed"


@router.get("/health")
async def health():
    """Unauthenticated liveness probe. Reveals nothing about the pool."""
    return {"ok": True}


@router.get("/next")
async def next_item(
    request: Request,
    response: Response,
    category: str = Query(...),
    tenant: str = Query(...),
    lang: str = Query("urdu"),
    authorization: Optional[str] = Header(default=None),
):
    """Reserve and return this tenant's next approved item, or 204 when exhausted."""
    reject_browser(request)
    require_service_key(authorization)
    if not _api_enabled():
        raise HTTPException(status_code=503, detail="AutoTube feed disabled")
    if category not in SERVED_CATEGORIES:
        raise HTTPException(status_code=400, detail="Unsupported category")
    tenant_id = clean_tenant(tenant)

    allowed, retry_after = await run_in_threadpool(vi.rate_limit_hit, tenant_id)
    if not allowed:
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded",
            headers={"Retry-After": str(max(retry_after, 1))},
        )

    # A crashed run must not permanently burn an item: anything reserved and past its
    # expiry goes back into this tenant's pool before we look for the next one.
    await run_in_threadpool(vi.sweep_expired, tenant_id)

    item = await run_in_threadpool(vi.reserve_next, tenant_id, category)
    if not item:
        # Not an error. The tenant has seen everything approved in this category —
        # serving an unapproved item instead is the one thing this API must never do.
        print(f"autotube: pool exhausted category={category} tenant={tenant_ref(tenant_id)}")
        response.status_code = 204
        return response

    citation = item.get("citation") or {}
    return {
        "item_id": item["id"],
        "category": item["category"],
        "text_ur": item["text_ur"],
        # Roman Urdu is optional in the pool; the caller asked for a language but we
        # never synthesise one that a human did not approve.
        "text_roman": item.get("text_roman"),
        "citation": {
            "kitab": citation.get("kitab"),
            "jild": citation.get("jild"),
            "safha": citation.get("safha"),
            "hadith_no": citation.get("hadith_no"),
            "rawi": citation.get("rawi"),
        },
        "citation_display": item.get("citation_display"),
        "darja": item.get("darja"),
        "word_count": item.get("word_count"),
        "attribution": ATTRIBUTION,
        "expires_at": item["expires_at"].isoformat().replace("+00:00", "Z"),
        "lang": lang,
    }


@router.post("/ack")
async def ack(
    request: Request,
    body: AckRequest,
    authorization: Optional[str] = Header(default=None),
):
    """`used` retires the item for this tenant, `failed` returns it to their pool."""
    reject_browser(request)
    require_service_key(authorization)
    if not _api_enabled():
        raise HTTPException(status_code=503, detail="AutoTube feed disabled")
    if body.status not in ("used", "failed"):
        raise HTTPException(status_code=400, detail="Invalid status")
    tenant_id = clean_tenant(body.tenant)

    result = await run_in_threadpool(
        vi.ack, tenant_id, body.item_id, body.status, body.run_id
    )
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail="No such reservation")
    return {"ok": True, "status": result["status"]}


@router.get("/stats")
async def stats(
    request: Request,
    tenant: str = Query(...),
    authorization: Optional[str] = Header(default=None),
):
    """Per-category: how much of the pool this tenant has left, and its total size.
    AutoTube warns a user before their queue runs dry."""
    reject_browser(request)
    require_service_key(authorization)
    tenant_id = clean_tenant(tenant)
    await run_in_threadpool(vi.sweep_expired, tenant_id)
    return {"categories": await run_in_threadpool(vi.tenant_stats, tenant_id)}
