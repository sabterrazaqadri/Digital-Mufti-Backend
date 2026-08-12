"""
Admin routes behind the video-item review queue (/admin/video-items in the Next app).

Approval is the entire security model of the AutoTube feed: an item only becomes
publishable because a human read it *against its source chunks* and said so. These
routes therefore do two things and nothing clever — they serve the draft with its raw
source text attached, and they record what the reviewer decided.

Authorization reuses the existing Better Auth JWT verification plus an env allowlist
(see auth.get_admin_user). No separate admin password, no new session mechanism.
"""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

import video_items as vi
from auth import get_admin_user, whoami

router = APIRouter(prefix="/api/admin", tags=["admin"])


class UpdateItemRequest(BaseModel):
    text_ur: Optional[str] = None
    text_roman: Optional[str] = None
    citation: Optional[Dict[str, Any]] = None
    citation_display: Optional[str] = None
    citation_incomplete: Optional[bool] = None
    darja: Optional[str] = None
    category: Optional[str] = None
    maslak_tag: Optional[str] = None


class ApproveRequest(BaseModel):
    allow_video: bool = False


class RejectRequest(BaseModel):
    review_note: str = Field(..., min_length=1, max_length=2000)


@router.get("/whoami")
async def admin_whoami(identity: dict = Depends(whoami)):
    """Your own verified identity — how you find the value for ADMIN_EMAILS."""
    return identity


@router.get("/video-items")
async def list_video_items(
    status: str = Query("draft"),
    category: Optional[str] = Query(None),
    citation_incomplete: Optional[bool] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _admin: str = Depends(get_admin_user),
):
    """The review queue, oldest first. Each item carries the raw source chunks it was
    extracted from — approval must be verification, not rubber-stamping a card."""
    if status not in vi.STATUSES:
        raise HTTPException(status_code=400, detail="Invalid status")
    if category and category not in vi.CATEGORIES:
        raise HTTPException(status_code=400, detail="Invalid category")
    items: List[Dict[str, Any]] = await run_in_threadpool(
        vi.list_items,
        status=status, category=category, citation_incomplete=citation_incomplete,
        limit=limit, offset=offset,
    )
    counts = await run_in_threadpool(vi.queue_counts)
    return {"items": items, "counts": counts}


@router.get("/video-items/stats")
async def video_item_stats(_admin: str = Depends(get_admin_user)):
    return await run_in_threadpool(vi.queue_counts)


@router.get("/video-items/{item_id}")
async def get_video_item(item_id: str, _admin: str = Depends(get_admin_user)):
    item = await run_in_threadpool(vi.get_item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Not found")
    return item


@router.patch("/video-items/{item_id}")
async def update_video_item(
    item_id: str, body: UpdateItemRequest, _admin: str = Depends(get_admin_user)
):
    fields = {k: v for k, v in body.model_dump(exclude_unset=True).items()}
    if fields.get("category") and fields["category"] not in vi.CATEGORIES:
        raise HTTPException(status_code=400, detail="Invalid category")
    if fields.get("darja") and fields["darja"] not in vi.DARJAT:
        raise HTTPException(status_code=400, detail="Invalid darja")
    item = await run_in_threadpool(vi.update_item, item_id, fields)
    if not item:
        raise HTTPException(status_code=404, detail="Not found")
    return item


@router.post("/video-items/{item_id}/approve")
async def approve_video_item(
    item_id: str, body: ApproveRequest, admin: str = Depends(get_admin_user)
):
    """The only route in this codebase that can set status='approved'."""
    result = await run_in_threadpool(vi.approve, item_id, body.allow_video, admin)
    if not result.get("ok"):
        if result.get("error") == "not_found":
            raise HTTPException(status_code=404, detail="Not found")
        raise HTTPException(
            status_code=422,
            detail={"error": "not_publishable", "failures": result.get("failures", [])},
        )
    return result["item"]


@router.post("/video-items/{item_id}/reject")
async def reject_video_item(
    item_id: str, body: RejectRequest, admin: str = Depends(get_admin_user)
):
    result = await run_in_threadpool(vi.reject, item_id, body.review_note, admin)
    if not result.get("ok"):
        if result.get("error") == "not_found":
            raise HTTPException(status_code=404, detail="Not found")
        raise HTTPException(status_code=400, detail="review_note is required")
    return result["item"]
