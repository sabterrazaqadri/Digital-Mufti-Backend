"""
WhatsApp bot for AI Mufti (Meta Cloud API).

The audience this product serves lives on WhatsApp, so this puts the same
source-grounded assistant behind a phone number: a user sends a mas'ala, gets the
answer plus its exact references, and nothing is answered that the library does
not support.

NOT ACTIVE until these are set — every route no-ops without them, so deploying
this file changes nothing on its own:

    WHATSAPP_VERIFY_TOKEN   any string you also type into the Meta webhook form
    WHATSAPP_TOKEN          permanent access token for the WhatsApp Business app
    WHATSAPP_PHONE_ID       the Phone Number ID (not the phone number itself)

Setup, once those exist:
  1. Meta App → WhatsApp → Configuration → Webhook
     Callback URL: https://<backend>/webhook/whatsapp
     Verify token: WHATSAPP_VERIFY_TOKEN
  2. Subscribe the app to the `messages` field.
  3. Add the sender number in Meta → API Setup (test numbers work immediately;
     a production number needs business verification).
"""
import os
import threading
from typing import Any, Dict, Optional

import requests
from fastapi import APIRouter, HTTPException, Request, Response

import rag

router = APIRouter()

VERIFY_TOKEN = (os.getenv("WHATSAPP_VERIFY_TOKEN") or "").strip()
ACCESS_TOKEN = (os.getenv("WHATSAPP_TOKEN") or "").strip()
PHONE_ID = (os.getenv("WHATSAPP_PHONE_ID") or "").strip()
GRAPH = os.getenv("WHATSAPP_GRAPH_URL", "https://graph.facebook.com/v21.0")

# WhatsApp hard-limits a text body; answers are trimmed rather than dropped.
MAX_BODY = 3900


def configured() -> bool:
    return bool(VERIFY_TOKEN and ACCESS_TOKEN and PHONE_ID)


def send_text(to: str, body: str) -> None:
    if not configured():
        return
    if len(body) > MAX_BODY:
        body = body[:MAX_BODY].rstrip() + "…"
    try:
        requests.post(
            f"{GRAPH}/{PHONE_ID}/messages",
            headers={"Authorization": f"Bearer {ACCESS_TOKEN}"},
            json={
                "messaging_product": "whatsapp",
                "to": to,
                "type": "text",
                "text": {"body": body, "preview_url": False},
            },
            timeout=20,
        )
    except Exception as exc:  # never raise into the webhook
        print(f"WhatsApp send failed: {exc}")


def _format_reply(answer: str, passages) -> str:
    """Answer plus its references. WhatsApp has no rich text, so the citations
    go in a plain block rather than a collapsible panel."""
    if not passages:
        return answer
    refs = "\n".join(
        f"• {p.get('reference') or p.get('title')}" for p in passages if p.get("reference") or p.get("title")
    )
    return f"{answer}\n\n— — —\n*حوالہ جات*\n{refs}"


def _answer(text: str) -> str:
    """Same pipeline as the web chat, so the two can never diverge in what they
    are willing to say."""
    # Imported here, not at module load: main imports this module, and importing
    # it back at the top would be circular.
    import main as app_main

    retrieval_attempted = app_main._should_retrieve(text)
    passages = []
    grounded = None
    if retrieval_attempted:
        verse_frag = rag.verse_request(text)
        if verse_frag is not None:
            passages, verse_mode = rag.verse_lookup(text, verse_frag)
            grounded = rag.build_verse_input(text, verse_frag, passages, verse_mode)
        else:
            intent = rag.browse_intent(text)
            if intent:
                passages = rag.browse(intent[0], intent[1])
            if not passages:
                passages = rag.retrieve(text)
    if grounded is None:
        grounded = rag.build_grounded_input(text, passages, retrieval_attempted)
    directive = app_main._language_directive(text)

    source = app_main._groq_chunks if app_main.LLM_PROVIDER == "groq" else app_main._gemini_chunks
    chunks = []
    for piece, _ in source(directive + grounded, []):
        if piece:
            chunks.append(piece)
    answer = "".join(chunks).strip()
    if not answer:
        return "معذرت، اس وقت جواب تیار نہیں ہو سکا۔ براہِ کرم دوبارہ کوشش کریں۔"
    return _format_reply(answer, rag.public_passages(passages) if passages else [])


def _handle_async(sender: str, text: str) -> None:
    """Meta retries a webhook it does not get a fast 200 for, which would answer
    the same question several times — so the reply is produced off-thread and the
    webhook returns immediately."""
    try:
        send_text(sender, _answer(text))
    except Exception as exc:
        print(f"WhatsApp handler failed: {exc}")
        send_text(sender, "معذرت، اس وقت دشواری ہو رہی ہے۔ تھوڑی دیر بعد کوشش کریں۔")


@router.get("/webhook/whatsapp")
async def verify(request: Request):
    """Meta's one-time subscription handshake."""
    params = request.query_params
    if params.get("hub.mode") == "subscribe" and params.get("hub.verify_token") == VERIFY_TOKEN:
        return Response(content=params.get("hub.challenge", ""), media_type="text/plain")
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/webhook/whatsapp")
async def receive(request: Request):
    """Inbound messages. Always 200: a non-200 makes Meta retry the delivery."""
    if not configured():
        return {"status": "not_configured"}

    try:
        payload: Dict[str, Any] = await request.json()
    except Exception:
        return {"status": "ignored"}

    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {}) or {}
            for msg in value.get("messages", []) or []:
                sender: Optional[str] = msg.get("from")
                if not sender or msg.get("type") != "text":
                    if sender:
                        send_text(
                            sender,
                            "معذرت، میں فی الحال صرف تحریری سوالات کا جواب دے سکتا ہوں۔",
                        )
                    continue
                body = (msg.get("text", {}) or {}).get("body", "").strip()
                if body:
                    threading.Thread(
                        target=_handle_async, args=(sender, body), daemon=True
                    ).start()

    return {"status": "ok"}
