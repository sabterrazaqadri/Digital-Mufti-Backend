"""
The review queue's three actions, at the layer the admin routes call.

    python test_admin_review.py

Checks that an edit, an approval and a rejection each persist correctly, that an
edited text re-hashes (so idempotency keys follow the text a human actually approved),
and — the important one — that approve() refuses to set allow_video on an item the
publishability rule blocks, instead of letting the request reach the DB and 500.

Test rows are marked and deleted in a finally block. Exit code 0 = all passed.
"""
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import database as db
import video_items as vi

MARK = "TEST-REVIEW-"
failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def seed(**over):
    item_id = str(uuid.uuid4())
    row = {"category": "hadith", "text_ur": " ".join(["لفظ"] * 40),
           "citation_incomplete": True, "darja": "na_maloom", "word_count": 40}
    row.update(over)
    with db.get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO video_items
                (id, category, text_ur, citation, citation_display, citation_incomplete,
                 darja, source_chunk_ids, content_hash, status, allow_video, word_count,
                 extractor_version)
            VALUES (%s, %s, %s, '{"kitab": null}'::jsonb, '', %s, %s, '{}', %s,
                    'draft', FALSE, %s, 'test');
            """,
            (item_id, row["category"], row["text_ur"], row["citation_incomplete"],
             row["darja"], MARK + item_id, row["word_count"]),
        )
    return item_id


def main():
    vi.init_video_items()
    created = []
    try:
        print("Editing a draft:")
        item_id = seed()
        created.append(item_id)
        new_text = " ".join(["کلمہ"] * 45)
        vi.update_item(item_id, {
            "text_ur": new_text,
            "citation": {"kitab": "Sahih Bukhari", "safha": "12", "hadith_no": "3"},
            "citation_display": "Sahih Bukhari, Hadees 3",
            "citation_incomplete": False,
            "darja": "sahih",
        })
        item = vi.get_item(item_id, with_source=False)
        check("text edit persisted", item["text_ur"] == new_text)
        check("word_count recomputed on edit", item["word_count"] == 45,
              f"got {item['word_count']}")
        check("content_hash follows the edited text",
              item["content_hash"] == vi.content_hash(new_text))
        check("citation fields persisted", item["citation"]["kitab"] == "Sahih Bukhari")
        check("still a draft after editing", item["status"] == "draft")
        check("editing never grants video", item["allow_video"] is False)

        print("\nApprove & allow video:")
        result = vi.approve(item_id, allow_video=True, approved_by="tester@example.com")
        item = vi.get_item(item_id, with_source=False)
        check("approve returned ok", result.get("ok") is True)
        check("status is approved", item["status"] == "approved")
        check("allow_video is true", item["allow_video"] is True)
        check("approved_by recorded", item["approved_by"] == "tester@example.com")
        check("approved_at recorded", item["approved_at"] is not None)

        print("\nApprove (no video) on an item that is not video-eligible:")
        blocked = seed()          # citation_incomplete, darja na_maloom
        created.append(blocked)
        refused = vi.approve(blocked, allow_video=True, approved_by="tester@example.com")
        check("approve+video is refused with reasons",
              refused.get("ok") is False and refused.get("failures"),
              str(refused.get("failures")))
        still = vi.get_item(blocked, with_source=False)
        check("refused approval left the row untouched",
              still["status"] == "draft" and still["allow_video"] is False)
        ok = vi.approve(blocked, allow_video=False, approved_by="tester@example.com")
        item = vi.get_item(blocked, with_source=False)
        check("approve without video succeeds", ok.get("ok") is True)
        check("approved but not video-enabled",
              item["status"] == "approved" and item["allow_video"] is False)

        print("\nReject:")
        bad = seed()
        created.append(bad)
        check("rejection without a note is refused",
              vi.reject(bad, "   ", "tester@example.com").get("ok") is False)
        vi.reject(bad, "Citation belongs to a different narration", "tester@example.com")
        item = vi.get_item(bad, with_source=False)
        check("status is rejected", item["status"] == "rejected")
        check("review_note stored",
              item["review_note"] == "Citation belongs to a different narration")

        print("\nQueue counters:")
        counts = vi.queue_counts()
        check("counts expose the video-ready total", "video_ready_total" in counts,
              str(counts.get("video_ready_by_category")))

        check_admin_allowlist()

    finally:
        if created:
            with db.get_cursor(commit=True) as cur:
                cur.execute("DELETE FROM video_items WHERE id = ANY(%s::uuid[]);", (created,))
            print(f"\nCleaned up {len(created)} test rows.")

    if failures:
        print(f"\n{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("\nAll checks passed.")


def check_admin_allowlist():
    """Who counts as an admin.

    JWT *verification* is the existing, already-trusted path, so it is stubbed here;
    what is new — and what this checks — is the allowlist decision made on top of the
    verified claims, including that an unconfigured allowlist locks everyone out
    rather than letting everyone in.
    """
    import os

    import auth
    from fastapi import HTTPException

    print("\nAdmin allowlist:")
    real_verify, real_env = auth._verify_claims, dict(os.environ)

    def stub(_token, claims):
        auth._verify_claims = lambda _t: claims

    try:
        os.environ["ADMIN_EMAILS"] = "boss@example.com"
        os.environ.pop("ADMIN_USER_IDS", None)

        stub("t", {"sub": "u1", "email": "BOSS@example.com"})
        check("allowlisted email is admin (case-insensitive)",
              auth.get_admin_user("Bearer t") == "boss@example.com")

        stub("t", {"sub": "u2", "email": "someone@example.com"})
        try:
            auth.get_admin_user("Bearer t")
            check("a signed-in non-admin is refused", False, "was allowed")
        except HTTPException as exc:
            check("a signed-in non-admin is refused", exc.status_code == 403,
                  f"status {exc.status_code}")

        os.environ.pop("ADMIN_EMAILS")
        os.environ["ADMIN_USER_IDS"] = "u9"
        stub("t", {"sub": "u9"})
        check("allowlisted user id is admin (token with no email claim)",
              auth.get_admin_user("Bearer t") == "u9")

        os.environ.pop("ADMIN_USER_IDS")
        stub("t", {"sub": "u9", "email": "boss@example.com"})
        try:
            auth.get_admin_user("Bearer t")
            check("unconfigured allowlist locks everyone out", False, "was allowed")
        except HTTPException as exc:
            check("unconfigured allowlist locks everyone out", exc.status_code == 503,
                  f"status {exc.status_code}")

        try:
            auth.get_admin_user(None)
            check("no token is refused", False, "was allowed")
        except HTTPException as exc:
            check("no token is refused", exc.status_code in (401, 503),
                  f"status {exc.status_code}")
    finally:
        auth._verify_claims = real_verify
        os.environ.clear()
        os.environ.update(real_env)


if __name__ == "__main__":
    main()
