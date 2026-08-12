"""
Proof that the publishability rule is enforced by the DATABASE, not by app code.

    python test_video_items_constraint.py

Every check here writes raw SQL, deliberately bypassing video_items.approve() — the
whole point is that an item cannot become `approved + allow_video` through ANY route,
including a buggy review UI, a psql session, or a future script that forgets the rule.

No test framework: this repo has none, and adding pytest for one file is not worth a
new dependency. Exit code 0 = all checks passed.

Test rows are marked with an obvious content_hash prefix and removed in a finally
block, so running this against the real database leaves nothing behind.
"""
import sys
import uuid

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import psycopg2

import database as db
import video_items as vi

MARK = "TEST-CONSTRAINT-"
failures = []


def _insert(**over):
    """Insert a draft row directly. Returns its id."""
    row = {
        "category": "hadith",
        "text_ur": " ".join(["لفظ"] * 40),   # 40 words: inside the 25-160 band
        "citation": '{"kitab": "Test", "safha": "1"}',
        "citation_display": "Test, Safha 1",
        "citation_incomplete": False,
        "darja": "sahih",
        "word_count": 40,
    }
    row.update(over)
    item_id = str(uuid.uuid4())
    with db.get_cursor(commit=True) as cur:
        cur.execute(
            """
            INSERT INTO video_items
                (id, category, text_ur, citation, citation_display, citation_incomplete,
                 darja, source_chunk_ids, content_hash, status, allow_video, word_count,
                 extractor_version)
            VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, '{}', %s, 'draft', FALSE, %s, 'test');
            """,
            (item_id, row["category"], row["text_ur"], row["citation"],
             row["citation_display"], row["citation_incomplete"], row["darja"],
             MARK + item_id, row["word_count"]),
        )
    return item_id


def _try_publish(item_id):
    """Attempt approved+allow_video via raw SQL. Returns True if the DB allowed it."""
    try:
        with db.get_cursor(commit=True) as cur:
            cur.execute(
                "UPDATE video_items SET status = 'approved', allow_video = TRUE, "
                "approved_at = NOW(), approved_by = 'test' WHERE id = %s;",
                (item_id,),
            )
        return True
    except psycopg2.errors.CheckViolation:
        return False


def check(name, item_id, should_publish):
    allowed = _try_publish(item_id)
    ok = allowed == should_publish
    verdict = "PASS" if ok else "FAIL"
    expected = "accepted" if should_publish else "rejected"
    got = "accepted" if allowed else "rejected"
    print(f"  [{verdict}] {name}: expected {expected}, DB {got}")
    if not ok:
        failures.append(name)


def main():
    vi.init_video_items()
    created = []
    print("DB CHECK constraint — approved + allow_video is only reachable when the "
          "item is genuinely publishable:")
    try:
        cases = [
            ("complete sahih hadith, 40 words", {}, True),
            ("citation_incomplete = true", {"citation_incomplete": True}, False),
            ("hadith graded zaeef", {"darja": "zaeef"}, False),
            ("hadith with no grading (na_maloom)", {"darja": "na_maloom"}, False),
            ("fiqh item", {"category": "fiqh", "darja": None}, False),
            ("too short (10 words)",
             {"word_count": 10, "text_ur": " ".join(["لفظ"] * 10)}, False),
            ("too long (200 words)",
             {"word_count": 200, "text_ur": " ".join(["لفظ"] * 200)}, False),
            ("tafsir, no darja, 40 words", {"category": "tafsir", "darja": None}, True),
        ]
        for name, over, should in cases:
            item_id = _insert(**over)
            created.append(item_id)
            check(name, item_id, should)

        # The app-level mirror of the constraint must agree with the DB, or the UI
        # would grey out a button the database would have accepted (or worse, offer
        # one it will reject).
        print("\npublishable_failures() agrees with the constraint:")
        for name, over, should in cases:
            item = {"citation_incomplete": False, "category": "hadith",
                    "darja": "sahih", "word_count": 40}
            item.update({k: v for k, v in over.items() if k != "text_ur"})
            fails = vi.publishable_failures(item)
            ok = (not fails) == should
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: "
                  f"{fails or 'publishable'}")
            if not ok:
                failures.append(f"mirror:{name}")
    finally:
        if created:
            with db.get_cursor(commit=True) as cur:
                cur.execute("DELETE FROM video_items WHERE id = ANY(%s::uuid[]);", (created,))
            print(f"\nCleaned up {len(created)} test rows.")

    if failures:
        print(f"\n{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
