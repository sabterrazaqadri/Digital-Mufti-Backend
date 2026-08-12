"""
Batch extraction agent: RAG chunks -> proposed video_items (status='draft').

    python -u extract_video_items.py --book anwaar-ul-hadees --category hadith --limit 200 --dry-run
    python -u extract_video_items.py --book anwaar-ul-hadees --category hadith --limit 200

THE MODEL NEVER AUTHORS ANYTHING. It is allowed to do exactly one thing: point at a
span of text that is physically present in its input and say "that is one complete
hadith, and here is the reference printed next to it". Everything it returns is then
checked mechanically against the input before it is written:

  * text_ur must occur verbatim in the window (diacritic/whitespace-insensitive) —
    a paraphrased, completed or "corrected" passage fails this check and is dropped.
  * every citation field must occur in the window (Urdu and ASCII digits are treated
    as the same digit). A field that does not is nulled and citation_incomplete is
    set. Guessing a hadith number is the single worst failure mode of this system,
    so the model's word for it is never sufficient.
  * citation_display is BUILT HERE from the verified fields, not taken from the
    model, so a display line can never carry a number that failed verification.
  * darja survives only if the grading word actually appears in the source text;
    otherwise na_maloom.

Everything is written as `draft` with allow_video=false. There is no flag, env var
or code path in this file that can approve anything.

Re-runs are idempotent: content_hash collides and the insert does nothing.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

import video_items as vi

EXTRACTOR_VERSION = "extract-v1"
# Copying text out of a page accurately is a comprehension task, so this uses a
# current flash model rather than the lite one the query-rewriter uses.
EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL", "gemini-3.6-flash")
# Seconds between model calls. Free-tier RPM is the binding constraint; with key
# rotation each key only sees 1/N of the calls (same trick as ingest_book.py).
CALL_SLEEP = float(os.getenv("EXTRACT_SLEEP", "2"))
MAX_QUOTA_SWEEPS = int(os.getenv("EXTRACT_QUOTA_SWEEPS", "20"))

CHECKPOINT_DIR = Path(__file__).with_name(".extract_checkpoints")


# ---------------------------------------------------------------------------
# THE EXTRACTION PROMPT — reviewed and tuned as a unit. Read it before changing it.
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = """You are a careful Islamic-text librarian, NOT an author.

You are given three consecutive excerpts from one Urdu Islamic book. The MIDDLE one
(marked CURRENT) is the one being processed; the PREVIOUS and NEXT excerpts are shown
ONLY so that a reference or a sentence that spills across the boundary is visible to
you. Each excerpt is preceded by its machine-generated REFERENCE line, which states
the book, volume and page it came from.

Your task: identify complete, self-contained items in the CURRENT excerpt that could
stand alone as a 30-60 second narration, and return them as strict JSON.

ABSOLUTE RULES — breaking any of these makes your output worthless:

1. COPY, NEVER WRITE. `text_ur` must be an EXACT substring of the input text —
   character for character. Do not paraphrase, summarise, shorten, complete,
   translate, re-spell, fix grammar, add or remove diacritics, or "clean up"
   anything. If you cannot copy it exactly, do not return it.

2. NEVER INVENT A CITATION. Fill a citation field ONLY if that exact value is
   printed in the text or in a REFERENCE line above. If the hadith number is not
   there, `hadith_no` is null. If the volume is not there, `jild` is null. Set
   `citation_incomplete` to true whenever ANY of kitab / jild / safha / hadith_no /
   rawi is null. NEVER guess a number. A guessed hadith number is worse than no item.

3. NEVER INVENT A GRADING. `darja` may be "sahih", "hasan" or "zaeef" ONLY if the
   text itself explicitly says so (صحیح / حسن / ضعیف about THIS narration). In every
   other case `darja` is "na_maloom". Your own knowledge of the hadith's status is
   irrelevant and must not be used.

4. COMPLETE UNITS ONLY. One item = one full hadith, OR one full tafseer passage about
   one ayah, OR one full hikayat (story of a wali/buzurg), OR one full fiqhi mas'ala.
   Never half of one. If the unit starts before the CURRENT excerpt or runs past its
   end, SKIP IT — a later window will contain it whole.

5. NEVER MERGE and NEVER SPLIT. Two narrations are two items, never one. One
   narration is one item, never two.

6. Only extract from the CURRENT excerpt. Use PREVIOUS/NEXT for context and citation
   only — never extract an item whose text lives there.

7. Classify `category` as exactly one of: "hadith", "tafsir", "hikayat", "fiqh".
   - hadith: words of the Prophet ﷺ narrated with or without a chain.
   - tafsir: explanation of a specific Qur'anic ayah.
   - hikayat: a narrated incident/story about a wali, buzurg or the Sahaba.
   - fiqh: a ruling / mas'ala.
   If you are not sure which one it is, SKIP the item. Do not guess.

8. Returning an empty array is a NORMAL, CORRECT answer. Most excerpts contain no
   complete, self-contained, citable unit. Extracting nothing is far better than
   extracting something imperfect. Do not try to find something.

OUTPUT FORMAT — output ONLY this JSON array, nothing else. No prose, no explanation,
no markdown code fences:

[
  {
    "category": "hadith",
    "text_ur": "<exact substring copied from the CURRENT excerpt>",
    "citation": {
      "kitab": "<book name as printed, or null>",
      "jild": "<volume as printed, or null>",
      "safha": "<page as printed, or null>",
      "hadith_no": "<hadith number as printed, or null>",
      "rawi": "<narrator named in the text, or null>"
    },
    "citation_incomplete": true,
    "darja": "na_maloom",
    "maslak_tag": null
  }
]
"""


# ---------------------------------------------------------------------------
# Model plumbing (same key-rotation pattern as ingest_book.py)
# ---------------------------------------------------------------------------
INGESTION_KEYS: List[str] = []
for _i in range(1, 9):
    _k = (os.getenv(f"INGESTION_KEY_{_i}") or "").strip()
    if _k:
        INGESTION_KEYS.append(_k)
if not INGESTION_KEYS:
    _main = (os.getenv("GEMINI_API_KEY") or "").strip()
    if _main:
        INGESTION_KEYS.append(_main)

_key_index = 0


def _advance_key():
    """Rotate to the next key so no single key absorbs the whole run's RPM."""
    global _key_index
    if not INGESTION_KEYS:
        raise SystemExit("No INGESTION_KEY_1..8 and no GEMINI_API_KEY in .env")
    import google.generativeai as genai

    _key_index = (_key_index + 1) % len(INGESTION_KEYS)
    genai.configure(api_key=INGESTION_KEYS[_key_index])


def _call_model(window_text: str) -> str:
    """One extraction call, with key rotation and quota backoff. Returns raw text."""
    import google.generativeai as genai

    sweeps = 0
    while True:
        try:
            _advance_key()
            model = genai.GenerativeModel(
                EXTRACTOR_MODEL,
                generation_config=genai.types.GenerationConfig(
                    # Deterministic: this is a copying task, not a creative one.
                    temperature=0.0,
                    # Urdu tokenizes heavily and an item is copied verbatim, so a
                    # window holding two long narrations can run past a small cap —
                    # a truncated reply is unparseable JSON and the window is lost.
                    max_output_tokens=int(os.getenv("EXTRACT_MAX_TOKENS", "8192")),
                    response_mime_type="application/json",
                ),
            )
            resp = model.generate_content(
                EXTRACTION_PROMPT + "\n\n=== INPUT ===\n" + window_text,
                request_options={"timeout": float(os.getenv("EXTRACT_TIMEOUT", "90"))},
            )
            return (resp.text or "").strip()
        except Exception as exc:
            msg = str(exc)
            # A safety/RECITATION block on scripture is not retryable — treat the
            # window as yielding nothing rather than burning the whole run on it.
            if "finish_reason" in msg or "RECITATION" in msg or "blocked" in msg.lower():
                print(f"  window blocked by the model ({msg[:80]}); skipping")
                return ""
            # A retired or misspelled model id will never start working; retrying it
            # 20 times just wastes 20 minutes before saying so.
            if "404" in msg or "not found" in msg.lower() or "no longer available" in msg:
                raise SystemExit(
                    f"Model '{EXTRACTOR_MODEL}' is not available to this key: {msg[:200]}\n"
                    f"Set EXTRACTOR_MODEL in .env to a current model."
                )
            sweeps += 1
            if sweeps > MAX_QUOTA_SWEEPS:
                raise
            wait = 60 if ("429" in msg or "quota" in msg.lower()) else min(15 * sweeps, 90)
            print(f"  model call failed ({msg[:120]}); retry in {wait}s")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def build_window(chunks: List[Dict[str, Any]], i: int) -> str:
    """Previous + CURRENT + next, each with its machine-generated reference line.

    The reference lines are generated at ingestion from the book/volume/page the text
    was scraped from, so they are trustworthy citation material — unlike anything the
    model might recall. Including them is what lets a citation be complete at all.
    """
    parts = []
    for offset, label in ((-1, "PREVIOUS"), (0, "CURRENT"), (1, "NEXT")):
        j = i + offset
        if j < 0 or j >= len(chunks):
            continue
        c = chunks[j]
        parts.append(
            f"--- {label} EXCERPT ---\n"
            f"REFERENCE: {c.get('reference') or '(none)'}\n"
            f"{c.get('content') or ''}"
        )
    return "\n\n".join(parts)


def verifiable_text(chunks: List[Dict[str, Any]], i: int) -> str:
    """The corpus a returned item is checked against — the same window, references
    included, so a citation drawn from a reference line verifies."""
    return build_window(chunks, i)


# ---------------------------------------------------------------------------
# Validation — the mechanical half of "the LLM never authors content"
# ---------------------------------------------------------------------------
_URDU_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

_DARJA_WORDS = {
    "sahih": ("صحیح", "صحيح"),
    "hasan": ("حسن",),
    "zaeef": ("ضعیف", "ضعيف"),
}


def _digit_fold(s: str) -> str:
    return vi.normalise(s).translate(_URDU_DIGITS)


def appears_in(value: Any, window: str) -> bool:
    """Is this citation value physically present in the window?

    Urdu-Indic and ASCII digits are folded together, because a page printed as
    "صفحہ ۲۳" and a model that writes "23" mean the same page — but "23" when the
    page says "۲۴" must still fail.
    """
    text = str(value or "").strip()
    if not text:
        return False
    return _digit_fold(text) in _digit_fold(window)


def parse_items(raw: str) -> Optional[List[Dict[str, Any]]]:
    """Defensive JSON parse: strip fences, then take the outermost array."""
    if not raw:
        return []
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("["), s.rfind("]")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(s[start:end + 1])
        except json.JSONDecodeError:
            return None
    if isinstance(data, dict):
        data = [data]
    return data if isinstance(data, list) else None


def insert_with_retry(**kwargs) -> Optional[str]:
    """Insert one draft, riding out Neon's idle-connection drops.

    Neon (serverless) closes connections during the pauses between model calls, and
    the drop only surfaces on the next statement. Every long-running script in this
    repo retries around it rather than losing the run; this does the same.
    """
    for attempt in range(12):
        try:
            return vi.insert_draft(**kwargs)
        except Exception as exc:
            print(f"  db insert failed ({exc}); retry in 20s")
            time.sleep(20)
    raise SystemExit("Database unreachable for 4 minutes; re-run to resume from the checkpoint.")


def build_citation_display(citation: Dict[str, Any]) -> str:
    """The on-screen citation line, assembled ONLY from verified fields.

    Deliberately not taken from the model: a display string is what a viewer actually
    reads, so it must be impossible for it to contain a value that failed verification.
    """
    bits = []
    if citation.get("kitab"):
        bits.append(str(citation["kitab"]))
    if citation.get("jild"):
        bits.append(f"Jild {citation['jild']}")
    if citation.get("safha"):
        bits.append(f"Safha {citation['safha']}")
    if citation.get("hadith_no"):
        bits.append(f"Hadees {citation['hadith_no']}")
    return ", ".join(bits)


CITATION_FIELDS = ("kitab", "jild", "safha", "hadith_no", "rawi")


def validate(candidate: Any, window: str, stats: Dict[str, int]) -> Optional[Dict[str, Any]]:
    """Return a clean, insertable item, or None (and count why it was dropped)."""
    if not isinstance(candidate, dict):
        stats["dropped_malformed"] += 1
        return None

    category = str(candidate.get("category") or "").strip().lower()
    if category not in vi.CATEGORIES:
        stats["dropped_bad_category"] += 1
        return None

    text_ur = candidate.get("text_ur")
    if not isinstance(text_ur, str) or not text_ur.strip():
        stats["dropped_malformed"] += 1
        return None
    text_ur = text_ur.strip()

    # The central check: did the model copy, or did it write?
    if not vi.is_verbatim(text_ur, window):
        stats["dropped_not_verbatim"] += 1
        return None

    words = vi.word_count(text_ur)
    if words < ARGS.min_words or words > ARGS.max_words:
        stats["dropped_length"] += 1
        return None

    raw_citation = candidate.get("citation")
    raw_citation = raw_citation if isinstance(raw_citation, dict) else {}
    citation: Dict[str, Any] = {}
    for field in CITATION_FIELDS:
        value = raw_citation.get(field)
        value = str(value).strip() if value not in (None, "", "null") else ""
        if value and appears_in(value, window):
            citation[field] = value
        else:
            if value:
                stats["citation_fields_rejected"] += 1
            citation[field] = None

    # `rawi` is frequently absent from a printed passage and is not part of locating
    # the text, so it does not by itself make a citation "incomplete". The locating
    # fields do — and for a hadith that includes its number.
    required = ["kitab", "safha"] if category != "hadith" else ["kitab", "safha", "hadith_no"]
    citation_incomplete = any(not citation.get(f) for f in required)

    darja = str(candidate.get("darja") or "na_maloom").strip().lower()
    if category != "hadith":
        darja = None
    else:
        if darja not in vi.DARJAT:
            darja = "na_maloom"
        if darja in _DARJA_WORDS and not any(w in window for w in _DARJA_WORDS[darja]):
            # The model asserted a grading the page does not state. That is exactly
            # the failure mode rule 3 exists for.
            stats["darja_rejected"] += 1
            darja = "na_maloom"

    maslak = candidate.get("maslak_tag")
    maslak = str(maslak).strip() if isinstance(maslak, str) and maslak.strip() else None

    return {
        "category": category,
        "text_ur": text_ur,
        "citation": citation,
        "citation_display": build_citation_display(citation),
        "citation_incomplete": citation_incomplete,
        "darja": darja,
        "maslak_tag": maslak,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
ARGS: argparse.Namespace


def checkpoint_path(book: str, jild: Optional[int]) -> Path:
    name = f"{book}{'-jild-' + str(jild) if jild is not None else ''}.json"
    return CHECKPOINT_DIR / name


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", required=True, help="book slug, e.g. anwaar-ul-hadees (tags[1])")
    ap.add_argument("--jild", type=int, default=None, help="restrict to one volume")
    ap.add_argument("--category", default=None, choices=list(vi.CATEGORIES),
                    help="keep only items the model classifies as this category")
    ap.add_argument("--limit", type=int, default=200, help="max windows to process this run")
    ap.add_argument("--start", type=int, default=None, help="window index to start at (overrides checkpoint)")
    ap.add_argument("--step", type=int, default=1, help="advance the window by N chunks")
    ap.add_argument("--min-words", type=int, default=15,
                    help="drop shorter candidates (publishable band is 25-160)")
    ap.add_argument("--max-words", type=int, default=220,
                    help="drop longer candidates (publishable band is 25-160)")
    ap.add_argument("--dry-run", action="store_true", help="print, write nothing")
    ap.add_argument("--no-checkpoint", action="store_true")
    ARGS = ap.parse_args()

    chunks = vi.ordered_chunks(ARGS.book, ARGS.jild)
    if not chunks:
        books = ", ".join(b["slug"] for b in vi.book_slugs()[:15])
        raise SystemExit(f"No chunks for book '{ARGS.book}'. Available: {books}, …")

    ckpt = checkpoint_path(ARGS.book, ARGS.jild)
    start = 0
    if ARGS.start is not None:
        start = ARGS.start
    elif not ARGS.no_checkpoint and ckpt.exists():
        start = int(json.loads(ckpt.read_text(encoding="utf-8")).get("next_window", 0))

    end = min(start + ARGS.limit * ARGS.step, len(chunks))
    print(f"{ARGS.book}{f' jild {ARGS.jild}' if ARGS.jild is not None else ''}: "
          f"{len(chunks)} chunks; windows {start}..{end} (step {ARGS.step}), "
          f"model={EXTRACTOR_MODEL}{' [DRY RUN]' if ARGS.dry_run else ''}")

    if not ARGS.dry_run:
        vi.init_video_items()

    if INGESTION_KEYS:
        globals()["_key_index"] = random.randrange(len(INGESTION_KEYS))

    stats = {
        "windows": 0, "proposed": 0, "inserted": 0, "duplicate": 0,
        "dropped_malformed": 0, "dropped_bad_category": 0, "dropped_not_verbatim": 0,
        "dropped_length": 0, "dropped_other_category": 0, "parse_failures": 0,
        "citation_fields_rejected": 0, "darja_rejected": 0, "citation_incomplete": 0,
    }
    seen_hashes = set()

    i = start
    while i < end:
        window = build_window(chunks, i)
        stats["windows"] += 1

        raw = _call_model(window)
        candidates = parse_items(raw)
        if candidates is None:
            # One retry, as the spec requires — a truncated or fenced reply is common
            # enough to be worth a second attempt before dropping the window.
            print(f"  [{i}] JSON parse failed; retrying once")
            time.sleep(CALL_SLEEP)
            candidates = parse_items(_call_model(window))
        if candidates is None:
            stats["parse_failures"] += 1
            print(f"  [{i}] unparseable after retry; skipping window")
            candidates = []

        stats["proposed"] += len(candidates)
        for cand in candidates:
            item = validate(cand, verifiable_text(chunks, i), stats)
            if not item:
                continue
            if ARGS.category and item["category"] != ARGS.category:
                stats["dropped_other_category"] += 1
                continue
            digest = vi.content_hash(item["text_ur"])
            if digest in seen_hashes:
                stats["duplicate"] += 1
                continue
            seen_hashes.add(digest)
            if item["citation_incomplete"]:
                stats["citation_incomplete"] += 1

            window_ids = [chunks[j]["id"] for j in (i - 1, i, i + 1) if 0 <= j < len(chunks)]

            if ARGS.dry_run:
                stats["inserted"] += 1
                head = item["text_ur"][:90].replace("\n", " ")
                print(f"  [{i}] {item['category']:8s} "
                      f"{'INCOMPLETE' if item['citation_incomplete'] else 'complete  '} "
                      f"{vi.word_count(item['text_ur']):3d}w  {item['citation_display'] or '(no citation)'}"
                      f"\n        {head}…")
                continue

            new_id = insert_with_retry(
                category=item["category"],
                text_ur=item["text_ur"],
                citation=item["citation"],
                citation_display=item["citation_display"],
                citation_incomplete=item["citation_incomplete"],
                darja=item["darja"],
                maslak_tag=item["maslak_tag"],
                source_chunk_ids=window_ids,
                extractor_version=EXTRACTOR_VERSION,
            )
            if new_id:
                stats["inserted"] += 1
            else:
                stats["duplicate"] += 1

        if not ARGS.dry_run and not ARGS.no_checkpoint:
            CHECKPOINT_DIR.mkdir(exist_ok=True)
            ckpt.write_text(
                json.dumps({"next_window": i + ARGS.step}, indent=1), encoding="utf-8"
            )
        if stats["windows"] % 10 == 0:
            print(f"  [{i}] {stats['windows']} windows, {stats['inserted']} inserted")
        i += ARGS.step
        time.sleep(CALL_SLEEP)

    print("\n=== summary ===")
    for key in ("windows", "proposed", "inserted", "duplicate", "dropped_not_verbatim",
                "dropped_length", "dropped_bad_category", "dropped_other_category",
                "dropped_malformed", "parse_failures", "citation_fields_rejected",
                "darja_rejected"):
        print(f"  {key:26s} {stats[key]}")

    kept = stats["inserted"]
    if kept:
        rate = 100.0 * stats["citation_incomplete"] / kept
        print(f"  {'citation_incomplete':26s} {stats['citation_incomplete']} ({rate:.0f}%)")
        if rate > 30:
            print(
                "\n*** WARNING: citation_incomplete is above 30%. Do NOT start reviewing "
                "yet.\n*** Most of these items cannot be approved for video no matter how "
                "good the text is,\n*** so a reviewer would burn hours on unusable cards. "
                "The window or the source's\n*** page metadata needs work first — check "
                "that this book's chunks carry safha\n*** numbers in their reference line, "
                "and consider a wider window (--step 1 with\n*** a 5-chunk window) for a "
                "book whose citations sit far from the text.\n"
            )
    if stats["dropped_not_verbatim"]:
        print(f"\nNote: {stats['dropped_not_verbatim']} candidate(s) were dropped because the "
              "text was not\nan exact substring of the source — that guard working as intended.")
    if not ARGS.dry_run:
        print(f"\nAll inserted rows are status='draft', allow_video=false. "
              f"Review them at /admin/video-items.")


if __name__ == "__main__":
    main()
