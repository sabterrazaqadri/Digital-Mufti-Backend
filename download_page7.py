"""Download the 5 selected page-7 books to data/<dir>/book.pdf.
Same direct-then-pdf-proxy fallback as download_page6.py."""
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import requests

BOOKS = [  # (slug, dir, display_name)
    ("aadab-e-risalat-ki-qadar-wa-manzilat", "../data/aadab_e_risalat_ki_qadar_wa_manzilat", "Aadab-e-Risalat-Ki-Qadar-wa-Manzilat"),
    ("aadab-ul-murshid-wal-mureed", "../data/aadab_ul_murshid_wal_mureed", "Aadab-ul-Murshid-wal-Mureed"),
    ("aainae-deoband", "../data/aainae_deoband", "Aainae-Deoband"),
    ("200-karamad-naseehatain", "../data/200_karamad_naseehatain", "200-Karamad-Naseehatain"),
    ("zia-ul-harmain", "../data/zia_ul_harmain", "Zia-ul-Harmain"),
]

H = {"User-Agent": "Mozilla/5.0", "Referer": "https://alahazrat.net/"}
s = requests.Session()
s.headers.update(H)


def get(url):
    for _ in range(4):
        try:
            return s.get(url, timeout=180)
        except Exception:
            time.sleep(5)
    return None


def pdf_url(slug):
    r = get(f"https://alahazrat.net/books-library/{slug}/")
    if not r:
        return None
    m = re.search(r"pdf-proxy\.php\?url=(https?://[^\"'&]+\.pdf)", r.text) or \
        re.search(r"(https?://[^\"']*alahazrat\.info[^\"']+\.pdf)", r.text)
    return m.group(1) if m else None


def fetch_pdf(u):
    """Direct first, then the pdf-proxy fallback."""
    for candidate in (u, f"https://alahazrat.net/pdf-proxy.php?url={u}"):
        r = get(candidate)
        if r and r.content[:4] == b"%PDF" and len(r.content) > 10000:
            return r.content
    return None


def main():
    bad = []
    for slug, d, name in BOOKS:
        p = Path(d)
        p.mkdir(parents=True, exist_ok=True)
        pdf = p / "book.pdf"
        if pdf.exists() and pdf.stat().st_size > 10000:
            print(f"{slug}: already have book.pdf", flush=True)
            continue
        u = pdf_url(slug)
        if not u:
            bad.append((slug, "no-url"))
            print(f"{slug}: NO PDF URL", flush=True)
            continue
        data = fetch_pdf(u)
        if not data:
            bad.append((slug, "dl-fail"))
            print(f"{slug}: DL FAIL ({u})", flush=True)
            continue
        pdf.write_bytes(data)
        print(f"{slug}: saved {len(data)} bytes  [{name}]", flush=True)
        time.sleep(0.5)
    print(f"\nBad: {bad}", flush=True)


if __name__ == "__main__":
    main()
