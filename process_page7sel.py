"""One-shot OCR -> clean -> ingest for the 5 selected page-7 books.
Smallest-first; resume-able (skips existing/blocked pages, checkpointed ingest).
205 pages total. Re-run to resume after any Neon drop / quota stall."""
import subprocess
import sys

BOOKS = [  # (dir, display_name) smallest-first by page count
    ("../data/200_karamad_naseehatain", "200-Karamad-Naseehatain"),                       # 11
    ("../data/aadab_e_risalat_ki_qadar_wa_manzilat", "Aadab-e-Risalat-Ki-Qadar-wa-Manzilat"),  # 18
    ("../data/aadab_ul_murshid_wal_mureed", "Aadab-ul-Murshid-wal-Mureed"),               # 27
    ("../data/aainae_deoband", "Aainae-Deoband"),                                         # 27
    ("../data/zia_ul_harmain", "Zia-ul-Harmain"),                                         # 122
]

def run(args):
    r = subprocess.run([sys.executable, "-u", *args], cwd=".")
    if r.returncode != 0:
        print(f"STEP FAILED ({args[0]}) rc={r.returncode}; stopping. Re-run to resume.", flush=True)
        sys.exit(1)

for d, _ in BOOKS:
    print(f"\n=== OCR {d} ===", flush=True)
    run(["ocr_pdf_book.py", f"{d}/book.pdf", d])

print("\n=== CLEAN ===", flush=True)
run(["clean_alahazrat_ocr.py", *[d for d, _ in BOOKS]])

for d, name in BOOKS:
    print(f"\n=== INGEST {name} ===", flush=True)
    run(["ingest_book.py", d, "--book", name, "--jild", "1"])

print("\n=== FIVE BOOKS DONE ===", flush=True)
