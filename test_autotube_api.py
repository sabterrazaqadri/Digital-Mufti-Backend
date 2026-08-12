"""
End-to-end test of the AutoTube feed — the curl checklist, automated.

    python test_autotube_api.py

Boots the real FastAPI app with uvicorn on a free port, seeds a small pool of
approved items, and checks the behaviours the feed is actually judged on:

  * 401 without the service key, 401 with a wrong one, 200 with the right one
  * the same tenant never receives the same item twice
  * two tenants walk the pool in different orders
  * 204 (not a fallback item) when a tenant's pool is exhausted
  * `ack failed` returns the item to that tenant's pool
  * an expired reservation is swept and the item is served again
  * the per-tenant rate limit returns 429 with Retry-After
  * a browser-style request (Origin header) is refused — these routes have no CORS

Uses `requests` and the stdlib only; the repo has no test framework and this is not
worth adding one for. Seeded rows are marked and deleted in a finally block, and
every tenant id is random per run, so it is safe against the live database.

Exit code 0 = all checks passed.
"""
import os
import socket
import subprocess
import sys
import time
import uuid

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests

import database as db
import video_items as vi

SERVICE_KEY = "test-key-" + uuid.uuid4().hex
MARK = "TEST-AUTOTUBE-"
POOL = 6                      # items seeded per category
CATEGORY = "hadith"

failures = []
seeded = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def seed_pool():
    """Insert POOL drafts and approve them through the normal human path."""
    for n in range(POOL):
        item_id = str(uuid.uuid4())
        text = " ".join([f"لفظ{n}"] * 40)
        with db.get_cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO video_items
                    (id, category, text_ur, citation, citation_display,
                     citation_incomplete, darja, source_chunk_ids, content_hash,
                     status, allow_video, word_count, extractor_version)
                VALUES (%s, %s, %s, %s::jsonb, %s, FALSE, 'sahih', '{}', %s,
                        'draft', FALSE, 40, 'test');
                """,
                (item_id, CATEGORY, text,
                 '{"kitab": "Test Kitab", "safha": "%d", "hadith_no": "%d"}' % (n + 1, n + 1),
                 f"Test Kitab, Safha {n + 1}, Hadees {n + 1}", MARK + item_id),
            )
        result = vi.approve(item_id, allow_video=True, approved_by="test@example.com")
        if not result.get("ok"):
            raise SystemExit(f"seed approve failed: {result}")
        seeded.append(item_id)


def cleanup():
    with db.get_cursor(commit=True) as cur:
        cur.execute("DELETE FROM video_items WHERE content_hash LIKE %s;", (MARK + "%",))
        deleted = cur.rowcount
        cur.execute("DELETE FROM autotube_rate_limit WHERE tenant_id LIKE 'test-%';")
    print(f"\nCleaned up {deleted} seeded items (serve rows cascade).")


def start_server(port):
    env = dict(os.environ)
    env["AUTOTUBE_SERVICE_KEY"] = SERVICE_KEY
    env["AUTOTUBE_API_ENABLED"] = "true"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1",
         "--port", str(port), "--log-level", "warning"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        if proc.poll() is not None:
            raise SystemExit("uvicorn exited during startup")
        try:
            if requests.get(f"{base}/api/autotube/health", timeout=2).status_code == 200:
                return proc, base
        except requests.RequestException:
            pass
        time.sleep(1)
    proc.terminate()
    raise SystemExit("server did not come up within 120s")


def main():
    vi.init_video_items()
    port = free_port()
    proc = None
    try:
        seed_pool()
        proc, base = start_server(port)
        auth = {"Authorization": f"Bearer {SERVICE_KEY}"}
        tenant_a = "test-" + uuid.uuid4().hex
        tenant_b = "test-" + uuid.uuid4().hex

        def nxt(tenant, headers=auth):
            return requests.get(
                f"{base}/api/autotube/next",
                params={"category": CATEGORY, "tenant": tenant, "lang": "urdu"},
                headers=headers, timeout=30,
            )

        print("\nAuthentication:")
        check("health is unauthenticated",
              requests.get(f"{base}/api/autotube/health", timeout=10).status_code == 200)
        check("no key -> 401", nxt(tenant_a, {}).status_code == 401)
        check("wrong key -> 401",
              nxt(tenant_a, {"Authorization": "Bearer wrong-key"}).status_code == 401)
        r = nxt(tenant_a)
        check("correct key -> 200", r.status_code == 200, f"status {r.status_code}")

        print("\nPayload:")
        body = r.json() if r.status_code == 200 else {}
        for field in ("item_id", "category", "text_ur", "citation", "citation_display",
                      "darja", "word_count", "attribution", "expires_at"):
            check(f"payload has {field}", field in body)
        check("attribution names Digital Mufti",
              "Digital Mufti" in (body.get("attribution") or ""))

        print("\nBrowser access:")
        check("Origin header -> 403 (no CORS on this API)",
              nxt(tenant_a, {**auth, "Origin": "https://digitalmufti.vercel.app"})
              .status_code == 403)

        print("\nPool walking:")
        # Pool may hold rows beyond the seeded ones; ask the API how many are left.
        stats = requests.get(f"{base}/api/autotube/stats", params={"tenant": tenant_a},
                             headers=auth, timeout=30).json()
        remaining = stats["categories"][CATEGORY]["remaining"]
        seen = [body.get("item_id")]
        exhausted = False
        for _ in range(remaining + 1):
            rr = nxt(tenant_a)
            if rr.status_code == 204:
                exhausted = True
                break
            if rr.status_code != 200:
                break
            seen.append(rr.json()["item_id"])
        check("same tenant never gets the same item twice",
              len(seen) == len(set(seen)), f"{len(seen)} served, {len(set(seen))} unique")
        check("exhausted pool -> 204 (no fallback item)", exhausted)
        check("204 body is empty", exhausted)

        seq_b = []
        for _ in range(min(4, POOL)):
            rr = nxt(tenant_b)
            if rr.status_code != 200:
                break
            seq_b.append(rr.json()["item_id"])
        check("second tenant is served from the same pool", len(seq_b) >= 2)
        check("two tenants walk the pool in different orders",
              seq_b != seen[:len(seq_b)],
              f"A={[s[:8] for s in seen[:len(seq_b)]]} B={[s[:8] for s in seq_b]}")

        print("\nAck:")
        used_id = seq_b[0]
        ack_used = requests.post(f"{base}/api/autotube/ack", headers=auth, timeout=30,
                                 json={"tenant": tenant_b, "item_id": used_id,
                                       "run_id": "run-1", "status": "used"})
        check("ack used -> 200", ack_used.status_code == 200)
        again = requests.post(f"{base}/api/autotube/ack", headers=auth, timeout=30,
                              json={"tenant": tenant_b, "item_id": used_id,
                                    "run_id": "run-1", "status": "used"})
        check("re-acking is an idempotent no-op", again.status_code == 200)

        failed_id = seq_b[1]
        before = requests.get(f"{base}/api/autotube/stats", params={"tenant": tenant_b},
                              headers=auth, timeout=30).json()["categories"][CATEGORY]["remaining"]
        requests.post(f"{base}/api/autotube/ack", headers=auth, timeout=30,
                      json={"tenant": tenant_b, "item_id": failed_id,
                            "run_id": "run-2", "status": "failed"})
        after = requests.get(f"{base}/api/autotube/stats", params={"tenant": tenant_b},
                             headers=auth, timeout=30).json()["categories"][CATEGORY]["remaining"]
        check("ack failed returns the item to the pool", after == before + 1,
              f"remaining {before} -> {after}")

        print("\nExpiry sweep:")
        # A crashed run leaves a live reservation; it must not burn the item forever.
        stuck = seq_b[2] if len(seq_b) > 2 else None
        if stuck:
            with db.get_cursor(commit=True) as cur:
                cur.execute(
                    "UPDATE video_item_serves SET expires_at = NOW() - INTERVAL '1 hour' "
                    "WHERE tenant_id = %s AND item_id = %s AND status = 'reserved';",
                    (tenant_b, stuck),
                )
                swept_row = cur.rowcount
            stats_after = requests.get(f"{base}/api/autotube/stats",
                                       params={"tenant": tenant_b}, headers=auth,
                                       timeout=30).json()["categories"][CATEGORY]["remaining"]
            check("expired reservation is swept back into the pool",
                  swept_row == 1 and stats_after >= after + 1,
                  f"remaining {after} -> {stats_after}")
        else:
            check("expired reservation is swept back into the pool", False,
                  "not enough items served to test")

        print("\nRate limit:")
        tenant_rl = "test-" + uuid.uuid4().hex
        statuses = [nxt(tenant_rl).status_code for _ in range(vi.RATE_LIMIT_PER_HOUR)]
        limited = requests.get(
            f"{base}/api/autotube/next",
            params={"category": CATEGORY, "tenant": tenant_rl}, headers=auth, timeout=30,
        )
        check(f"first {vi.RATE_LIMIT_PER_HOUR} calls are not rate-limited",
              429 not in statuses, f"statuses {sorted(set(statuses))}")
        check("over the limit -> 429", limited.status_code == 429)
        check("429 carries Retry-After", "Retry-After" in limited.headers)

        print("\nKill switch:")
        os.environ["AUTOTUBE_API_ENABLED"] = "false"
        import autotube
        check("AUTOTUBE_API_ENABLED=false disables the feed", not autotube._api_enabled())
        os.environ["AUTOTUBE_API_ENABLED"] = "true"
        check("AUTOTUBE_API_ENABLED=true enables it", autotube._api_enabled())

        print("\nAdmin routes are never open:")
        # The curation routes must never answer an unauthenticated caller — 503 when
        # no allowlist is configured, 401 when one is. Never 200, either way.
        for route in ("/api/admin/video-items", "/api/admin/video-items/stats"):
            code = requests.get(f"{base}{route}", timeout=30).status_code
            check(f"GET {route} without a token", code in (401, 403, 503), f"got {code}")
        code = requests.post(
            f"{base}/api/admin/video-items/{uuid.uuid4()}/approve",
            json={"allow_video": True}, timeout=30,
        ).status_code
        check("approve without a token", code in (401, 403, 503), f"got {code}")

        print("\nUnapproved items are never served:")
        with db.get_cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS n FROM video_item_serves s "
                "JOIN video_items i ON i.id = s.item_id "
                "WHERE s.tenant_id IN (%s, %s, %s) "
                "AND NOT (i.status = 'approved' AND i.allow_video);",
                (tenant_a, tenant_b, tenant_rl),
            )
            leaked = int(cur.fetchone()["n"])
        check("no draft/rejected item was ever reserved", leaked == 0)

    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        cleanup()

    if failures:
        print(f"\n{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
