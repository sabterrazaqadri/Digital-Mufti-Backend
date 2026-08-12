"""
Better Auth session-token verification for the FastAPI backend.

Identity is derived ONLY from a cryptographically verified Better Auth JWT
(signed by the Next.js app and exposed via its JWKS endpoint) — never from a
client-supplied user_id. This closes the IDOR hole where any caller could
read/delete another user's chats by passing their id.
"""
import os
from typing import Optional

import jwt
from jwt import PyJWKClient
from fastapi import Header, HTTPException
from dotenv import load_dotenv

load_dotenv()

# Base URL(s) of the Next.js app that issue tokens (Better Auth `baseURL`).
# Accepts a comma-separated list so a single backend can trust tokens from both
# production and local dev (e.g. "https://digitalmufti.vercel.app,http://localhost:3000")
# — both share the same DB/signing keys, only the `iss` claim differs.
# .strip() guards against a trailing newline/space in the host's secret value —
# otherwise the JWKS hostname becomes "...vercel.app\n" and DNS fails with
# "Name or service not known", silently 401-ing every signed-in request.
_AUTH_URLS = [
    u.strip().rstrip("/")
    for u in (os.getenv("BETTER_AUTH_URL") or "").split(",")
    if u.strip()
]
# Primary URL (first entry) is used to locate the JWKS endpoint.
BETTER_AUTH_URL = _AUTH_URLS[0] if _AUTH_URLS else ""
# Every configured URL is an accepted token issuer.
ALLOWED_ISSUERS = set(_AUTH_URLS)
# JWKS published by the Better Auth `jwt` plugin.
JWKS_URL = (os.getenv("BETTER_AUTH_JWKS_URL") or "").strip() or (
    f"{BETTER_AUTH_URL}/api/auth/jwks" if BETTER_AUTH_URL else None
)

# Better Auth signs with EdDSA (Ed25519) by default; allow common alternates.
ALGORITHMS = ["EdDSA", "ES256", "RS256"]

_jwk_client: Optional[PyJWKClient] = PyJWKClient(JWKS_URL) if JWKS_URL else None


def auth_configured() -> bool:
    return _jwk_client is not None


def _verify_claims(token: str) -> dict:
    """Verify a Better Auth JWT and return its claims. Raises on failure."""
    if not _jwk_client:
        raise HTTPException(status_code=503, detail="Authentication not configured")
    try:
        signing_key = _jwk_client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=ALGORITHMS,
            options={
                "verify_aud": False,
                # Issuer is checked manually below against the allowed set, so
                # one backend can accept tokens from prod and localhost alike.
                "verify_iss": False,
                "require": ["exp", "sub"],
            },
            leeway=10,
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")

    if ALLOWED_ISSUERS and claims.get("iss") not in ALLOWED_ISSUERS:
        raise HTTPException(status_code=401, detail="Invalid token: Invalid issuer")

    if not claims.get("sub"):
        raise HTTPException(status_code=401, detail="Token missing subject")
    return claims


def _verify_token(token: str) -> str:
    """Verify a Better Auth JWT and return the user id (sub)."""
    return _verify_claims(token)["sub"]


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
        return parts[1].strip()
    return None


def get_current_user_id(authorization: Optional[str] = Header(default=None)) -> str:
    """Dependency: require a valid signed-in user. 401 otherwise."""
    token = _extract_bearer(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    return _verify_token(token)


def get_optional_user_id(authorization: Optional[str] = Header(default=None)) -> Optional[str]:
    """Dependency: return verified user id if present, else None (guest)."""
    token = _extract_bearer(authorization)
    if not token:
        return None
    return _verify_token(token)


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------
# There is no admin ROLE in Better Auth here, and adding one would mean a schema
# change to the auth tables. An env allowlist checked on top of the same verified
# JWT gives the same guarantee with no new auth mechanism: identity still comes
# only from a signed token, the allowlist just says which identities may curate.
#
#   ADMIN_EMAILS    comma-separated, matched against the token's `email` claim
#   ADMIN_USER_IDS  comma-separated, matched against `sub` (use when the token
#                   carries no email claim)
#
# With NEITHER set, every admin route 503s — an unconfigured allowlist must never
# mean "everyone is an admin".

def _csv_env(name: str) -> set:
    return {v.strip().lower() for v in (os.getenv(name) or "").split(",") if v.strip()}


def admin_configured() -> bool:
    return bool(_csv_env("ADMIN_EMAILS") or _csv_env("ADMIN_USER_IDS"))


def get_admin_user(authorization: Optional[str] = Header(default=None)) -> str:
    """Dependency: require a signed-in user who is on the admin allowlist.

    Returns an identifier for the reviewer, stored as `approved_by` so every
    approval is attributable to a person."""
    if not admin_configured():
        raise HTTPException(status_code=503, detail="Admin access not configured")
    token = _extract_bearer(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    claims = _verify_claims(token)

    email = str(claims.get("email") or "").strip().lower()
    sub = str(claims.get("sub") or "")
    if email and email in _csv_env("ADMIN_EMAILS"):
        return email
    if sub and sub.lower() in _csv_env("ADMIN_USER_IDS"):
        return sub
    raise HTTPException(status_code=403, detail="Admin access required")


def whoami(authorization: Optional[str] = Header(default=None)) -> dict:
    """Verified identity of the caller — used once, to find the value to put in
    ADMIN_EMAILS/ADMIN_USER_IDS. Returns only the caller's own claims."""
    token = _extract_bearer(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    claims = _verify_claims(token)
    return {
        "user_id": claims.get("sub"),
        "email": claims.get("email"),
        "is_admin": bool(
            (str(claims.get("email") or "").lower() in _csv_env("ADMIN_EMAILS"))
            or (str(claims.get("sub") or "").lower() in _csv_env("ADMIN_USER_IDS"))
        ),
        "admin_configured": admin_configured(),
    }
