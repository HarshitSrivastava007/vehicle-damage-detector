import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.db_models import User, UserSession

SESSION_COOKIE_NAME = "vdd_session"
SESSION_TTL = timedelta(days=7)


def utcnow() -> datetime:
    # Naive UTC, not datetime.now(timezone.utc): SQLite silently drops
    # tzinfo on round-trip (confirmed — a stored aware datetime comes back
    # naive), so comparing a fresh aware "now" against a DB-loaded value
    # raises TypeError. Every expires_at/created_at comparison in this
    # module must use this same naive-UTC convention consistently.
    return datetime.now(timezone.utc).replace(tzinfo=None)


def hash_password(password: str) -> str:
    # bcrypt, not sha256: passwords are low-entropy human-chosen secrets
    # vulnerable to offline brute force if a hash dump ever leaks — the
    # deliberate slowness + per-password random salt is what defends
    # against that. Contrast with hash_session_token() below and
    # app/auth.py::hash_api_key(), both of which hash high-entropy
    # server-generated tokens and deliberately use a fast, unsalted,
    # indexable hash instead.
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def generate_session_token() -> str:
    return secrets.token_urlsafe(32)


def hash_session_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def resolve_session_user(raw_token: str | None, db: Session) -> User | None:
    """Look up and validate a raw session token. Returns None (never
    raises) so callers that want to try a different auth mechanism
    afterwards can."""
    if not raw_token:
        return None

    token_hash = hash_session_token(raw_token)
    session = db.scalars(select(UserSession).where(UserSession.session_token_hash == token_hash)).one_or_none()
    if session is None or session.revoked_at is not None or session.expires_at < utcnow():
        return None
    # Checked on every request (not just login) so deactivating a user
    # immediately blocks their existing session, not just future logins.
    if not session.user.is_active:
        return None

    return session.user


def get_current_session_user(request: Request, db: Session = Depends(get_db)) -> User:
    user = resolve_session_user(request.cookies.get(SESSION_COOKIE_NAME), db)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not logged in or session expired")
    return user


def require_admin(user: User = Depends(get_current_session_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin privileges required")
    return user
