import hashlib
import secrets
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.db_models import APIKey, User

API_KEY_PREFIX = "vdd_"

# auto_error=False: HTTPBearer's default raises 403 on a missing header,
# but a missing/invalid API key should be 401.
_bearer_scheme = HTTPBearer(auto_error=False)


def generate_api_key() -> str:
    return f"{API_KEY_PREFIX}{secrets.token_urlsafe(32)}"


def hash_api_key(raw_key: str) -> str:
    # sha256, not bcrypt/argon2: those defend low-entropy human passwords
    # via deliberate slowness + random salting. This key already has ~256
    # bits of CSPRNG entropy (brute-forcing the hash is infeasible either
    # way), and the random salt those algorithms add would make an indexed
    # "WHERE key_hash = ?" lookup impossible on every authenticated request.
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def display_prefix(raw_key: str) -> str:
    return raw_key[: len(API_KEY_PREFIX) + 8]


def get_current_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> APIKey:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    key_hash = hash_api_key(credentials.credentials)
    api_key = db.scalars(select(APIKey).where(APIKey.key_hash == key_hash)).one_or_none()
    if api_key is None or api_key.revoked_at is not None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "invalid or revoked API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    api_key.last_used_at = datetime.now(timezone.utc)
    db.commit()
    return api_key


def get_current_user(api_key: APIKey = Depends(get_current_api_key)) -> User:
    return api_key.user
