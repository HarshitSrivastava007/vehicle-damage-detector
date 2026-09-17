from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.auth import resolve_api_key
from app.db import get_db
from app.db_models import APIKey, User
from app.session_auth import SESSION_COOKIE_NAME, resolve_session_user


def get_current_user_flexible(request: Request, db: Session = Depends(get_db)) -> tuple[User, APIKey | None]:
    """Accepts either a Bearer API key or a session cookie — used by
    endpoints the SPA needs to call (which only ever has a session
    cookie, never an API key) that are also part of the programmatic API
    (which only ever has an API key). Returns (user, api_key) — api_key
    is None when authenticated via session."""
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        api_key = resolve_api_key(auth_header[7:].strip(), db)
        if api_key is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or revoked API key")
        return api_key.user, api_key

    user = resolve_session_user(request.cookies.get(SESSION_COOKIE_NAME), db)
    if user is not None:
        return user, None

    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "authentication required (API key or session)")
