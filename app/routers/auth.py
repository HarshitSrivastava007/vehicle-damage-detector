from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import display_prefix, generate_api_key, get_current_user, hash_api_key
from app.config import Settings, get_settings
from app.current_user import get_current_user_flexible
from app.db import get_db
from app.db_models import APIKey, User, UserSession
from app.schemas import (
    APIKeyCreateResponse,
    APIKeyOut,
    LoginRequest,
    RegisterRequest,
    RegisterResponse,
    SessionUserOut,
    UserOut,
)
from app.session_auth import (
    SESSION_COOKIE_NAME,
    SESSION_TTL,
    generate_session_token,
    get_current_session_user,
    hash_password,
    hash_session_token,
    utcnow,
    verify_password,
)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

_GENERIC_LOGIN_ERROR = "invalid email or password"


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)) -> RegisterResponse:
    user = User(email=body.email)
    if body.password:
        user.password_hash = hash_password(body.password)
    db.add(user)
    try:
        db.flush()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "email already registered") from exc

    raw_key = generate_api_key()
    db.add(
        APIKey(
            user_id=user.id,
            key_hash=hash_api_key(raw_key),
            prefix=display_prefix(raw_key),
        )
    )
    db.commit()

    return RegisterResponse(user_id=user.id, email=user.email, api_key=raw_key)


@router.post("/login", response_model=SessionUserOut)
def login(
    body: LoginRequest,
    response: Response,
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
) -> User:
    user = db.scalars(select(User).where(User.email == body.email)).one_or_none()
    if user is None or user.password_hash is None or not verify_password(body.password, user.password_hash):
        # Same message and status for "no such user", "no password set",
        # and "wrong password" — distinguishing them would let an attacker
        # enumerate registered emails.
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _GENERIC_LOGIN_ERROR)

    if not user.is_active:
        # Safe to be specific here (unlike the branch above): the caller
        # has already proven they know the correct password, so this
        # doesn't leak anything an attacker without the password could use.
        raise HTTPException(status.HTTP_403_FORBIDDEN, "account is inactive")

    raw_token = generate_session_token()
    db.add(
        UserSession(
            user_id=user.id,
            session_token_hash=hash_session_token(raw_token),
            expires_at=utcnow() + SESSION_TTL,
        )
    )
    db.commit()

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=raw_token,
        httponly=True,
        secure=settings.environment == "production",
        samesite="lax",
        path="/",
        max_age=int(SESSION_TTL.total_seconds()),
    )
    return user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request, response: Response, db: Session = Depends(get_db)) -> None:
    raw_token = request.cookies.get(SESSION_COOKIE_NAME)
    if raw_token:
        token_hash = hash_session_token(raw_token)
        session = db.scalars(
            select(UserSession).where(UserSession.session_token_hash == token_hash)
        ).one_or_none()
        if session is not None and session.revoked_at is None:
            session.revoked_at = utcnow()
            db.commit()

    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")


@router.get("/session", response_model=SessionUserOut)
def session_info(user: User = Depends(get_current_session_user)) -> User:
    return user


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> User:
    return user


@router.post("/keys", response_model=APIKeyCreateResponse, status_code=status.HTTP_201_CREATED)
def create_key(
    user_and_key: tuple[User, APIKey | None] = Depends(get_current_user_flexible),
    db: Session = Depends(get_db),
) -> APIKeyCreateResponse:
    user, _ = user_and_key
    raw_key = generate_api_key()
    api_key = APIKey(
        user_id=user.id,
        key_hash=hash_api_key(raw_key),
        prefix=display_prefix(raw_key),
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    return APIKeyCreateResponse(id=api_key.id, api_key=raw_key, created_at=api_key.created_at)


@router.get("/keys", response_model=list[APIKeyOut])
def list_keys(
    user_and_key: tuple[User, APIKey | None] = Depends(get_current_user_flexible),
    db: Session = Depends(get_db),
) -> list[APIKey]:
    user, _ = user_and_key
    return list(db.scalars(select(APIKey).where(APIKey.user_id == user.id).order_by(APIKey.created_at.desc())))


@router.delete("/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_key(
    key_id: int,
    user_and_key: tuple[User, APIKey | None] = Depends(get_current_user_flexible),
    db: Session = Depends(get_db),
) -> None:
    user, _ = user_and_key
    api_key = db.scalars(select(APIKey).where(APIKey.id == key_id, APIKey.user_id == user.id)).one_or_none()
    if api_key is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "API key not found")
    if api_key.revoked_at is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "API key already revoked")

    api_key.revoked_at = datetime.now(timezone.utc)
    db.commit()
