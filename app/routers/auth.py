from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import display_prefix, generate_api_key, get_current_user, hash_api_key
from app.db import get_db
from app.db_models import APIKey, User
from app.schemas import (
    APIKeyCreateResponse,
    APIKeyOut,
    RegisterRequest,
    RegisterResponse,
    UserOut,
)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)) -> RegisterResponse:
    user = User(email=body.email)
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


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> User:
    return user


@router.post("/keys", response_model=APIKeyCreateResponse, status_code=status.HTTP_201_CREATED)
def create_key(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> APIKeyCreateResponse:
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
def list_keys(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[APIKey]:
    return list(db.scalars(select(APIKey).where(APIKey.user_id == user.id).order_by(APIKey.created_at.desc())))


@router.delete("/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_key(key_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> None:
    api_key = db.scalars(select(APIKey).where(APIKey.id == key_id, APIKey.user_id == user.id)).one_or_none()
    if api_key is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "API key not found")
    if api_key.revoked_at is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "API key already revoked")

    api_key.revoked_at = datetime.now(timezone.utc)
    db.commit()
