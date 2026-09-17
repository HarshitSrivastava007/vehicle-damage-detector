from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import display_prefix, generate_api_key, hash_api_key
from app.db import get_db
from app.db_models import APIKey, DetectionLog, User
from app.schemas import (
    AdminCreateUserRequest,
    AdminCreateUserResponse,
    AdminUserOut,
    UpdateCostRequest,
    UpdateUserStatusRequest,
)
from app.session_auth import hash_password, require_admin

router = APIRouter(prefix="/api/v1/admin", tags=["admin"], dependencies=[Depends(require_admin)])


def _to_admin_user_out(user: User, total_detections: int) -> AdminUserOut:
    return AdminUserOut(
        id=user.id,
        email=user.email,
        is_admin=user.is_admin,
        is_active=user.is_active,
        cost_per_call=user.cost_per_call,
        created_at=user.created_at,
        total_detections=total_detections,
    )


def _count_detections(db: Session, user_id: int) -> int:
    return db.scalar(select(func.count(DetectionLog.id)).where(DetectionLog.user_id == user_id)) or 0


@router.get("/users", response_model=list[AdminUserOut])
def list_users(db: Session = Depends(get_db)) -> list[AdminUserOut]:
    stmt = (
        select(User, func.count(DetectionLog.id))
        .outerjoin(DetectionLog, DetectionLog.user_id == User.id)
        .group_by(User.id)
        .order_by(User.created_at)
    )
    return [_to_admin_user_out(user, total_detections) for user, total_detections in db.execute(stmt).all()]


@router.post("/users", response_model=AdminCreateUserResponse, status_code=status.HTTP_201_CREATED)
def create_user(body: AdminCreateUserRequest, db: Session = Depends(get_db)) -> AdminCreateUserResponse:
    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        is_admin=body.is_admin,
        cost_per_call=body.cost_per_call,
    )
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

    return AdminCreateUserResponse(
        **_to_admin_user_out(user, total_detections=0).model_dump(),
        api_key=raw_key,
    )


@router.patch("/users/{user_id}/cost", response_model=AdminUserOut)
def update_user_cost(user_id: int, body: UpdateCostRequest, db: Session = Depends(get_db)) -> AdminUserOut:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")

    user.cost_per_call = body.cost_per_call
    db.commit()

    return _to_admin_user_out(user, _count_detections(db, user.id))


@router.patch("/users/{user_id}/status", response_model=AdminUserOut)
def update_user_status(
    user_id: int,
    body: UpdateUserStatusRequest,
    db: Session = Depends(get_db),
    acting_admin: User = Depends(require_admin),
) -> AdminUserOut:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")
    if user.id == acting_admin.id and not body.is_active:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "cannot deactivate your own account")

    user.is_active = body.is_active
    db.commit()

    return _to_admin_user_out(user, _count_detections(db, user.id))
