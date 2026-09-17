from decimal import Decimal

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.db_models import DetectionLog, User
from app.schemas import UsageSummaryOut
from app.session_auth import get_current_session_user, utcnow

router = APIRouter(prefix="/api/v1/usage", tags=["usage"])


@router.get("/summary", response_model=UsageSummaryOut)
def usage_summary(
    user: User = Depends(get_current_session_user),
    db: Session = Depends(get_db),
) -> UsageSummaryOut:
    month_start = utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    all_time = db.execute(
        select(func.count(DetectionLog.id), func.coalesce(func.sum(DetectionLog.cost), 0)).where(
            DetectionLog.user_id == user.id
        )
    ).one()
    this_month = db.execute(
        select(func.count(DetectionLog.id), func.coalesce(func.sum(DetectionLog.cost), 0)).where(
            DetectionLog.user_id == user.id, DetectionLog.created_at >= month_start
        )
    ).one()

    return UsageSummaryOut(
        calls_this_month=this_month[0],
        # str(...) first: SQLite's sum() returns a plain float for Numeric
        # columns (no native decimal type), and Decimal(some_float) would
        # capture that float's binary imprecision (e.g. Decimal(1.1) ->
        # Decimal('1.100000000000000088817841970012523233890533447265625')).
        cost_this_month=Decimal(str(this_month[1])),
        calls_all_time=all_time[0],
        cost_all_time=Decimal(str(all_time[1])),
    )
