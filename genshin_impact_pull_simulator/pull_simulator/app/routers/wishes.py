from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db import get_session
from app.schemas.common import Page
from app.schemas.wish import HistoryEntryView, InventoryEntryView, PityView, StatsView
from app.services import history_service, inventory_service, wish_service
from app.shared.security import CurrentUserId

router = APIRouter(prefix="/api", tags=["wishes"])
SessionDep = Annotated[Session, Depends(get_session)]


@router.get("/pity")
def pity(user_id: CurrentUserId, session: SessionDep) -> PityView:
    return wish_service.get_pity(session, user_id)


@router.get("/wishes/history")
def history(
    user_id: CurrentUserId,
    session: SessionDep,
    page: Annotated[int, Query(ge=0)] = 0,
    size: Annotated[int, Query(ge=1, le=100)] = 20,
    rarity: Annotated[int | None, Query(ge=3, le=5)] = None,
    banner: str | None = None,
) -> Page[HistoryEntryView]:
    return history_service.history(session, user_id, page, size, rarity, banner)


@router.get("/wishes/stats")
def stats(user_id: CurrentUserId, session: SessionDep) -> StatsView:
    return history_service.stats(session, user_id)


@router.get("/inventory")
def inventory(
    user_id: CurrentUserId,
    session: SessionDep,
    type: Annotated[str | None, Query(pattern="^(CHARACTER|WEAPON)$")] = None,
) -> list[InventoryEntryView]:
    return inventory_service.inventory(session, user_id, type)
