from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db import get_session
from app.schemas.common import Page
from app.schemas.wallet import BuyFatesRequest, TopUpRequest, WalletTransactionView, WalletView
from app.services import wallet_service
from app.shared.security import CurrentUserId

router = APIRouter(prefix="/api/wallet", tags=["wallet"])
SessionDep = Annotated[Session, Depends(get_session)]


@router.get("")
def get_wallet(user_id: CurrentUserId, session: SessionDep) -> WalletView:
    return wallet_service.get_wallet(session, user_id)


@router.post("/top-ups")
def top_up(body: TopUpRequest, user_id: CurrentUserId, session: SessionDep) -> WalletView:
    return wallet_service.top_up(session, user_id, body.amount)


@router.post("/fates")
def buy_fates(body: BuyFatesRequest, user_id: CurrentUserId, session: SessionDep) -> WalletView:
    return wallet_service.buy_fates(session, user_id, body.count, body.pay_with)


@router.get("/transactions")
def transactions(
    user_id: CurrentUserId,
    session: SessionDep,
    page: Annotated[int, Query(ge=0)] = 0,
    size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> Page[WalletTransactionView]:
    return wallet_service.transactions(session, user_id, page, size)
