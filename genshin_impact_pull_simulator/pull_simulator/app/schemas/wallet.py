from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class WalletView(BaseModel):
    primogems: int
    intertwined_fates: int
    starglitter: int
    stardust: int
    total_primogems_added: int
    total_pulls: int
    stardust_fates_left_this_month: int


class WalletBalancesView(BaseModel):
    primogems: int
    intertwined_fates: int
    starglitter: int
    stardust: int


class TopUpRequest(BaseModel):
    amount: int


class BuyFatesRequest(BaseModel):
    count: int = Field(ge=1, le=1000)
    pay_with: Literal["PRIMOGEM", "STARGLITTER", "STARDUST"] = "PRIMOGEM"


class WalletTransactionView(BaseModel):
    id: int
    type: str
    currency: str
    amount: int
    balance_after: int
    wish_id: str | None
    created_at: datetime
