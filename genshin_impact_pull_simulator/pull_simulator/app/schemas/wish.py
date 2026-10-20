from datetime import datetime

from pydantic import BaseModel

from app.schemas.catalog import ItemView
from app.schemas.wallet import WalletBalancesView


class WishRequest(BaseModel):
    count: int
    auto_buy: bool = False


class PityView(BaseModel):
    pool: str = "CHARACTER_EVENT"
    pity5: int
    pity4: int
    guaranteed5: bool
    guaranteed4: bool
    next_5star_rate: float
    pulls_to_hard_pity: int
    pulls_to_soft_pity: int


class WishResultView(BaseModel):
    position: int
    item: ItemView
    rarity: int
    pity: int | None
    featured: bool
    outcome: str | None
    is_new: bool
    constellation_after: int | None
    starglitter: int
    stardust: int


class WishView(BaseModel):
    wish_id: str
    banner_code: str
    count: int
    max_rarity: int
    fates_bought: int
    results: list[WishResultView]
    pity_after: PityView
    wallet_after: WalletBalancesView


class HistoryEntryView(BaseModel):
    id: int
    wish_id: str
    banner_code: str
    banner_name: str
    banner_name_zh: str | None
    item: ItemView
    rarity: int
    pity: int | None
    featured: bool
    outcome: str | None
    created_at: datetime


class FiftyFiftyView(BaseModel):
    won: int
    lost: int
    guaranteed: int


class StatsView(BaseModel):
    total_pulls: int
    fates_spent: int
    primogems_equivalent: int
    five_star_count: int
    four_star_count: int
    average_five_star_pity: float | None
    average_four_star_pity: float | None
    fifty_fifty: FiftyFiftyView
    featured_five_star_count: int
    pulls_per_featured_five_star: float | None
    luckiest_five_star_pity: int | None
    unluckiest_five_star_pity: int | None


class InventoryEntryView(BaseModel):
    item: ItemView
    copies: int
    constellation: int | None
    first_obtained_at: datetime
    last_obtained_at: datetime
