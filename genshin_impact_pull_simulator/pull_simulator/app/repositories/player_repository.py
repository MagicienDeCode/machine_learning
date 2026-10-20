from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import InventoryEntry, PityState, Wallet


def wallet_for_update(session: Session, user_id: int) -> Wallet:
    """The user's wallet, row-locked until the end of the transaction; created if missing."""
    wallet = session.scalar(select(Wallet).where(Wallet.user_id == user_id).with_for_update())
    if wallet is None:
        wallet = Wallet(
            user_id=user_id,
            primogems=0,
            intertwined_fates=0,
            starglitter=0,
            stardust=0,
            total_primogems_added=0,
            total_pulls=0,
            stardust_fates_bought=0,
        )
        session.add(wallet)
        session.flush()
    return wallet


def pity_for_update(session: Session, user_id: int) -> PityState:
    pity = session.scalar(
        select(PityState)
        .where(PityState.user_id == user_id, PityState.pool == "CHARACTER_EVENT")
        .with_for_update()
    )
    if pity is None:
        pity = PityState(
            user_id=user_id,
            pool="CHARACTER_EVENT",
            pity5=0,
            pity4=0,
            guaranteed5=False,
            guaranteed4=False,
        )
        session.add(pity)
        session.flush()
    return pity


def inventory_of(session: Session, user_id: int, item_ids: set[int]) -> dict[int, InventoryEntry]:
    entries = session.scalars(
        select(InventoryEntry).where(
            InventoryEntry.user_id == user_id, InventoryEntry.item_id.in_(item_ids)
        )
    )
    return {e.item_id: e for e in entries}
