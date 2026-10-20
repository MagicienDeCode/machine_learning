from sqlalchemy import select
from sqlalchemy.orm import Session

from app.gacha import rules
from app.models import InventoryEntry, Item
from app.schemas.catalog import ItemView
from app.schemas.wish import InventoryEntryView


def inventory(session: Session, user_id: int, item_type: str | None) -> list[InventoryEntryView]:
    query = (
        select(InventoryEntry)
        .join(Item, Item.id == InventoryEntry.item_id)
        .where(InventoryEntry.user_id == user_id)
        .order_by(Item.rarity.desc(), Item.name)
    )
    if item_type:
        query = query.where(Item.item_type == item_type)
    return [
        InventoryEntryView(
            item=ItemView.of(e.item),
            copies=e.copies,
            constellation=(
                min(e.copies - 1, rules.MAX_CONSTELLATION)
                if e.item.item_type == "CHARACTER"
                else None
            ),
            first_obtained_at=e.first_obtained_at,
            last_obtained_at=e.last_obtained_at,
        )
        for e in session.scalars(query).unique()
    ]
