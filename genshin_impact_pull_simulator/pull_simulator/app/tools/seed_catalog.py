"""Upserts data/items.json and data/banners.json into items, banners and banner_items.

Safe to re-run: rows are matched by slug / code, so the ids that wishes and inventory point to
never change. Image paths found in static/ (see download_assets.py) are filled in too.

    python -m app.tools.seed_catalog
"""

import json
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import SessionLocal
from app.models import Banner, BannerItem, Item

POOL_KEYS = ("standard5", "off_banner4_characters", "off_banner4_weapons", "three_star_weapons")


def image_path(relative: str) -> str | None:
    return relative if (get_settings().static_dir / relative).is_file() else None


def load_catalog(session: Session) -> tuple[int, int]:
    """Loads the catalogue inside the caller's transaction; returns (items, banners) counts."""
    data_dir = get_settings().data_dir
    items_data = json.loads((data_dir / "items.json").read_text())
    banners_data = json.loads((data_dir / "banners.json").read_text())

    items = {i.slug: i for i in session.scalars(select(Item))}
    for d in items_data:
        item = items.get(d["slug"]) or Item(slug=d["slug"])
        item.name = d["name"]
        item.name_zh = d.get("name_zh")
        item.item_type = d["item_type"]
        item.rarity = d["rarity"]
        item.element = d.get("element")
        item.weapon_type = d["weapon_type"]
        item.release_date = date.fromisoformat(d["release_date"])
        item.icon_path = image_path(f"items/{d['slug']}/icon.png")
        item.splash_path = image_path(f"items/{d['slug']}/splash.png")
        session.add(item)
        items[d["slug"]] = item
    session.flush()

    banners = {b.code: b for b in session.scalars(select(Banner))}
    for d in banners_data:
        banner = banners.get(d["code"]) or Banner(code=d["code"])
        banner.name = d["name"]
        banner.name_zh = d.get("name_zh")
        banner.version = d["version"]
        banner.phase = d["phase"]
        banner.slot = d["slot"]
        banner.start_date = date.fromisoformat(d["start_date"])
        banner.end_date = date.fromisoformat(d["end_date"])
        banner.banner_image_path = image_path(f"banners/{d['code']}.png")
        session.add(banner)
        session.flush()

        roles = {items[s].id: "STANDARD" for key in POOL_KEYS for s in d[key]}
        roles.update({items[s].id: "FEATURED" for s in [d["featured5"], *d["featured4"]]})
        # Sync the pool through the relationship (delete-orphan removes what left the JSON).
        existing = {e.item_id: e for e in banner.items}
        banner.items = [
            _with_role(existing.get(item_id) or BannerItem(item_id=item_id), role)
            for item_id, role in roles.items()
        ]
    session.flush()
    return len(items_data), len(banners_data)


def _with_role(entry: BannerItem, role: str) -> BannerItem:
    entry.role = role
    return entry


def main() -> None:
    with SessionLocal.begin() as session:
        items_count, banners_count = load_catalog(session)
    print(f"{items_count} items, {banners_count} banners loaded.")


if __name__ == "__main__":
    main()
