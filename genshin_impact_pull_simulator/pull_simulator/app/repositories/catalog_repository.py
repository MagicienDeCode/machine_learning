from sqlalchemy import select
from sqlalchemy.orm import Session

from app.gacha.pool import BannerPool, PoolItem
from app.models import Banner

# The catalogue is read-only at run time, so a banner's pool is built once per process.
_pool_cache: dict[int, BannerPool] = {}


def list_banners(session: Session, version: str | None = None) -> list[Banner]:
    query = select(Banner).order_by(Banner.start_date.desc(), Banner.slot)
    if version:
        query = query.where(Banner.version == version)
    return list(session.scalars(query))


def find_banner(session: Session, code: str) -> Banner | None:
    return session.scalar(select(Banner).where(Banner.code == code))


def pool_of(banner: Banner) -> BannerPool:
    pool = _pool_cache.get(banner.id)
    if pool is None:
        pool = BannerPool.from_entries(
            [
                (
                    PoolItem(e.item.id, e.item.slug, e.item.rarity, e.item.item_type),
                    e.role == "FEATURED",
                )
                for e in banner.items
            ]
        )
        _pool_cache[banner.id] = pool
    return pool


def clear_pool_cache() -> None:
    _pool_cache.clear()
