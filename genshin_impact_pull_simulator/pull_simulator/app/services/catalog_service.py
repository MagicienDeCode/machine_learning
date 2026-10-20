from sqlalchemy.orm import Session

from app.gacha import rules
from app.models import Banner
from app.repositories import catalog_repository
from app.schemas.catalog import (
    BannerDetailsView,
    BannerPoolView,
    BannerView,
    ItemView,
    RatesView,
    static_url,
)
from app.shared.errors import NotFound

RULES_TEXT = [
    "The base rate of a 5★ is 0.6 % (1.6 % consolidated with pity); a 5★ is certain within 90 "
    "pulls, and the rate rises from pull 74 on.",
    "The first 5★ has a 50 % chance of being the featured character. If it is not, the next 5★ "
    "is the featured character for sure.",
    "The base rate of a 4★ is 5.1 % (13 % consolidated); a 4★ or better is certain within 10 "
    "pulls.",
    "A 4★ has a 50 % chance of being one of the featured 4★ characters. If it is not, the next "
    "4★ is a featured one for sure.",
    "Pity and guarantees are shared by every Character Event Wish and carry over between banners.",
]


def get_banner(session: Session, code: str) -> Banner:
    banner = catalog_repository.find_banner(session, code)
    if banner is None:
        raise NotFound(f"Unknown banner {code}")
    return banner


def banner_view(banner: Banner) -> BannerView:
    featured = [e.item for e in banner.items if e.role == "FEATURED"]
    return BannerView(
        code=banner.code,
        name=banner.name,
        name_zh=banner.name_zh,
        version=banner.version,
        phase=banner.phase,
        slot=banner.slot,
        start_date=banner.start_date,
        end_date=banner.end_date,
        banner_image_url=static_url(banner.banner_image_path),
        featured5=ItemView.of(next(i for i in featured if i.rarity == 5)),
        featured4=[ItemView.of(i) for i in sorted(featured, key=lambda i: i.name) if i.rarity == 4],
    )


def list_banners(session: Session, version: str | None) -> list[BannerView]:
    return [banner_view(b) for b in catalog_repository.list_banners(session, version)]


def banner_details(session: Session, code: str) -> BannerDetailsView:
    banner = get_banner(session, code)
    items = {e.item.id: e.item for e in banner.items}
    pool = catalog_repository.pool_of(banner)

    def views(pool_items) -> list[ItemView]:
        return sorted((ItemView.of(items[p.id]) for p in pool_items), key=lambda v: v.name)

    return BannerDetailsView(
        banner=banner_view(banner),
        rates=RatesView(
            five_star_base=rules.FIVE_STAR_BASE_RATE,
            five_star_consolidated=0.016,
            five_star_soft_pity_start=rules.FIVE_STAR_SOFT_PITY_START,
            five_star_hard_pity=rules.FIVE_STAR_HARD_PITY,
            four_star_base=rules.FOUR_STAR_BASE_RATE,
            four_star_consolidated=0.13,
            four_star_hard_pity=rules.FOUR_STAR_HARD_PITY,
            featured_chance=rules.FEATURED_CHANCE,
        ),
        rules=RULES_TEXT,
        pool=BannerPoolView(
            featured5=ItemView.of(items[pool.featured5.id]),
            featured4=views(pool.featured4),
            standard5=views(pool.standard5),
            off_banner4_characters=views(pool.off_banner4_characters),
            off_banner4_weapons=views(pool.off_banner4_weapons),
            three_stars=views(pool.three_stars),
        ),
    )
