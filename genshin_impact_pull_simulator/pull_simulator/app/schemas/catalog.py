from datetime import date

from pydantic import BaseModel

from app.models import Item


def static_url(path: str | None) -> str | None:
    return f"/static/{path}" if path else None


class ItemView(BaseModel):
    slug: str
    name: str
    name_zh: str | None
    rarity: int
    item_type: str
    element: str | None
    weapon_type: str
    icon_url: str | None
    splash_url: str | None

    @classmethod
    def of(cls, item: Item) -> "ItemView":
        return cls(
            slug=item.slug,
            name=item.name,
            name_zh=item.name_zh,
            rarity=item.rarity,
            item_type=item.item_type,
            element=item.element,
            weapon_type=item.weapon_type,
            icon_url=static_url(item.icon_path),
            splash_url=static_url(item.splash_path),
        )


class BannerView(BaseModel):
    code: str
    name: str
    name_zh: str | None
    version: str
    phase: int
    slot: int
    start_date: date
    end_date: date
    banner_image_url: str | None
    featured5: ItemView
    featured4: list[ItemView]


class RatesView(BaseModel):
    five_star_base: float
    five_star_consolidated: float
    five_star_soft_pity_start: int
    five_star_hard_pity: int
    four_star_base: float
    four_star_consolidated: float
    four_star_hard_pity: int
    featured_chance: float


class BannerPoolView(BaseModel):
    featured5: ItemView
    featured4: list[ItemView]
    standard5: list[ItemView]
    off_banner4_characters: list[ItemView]
    off_banner4_weapons: list[ItemView]
    three_stars: list[ItemView]


class BannerDetailsView(BaseModel):
    banner: BannerView
    rates: RatesView
    rules: list[str]
    pool: BannerPoolView
