from dataclasses import dataclass


@dataclass(frozen=True)
class PoolItem:
    id: int
    slug: str
    rarity: int
    item_type: str  # CHARACTER or WEAPON


@dataclass(frozen=True)
class BannerPool:
    """What one banner can drop, split the way the draw needs it."""

    featured5: PoolItem
    featured4: tuple[PoolItem, ...]
    standard5: tuple[PoolItem, ...]
    off_banner4_characters: tuple[PoolItem, ...]
    off_banner4_weapons: tuple[PoolItem, ...]
    three_stars: tuple[PoolItem, ...]

    @classmethod
    def from_entries(cls, entries: list[tuple[PoolItem, bool]]) -> "BannerPool":
        """`entries` = (item, featured) for every item of the banner's pool."""
        featured5 = [i for i, f in entries if f and i.rarity == 5]
        featured4 = tuple(i for i, f in entries if f and i.rarity == 4)
        if len(featured5) != 1 or not featured4:
            raise ValueError("a banner needs exactly one featured 5★ and at least one featured 4★")
        standard = [i for i, f in entries if not f]
        return cls(
            featured5=featured5[0],
            featured4=featured4,
            standard5=tuple(i for i in standard if i.rarity == 5),
            off_banner4_characters=tuple(
                i for i in standard if i.rarity == 4 and i.item_type == "CHARACTER"
            ),
            off_banner4_weapons=tuple(
                i for i in standard if i.rarity == 4 and i.item_type == "WEAPON"
            ),
            three_stars=tuple(i for i in standard if i.rarity == 3),
        )
