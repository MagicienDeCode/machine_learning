"""What a pull gives back, and the constellation it brings a character to (spec §2.6)."""

from dataclasses import dataclass

from app.gacha import rules


@dataclass(frozen=True)
class Refund:
    starglitter: int
    stardust: int
    constellation_after: int | None  # characters only


def refund_for(item_type: str, rarity: int, copies_before: int) -> Refund:
    """`copies_before` = copies the player already owned before this pull."""
    if item_type == "WEAPON":
        if rarity == 3:
            return Refund(0, rules.THREE_STAR_WEAPON_STARDUST, None)
        if rarity == 4:
            return Refund(rules.FOUR_STAR_WEAPON_STARGLITTER, 0, None)
        return Refund(rules.FIVE_STAR_WEAPON_STARGLITTER, 0, None)

    constellation = min(copies_before, rules.MAX_CONSTELLATION)
    if copies_before == 0:
        starglitter = 0
    elif copies_before <= rules.MAX_CONSTELLATION:
        starglitter = rules.DUPLICATE_STARGLITTER[rarity]
    else:
        starglitter = rules.MAXED_STARGLITTER[rarity]
    return Refund(starglitter, 0, constellation)
