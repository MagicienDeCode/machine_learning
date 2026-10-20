"""The draw of one pull (spec §2.1-2.4): a pure function of the pity state, the pool and the RNG."""

import random
from dataclasses import dataclass, replace
from enum import StrEnum

from app.gacha import rules
from app.gacha.pool import BannerPool, PoolItem


class Outcome(StrEnum):
    WON_50_50 = "WON_50_50"
    LOST_50_50 = "LOST_50_50"
    GUARANTEED = "GUARANTEED"


@dataclass(frozen=True)
class PityState:
    pity5: int = 0  # pulls since the last 5★
    pity4: int = 0  # pulls since the last 4★ (a 5★ does not reset it, see pull())
    guaranteed5: bool = False
    guaranteed4: bool = False


@dataclass(frozen=True)
class PullResult:
    item: PoolItem
    rarity: int
    pity: int | None  # the pity at which a 5★ / 4★ came, None for a 3★
    featured: bool
    outcome: Outcome | None


def pull(state: PityState, pool: BannerPool, rng: random.Random) -> tuple[PullResult, PityState]:
    pity5 = state.pity5 + 1
    pity4 = state.pity4 + 1
    p5 = rules.five_star_rate(pity5)
    p4 = rules.four_star_rate(pity4)
    r = rng.random()

    if r < p5:
        # A 5★ resets its own pity only: the 4★ counter keeps running, so a 5★ landing on the
        # 10th pull of the 4★ pity pushes the 4★ to the next pull, as in the game.
        if state.guaranteed5:
            item, featured, outcome, guaranteed5 = pool.featured5, True, Outcome.GUARANTEED, False
        elif rng.random() < rules.FEATURED_CHANCE:
            item, featured, outcome, guaranteed5 = pool.featured5, True, Outcome.WON_50_50, False
        else:
            item, featured = rng.choice(pool.standard5), False
            outcome, guaranteed5 = Outcome.LOST_50_50, True
        result = PullResult(item, 5, pity5, featured, outcome)
        return result, replace(state, pity5=0, pity4=pity4, guaranteed5=guaranteed5)

    if r < p5 + p4:
        if state.guaranteed4:
            item, featured, outcome = rng.choice(pool.featured4), True, Outcome.GUARANTEED
            guaranteed4 = False
        elif rng.random() < rules.FEATURED_CHANCE:
            item, featured, outcome = rng.choice(pool.featured4), True, Outcome.WON_50_50
            guaranteed4 = False
        else:
            item, featured = _off_banner_four_star(pool, rng), False
            outcome, guaranteed4 = Outcome.LOST_50_50, True
        result = PullResult(item, 4, pity4, featured, outcome)
        return result, replace(state, pity5=pity5, pity4=0, guaranteed4=guaranteed4)

    result = PullResult(rng.choice(pool.three_stars), 3, None, False, None)
    return result, replace(state, pity5=pity5, pity4=pity4)


def pull_many(
    state: PityState, pool: BannerPool, rng: random.Random, count: int
) -> tuple[list[PullResult], PityState]:
    results = []
    for _ in range(count):
        result, state = pull(state, pool, rng)
        results.append(result)
    return results, state


def _off_banner_four_star(pool: BannerPool, rng: random.Random) -> PoolItem:
    characters, weapons = pool.off_banner4_characters, pool.off_banner4_weapons
    if characters and (not weapons or rng.random() < rules.OFF_BANNER_FOUR_STAR_CHARACTER_CHANCE):
        return rng.choice(characters)
    return rng.choice(weapons)
