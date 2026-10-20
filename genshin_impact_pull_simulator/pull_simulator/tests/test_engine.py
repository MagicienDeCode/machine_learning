import random

import pytest

from app.gacha import rules
from app.gacha.engine import Outcome, PityState, pull, pull_many
from app.gacha.pool import BannerPool, PoolItem
from app.gacha.refunds import refund_for


def make_pool() -> BannerPool:
    ids = iter(range(1, 100))

    def items(n, rarity, item_type="CHARACTER"):
        return tuple(
            PoolItem(next(ids), f"{item_type}-{rarity}-{i}", rarity, item_type) for i in range(n)
        )

    return BannerPool(
        featured5=items(1, 5)[0],
        featured4=items(3, 4),
        standard5=items(7, 5),
        off_banner4_characters=items(20, 4),
        off_banner4_weapons=items(18, 4, "WEAPON"),
        three_stars=items(13, 3, "WEAPON"),
    )


POOL = make_pool()


def test_five_star_rate_follows_soft_and_hard_pity():
    assert rules.five_star_rate(1) == pytest.approx(0.006)
    assert rules.five_star_rate(73) == pytest.approx(0.006)
    assert rules.five_star_rate(74) == pytest.approx(0.066)
    assert rules.five_star_rate(80) == pytest.approx(0.426)
    assert rules.five_star_rate(89) == pytest.approx(0.966)
    assert rules.five_star_rate(90) == 1.0


def test_four_star_rate_follows_soft_and_hard_pity():
    assert rules.four_star_rate(8) == pytest.approx(0.051)
    assert rules.four_star_rate(9) == pytest.approx(0.561)
    assert rules.four_star_rate(10) == 1.0
    assert rules.four_star_rate(11) == 1.0


class FixedRng(random.Random):
    """random() answers from a list; choice() takes the first element."""

    def __init__(self, values):
        super().__init__(0)
        self.values = list(values)

    def random(self):
        return self.values.pop(0)

    def choice(self, seq):
        return seq[0]


def test_lost_fifty_fifty_guarantees_the_next_five_star():
    # r=0 -> 5★; then 0.9 -> lost 50/50.
    result, state = pull(PityState(), POOL, FixedRng([0.0, 0.9]))
    assert result.rarity == 5 and result.outcome == Outcome.LOST_50_50 and not result.featured
    assert state.guaranteed5 and state.pity5 == 0

    result, state = pull(state, POOL, FixedRng([0.0]))
    assert result.item == POOL.featured5 and result.outcome == Outcome.GUARANTEED
    assert not state.guaranteed5


def test_won_fifty_fifty_keeps_no_guarantee():
    result, state = pull(PityState(), POOL, FixedRng([0.0, 0.1]))
    assert result.item == POOL.featured5 and result.outcome == Outcome.WON_50_50
    assert not state.guaranteed5


def test_hard_pity_gives_a_five_star_on_pull_90():
    result, state = pull(PityState(pity5=89), POOL, FixedRng([0.999999, 0.1]))
    assert result.rarity == 5 and result.pity == 90


def test_five_star_does_not_reset_four_star_pity():
    result, state = pull(PityState(pity5=89, pity4=9), POOL, FixedRng([0.5, 0.1]))
    assert result.rarity == 5
    assert state.pity4 == 10
    result, state = pull(state, POOL, FixedRng([0.999, 0.1]))
    assert result.rarity == 4 and result.pity == 11 and state.pity4 == 0


def test_lost_four_star_guarantees_a_featured_four_star():
    # r=0.01 -> 4★ (above 0.6 %, below 5.7 %); 0.9 -> lost; 0.1 -> character.
    result, state = pull(PityState(), POOL, FixedRng([0.01, 0.9, 0.1]))
    assert result.rarity == 4 and result.outcome == Outcome.LOST_50_50
    assert result.item in POOL.off_banner4_characters and state.guaranteed4
    result, state = pull(state, POOL, FixedRng([0.01]))
    assert result.item in POOL.featured4 and result.outcome == Outcome.GUARANTEED


def test_statistics_over_two_million_pulls():
    rng = random.Random(20261019)
    state = PityState()
    n = 2_000_000
    five = four = won = lost = 0
    since5 = threes_in_a_row = 0
    max5 = max_threes_in_a_row = 0
    last_five_standard = False
    featured_gap = max_featured_gap = 0
    for _ in range(n):
        result, state = pull(state, POOL, rng)
        since5 += 1
        featured_gap += 1
        if result.rarity == 5:
            five += 1
            max5 = max(max5, since5)
            since5 = 0
            if result.outcome == Outcome.WON_50_50:
                won += 1
            elif result.outcome == Outcome.LOST_50_50:
                lost += 1
            standard = not result.featured
            assert not (standard and last_five_standard), "two standard 5★ in a row"
            last_five_standard = standard
            if result.featured:
                max_featured_gap = max(max_featured_gap, featured_gap)
                featured_gap = 0
        if result.rarity == 4:
            four += 1
        if result.rarity == 3:
            threes_in_a_row += 1
            max_threes_in_a_row = max(max_threes_in_a_row, threes_in_a_row)
        else:
            threes_in_a_row = 0

    assert five / n == pytest.approx(0.016, abs=0.0005)
    assert n / five == pytest.approx(62.3, abs=1.5)
    assert four / n == pytest.approx(0.13, abs=0.003)
    assert max5 <= 90
    assert max_threes_in_a_row <= 9  # a 4★ or better within every 10 pulls
    assert won / (won + lost) == pytest.approx(0.5, abs=0.02)
    assert max_featured_gap <= 180


def test_pull_many_chains_the_state():
    results, state = pull_many(PityState(), POOL, random.Random(1), 10)
    assert len(results) == 10
    assert any(r.rarity >= 4 for r in results)


@pytest.mark.parametrize(
    ("item_type", "rarity", "copies_before", "expected"),
    [
        ("WEAPON", 3, 0, (0, 15, None)),
        ("WEAPON", 4, 5, (2, 0, None)),
        ("CHARACTER", 4, 0, (0, 0, 0)),
        ("CHARACTER", 4, 1, (2, 0, 1)),
        ("CHARACTER", 4, 6, (2, 0, 6)),
        ("CHARACTER", 4, 7, (5, 0, 6)),
        ("CHARACTER", 5, 0, (0, 0, 0)),
        ("CHARACTER", 5, 3, (10, 0, 3)),
        ("CHARACTER", 5, 9, (25, 0, 6)),
    ],
)
def test_refunds_and_constellations(item_type, rarity, copies_before, expected):
    refund = refund_for(item_type, rarity, copies_before)
    assert (refund.starglitter, refund.stardust, refund.constellation_after) == expected
