"""Every figure of the Character Event Wish, in one place (spec §2)."""

# 5★: 0.6 % base, +6 % per pull from pull 74, certain on pull 90.
FIVE_STAR_BASE_RATE = 0.006
FIVE_STAR_SOFT_PITY_START = 74
FIVE_STAR_SOFT_PITY_STEP = 0.06
FIVE_STAR_HARD_PITY = 90

# 4★: 5.1 % base, +51 % per pull from pull 9, certain on pull 10.
FOUR_STAR_BASE_RATE = 0.051
FOUR_STAR_SOFT_PITY_START = 9
FOUR_STAR_SOFT_PITY_STEP = 0.51
FOUR_STAR_HARD_PITY = 10

# Chance that a 5★ / 4★ without guarantee is the featured one.
FEATURED_CHANCE = 0.5
# Chance that an off-banner 4★ is a character rather than a weapon.
OFF_BANNER_FOUR_STAR_CHARACTER_CHANCE = 0.5

WISH_COUNTS = (1, 10)
PRIMOGEMS_PER_FATE = 160
TOP_UP_AMOUNTS = (1600, 16000, 160000)

# Paimon's Bargains.
STARGLITTER_PER_FATE = 5
STARDUST_PER_FATE = 75
STARDUST_FATES_PER_MONTH = 5

MAX_CONSTELLATION = 6

# Refunds of a pull.
THREE_STAR_WEAPON_STARDUST = 15
FOUR_STAR_WEAPON_STARGLITTER = 2
FIVE_STAR_WEAPON_STARGLITTER = 10
DUPLICATE_STARGLITTER = {4: 2, 5: 10}  # C1 to C6
MAXED_STARGLITTER = {4: 5, 5: 25}  # copies beyond C6


def five_star_rate(pity5: int) -> float:
    """Chance of a 5★ on the pull that is the `pity5`-th since the last 5★ (1-based)."""
    if pity5 >= FIVE_STAR_HARD_PITY:
        return 1.0
    if pity5 < FIVE_STAR_SOFT_PITY_START:
        return FIVE_STAR_BASE_RATE
    steps = pity5 - FIVE_STAR_SOFT_PITY_START + 1
    return min(1.0, FIVE_STAR_BASE_RATE + FIVE_STAR_SOFT_PITY_STEP * steps)


def four_star_rate(pity4: int) -> float:
    """Chance of a 4★ on the pull that is the `pity4`-th since the last 4★ or better (1-based)."""
    if pity4 >= FOUR_STAR_HARD_PITY:
        return 1.0
    if pity4 < FOUR_STAR_SOFT_PITY_START:
        return FOUR_STAR_BASE_RATE
    steps = pity4 - FOUR_STAR_SOFT_PITY_START + 1
    return min(1.0, FOUR_STAR_BASE_RATE + FOUR_STAR_SOFT_PITY_STEP * steps)
