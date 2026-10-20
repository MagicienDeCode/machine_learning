"""One wish, start to end, in one transaction (spec §4.4)."""

import random
import secrets
import uuid

from sqlalchemy.orm import Session

from app.gacha import engine, rules
from app.gacha.refunds import refund_for
from app.models import InventoryEntry, PityState, Wish, WishResult
from app.repositories import catalog_repository, player_repository
from app.schemas.catalog import ItemView
from app.schemas.wish import PityView, WishResultView, WishView
from app.services import catalog_service, wallet_service
from app.shared.clock import utcnow
from app.shared.errors import BadRequest, InsufficientFunds

# OS entropy, the counterpart of anti-gamble's SecureRandom. Tests pass a seeded random.Random.
system_rng: random.Random = secrets.SystemRandom()


def make_wish(
    session: Session,
    user_id: int,
    banner_code: str,
    count: int,
    auto_buy: bool,
    rng: random.Random = system_rng,
) -> WishView:
    if count not in rules.WISH_COUNTS:
        raise BadRequest("A wish is 1 or 10 pulls")

    with session.begin():
        banner = catalog_service.get_banner(session, banner_code)
        pool = catalog_repository.pool_of(banner)
        items = {e.item.id: e.item for e in banner.items}

        # Locked until commit: two wishes at once can neither spend the same fates nor read
        # the same pity.
        wallet = player_repository.wallet_for_update(session, user_id)
        pity = player_repository.pity_for_update(session, user_id)

        missing = count - wallet.intertwined_fates
        fates_bought = 0
        if missing > 0:
            if not auto_buy:
                raise InsufficientFunds("Not enough Intertwined Fates")
            if wallet.primogems < missing * rules.PRIMOGEMS_PER_FATE:
                raise InsufficientFunds("Not enough Intertwined Fates or primogems")
            wallet_service.buy_fates_in(session, wallet, missing, "PRIMOGEM")
            fates_bought = missing

        wish = Wish(
            id=str(uuid.uuid4()),
            user_id=user_id,
            banner_id=banner.id,
            count=count,
            fates_spent=count,
        )
        session.add(wish)
        wallet_service.move(session, wallet, "WISH", "FATE", -count, wish.id)

        results, new_state = engine.pull_many(_state_of(pity), pool, rng, count)

        inventory = player_repository.inventory_of(session, user_id, {r.item.id for r in results})
        now = utcnow()
        views: list[WishResultView] = []
        starglitter = stardust = 0
        for position, result in enumerate(results, start=1):
            entry = inventory.get(result.item.id)
            if entry is None:
                entry = InventoryEntry(
                    user_id=user_id,
                    item_id=result.item.id,
                    copies=0,
                    first_obtained_at=now,
                    last_obtained_at=now,
                )
                session.add(entry)
                inventory[result.item.id] = entry
            refund = refund_for(result.item.item_type, result.rarity, entry.copies)
            is_new = entry.copies == 0
            entry.copies += 1
            entry.last_obtained_at = now
            starglitter += refund.starglitter
            stardust += refund.stardust

            outcome = result.outcome.value if result.outcome else None
            session.add(
                WishResult(
                    wish_id=wish.id,
                    user_id=user_id,
                    banner_id=banner.id,
                    position=position,
                    item_id=result.item.id,
                    rarity=result.rarity,
                    pity=result.pity,
                    featured=result.featured,
                    outcome=outcome,
                    is_new=is_new,
                    constellation_after=refund.constellation_after,
                    starglitter=refund.starglitter,
                    stardust=refund.stardust,
                    created_at=now,
                )
            )
            views.append(
                WishResultView(
                    position=position,
                    item=ItemView.of(items[result.item.id]),
                    rarity=result.rarity,
                    pity=result.pity,
                    featured=result.featured,
                    outcome=outcome,
                    is_new=is_new,
                    constellation_after=refund.constellation_after,
                    starglitter=refund.starglitter,
                    stardust=refund.stardust,
                )
            )

        if starglitter:
            wallet_service.move(session, wallet, "REFUND", "STARGLITTER", starglitter, wish.id)
        if stardust:
            wallet_service.move(session, wallet, "REFUND", "STARDUST", stardust, wish.id)
        wallet.total_pulls += count
        _save_state(pity, new_state)

        return WishView(
            wish_id=wish.id,
            banner_code=banner.code,
            count=count,
            max_rarity=max(r.rarity for r in results),
            fates_bought=fates_bought,
            results=views,
            pity_after=pity_view(pity),
            wallet_after=wallet_service.balances(wallet),
        )


def pity_view(pity: PityState) -> PityView:
    next_pull = pity.pity5 + 1
    return PityView(
        pity5=pity.pity5,
        pity4=pity.pity4,
        guaranteed5=pity.guaranteed5,
        guaranteed4=pity.guaranteed4,
        next_5star_rate=round(rules.five_star_rate(next_pull), 6),
        pulls_to_hard_pity=rules.FIVE_STAR_HARD_PITY - pity.pity5,
        pulls_to_soft_pity=max(0, rules.FIVE_STAR_SOFT_PITY_START - next_pull),
    )


def get_pity(session: Session, user_id: int) -> PityView:
    with session.begin():
        return pity_view(player_repository.pity_for_update(session, user_id))


def _state_of(pity: PityState) -> engine.PityState:
    return engine.PityState(pity.pity5, pity.pity4, pity.guaranteed5, pity.guaranteed4)


def _save_state(pity: PityState, state: engine.PityState) -> None:
    pity.pity5 = state.pity5
    pity.pity4 = state.pity4
    pity.guaranteed5 = state.guaranteed5
    pity.guaranteed4 = state.guaranteed4
