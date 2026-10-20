from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.gacha import rules
from app.models import Wallet, WalletTransaction
from app.repositories import player_repository
from app.schemas.common import Page
from app.schemas.wallet import WalletBalancesView, WalletTransactionView, WalletView
from app.shared.clock import utcnow
from app.shared.errors import BadRequest, ExchangeLimitReached, InsufficientFunds

_BALANCE_FIELD = {
    "PRIMOGEM": "primogems",
    "FATE": "intertwined_fates",
    "STARGLITTER": "starglitter",
    "STARDUST": "stardust",
}
_PRICE = {
    "PRIMOGEM": rules.PRIMOGEMS_PER_FATE,
    "STARGLITTER": rules.STARGLITTER_PER_FATE,
    "STARDUST": rules.STARDUST_PER_FATE,
}


def move(
    session: Session,
    wallet: Wallet,
    type_: str,
    currency: str,
    amount: int,
    wish_id: str | None = None,
) -> None:
    """Applies one movement to the wallet and records it. Callers check funds first."""
    field = _BALANCE_FIELD[currency]
    balance = getattr(wallet, field) + amount
    if balance < 0:
        raise InsufficientFunds(f"Not enough {currency.lower()}")
    setattr(wallet, field, balance)
    session.add(
        WalletTransaction(
            user_id=wallet.user_id,
            type=type_,
            currency=currency,
            amount=amount,
            balance_after=balance,
            wish_id=wish_id,
        )
    )


def view(wallet: Wallet) -> WalletView:
    return WalletView(
        primogems=wallet.primogems,
        intertwined_fates=wallet.intertwined_fates,
        starglitter=wallet.starglitter,
        stardust=wallet.stardust,
        total_primogems_added=wallet.total_primogems_added,
        total_pulls=wallet.total_pulls,
        stardust_fates_left_this_month=rules.STARDUST_FATES_PER_MONTH
        - _stardust_fates_bought(wallet),
    )


def balances(wallet: Wallet) -> WalletBalancesView:
    return WalletBalancesView(
        primogems=wallet.primogems,
        intertwined_fates=wallet.intertwined_fates,
        starglitter=wallet.starglitter,
        stardust=wallet.stardust,
    )


def get_wallet(session: Session, user_id: int) -> WalletView:
    with session.begin():
        return view(player_repository.wallet_for_update(session, user_id))


def top_up(session: Session, user_id: int, amount: int) -> WalletView:
    if amount not in rules.TOP_UP_AMOUNTS:
        allowed = ", ".join(str(a) for a in rules.TOP_UP_AMOUNTS)
        raise BadRequest(f"A top-up is one of {allowed} primogems")
    with session.begin():
        wallet = player_repository.wallet_for_update(session, user_id)
        move(session, wallet, "TOP_UP", "PRIMOGEM", amount)
        wallet.total_primogems_added += amount
        return view(wallet)


def buy_fates(session: Session, user_id: int, count: int, pay_with: str) -> WalletView:
    with session.begin():
        wallet = player_repository.wallet_for_update(session, user_id)
        buy_fates_in(session, wallet, count, pay_with)
        return view(wallet)


def buy_fates_in(session: Session, wallet: Wallet, count: int, pay_with: str) -> None:
    """Buys fates inside the caller's transaction (also used by a wish's auto-buy)."""
    cost = _PRICE[pay_with] * count
    if getattr(wallet, _BALANCE_FIELD[pay_with]) < cost:
        raise InsufficientFunds(
            f"{count} fate(s) cost {cost} {pay_with.lower()}, the wallet does not cover it"
        )
    if pay_with == "STARDUST":
        bought = _stardust_fates_bought(wallet)
        if bought + count > rules.STARDUST_FATES_PER_MONTH:
            left = rules.STARDUST_FATES_PER_MONTH - bought
            raise ExchangeLimitReached(
                f"Only {left} more fate(s) can be bought with stardust this month"
            )
        wallet.stardust_fates_month = _current_month()
        wallet.stardust_fates_bought = bought + count
    type_ = "BUY_FATE" if pay_with == "PRIMOGEM" else "EXCHANGE"
    move(session, wallet, type_, pay_with, -cost)
    move(session, wallet, type_, "FATE", count)


def transactions(
    session: Session, user_id: int, page: int, size: int
) -> Page[WalletTransactionView]:
    where = WalletTransaction.user_id == user_id
    total = session.scalar(select(func.count()).select_from(WalletTransaction).where(where))
    rows = session.scalars(
        select(WalletTransaction)
        .where(where)
        .order_by(WalletTransaction.id.desc())
        .offset(page * size)
        .limit(size)
    )
    items = [
        WalletTransactionView(
            id=t.id,
            type=t.type,
            currency=t.currency,
            amount=t.amount,
            balance_after=t.balance_after,
            wish_id=t.wish_id,
            created_at=t.created_at,
        )
        for t in rows
    ]
    return Page(items=items, page=page, size=size, total=total or 0)


def _current_month() -> str:
    return utcnow().strftime("%Y-%m")


def _stardust_fates_bought(wallet: Wallet) -> int:
    return wallet.stardust_fates_bought if wallet.stardust_fates_month == _current_month() else 0
