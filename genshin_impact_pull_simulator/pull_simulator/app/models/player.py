"""What a player has: wallet and its history, pity, wishes and their results, inventory."""

from datetime import datetime

from sqlalchemy import (
    CHAR,
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base, BigId
from app.models.catalog import Banner, Item

TRANSACTION_TYPES = ("TOP_UP", "BUY_FATE", "WISH", "REFUND", "EXCHANGE")
CURRENCIES = ("PRIMOGEM", "FATE", "STARGLITTER", "STARDUST")
OUTCOMES = ("WON_50_50", "LOST_50_50", "GUARANTEED")


class Wallet(Base):
    """One row per user. All currencies are simulated; primogems are topped up for free."""

    __tablename__ = "wallets"

    user_id: Mapped[int] = mapped_column(
        BigId, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    primogems: Mapped[int] = mapped_column(BigInteger, default=0)
    intertwined_fates: Mapped[int] = mapped_column(Integer, default=0)
    starglitter: Mapped[int] = mapped_column(Integer, default=0)
    stardust: Mapped[int] = mapped_column(Integer, default=0)
    total_primogems_added: Mapped[int] = mapped_column(BigInteger, default=0)
    total_pulls: Mapped[int] = mapped_column(Integer, default=0)
    # Paimon's Bargains sells at most 5 fates for stardust per calendar month ("2026-10").
    stardust_fates_month: Mapped[str | None] = mapped_column(CHAR(7))
    stardust_fates_bought: Mapped[int] = mapped_column(SmallInteger, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class WalletTransaction(Base):
    """Every movement of one currency of a wallet, never changed afterwards."""

    __tablename__ = "wallet_transactions"
    __table_args__ = (Index("idx_wallet_transactions_user", "user_id", "id"),)

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigId, ForeignKey("users.id", ondelete="CASCADE"))
    type: Mapped[str] = mapped_column(Enum(*TRANSACTION_TYPES, name="transaction_type"))
    currency: Mapped[str] = mapped_column(Enum(*CURRENCIES, name="currency"))
    amount: Mapped[int] = mapped_column(Integer)
    balance_after: Mapped[int] = mapped_column(BigInteger)
    wish_id: Mapped[str | None] = mapped_column(CHAR(36))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class PityState(Base):
    """Pity of one user in one pool. Every character event banner shares CHARACTER_EVENT."""

    __tablename__ = "pity_states"

    user_id: Mapped[int] = mapped_column(
        BigId, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    pool: Mapped[str] = mapped_column(
        Enum("CHARACTER_EVENT", name="pity_pool"), primary_key=True, default="CHARACTER_EVENT"
    )
    pity5: Mapped[int] = mapped_column(SmallInteger, default=0)
    pity4: Mapped[int] = mapped_column(SmallInteger, default=0)
    guaranteed5: Mapped[bool] = mapped_column(Boolean, default=False)
    guaranteed4: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Wish(Base):
    """One single or 10 pull."""

    __tablename__ = "wishes"

    id: Mapped[str] = mapped_column(CHAR(36), primary_key=True)
    user_id: Mapped[int] = mapped_column(BigId, ForeignKey("users.id", ondelete="CASCADE"))
    banner_id: Mapped[int] = mapped_column(BigId, ForeignKey("banners.id"))
    count: Mapped[int] = mapped_column(SmallInteger)
    fates_spent: Mapped[int] = mapped_column(SmallInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class WishResult(Base):
    """One item obtained, in pull order: the wish history."""

    __tablename__ = "wish_results"
    __table_args__ = (
        Index("idx_wish_results_user", "user_id", "id"),
        Index("idx_wish_results_user_rarity", "user_id", "rarity", "id"),
    )

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    wish_id: Mapped[str] = mapped_column(CHAR(36), ForeignKey("wishes.id", ondelete="CASCADE"))
    user_id: Mapped[int] = mapped_column(BigId, ForeignKey("users.id", ondelete="CASCADE"))
    banner_id: Mapped[int] = mapped_column(BigId, ForeignKey("banners.id"))
    position: Mapped[int] = mapped_column(SmallInteger)
    item_id: Mapped[int] = mapped_column(BigId, ForeignKey("items.id"))
    rarity: Mapped[int] = mapped_column(SmallInteger)
    # 5★ pity for a 5★, 4★ pity for a 4★, NULL for a 3★.
    pity: Mapped[int | None] = mapped_column(SmallInteger)
    featured: Mapped[bool] = mapped_column(Boolean)
    outcome: Mapped[str | None] = mapped_column(Enum(*OUTCOMES, name="wish_outcome"))
    is_new: Mapped[bool] = mapped_column(Boolean)
    constellation_after: Mapped[int | None] = mapped_column(SmallInteger)
    starglitter: Mapped[int] = mapped_column(SmallInteger, default=0)
    stardust: Mapped[int] = mapped_column(SmallInteger, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    item: Mapped[Item] = relationship(lazy="joined")
    banner: Mapped[Banner] = relationship(lazy="joined")


class InventoryEntry(Base):
    __tablename__ = "inventory"

    user_id: Mapped[int] = mapped_column(
        BigId, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    item_id: Mapped[int] = mapped_column(BigId, ForeignKey("items.id"), primary_key=True)
    copies: Mapped[int] = mapped_column(Integer, default=0)
    first_obtained_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_obtained_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    item: Mapped[Item] = relationship(lazy="joined")
