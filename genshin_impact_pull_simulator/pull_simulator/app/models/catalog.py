"""Catalogue: seeded from data/*.json by app/tools/seed_catalog.py, read-only at run time."""

from datetime import date

from sqlalchemy import Date, Enum, ForeignKey, SmallInteger, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base, BigId

ITEM_TYPES = ("CHARACTER", "WEAPON")
ELEMENTS = ("ANEMO", "GEO", "ELECTRO", "DENDRO", "HYDRO", "PYRO", "CRYO")
WEAPON_TYPES = ("SWORD", "CLAYMORE", "POLEARM", "BOW", "CATALYST")
BANNER_ITEM_ROLES = ("FEATURED", "STANDARD")


class Item(Base):
    """A character or weapon that can drop."""

    __tablename__ = "items"

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(80), unique=True)
    name: Mapped[str] = mapped_column(String(120))
    name_zh: Mapped[str | None] = mapped_column(String(120))
    item_type: Mapped[str] = mapped_column(Enum(*ITEM_TYPES, name="item_type"))
    rarity: Mapped[int] = mapped_column(SmallInteger)
    element: Mapped[str | None] = mapped_column(Enum(*ELEMENTS, name="element"))
    weapon_type: Mapped[str] = mapped_column(Enum(*WEAPON_TYPES, name="weapon_type"))
    release_date: Mapped[date] = mapped_column(Date)
    icon_path: Mapped[str | None] = mapped_column(String(255))
    splash_path: Mapped[str | None] = mapped_column(String(255))


class Banner(Base):
    __tablename__ = "banners"

    id: Mapped[int] = mapped_column(BigId, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(60), unique=True)
    name: Mapped[str] = mapped_column(String(120))
    name_zh: Mapped[str | None] = mapped_column(String(120))
    version: Mapped[str] = mapped_column(String(8))
    phase: Mapped[int] = mapped_column(SmallInteger)
    slot: Mapped[int] = mapped_column(SmallInteger)
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    banner_image_path: Mapped[str | None] = mapped_column(String(255))

    items: Mapped[list["BannerItem"]] = relationship(
        back_populates="banner", lazy="selectin", cascade="all, delete-orphan"
    )


class BannerItem(Base):
    """One row per item a banner can drop, as listed in the banner's official item pool:
    FEATURED (the rate-up 5★ and 4★) or STANDARD (what a lost 50/50 or a 3★ draws from)."""

    __tablename__ = "banner_items"

    banner_id: Mapped[int] = mapped_column(
        BigId, ForeignKey("banners.id", ondelete="CASCADE"), primary_key=True
    )
    item_id: Mapped[int] = mapped_column(BigId, ForeignKey("items.id"), primary_key=True)
    role: Mapped[str] = mapped_column(Enum(*BANNER_ITEM_ROLES, name="banner_item_role"))

    banner: Mapped[Banner] = relationship(back_populates="items")
    item: Mapped[Item] = relationship(lazy="joined")
