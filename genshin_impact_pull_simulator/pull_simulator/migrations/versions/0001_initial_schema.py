"""initial schema: catalogue, accounts, player state (spec §3)

Revision ID: 0001
Revises:
Create Date: 2026-10-19 20:52:19.349695
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "banners",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("code", sa.String(length=60), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("name_zh", sa.String(length=120), nullable=True),
        sa.Column("version", sa.String(length=8), nullable=False),
        sa.Column("phase", sa.SmallInteger(), nullable=False),
        sa.Column("slot", sa.SmallInteger(), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column("banner_image_path", sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("code"),
    )
    op.create_table(
        "items",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("slug", sa.String(length=80), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("name_zh", sa.String(length=120), nullable=True),
        sa.Column("item_type", sa.Enum("CHARACTER", "WEAPON", name="item_type"), nullable=False),
        sa.Column("rarity", sa.SmallInteger(), nullable=False),
        sa.Column(
            "element",
            sa.Enum("ANEMO", "GEO", "ELECTRO", "DENDRO", "HYDRO", "PYRO", "CRYO", name="element"),
            nullable=True,
        ),
        sa.Column(
            "weapon_type",
            sa.Enum("SWORD", "CLAYMORE", "POLEARM", "BOW", "CATALYST", name="weapon_type"),
            nullable=False,
        ),
        sa.Column("release_date", sa.Date(), nullable=False),
        sa.Column("icon_path", sa.String(length=255), nullable=True),
        sa.Column("splash_path", sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug"),
    )
    op.create_table(
        "users",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=150), nullable=False),
        sa.Column("status", sa.Enum("ACTIVE", "DISABLED", name="user_status"), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_table(
        "banner_items",
        sa.Column(
            "banner_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False
        ),
        sa.Column("item_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False),
        sa.Column("role", sa.Enum("FEATURED", "STANDARD", name="banner_item_role"), nullable=False),
        sa.ForeignKeyConstraint(["banner_id"], ["banners.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["item_id"],
            ["items.id"],
        ),
        sa.PrimaryKeyConstraint("banner_id", "item_id"),
    )
    op.create_table(
        "inventory",
        sa.Column("user_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False),
        sa.Column("item_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False),
        sa.Column("copies", sa.Integer(), nullable=False),
        sa.Column("first_obtained_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("last_obtained_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["item_id"],
            ["items.id"],
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id", "item_id"),
    )
    op.create_table(
        "pity_states",
        sa.Column("user_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False),
        sa.Column("pool", sa.Enum("CHARACTER_EVENT", name="pity_pool"), nullable=False),
        sa.Column("pity5", sa.SmallInteger(), nullable=False),
        sa.Column("pity4", sa.SmallInteger(), nullable=False),
        sa.Column("guaranteed5", sa.Boolean(), nullable=False),
        sa.Column("guaranteed4", sa.Boolean(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id", "pool"),
    )
    op.create_table(
        "refresh_tokens",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("user_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False),
        sa.Column("token_hash", sa.CHAR(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_table(
        "wallet_transactions",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("user_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False),
        sa.Column(
            "type",
            sa.Enum("TOP_UP", "BUY_FATE", "WISH", "REFUND", "EXCHANGE", name="transaction_type"),
            nullable=False,
        ),
        sa.Column(
            "currency",
            sa.Enum("PRIMOGEM", "FATE", "STARGLITTER", "STARDUST", name="currency"),
            nullable=False,
        ),
        sa.Column("amount", sa.Integer(), nullable=False),
        sa.Column("balance_after", sa.BigInteger(), nullable=False),
        sa.Column("wish_id", sa.CHAR(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_wallet_transactions_user", "wallet_transactions", ["user_id", "id"], unique=False
    )
    op.create_table(
        "wallets",
        sa.Column("user_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False),
        sa.Column("primogems", sa.BigInteger(), nullable=False),
        sa.Column("intertwined_fates", sa.Integer(), nullable=False),
        sa.Column("starglitter", sa.Integer(), nullable=False),
        sa.Column("stardust", sa.Integer(), nullable=False),
        sa.Column("total_primogems_added", sa.BigInteger(), nullable=False),
        sa.Column("total_pulls", sa.Integer(), nullable=False),
        sa.Column("stardust_fates_month", sa.CHAR(length=7), nullable=True),
        sa.Column("stardust_fates_bought", sa.SmallInteger(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id"),
    )
    op.create_table(
        "wishes",
        sa.Column("id", sa.CHAR(length=36), nullable=False),
        sa.Column("user_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False),
        sa.Column(
            "banner_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False
        ),
        sa.Column("count", sa.SmallInteger(), nullable=False),
        sa.Column("fates_spent", sa.SmallInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["banner_id"],
            ["banners.id"],
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "wish_results",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("wish_id", sa.CHAR(length=36), nullable=False),
        sa.Column("user_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False),
        sa.Column(
            "banner_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False
        ),
        sa.Column("position", sa.SmallInteger(), nullable=False),
        sa.Column("item_id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), nullable=False),
        sa.Column("rarity", sa.SmallInteger(), nullable=False),
        sa.Column("pity", sa.SmallInteger(), nullable=True),
        sa.Column("featured", sa.Boolean(), nullable=False),
        sa.Column(
            "outcome",
            sa.Enum("WON_50_50", "LOST_50_50", "GUARANTEED", name="wish_outcome"),
            nullable=True,
        ),
        sa.Column("is_new", sa.Boolean(), nullable=False),
        sa.Column("constellation_after", sa.SmallInteger(), nullable=True),
        sa.Column("starglitter", sa.SmallInteger(), nullable=False),
        sa.Column("stardust", sa.SmallInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["banner_id"],
            ["banners.id"],
        ),
        sa.ForeignKeyConstraint(
            ["item_id"],
            ["items.id"],
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["wish_id"], ["wishes.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_wish_results_user", "wish_results", ["user_id", "id"], unique=False)
    op.create_index(
        "idx_wish_results_user_rarity", "wish_results", ["user_id", "rarity", "id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("idx_wish_results_user_rarity", table_name="wish_results")
    op.drop_index("idx_wish_results_user", table_name="wish_results")
    op.drop_table("wish_results")
    op.drop_table("wishes")
    op.drop_table("wallets")
    op.drop_index("idx_wallet_transactions_user", table_name="wallet_transactions")
    op.drop_table("wallet_transactions")
    op.drop_table("refresh_tokens")
    op.drop_table("pity_states")
    op.drop_table("inventory")
    op.drop_table("banner_items")
    op.drop_table("users")
    op.drop_table("items")
    op.drop_table("banners")
