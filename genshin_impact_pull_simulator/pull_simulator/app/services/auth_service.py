from datetime import timedelta

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import (
    InventoryEntry,
    PityState,
    RefreshToken,
    User,
    Wallet,
    WalletTransaction,
    Wish,
    WishResult,
)
from app.schemas.auth import AccessTokenView, UserView
from app.shared import security
from app.shared.clock import utcnow
from app.shared.errors import NotFound, Unauthorized


def login(session: Session, email: str, password: str) -> tuple[AccessTokenView, str]:
    """Returns the access token and the raw refresh token (for the cookie)."""
    with session.begin():
        user = session.scalar(select(User).where(User.email == email))
        if user is None or not security.verify_password(password, user.password_hash):
            raise Unauthorized("Invalid email or password")
        if user.status != "ACTIVE":
            raise Unauthorized("Account is not active")
        return _issue(session, user.id)


def refresh(session: Session, raw_token: str | None) -> tuple[AccessTokenView, str]:
    """Rotates the refresh token: the one presented is revoked, a new one is issued."""
    if not raw_token:
        raise Unauthorized("No refresh token")
    with session.begin():
        token = session.scalar(
            select(RefreshToken)
            .where(RefreshToken.token_hash == security.hash_refresh_token(raw_token))
            .with_for_update()
        )
        if token is None or token.revoked_at is not None or token.expires_at < utcnow():
            raise Unauthorized("Invalid or expired refresh token")
        token.revoked_at = utcnow()
        return _issue(session, token.user_id)


def logout(session: Session, raw_token: str | None) -> None:
    if not raw_token:
        return
    with session.begin():
        session.execute(
            update(RefreshToken)
            .where(
                RefreshToken.token_hash == security.hash_refresh_token(raw_token),
                RefreshToken.revoked_at.is_(None),
            )
            .values(revoked_at=utcnow())
        )


def me(session: Session, user_id: int) -> UserView:
    user = session.get(User, user_id)
    if user is None:
        raise NotFound("User not found")
    return UserView(id=user.id, email=user.email, display_name=user.display_name)


def reset_progress(session: Session, user_id: int) -> None:
    """Start over: wallet, pity, history and inventory of the user are wiped."""
    with session.begin():
        for model in (WishResult, Wish, InventoryEntry, WalletTransaction, PityState, Wallet):
            session.execute(delete(model).where(model.user_id == user_id))


def _issue(session: Session, user_id: int) -> tuple[AccessTokenView, str]:
    access_token, expires_at = security.issue_access_token(user_id)
    raw = security.new_refresh_token()
    session.add(
        RefreshToken(
            user_id=user_id,
            token_hash=security.hash_refresh_token(raw),
            expires_at=utcnow() + timedelta(days=get_settings().jwt_refresh_ttl_days),
        )
    )
    return AccessTokenView(access_token=access_token, access_token_expires_at=expires_at), raw
