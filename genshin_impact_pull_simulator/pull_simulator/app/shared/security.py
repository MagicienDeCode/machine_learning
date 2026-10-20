import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Annotated

import bcrypt
import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings
from app.shared.errors import Unauthorized

ALGORITHM = "HS256"
_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def issue_access_token(user_id: int) -> tuple[str, datetime]:
    settings = get_settings()
    expires_at = datetime.now(UTC) + timedelta(minutes=settings.jwt_access_ttl_minutes)
    token = jwt.encode(
        {"sub": str(user_id), "exp": expires_at}, settings.jwt_secret, algorithm=ALGORITHM
    )
    return token, expires_at


def new_refresh_token() -> str:
    return secrets.token_urlsafe(48)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def current_user_id(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> int:
    """Dependency: the id of the logged-in user, from `Authorization: Bearer <token>`."""
    if credentials is None:
        raise Unauthorized("Authentication required")
    try:
        payload = jwt.decode(
            credentials.credentials, get_settings().jwt_secret, algorithms=[ALGORITHM]
        )
        return int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError) as exc:
        raise Unauthorized("Invalid or expired access token") from exc


CurrentUserId = Annotated[int, Depends(current_user_id)]
