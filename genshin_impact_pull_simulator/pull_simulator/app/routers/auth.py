from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, Response
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_session
from app.schemas.auth import AccessTokenView, LoginRequest
from app.services import auth_service

router = APIRouter(prefix="/api/auth", tags=["auth"])

COOKIE_NAME = "refresh_token"
COOKIE_PATH = "/api/auth"


def _set_cookie(response: Response, raw_token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        COOKIE_NAME,
        raw_token,
        max_age=settings.jwt_refresh_ttl_days * 24 * 3600,
        httponly=True,
        secure=settings.jwt_cookie_secure,
        samesite="lax",
        path=COOKIE_PATH,
    )


@router.post("/login")
def login(
    body: LoginRequest, response: Response, session: Annotated[Session, Depends(get_session)]
) -> AccessTokenView:
    access, raw = auth_service.login(session, body.email, body.password)
    _set_cookie(response, raw)
    return access


@router.post("/refresh")
def refresh(
    response: Response,
    session: Annotated[Session, Depends(get_session)],
    refresh_token: Annotated[str | None, Cookie()] = None,
) -> AccessTokenView:
    access, raw = auth_service.refresh(session, refresh_token)
    _set_cookie(response, raw)
    return access


@router.post("/logout", status_code=204)
def logout(
    response: Response,
    session: Annotated[Session, Depends(get_session)],
    refresh_token: Annotated[str | None, Cookie()] = None,
) -> None:
    auth_service.logout(session, refresh_token)
    response.delete_cookie(COOKIE_NAME, path=COOKIE_PATH)
