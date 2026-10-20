from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_session
from app.schemas.auth import UserView
from app.services import auth_service
from app.shared.security import CurrentUserId

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("/me")
def me(user_id: CurrentUserId, session: Annotated[Session, Depends(get_session)]) -> UserView:
    return auth_service.me(session, user_id)


@router.post("/me/reset", status_code=204)
def reset(user_id: CurrentUserId, session: Annotated[Session, Depends(get_session)]) -> None:
    auth_service.reset_progress(session, user_id)
