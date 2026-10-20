from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_session
from app.schemas.catalog import BannerDetailsView, BannerView
from app.schemas.wish import WishRequest, WishView
from app.services import catalog_service, wish_service
from app.shared.security import CurrentUserId

router = APIRouter(prefix="/api/banners", tags=["banners"])
SessionDep = Annotated[Session, Depends(get_session)]


@router.get("")
def list_banners(session: SessionDep, version: str | None = None) -> list[BannerView]:
    return catalog_service.list_banners(session, version)


@router.get("/{code}")
def get_banner(code: str, session: SessionDep) -> BannerView:
    return catalog_service.banner_view(catalog_service.get_banner(session, code))


@router.get("/{code}/details")
def banner_details(code: str, session: SessionDep) -> BannerDetailsView:
    return catalog_service.banner_details(session, code)


@router.post("/{code}/wishes", status_code=201)
def make_wish(
    code: str, body: WishRequest, user_id: CurrentUserId, session: SessionDep
) -> WishView:
    return wish_service.make_wish(session, user_id, code, body.count, body.auto_buy)
