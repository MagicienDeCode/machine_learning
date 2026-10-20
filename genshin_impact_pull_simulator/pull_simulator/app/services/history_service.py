from sqlalchemy import Select, and_, case, func, select
from sqlalchemy.orm import Session

from app.gacha import rules
from app.models import Banner, Wish, WishResult
from app.schemas.catalog import ItemView
from app.schemas.common import Page
from app.schemas.wish import FiftyFiftyView, HistoryEntryView, StatsView


def history(
    session: Session,
    user_id: int,
    page: int,
    size: int,
    rarity: int | None,
    banner_code: str | None,
) -> Page[HistoryEntryView]:
    def filtered(query: Select) -> Select:
        query = query.where(WishResult.user_id == user_id)
        if rarity is not None:
            query = query.where(WishResult.rarity == rarity)
        if banner_code:
            query = query.join(Banner, Banner.id == WishResult.banner_id).where(
                Banner.code == banner_code
            )
        return query

    total = session.scalar(filtered(select(func.count()).select_from(WishResult)))
    rows = session.scalars(
        filtered(select(WishResult)).order_by(WishResult.id.desc()).offset(page * size).limit(size)
    ).unique()
    items = [
        HistoryEntryView(
            id=r.id,
            wish_id=r.wish_id,
            banner_code=r.banner.code,
            banner_name=r.banner.name,
            banner_name_zh=r.banner.name_zh,
            item=ItemView.of(r.item),
            rarity=r.rarity,
            pity=r.pity,
            featured=r.featured,
            outcome=r.outcome,
            created_at=r.created_at,
        )
        for r in rows
    ]
    return Page(items=items, page=page, size=size, total=total or 0)


def stats(session: Session, user_id: int) -> StatsView:
    mine = WishResult.user_id == user_id
    five = WishResult.rarity == 5
    row = session.execute(
        select(
            func.count(),
            func.sum(case((five, 1), else_=0)),
            func.sum(case((WishResult.rarity == 4, 1), else_=0)),
            func.avg(case((five, WishResult.pity))),
            func.avg(case((WishResult.rarity == 4, WishResult.pity))),
            func.min(case((five, WishResult.pity))),
            func.max(case((five, WishResult.pity))),
            func.sum(case((and_(five, WishResult.featured.is_(True)), 1), else_=0)),
            func.sum(case((and_(five, WishResult.outcome == "WON_50_50"), 1), else_=0)),
            func.sum(case((and_(five, WishResult.outcome == "LOST_50_50"), 1), else_=0)),
            func.sum(case((and_(five, WishResult.outcome == "GUARANTEED"), 1), else_=0)),
        ).where(mine)
    ).one()
    total, n5, n4, avg5, avg4, min5, max5, featured5, won, lost, guaranteed = row
    fates = session.scalar(select(func.sum(Wish.fates_spent)).where(Wish.user_id == user_id)) or 0
    featured5 = int(featured5 or 0)
    return StatsView(
        total_pulls=total,
        fates_spent=int(fates),
        primogems_equivalent=int(fates) * rules.PRIMOGEMS_PER_FATE,
        five_star_count=int(n5 or 0),
        four_star_count=int(n4 or 0),
        average_five_star_pity=round(float(avg5), 1) if avg5 is not None else None,
        average_four_star_pity=round(float(avg4), 1) if avg4 is not None else None,
        fifty_fifty=FiftyFiftyView(
            won=int(won or 0), lost=int(lost or 0), guaranteed=int(guaranteed or 0)
        ),
        featured_five_star_count=featured5,
        pulls_per_featured_five_star=round(total / featured5, 1) if featured5 else None,
        luckiest_five_star_pity=min5,
        unluckiest_five_star_pity=max5,
    )
