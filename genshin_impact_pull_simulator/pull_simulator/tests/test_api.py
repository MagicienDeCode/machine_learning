import random

from sqlalchemy import select

from app.models import WalletTransaction
from app.services import wish_service


def top_up(client, auth, amount=16000):
    response = client.post("/api/wallet/top-ups", json={"amount": amount}, headers=auth)
    assert response.status_code == 200, response.text
    return response.json()


def buy(client, auth, count, pay_with="PRIMOGEM"):
    return client.post(
        "/api/wallet/fates", json={"count": count, "pay_with": pay_with}, headers=auth
    )


def wish(client, auth, count=10, banner="2.1-1-raiden-shogun", auto_buy=False):
    return client.post(
        f"/api/banners/{banner}/wishes", json={"count": count, "auto_buy": auto_buy}, headers=auth
    )


def test_login_refresh_logout(client):
    response = client.post(
        "/api/auth/login", json={"email": "traveler@test.local", "password": "123456"}
    )
    assert response.status_code == 200
    assert "refresh_token" in response.cookies
    refreshed = client.post("/api/auth/refresh")
    assert refreshed.status_code == 200 and refreshed.json()["access_token"]
    token = refreshed.json()["access_token"]
    assert (
        client.get("/api/users/me", headers={"Authorization": f"Bearer {token}"}).json()["email"]
        == "traveler@test.local"
    )
    assert client.post("/api/auth/logout").status_code == 204
    assert client.post("/api/auth/refresh").status_code == 401


def test_wrong_password_and_missing_token(client):
    response = client.post(
        "/api/auth/login", json={"email": "traveler@test.local", "password": "nope"}
    )
    assert response.status_code == 401
    assert response.json()["message"] == "Invalid email or password"
    assert client.get("/api/wallet").status_code == 401


def test_banners_are_listed_newest_first(client):
    banners = client.get("/api/banners").json()
    assert [b["code"] for b in banners] == [
        "5.3-1-mavuika",
        "4.2-1-furina",
        "3.2-1-nahida",
        "2.1-1-raiden-shogun",
        "1.1-2-zhongli",
        "1.0-1-venti",
    ]
    raiden = next(b for b in banners if b["code"] == "2.1-1-raiden-shogun")
    assert raiden["featured5"]["slug"] == "raiden-shogun"
    assert {i["slug"] for i in raiden["featured4"]} == {"kujou-sara", "xiangling", "sucrose"}


def test_banner_details_pool_follows_the_version(client):
    venti = client.get("/api/banners/1.0-1-venti/details").json()["pool"]
    slugs = {i["slug"] for i in venti["off_banner4_characters"]}
    assert "xinyan" not in slugs  # 1.1
    assert {i["slug"] for i in venti["standard5"]} == {"diluc", "jean", "keqing", "mona", "qiqi"}
    assert len(venti["three_stars"]) == 13
    assert client.get("/api/banners/nope/details").status_code == 404


def test_top_up_amounts(client, auth):
    assert top_up(client, auth, 1600)["primogems"] == 1600
    response = client.post("/api/wallet/top-ups", json={"amount": 100}, headers=auth)
    assert response.status_code == 400


def test_wish_without_fates_is_refused_and_changes_nothing(client, auth, session_factory):
    response = wish(client, auth)
    assert response.status_code == 402
    assert client.get("/api/pity", headers=auth).json()["pity5"] == 0
    with session_factory() as session:
        assert session.scalars(select(WalletTransaction)).all() == []


def test_ten_pull_writes_results_wallet_rows_and_pity(client, auth, session_factory):
    top_up(client, auth)
    assert buy(client, auth, 10).json()["intertwined_fates"] == 10
    response = wish(client, auth)
    assert response.status_code == 201, response.text
    body = response.json()
    assert len(body["results"]) == 10
    assert [r["position"] for r in body["results"]] == list(range(1, 11))
    assert body["wallet_after"]["intertwined_fates"] == 0
    assert body["max_rarity"] >= 4  # a 4★ is certain within 10 pulls
    starglitter = sum(r["starglitter"] for r in body["results"])
    stardust = sum(r["stardust"] for r in body["results"])
    assert body["wallet_after"]["starglitter"] == starglitter
    assert body["wallet_after"]["stardust"] == stardust

    history = client.get("/api/wishes/history", headers=auth).json()
    assert history["total"] == 10
    assert client.get("/api/wallet", headers=auth).json()["total_pulls"] == 10
    stats = client.get("/api/wishes/stats", headers=auth).json()
    assert stats["total_pulls"] == 10 and stats["primogems_equivalent"] == 1600


def test_auto_buy_uses_primogems(client, auth):
    top_up(client, auth, 1600)
    response = wish(client, auth, auto_buy=True)
    assert response.status_code == 201
    assert response.json()["fates_bought"] == 10
    assert response.json()["wallet_after"]["primogems"] == 0
    assert wish(client, auth, count=1, auto_buy=True).status_code == 402


def test_wish_count_must_be_one_or_ten(client, auth):
    top_up(client, auth)
    assert wish(client, auth, count=5, auto_buy=True).status_code == 400


def test_pity_is_shared_between_banners(client, auth):
    top_up(client, auth, 160000)
    wish(client, auth, banner="1.0-1-venti", auto_buy=True)
    pity_after_venti = client.get("/api/pity", headers=auth).json()
    body = wish(client, auth, count=1, banner="4.2-1-furina", auto_buy=True).json()
    if body["results"][0]["rarity"] != 5:
        assert body["pity_after"]["pity5"] == pity_after_venti["pity5"] + 1
    else:
        assert body["results"][0]["pity"] == pity_after_venti["pity5"] + 1


def test_ninety_pulls_always_give_a_five_star(client, auth):
    top_up(client, auth, 160000)
    for _ in range(9):
        assert wish(client, auth, auto_buy=True).status_code == 201
    history = client.get("/api/wishes/history?rarity=5&size=100", headers=auth).json()
    assert history["total"] >= 1
    inventory = client.get("/api/inventory?type=CHARACTER", headers=auth).json()
    assert any(e["item"]["rarity"] == 5 for e in inventory)


def test_stardust_exchange_is_limited_to_five_a_month(client, auth, session_factory):
    from app.models import Wallet

    with session_factory.begin() as session:
        session.add(
            Wallet(
                user_id=1,
                primogems=0,
                intertwined_fates=0,
                starglitter=10,
                stardust=1000,
                total_primogems_added=0,
                total_pulls=0,
                stardust_fates_bought=0,
            )
        )
    assert buy(client, auth, 4, "STARDUST").status_code == 200
    assert buy(client, auth, 2, "STARDUST").status_code == 409
    assert buy(client, auth, 1, "STARDUST").json()["stardust_fates_left_this_month"] == 0
    assert buy(client, auth, 2, "STARGLITTER").json()["intertwined_fates"] == 7
    assert buy(client, auth, 1, "STARGLITTER").status_code == 402


def test_reset_wipes_progress(client, auth):
    top_up(client, auth)
    wish(client, auth, auto_buy=True)
    assert client.post("/api/users/me/reset", headers=auth).status_code == 204
    assert client.get("/api/wallet", headers=auth).json()["primogems"] == 0
    assert client.get("/api/wishes/history", headers=auth).json()["total"] == 0
    assert client.get("/api/inventory", headers=auth).json() == []


def test_seeded_rng_makes_the_service_reproducible(session_factory):
    def run():
        with session_factory() as session:
            from app.services import wallet_service

            wallet_service.top_up(session, 1, 160000)
        with session_factory() as session:
            view = wish_service.make_wish(session, 1, "5.3-1-mavuika", 10, True, random.Random(7))
        with session_factory() as session:
            from app.services import auth_service

            auth_service.reset_progress(session, 1)
        return [r.item.slug for r in view.results]

    assert run() == run()
