"""Downloads the character and weapon pictures into static/ (spec §4.6).

They are HoYoverse artwork, fetched from Project Amber (gi.yatta.moe, a fan database of the game
files) for local use only, and never committed (static/ is git-ignored). Files already there are
skipped, so re-running only fetches what is missing. The catalogue is then re-seeded so the API
serves the new pictures.

Banners have no artwork of their own here: the front draws them from the featured 5★ splash.
(The Fandom wiki's image CDN sits behind a bot challenge, so it is not used.)

    python -m app.tools.download_assets
"""

import json
import sys
import time
from pathlib import Path

import httpx

from app.config import get_settings

API = "https://gi.yatta.moe/api/v2/en"
ASSETS = "https://gi.yatta.moe/assets/UI"
HEADERS = {"User-Agent": "genshin-pull-simulator/0.1 (local, non-commercial fan project)"}
DELAY_SECONDS = 0.3
# Currency icons shown by the front: static/currency/<name>.png <- UI_ItemIcon_<material id>.
CURRENCY_ICONS = {
    "primogem": "UI_ItemIcon_201",
    "fate": "UI_ItemIcon_223",
    "starglitter": "UI_ItemIcon_221",
    "stardust": "UI_ItemIcon_222",
}


def asset_names(client: httpx.Client) -> dict[str, dict[str, str]]:
    """Item name -> {local file: asset name}, from Project Amber's character and weapon lists."""
    names: dict[str, dict[str, str]] = {}
    avatars = client.get(f"{API}/avatar").raise_for_status().json()["data"]["items"]
    for avatar in avatars.values():
        key = avatar["icon"].removeprefix("UI_AvatarIcon_")
        names[avatar["name"]] = {
            "icon.png": avatar["icon"],
            "splash.png": f"UI_Gacha_AvatarImg_{key}",
        }
    weapons = client.get(f"{API}/weapon").raise_for_status().json()["data"]["items"]
    for weapon in weapons.values():
        names.setdefault(weapon["name"], {"icon.png": weapon["icon"]})
    return names


def main() -> int:
    settings = get_settings()
    items = json.loads((settings.data_dir / "items.json").read_text())

    wanted: dict[Path, str] = {
        settings.static_dir / "currency" / f"{name}.png": asset
        for name, asset in CURRENCY_ICONS.items()
    }
    missing: list[str] = []
    with httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True) as client:
        names = asset_names(client)
        for item in items:
            files = names.get(item["name"])
            if files is None:
                missing.append(f"{item['name']} (not in Project Amber)")
                continue
            if item["item_type"] == "WEAPON":
                files = {"icon.png": files["icon.png"]}
            for local, asset in files.items():
                wanted[settings.static_dir / "items" / item["slug"] / local] = asset

        fetched = 0
        for path, asset in wanted.items():
            if path.is_file():
                continue
            response = client.get(f"{ASSETS}/{asset}.png")
            if response.status_code != 200:
                missing.append(
                    f"{path.relative_to(settings.static_dir)} (HTTP {response.status_code})"
                )
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            fetched += 1
            print(f"  {path.relative_to(settings.static_dir)}")
            time.sleep(DELAY_SECONDS)

    print(f"{fetched} pictures fetched.")
    if missing:
        print("Missing:")
        for line in missing:
            print(f"  {line}")

    try:
        from app.tools import seed_catalog

        seed_catalog.main()
    except Exception as exc:  # the database may simply not be up yet
        print(f"Catalogue not re-seeded ({type(exc).__name__}); run app.tools.seed_catalog later.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
