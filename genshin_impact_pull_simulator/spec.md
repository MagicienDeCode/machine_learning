# Genshin impact pull simulator

A local, for-fun simulator of Genshin Impact's **Character Event Wish**: the player tops up
simulated primogems, converts them to Intertwined Fates and pulls on a character banner. The
backend applies the real rates, pity and 50/50 rules and keeps every pull.
Primogems are free: the player tops up as much as they like, as with anti-gamble's balance. No
real money is involved.

Laid out like `anti-gamble` / `anti-gamble-front`: a backend that owns every random draw and
all state, and a thin front that only shows results.

```
genshin_impact_pull_simulator/
  spec.md
  pull_simulator/         Python 3.12+, FastAPI, SQLAlchemy 2, Alembic, MySQL 8
  pull_simulator_front/   Vue 3 + Vite + TypeScript + ant-design-vue (same stack as anti-gamble-front)
```

## 1. Scope

In scope:

- Character Event Wish banners only (both "Character Event Wish" and "Character Event Wish-2"
  of a phase, sharing one pity, as in the game).
- Single pull and 10 pull.
- Pity for 5★ and 4★, soft pity, 50/50 and guarantee for 5★ and 4★.
- Pool as on a real character banner: featured 5★, 3 featured 4★ characters, the standard 5★
  characters, the 4★ characters and 4★ weapons, the 3★ weapons. Weapons appear because the
  character banner drops them; there is no weapon banner.
- Wallet: primogems, Intertwined Fates, Masterless Starglitter and Stardust. Free top-up of
  primogems, as anti-gamble's top-up.
- Duplicates: constellations C0–C6 and starglitter/stardust refunds; exchange of
  starglitter/stardust for fates (Paimon's Bargains).
- Wish history, pity status, statistics, owned characters.
- Login like anti-gamble (several local accounts, each with its own wallet and pity).

Out of scope: Capturing Radiance (the 5.0+ rescue of a lost 50/50), real money / Genesis
Crystal packs, Standard Wish, Epitomized Path weapon banner, Chronicled Wish, Beginners' Wish,
Welkin / Battle Pass, wish animation (meteor colours are enough), weapon refinement display.

## 2. Wish rules

All figures in one module, `app/gacha/rules.py`, so they can be tuned in one place.

### 2.1 Rarity draw

`pity5` = pulls since the last 5★, counting the current one (1..90).
`pity4` = pulls since the last 4★, counting the current one (1..10; 11 or more only right after a
5★ took the 10th pull, see below).

| Rarity | Base rate | Soft pity                              | Hard pity |
|--------|-----------|----------------------------------------|-----------|
| 5★     | 0.6 %     | from pull 74: `0.6 % + 6 % × (pity5 − 73)` | 90        |
| 4★     | 5.1 %     | from pull 9: `5.1 % + 51 % × (pity4 − 8)`  | 10        |
| 3★     | the rest  |                                        |           |

One uniform `r ∈ [0, 1)` per pull: `r < p5` → 5★, else `r < p5 + p4` → 4★, else 3★.
`p5 + p4` is capped at 1. These give the official consolidated rates (5★ 1.6 %, 4★ 13 %),
which tests check (§7).

When a 5★ comes out, `pity5` resets and `pity4` keeps counting — a 5★ on pull 10 of the 4★
pity pushes the 4★ to the next pull, as in the game. When a 4★ comes out, `pity4` resets.

### 2.2 5★ featured or not

| State before the 5★                   | Result                                                       |
|---------------------------------------|--------------------------------------------------------------|
| `guaranteed5 = true`                  | Featured 5★ (`GUARANTEED`), `guaranteed5 = false`            |
| `guaranteed5 = false`                 | 50/50: featured (`WON_50_50`) or a standard 5★ (`LOST_50_50`, then `guaranteed5 = true`) |

So the first 5★ is a 50/50; if it is lost, the next 5★ is the featured character for sure, then
it starts over with a 50/50. Over many pulls half of the 50/50s are won, and a featured 5★ takes
at most 180 pulls.

Standard 5★ characters lost to: those of the banner's official item pool (Diluc, Jean, Keqing,
Mona, Qiqi from 1.0, then Tighnari, Dehya… from the version they joined), stored per banner in
`banner_items` (§3.1).

### 2.3 4★ featured or not

- `guaranteed4 = true` → one of the 3 featured 4★ characters, uniform; `guaranteed4 = false`.
- Otherwise 50 %: featured (uniform among the 3); 50 %: off-banner, then `guaranteed4 = true`.
- Off-banner 4★: 50 % a 4★ character, 50 % a 4★ weapon, uniform inside each group. Both lists
  are the banner's official item pool, copied from the banner's wiki page: a new 4★ only joins
  it the version after its debut (Kujou Sara is not off-banner on her own 2.1 banner). Weapons
  are the 18 standard 4★ weapons (Favonius / Sacrificial series, The Flute, Rust, …).

### 2.4 3★

Uniform among the 13 3★ weapons of the banner's pool.

### 2.5 Pity sharing

One pity state per user for the whole `CHARACTER_EVENT` pool: it carries over between the two
banners of a phase and from one banner to the next, as in the game.

### 2.6 Currency and refunds

| Thing                                 | Value                                                     |
|---------------------------------------|-----------------------------------------------------------|
| 1 Intertwined Fate                    | 160 primogems                                             |
| 1 pull / 10 pull                      | 1 / 10 Intertwined Fates                                  |
| 3★ weapon                             | +15 Stardust                                              |
| 4★ weapon                             | +2 Starglitter                                            |
| 4★ character, new                     | nothing                                                   |
| 4★ character, duplicate C1–C6 / after C6 | +2 / +5 Starglitter                                   |
| 5★ character, new                     | nothing                                                   |
| 5★ character, duplicate C1–C6 / after C6 | +10 / +25 Starglitter                                 |
| Paimon's Bargains                     | 5 Starglitter = 1 fate; 75 Stardust = 1 fate (max 5 per calendar month) |

Constellation of a character = `min(copies − 1, 6)`.

Top-up: free, `{amount}` = 1600, 16000 or 160000 primogems (10, 100 or 1000 pulls), like
anti-gamble's 100 / 1000 € top-ups. Any other amount → 400.

### 2.7 Randomness

`secrets.SystemRandom()` (OS entropy, the Python counterpart of anti-gamble's `SecureRandom`),
injected as a dependency so tests can pass a seeded `random.Random`.

## 3. Database design

MySQL 8 in Docker on `localhost:3308` (3307 is anti-gamble's), database `genshinwish`. Tables are
created by the Alembic migration on `alembic upgrade head`. Tests run on in-memory SQLite, the
models use only portable types. The server runs with `utf8mb4`.

The same schema and data, written by hand as step-by-step SQL for teaching, is in
`database_design/` (its own `genshinwish_lesson` database).

```
users ──< refresh_tokens
  │
  ├──1 wallets ──< wallet_transactions
  ├──1 pity_states
  ├──< wishes ──< wish_results >── items
  └──< inventory >── items
banners ──< banner_items >── items
```

### 3.1 Catalogue (seeded from `pull_simulator/data/*.json`, read-only at run time)

**items** — every character and weapon that can drop.

| Column                 | Type                                   | Notes                              |
|------------------------|----------------------------------------|------------------------------------|
| id                     | BIGINT PK AI                           |                                    |
| slug                   | VARCHAR(80) UNIQUE                     | `raiden-shogun`, `favonius-sword`  |
| name                   | VARCHAR(120)                           | `Raiden Shogun`                    |
| name_zh                | VARCHAR(120) NULL                      | `雷电将军`, shown when the UI is in Chinese |
| item_type              | ENUM('CHARACTER','WEAPON')             |                                    |
| rarity                 | SMALLINT                               | 3, 4, 5                            |
| element                | ENUM('ANEMO','GEO','ELECTRO','DENDRO','HYDRO','PYRO','CRYO') NULL | characters only |
| weapon_type            | ENUM('SWORD','CLAYMORE','POLEARM','BOW','CATALYST') | character's weapon or the weapon's type |
| release_date           | DATE                                   | for information                    |
| icon_path              | VARCHAR(255) NULL                      | relative to `/static`, e.g. `items/raiden-shogun/icon.png` |
| splash_path            | VARCHAR(255) NULL                      | wish splash art                    |

**banners**

| Column          | Type                    | Notes                                         |
|-----------------|-------------------------|-----------------------------------------------|
| id              | BIGINT PK AI            |                                               |
| code            | VARCHAR(60) UNIQUE      | `2.1-1-raiden-shogun` (version-phase-featured) |
| name            | VARCHAR(120)            | `Reign of Serenity`                           |
| name_zh         | VARCHAR(120) NULL       | `影寂天下人`                                  |
| version         | VARCHAR(8)              | `2.1`                                         |
| phase           | SMALLINT                | 1 or 2                                        |
| slot            | SMALLINT                | 1 = Character Event Wish, 2 = Event Wish-2    |
| start_date      | DATE                    | historical, for display and sorting           |
| end_date        | DATE                    |                                               |
| banner_image_path | VARCHAR(255) NULL     | relative to `/static`; NULL = drawn from the featured 5★ splash |

**banner_items** — the official item pool of each banner, as listed on its wiki page.

| Column    | Type                            | Notes |
|-----------|---------------------------------|-------|
| banner_id | BIGINT FK, PK                   |       |
| item_id   | BIGINT FK, PK                   |       |
| role      | ENUM('FEATURED','STANDARD')     | FEATURED: the 5★ and the three 4★ of the rate-up; STANDARD: what a lost 50/50 and a 3★ draw from |

The engine's `BannerPool` (featured 5★, featured 4★, standard 5★, off-banner 4★ characters,
off-banner 4★ weapons, 3★) is built from these rows and cached in memory per banner.

### 3.2 Accounts (as in anti-gamble)

**users** — `id, email UNIQUE, password_hash (bcrypt), display_name, status ENUM('ACTIVE','DISABLED'), created_at, updated_at`. No registration: the seed script creates `traveler@test.local` / `123456`.

**refresh_tokens** — `id, user_id FK, token_hash CHAR(64) UNIQUE (SHA-256), expires_at, revoked_at NULL, created_at`.

### 3.3 Player state

**wallets** — one row per user, created with the user.

| Column                  | Type            | Notes                                     |
|-------------------------|-----------------|-------------------------------------------|
| user_id                 | BIGINT PK FK    |                                           |
| primogems               | BIGINT UNSIGNED |                                           |
| intertwined_fates       | INT UNSIGNED    |                                           |
| starglitter             | INT UNSIGNED    |                                           |
| stardust                | INT UNSIGNED    |                                           |
| total_primogems_added   | BIGINT UNSIGNED | sum of the top-ups                        |
| total_pulls             | INT UNSIGNED    |                                           |
| stardust_fates_month    | CHAR(7) NULL    | `2026-10`, month of the counter below     |
| stardust_fates_bought   | TINYINT         | fates bought with stardust that month (≤ 5) |
| updated_at              | DATETIME        |                                           |

**wallet_transactions** — every movement, never updated afterwards.

| Column        | Type                                                                 | Notes |
|---------------|----------------------------------------------------------------------|-------|
| id            | BIGINT PK AI                                                         |       |
| user_id       | BIGINT FK                                                            |       |
| type          | ENUM('TOP_UP','BUY_FATE','WISH','REFUND','EXCHANGE') | |
| currency      | ENUM('PRIMOGEM','FATE','STARGLITTER','STARDUST')                     |       |
| amount        | INT                                                                  | signed |
| balance_after | BIGINT UNSIGNED                                                      | of that currency |
| wish_id       | CHAR(36) NULL                                                        | `WISH` and `REFUND` rows |
| created_at    | DATETIME                                                             |       |

Index `(user_id, id)`. One operation can write several rows (buying 10 fates: −1600 PRIMOGEM
and +10 FATE).

**pity_states** — one row per user and pool.

| Column           | Type                          | Notes                          |
|------------------|-------------------------------|--------------------------------|
| user_id          | BIGINT FK, PK                 |                                |
| pool             | ENUM('CHARACTER_EVENT'), PK   | room for other pools later     |
| pity5            | TINYINT                       | pulls since last 5★ (0..89)    |
| pity4            | TINYINT                       | pulls since last 4★+ (0..9, can reach 10 after a 5★) |
| guaranteed5      | BOOLEAN                       |                                |
| guaranteed4      | BOOLEAN                       |                                |
| updated_at       | DATETIME                      |                                |

**wishes** — one row per single or 10 pull.

| Column       | Type            | Notes                           |
|--------------|-----------------|---------------------------------|
| id           | CHAR(36) PK     | UUID                            |
| user_id      | BIGINT FK       |                                 |
| banner_id    | BIGINT FK       |                                 |
| count        | TINYINT         | 1 or 10                         |
| fates_spent  | TINYINT         |                                 |
| created_at   | DATETIME        |                                 |

**wish_results** — one row per item obtained, in pull order. This is the wish history.

| Column              | Type                                                         | Notes |
|---------------------|--------------------------------------------------------------|-------|
| id                  | BIGINT PK AI                                                 | global pull order |
| wish_id             | CHAR(36) FK                                                  |       |
| user_id             | BIGINT FK                                                    | denormalised for history queries |
| banner_id           | BIGINT FK                                                    |       |
| position            | TINYINT                                                      | 1..10 inside the wish |
| item_id             | BIGINT FK                                                    |       |
| rarity              | TINYINT                                                      | copied from the item |
| pity                | TINYINT                                                      | the 5★ pity (for 5★) or 4★ pity (for 4★) at which it came; NULL for 3★ |
| featured            | BOOLEAN                                                      |       |
| outcome             | ENUM('WON_50_50','LOST_50_50','GUARANTEED') NULL | 5★ and 4★ |
| is_new              | BOOLEAN                                                      | first copy |
| constellation_after | TINYINT NULL                                                 | characters |
| starglitter         | TINYINT                                                      | refund of this pull |
| stardust            | TINYINT                                                      |       |
| created_at          | DATETIME                                                     |       |

Indexes `(user_id, id)`, `(user_id, rarity, id)`.

**inventory** — what the player owns.

| Column            | Type          |
|-------------------|---------------|
| user_id           | BIGINT FK, PK |
| item_id           | BIGINT FK, PK |
| copies            | INT UNSIGNED  |
| first_obtained_at | DATETIME      |
| last_obtained_at  | DATETIME      |

## 4. Backend design

### 4.1 Stack

Python 3.12+, FastAPI, Uvicorn, SQLAlchemy 2 (sync, `Session`), Alembic, PyMySQL, Pydantic v2 +
pydantic-settings, `pyjwt`, `bcrypt`, `httpx` (asset script). Dependencies in `pyproject.toml`,
installed in a `.venv` with pip (or `uv sync`). Tests: pytest + FastAPI `TestClient` on SQLite.
Lint/format: ruff.

### 4.2 Layout

Same split as anti-gamble (`api` / `core` / `spi`), in Python terms:

```
pull_simulator/
  pyproject.toml
  alembic.ini
  migrations/versions/           0001_initial_schema.py
  data/
    items.json                   characters and weapons (name, name_zh, rarity, element, weapon type)
    banners.json                 banners and their full item pool (from each banner's wiki page)
  static/                        downloaded images (git-ignored)
  start.sh                       venv if missing + init-db.sh + pictures once + uvicorn
  script/dev/
    docker-db.sh                 MySQL container on 3308
    create-db.sh                 drops and recreates the empty database
    migrate-db.sh                alembic upgrade head
    seed-catalog.sh              data/*.json -> items / banners / banner_items
    seed-test-users.sh/.sql      upserts the test login (bcrypt hash of 123456, checked by a test)
    download-assets.sh           fetches images into static/ (§4.6)
    init-db.sh                   docker-db + migrate + catalogue + test user (--reset drops first)
  app/
    main.py                      FastAPI app, routers, CORS, /static mount, error handlers
    config.py                    Settings (env vars)
    db.py                        engine, SessionLocal, get_session dependency
    models/                      SQLAlchemy ORM classes, one per table
    schemas/                     Pydantic request/response models ("views")
    routers/                     auth, users, wallet, banners, wishes, inventory — HTTP only
    services/                    use cases, own the transactions
      auth_service.py
      wallet_service.py          top-up, buy fates, exchange, refunds
      wish_service.py            orchestrates one wish (§4.4)
      history_service.py         history, stats
      inventory_service.py
    repositories/                DB access, one per aggregate
    tools/                       seed_catalog.py, download_assets.py (run by script/dev)
    gacha/                       PURE: no DB, no FastAPI
      rules.py                   all constants of §2
      pool.py                    BannerPool built from a banner + items
      engine.py                  PityState, PullResult, pull(state, pool, rng) -> (PullResult, PityState)
      refunds.py                 starglitter / stardust / constellation
    shared/
      security.py                JWT, password hashing, current_user dependency
      errors.py                  AppError hierarchy → HTTP status
  tests/
```

### 4.3 Gacha engine

`gacha/engine.py` is a pure function, the only place the rules of §2 live:

```python
@dataclass(frozen=True)
class PityState:
    pity5: int; pity4: int; guaranteed5: bool; guaranteed4: bool

@dataclass(frozen=True)
class PullResult:
    item: PoolItem; rarity: int; pity: int | None; featured: bool; outcome: Outcome | None

def pull(state: PityState, pool: BannerPool, rng: random.Random) -> tuple[PullResult, PityState]: ...
def pull_many(state, pool, rng, count) -> tuple[list[PullResult], PityState]: ...
```

Being pure, it is tested in isolation and could also run millions of pulls for statistics
without the DB.

### 4.4 Life of a wish

`POST /api/banners/{code}/wishes {count, auto_buy}` → `WishService.make_wish`, one DB
transaction:

1. Lock the user's `wallets` and `pity_states` rows (`SELECT … FOR UPDATE`) so two calls cannot
   spend the same fates or read the same pity.
2. Fates short? With `auto_buy`, buy the missing ones with primogems (writes `BUY_FATE` rows);
   still short → 402 « Not enough Intertwined Fates », nothing changes.
3. Take the fates (`WISH` row), run `pull_many` with the stored pity.
4. For each result, update `inventory`, compute constellation and refund, write
   `wish_results`.
5. Add refunds to the wallet (`REFUND` rows), save the new pity, `total_pulls += count`.
6. Commit and return the results, the pity and the wallet after.

The result is final once returned: unlike anti-gamble's tickets there is nothing to "finish",
the front just plays the reveal.

### 4.5 Errors

`AppError(status, message)` subclasses, one handler, body like anti-gamble's
`{timestamp, status, error, message, path}`:

| Exception               | Status |
|-------------------------|--------|
| `NotFound`              | 404 (unknown banner) |
| `BadRequest`            | 400 (count not 1/10, invalid amount); FastAPI validation errors are mapped to the same body |
| `InsufficientFunds`     | 402    |
| `ExchangeLimitReached`  | 409 (stardust monthly limit) |
| `Unauthorized`          | 401    |

### 4.6 Images

The user runs this locally only; images are HoYoverse artwork, downloaded on demand, never
committed (`backend/static/` in `.gitignore`), same note as anti-gamble-front's README.

`app/tools/download_assets.py` (`script/dev/download-assets.sh`) reads `data/items.json` and downloads into `static/items/<slug>/`,
from Project Amber (gi.yatta.moe, a fan database of the game files). Its character and weapon
lists (`/api/v2/en/avatar`, `/api/v2/en/weapon`) map each name to the game's asset name:

| File        | Asset (`https://gi.yatta.moe/assets/UI/<asset>.png`) |
|-------------|------------------------------------------------------|
| `icon.png`  | `UI_AvatarIcon_<key>` (characters), `UI_EquipIcon_<key>` (weapons) |
| `splash.png`| `UI_Gacha_AvatarImg_<key>` (character wish splash)   |

Already-downloaded files are skipped; missing ones are listed at the end and the item keeps a
NULL path (the front then shows a rarity-coloured placeholder). Throttled, with a descriptive
`User-Agent`. The catalogue is re-seeded at the end so the paths are filled in.

The Fandom wiki is where the catalogue data comes from, but its image CDN answers bots with a
Cloudflare challenge, so it is not used for pictures. Banners have no art of their own: the front
draws each banner from its featured 5★ splash and its element colour.

FastAPI serves the folder at `/static`; API responses give `icon_url` / `splash_url` already
prefixed.

### 4.7 Settings

Env vars with local defaults, as anti-gamble: `DB_HOST` (localhost), `DB_PORT` (3308),
`DB_NAME`, `DB_USERNAME`, `DB_PASSWORD`, `JWT_SECRET`, `JWT_ACCESS_TTL_MINUTES` (15),
`JWT_REFRESH_TTL_DAYS` (7), `JWT_COOKIE_SECURE` (false), `CORS_ALLOWED_ORIGINS`
(`http://localhost:5173`), `PORT` (8000).

### 4.8 Run locally

```sh
cd pull_simulator
./start.sh                  # venv, MySQL on 3308, migrations, catalogue, test user, pictures, API
./start.sh --reset          # same, database dropped first
.venv/bin/python -m pytest
```

## 5. API design

Base path `/api`, JSON, snake_case. Login-protected calls take `Authorization: Bearer <token>`.
OpenAPI at `/docs`. Lists are paged with `page` (from 0) and `size` (default 20, max 100) and
answer `{items, page, size, total}`.

| Method | Path                                  | Login | Description |
|--------|---------------------------------------|-------|-------------|
| POST   | `/api/auth/login`                     | no    | `{email, password}` → `{access_token}` + refresh cookie |
| POST   | `/api/auth/refresh`                   | no    | New access token, cookie rotated |
| POST   | `/api/auth/logout`                    | no    | Revoke refresh token, clear cookie |
| GET    | `/api/users/me`                       | yes   | The logged-in user |
| POST   | `/api/users/me/reset`                 | yes   | Wipes wallet, pity, history, inventory of the user (start over) |
| GET    | `/api/wallet`                         | yes   | Balances and totals |
| POST   | `/api/wallet/top-ups`                 | yes   | `{amount: 1600 \| 16000 \| 160000}` → adds free primogems |
| POST   | `/api/wallet/fates`                   | yes   | `{count, pay_with: "PRIMOGEM" \| "STARGLITTER" \| "STARDUST"}` — buy fates; 402, 409 |
| GET    | `/api/wallet/transactions`            | yes   | Movements, most recent first (paged) |
| GET    | `/api/banners`                        | no    | All banners, newest first, with featured items (`?version=2.1`) |
| GET    | `/api/banners/{code}`                 | no    | One banner |
| GET    | `/api/banners/{code}/details`         | no    | The in-game "Details": rates, rules text, the full pool by rarity |
| POST   | `/api/banners/{code}/wishes`          | yes   | `{count: 1 \| 10, auto_buy: false}` → the wish; 402 without fates |
| GET    | `/api/pity`                           | yes   | Current pity state and the 5★ rate of the next pull |
| GET    | `/api/wishes/history`                 | yes   | Pull history, most recent first (paged; `?rarity=5`, `?banner=<code>`) |
| GET    | `/api/wishes/stats`                   | yes   | Totals, averages, 50/50 record, spending |
| GET    | `/api/inventory`                      | yes   | Owned items with copies and constellation (`?type=CHARACTER`) |

### 5.1 Main payloads

`GET /api/banners` item:

```json
{
  "code": "2.1-1-raiden-shogun",
  "name": "Reign of Serenity",
  "version": "2.1", "phase": 1, "slot": 1,
  "start_date": "2021-09-01", "end_date": "2021-09-21",
  "banner_image_url": "/static/banners/2.1-1-raiden-shogun.png",
  "featured5": { "slug": "raiden-shogun", "name": "Raiden Shogun", "rarity": 5,
                 "item_type": "CHARACTER", "element": "ELECTRO", "weapon_type": "POLEARM",
                 "icon_url": "/static/items/raiden-shogun/icon.png",
                 "splash_url": "/static/items/raiden-shogun/splash.png" },
  "featured4": [ { "slug": "...", "...": "..." } ]
}
```

`POST /api/banners/{code}/wishes` → 201:

```json
{
  "wish_id": "6f1c…",
  "banner_code": "2.1-1-raiden-shogun",
  "count": 10,
  "max_rarity": 5,
  "fates_bought": 0,
  "results": [
    { "position": 1, "item": { "slug": "favonius-sword", "name": "Favonius Sword", "rarity": 4,
                               "item_type": "WEAPON", "icon_url": "…" },
      "rarity": 4, "pity": 7, "featured": false, "outcome": "LOST_50_50",
      "is_new": false, "constellation_after": null, "starglitter": 2, "stardust": 0 },
    { "position": 2, "item": { "slug": "raiden-shogun", "...": "..." },
      "rarity": 5, "pity": 77, "featured": true, "outcome": "WON_50_50",
      "is_new": true, "constellation_after": 0, "starglitter": 0, "stardust": 0 }
  ],
  "pity_after": { "pity5": 8, "pity4": 2, "guaranteed5": false, "guaranteed4": true,
                  "next_5star_rate": 0.006 },
  "wallet_after": { "primogems": 3200, "intertwined_fates": 0, "starglitter": 14, "stardust": 105 }
}
```

`results` are in pull order; the front sorts them for display. `max_rarity` drives the meteor
colour (blue / purple / gold). `fates_bought` is how many fates `auto_buy` bought first.

`GET /api/wallet`:

```json
{ "primogems": 3200, "intertwined_fates": 0, "starglitter": 14, "stardust": 105,
  "total_primogems_added": 33600, "total_pulls": 200,
  "stardust_fates_left_this_month": 5 }
```

`GET /api/wishes/stats`:

```json
{
  "total_pulls": 200, "fates_spent": 200, "primogems_equivalent": 32000,
  "five_star_count": 3, "four_star_count": 25,
  "average_five_star_pity": 66.3, "average_four_star_pity": 7.9,
  "fifty_fifty": { "won": 1, "lost": 1, "guaranteed": 1 },
  "featured_five_star_count": 2, "pulls_per_featured_five_star": 100,
  "luckiest_five_star_pity": 41, "unluckiest_five_star_pity": 88
}
```

`GET /api/pity`:

```json
{ "pool": "CHARACTER_EVENT", "pity5": 8, "pity4": 2, "guaranteed5": false,
  "guaranteed4": true, "next_5star_rate": 0.006,
  "pulls_to_hard_pity": 82, "pulls_to_soft_pity": 65 }
```

## 6. Front end

Vue 3 + Vite + TypeScript + ant-design-vue + Pinia + vue-router + axios, as anti-gamble-front
(reuse its `api/client.ts` refresh logic and auth store). Only shows what the backend returns —
no random draw, no rule in the front except display text. `VITE_API_BASE_URL=http://localhost:8000`.

Pages:

- **Login** — as anti-gamble.
- **Banners** (`/`) — banner art of each banner, grouped by version; filter by version.
- **Wish** (`/banners/:code`) — banner art, the featured 4★, wallet bar (primogems, fates,
  starglitter), buttons "Wish ×1" / "Wish ×10" with fate cost, "auto-buy fates" switch, pity
  panel (5★ pity, 4★ pity, guaranteed flags). After a wish: a short gold/purple/blue flash, then
  the results grid (10 cards sorted 5★ → 3★, card colour by rarity, NEW badge, constellation,
  50/50 tag such as "Won 50/50" / "Guaranteed"), then back.
- **History** (`/history`) — table like the in-game history: time, item, rarity, banner, pity;
  5★ rows gold, 4★ purple; filter by rarity.
- **Stats** (`/stats`) — numbers of `GET /api/wishes/stats`, pulls per featured 5★ in big letters.
- **Characters** (`/inventory`) — owned characters, icon + constellation.
- **Shop** (modal) — free primogem top-up, buy fates, Paimon's Bargains.

i18n zh (default) / en / fr like anti-gamble-front; item and banner names switch to `name_zh`
in Chinese.

## 7. Tests

- `gacha/engine.py`, seeded RNG, 2 million pulls: consolidated 5★ rate 1.6 % ± 0.05 %, 4★
  13 % ± 0.3 %, mean pulls per 5★ ≈ 62.3, no 5★ pity > 90, never more than 9 3★ in a row,
  share of 50/50 won ≈ 50 %, never two standard 5★ in a row, featured 5★ within 180 pulls.
- Unit: soft-pity formula per pull, guarantee transitions,
  refunds and constellations (C6 overflow), pool by version (no 4★ from a later version).
- API on SQLite: wish with/without fates (402, nothing changed), auto-buy, 10 pull writes 10
  results and the right wallet rows, pity carried across banners, history paging, top-up amounts, stardust monthly limit, reset.

## 8. Catalogue to seed — first version: the Archons

One banner per Archon, its first run (debut banner, slot 1). The pity state is shared, so the
player can hop between them as in the game.

Checked against each banner's page on the Genshin Impact Fandom wiki:

| Code                 | Banner                       | Featured 5★     | Version / phase | Dates                    | Featured 4★ |
|----------------------|------------------------------|-----------------|-----------------|--------------------------|-------------|
| `1.0-1-venti`        | Ballad in Goblets            | Venti (Anemo)   | 1.0 / 1         | 2020-09-28 → 2020-10-18  | Barbara, Fischl, Xiangling |
| `1.1-2-zhongli`      | Gentry of Hermitage          | Zhongli (Geo)   | 1.1 / 2         | 2020-12-01 → 2020-12-22  | Xinyan, Razor, Chongyun |
| `2.1-1-raiden-shogun`| Reign of Serenity            | Raiden Shogun (Electro) | 2.1 / 1 | 2021-09-01 → 2021-09-21  | Kujou Sara, Xiangling, Sucrose |
| `3.2-1-nahida`       | The Moongrass' Enlightenment | Nahida (Dendro) | 3.2 / 1         | 2022-11-02 → 2022-11-18  | Razor, Noelle, Bennett |
| `4.2-1-furina`       | Chanson of Many Waters       | Furina (Hydro)  | 4.2 / 1         | 2023-11-08 → 2023-11-28  | Charlotte, Collei, Beidou |
| `5.3-1-mavuika`      | Ancient Flame Ablaze         | Mavuika (Pyro)  | 5.3 / 1         | 2025-01-01 → 2025-01-21  | Kachina, Bennett, Diona |

The Cryo Archon (The Tsaritsa) has no character banner on the wiki, so she is not playable and
not included. Mavuika's banner has no Chinese name in the data (not sure of it), so the Chinese
UI shows its English name.

Each banner's pool is the one its wiki page lists (§2.2–2.4): Venti's 1.0 banner can only drop
the 4★ characters of 1.0, Mavuika's can drop every 4★ up to 5.3, and the standard 5★ grow the
same way (Tighnari and Dehya only on the later banners, once they joined the standard pool).

`data/items.json` therefore holds 83 items: the 6 Archons, the 7 standard 5★ characters, the 39
4★ characters, the 18 standard 4★ weapons and 13 3★ weapons, each with its Chinese name. More
banners are later just entries in `banners.json` (and new items if needed), then
`script/dev/seed-catalog.sh` and `script/dev/download-assets.sh` again.

## 9. Decisions

1. Login: kept, as anti-gamble (JWT + refresh cookie, seeded local accounts).
2. Database: MySQL 8 in Docker.
3. 50/50: classic rule — lose it once, the next 5★ is the featured one (§2.2). No Capturing Radiance.
4. Money: free primogem top-ups only, no real-money packs.
5. First banners: the six playable Archons (§8).
