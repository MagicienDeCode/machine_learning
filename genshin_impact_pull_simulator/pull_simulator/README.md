# pull_simulator

FastAPI backend of a local, for-fun Genshin Impact **Character Event Wish** simulator: login, a
wallet of free simulated primogems, the real rates / pity / 50-50 rules, the wish history,
statistics and owned characters. No real money is involved. The front is in
`../pull_simulator_front`, the design in `../spec.md`.

## Run locally

Python 3.12+ and Docker. One command:

```sh
./start.sh                 # http://localhost:8000, OpenAPI at /docs
./start.sh --reset         # drops the database first: every player starts over
./start.sh --no-assets     # skips the picture download

.venv/bin/python -m pytest # needs no Docker: in-memory SQLite
```

`start.sh` creates the `.venv` if missing (`pip install -e .`; `uv sync` works too, it reads the
same `pyproject.toml`), runs `script/dev/init-db.sh`, downloads the pictures the first time,
then starts Uvicorn with auto-reload (`PORT` to change the port). Every step is safe to re-run
and keeps the data.

`init-db.sh` chains the smaller scripts of `script/dev`, which can also be run one by one:

| Script                | What it does                                                        |
|-----------------------|---------------------------------------------------------------------|
| `docker-db.sh`        | Starts (or creates) the `genshin-wish-mysql` container, port 3308   |
| `create-db.sh`        | Drops and recreates the empty database (`init-db.sh --reset`)       |
| `migrate-db.sh`       | Creates / updates the tables (Alembic migrations)                   |
| `seed-catalog.sh`     | Loads `data/*.json`: items, banners and their pools                 |
| `seed-test-users.sh`  | Creates or resets the test login (`seed-test-users.sql`)            |
| `download-assets.sh`  | Fetches the pictures into `static/` (not part of `init-db.sh`)      |

There is no registration: `seed-test-users.sql` creates `traveler@test.local`, password
`123456` — local dev only. Its wallet and history are kept; `POST /api/users/me/reset` (or
"start over" in the shop) wipes them. Settings are environment variables (see `app/config.py`): `DB_HOST`, `DB_PORT`,
`DB_NAME`, `DB_USERNAME`, `DB_PASSWORD`, `DATABASE_URL`, `JWT_SECRET`, `JWT_COOKIE_SECURE`,
`CORS_ALLOWED_ORIGINS`. The defaults only suit local dev.

## API

| Method | Path                            | Login | Description |
|--------|---------------------------------|-------|-------------|
| POST   | `/api/auth/login`               | no    | `{email, password}` → access token + refresh cookie |
| POST   | `/api/auth/refresh`             | no    | New access token from the refresh cookie, which is rotated |
| POST   | `/api/auth/logout`              | no    | Revokes the refresh token, clears the cookie |
| GET    | `/api/users/me`                 | yes   | The logged-in user |
| POST   | `/api/users/me/reset`           | yes   | Wipes wallet, pity, history and inventory (start over) |
| GET    | `/api/wallet`                   | yes   | Primogems, fates, starglitter, stardust, totals |
| POST   | `/api/wallet/top-ups`           | yes   | `{amount: 1600 \| 16000 \| 160000}` free primogems |
| POST   | `/api/wallet/fates`             | yes   | `{count, pay_with: PRIMOGEM \| STARGLITTER \| STARDUST}`; 402, 409 (stardust: 5 a month) |
| GET    | `/api/wallet/transactions`      | yes   | Wallet movements, most recent first (`page`, `size`) |
| GET    | `/api/banners`                  | no    | Banners, newest first, with their featured items |
| GET    | `/api/banners/{code}`           | no    | One banner |
| GET    | `/api/banners/{code}/details`   | no    | Rates, rules and the full item pool |
| POST   | `/api/banners/{code}/wishes`    | yes   | `{count: 1 \| 10, auto_buy}` → the results; 402 without fates |
| GET    | `/api/pity`                     | yes   | Pity counters, guarantees, rate of the next pull |
| GET    | `/api/wishes/history`           | yes   | Every pull, most recent first (`page`, `size`, `rarity`, `banner`) |
| GET    | `/api/wishes/stats`             | yes   | Totals, average pity, 50/50 record |
| GET    | `/api/inventory`                | yes   | Owned items with copies and constellation (`type`) |

Errors all have the body `{timestamp, status, error, message, path}`.

## Rules

All figures are in `app/gacha/rules.py`; the draw itself is the pure function
`app/gacha/engine.py::pull` (no database), tested over two million pulls in
`tests/test_engine.py`.

- 5★: 0.6 %, +6 % per pull from pull 74, certain on pull 90 (1.6 % overall, ~62 pulls per 5★).
- 4★: 5.1 %, 56.1 % on pull 9, certain on pull 10 (13 % overall). A 5★ does not reset the 4★
  counter.
- 50/50: a 5★ is the featured one half the time; once lost, the next 5★ is the featured one.
  Same for 4★ with the three featured 4★. No Capturing Radiance.
- Pity and guarantees are shared by every banner.
- Duplicates give constellations (C0–C6) and starglitter; 3★ weapons give stardust; both buy
  fates at Paimon's Bargains.

Randomness comes from `secrets.SystemRandom()` (OS entropy).

## Catalogue

`data/items.json` and `data/banners.json` hold the six Archons' debut banners (Venti, Zhongli,
Raiden Shogun, Nahida, Furina, Mavuika) and every item they can drop. The pool of each banner
— featured, standard 5★, off-banner 4★ characters and weapons, 3★ weapons — is copied from the
banner's page on the Genshin Impact Fandom wiki, so a character only appears off-banner from
the version it really joined the pool. `script/dev/seed-catalog.sh` loads both files (safe to
re-run); adding a banner is adding its entry (and any new item) and re-running it.

## Pictures

`script/dev/download-assets.sh` (`app/tools/download_assets.py`) fetches each character's icon and wish splash and each weapon's
icon from Project Amber (gi.yatta.moe), then re-seeds the catalogue so the API serves them under
`/static`. They are HoYoverse artwork: fine for a private, local project, never committed
(`static/` is git-ignored). Banners have no art of their own; the front draws them from the
featured character's splash.

## Tables

| Table                 | Content |
|-----------------------|---------|
| `items`               | Characters and weapons (rarity, element, weapon type, picture paths) |
| `banners`             | Banners (version, phase, dates) |
| `banner_items`        | The item pool of each banner, `FEATURED` or `STANDARD` |
| `users`, `refresh_tokens` | Login, as in anti-gamble |
| `wallets`             | One row per player: balances and totals |
| `wallet_transactions` | Every movement of one currency, never changed afterwards |
| `pity_states`         | Pity counters and guarantees of each player |
| `wishes`              | One row per single or 10 pull |
| `wish_results`        | One row per item obtained, in pull order: the wish history |
| `inventory`           | Copies owned of each item |

The schema is the Alembic migration `migrations/versions/0001_initial_schema.py`.
