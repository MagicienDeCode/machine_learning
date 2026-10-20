# 原神抽卡模拟器

一个本地运行、纯属娱乐的原神 **角色活动祈愿** 模拟器：玩家充值模拟的原石，兑换成纠缠之缘，
在角色卡池中抽卡。后端按真实的概率、保底和 50/50（小保底）规则抽取，并记录每一抽。
原石免费：玩家想充多少就充多少，和 anti-gamble 的余额一样。不涉及任何真实金钱。

结构参照 `anti-gamble` / `anti-gamble-front`：后端负责所有随机抽取和全部状态，前端很薄，
只负责展示结果。

```
genshin_impact_pull_simulator/
  spec.md
  pull_simulator/         Python 3.12+, FastAPI, SQLAlchemy 2, Alembic, MySQL 8
  pull_simulator_front/   Vue 3 + Vite + TypeScript + ant-design-vue（与 anti-gamble-front 技术栈相同）
```

## 1. 范围

包含：

- 仅角色活动祈愿卡池（同一期的「角色活动祈愿」和「角色活动祈愿-2」，与游戏中一样共享保底）。
- 单抽和十连。
- 5★ 和 4★ 的保底、软保底，5★ 和 4★ 的 50/50 及大保底。
- 卡池与真实角色池一致：UP 5★、3 个 UP 4★ 角色、常驻 5★ 角色、4★ 角色和 4★ 武器、3★ 武器。
  武器之所以出现，是因为角色池本身会出武器；没有武器池。
- 钱包：原石、纠缠之缘、无主的星辉和无主的星尘。免费充值原石，与 anti-gamble 的充值一样。
- 重复获得：命之座 C0–C6，以及星辉/星尘返还；用星辉/星尘兑换纠缠之缘（派蒙百货）。
- 祈愿记录、保底状态、统计、已拥有角色。
- 登录与 anti-gamble 相同（多个本地账号，每个账号有自己的钱包和保底）。

不包含：捕获明光（5.0 起对歪掉的 50/50 的「救回」机制）、真实金钱 / 创世结晶礼包、常驻祈愿、
神铸赋形武器池（定轨）、集录祈愿、新手祈愿、空月祝福 / 纪行、祈愿动画（流星颜色就够了）、
武器精炼显示。

## 2. 祈愿规则

所有数值集中在一个模块 `app/gacha/rules.py` 中，方便统一调整。

### 2.1 稀有度抽取

`pity5` = 距上一个 5★ 的抽数，包含本抽（1..90）。
`pity4` = 距上一个 4★ 的抽数，包含本抽（1..10；只有在 5★ 占用了第 10 抽之后才会达到 11 或更多，见下文）。

| 稀有度 | 基础概率 | 软保底                                     | 硬保底 |
|--------|----------|--------------------------------------------|--------|
| 5★     | 0.6 %    | 第 74 抽起：`0.6 % + 6 % × (pity5 − 73)`   | 90     |
| 4★     | 5.1 %    | 第 9 抽起：`5.1 % + 51 % × (pity4 − 8)`    | 10     |
| 3★     | 其余     |                                            |        |

每抽取一个均匀随机数 `r ∈ [0, 1)`：`r < p5` → 5★，否则 `r < p5 + p4` → 4★，否则 3★。
`p5 + p4` 上限为 1。这样得到的综合概率与官方一致（5★ 1.6 %，4★ 13 %），由测试验证（§7）。

出 5★ 时，`pity5` 重置，`pity4` 继续累加——如果 5★ 出在 4★ 保底的第 10 抽，4★ 顺延到下一抽，
与游戏一致。出 4★ 时，`pity4` 重置。

### 2.2 5★ 是否为 UP

| 出 5★ 前的状态          | 结果                                                                 |
|-------------------------|----------------------------------------------------------------------|
| `guaranteed5 = true`    | UP 5★（`GUARANTEED`），`guaranteed5 = false`                         |
| `guaranteed5 = false`   | 50/50：UP（`WON_50_50`）或常驻 5★（`LOST_50_50`，之后 `guaranteed5 = true`） |

也就是说，第一个 5★ 是 50/50；如果歪了，下一个 5★ 必定是 UP 角色，然后重新从 50/50 开始。
长期来看一半的 50/50 会赢，获得 UP 5★ 最多需要 180 抽。

歪出的常驻 5★：该卡池官方物品池中的常驻角色（1.0 起的迪卢克、琴、刻晴、莫娜、七七，之后的提纳里、
迪希雅……从其加入的版本起），按卡池保存在 `banner_items` 中（§3.1）。

### 2.3 4★ 是否为 UP

- `guaranteed4 = true` → 3 个 UP 4★ 角色之一，均匀随机；`guaranteed4 = false`。
- 否则 50 %：UP（3 个中均匀随机）；50 %：非 UP，之后 `guaranteed4 = true`。
- 非 UP 4★：50 % 为 4★ 角色，50 % 为 4★ 武器，各组内均匀随机。两份名单都取自该卡池 wiki 页面上的
  官方物品池：新的 4★ 要到首发的下一个版本才会进入（九条裟罗在她自己的 2.1 卡池里不会作为非 UP 出现）。
  武器为 18 把常驻 4★ 武器（西风 / 祭礼系列、笛剑、弓藏……）。

### 2.4 3★

在该卡池物品池的 13 把 3★ 武器中均匀随机。

### 2.5 保底共享

每个用户在整个 `CHARACTER_EVENT` 池中只有一份保底状态：同一期的两个卡池之间、以及一个卡池到下一个
卡池之间都会继承，与游戏一致。

### 2.6 货币与返还

| 项目                               | 数值                                                        |
|------------------------------------|-------------------------------------------------------------|
| 1 纠缠之缘                         | 160 原石                                                    |
| 单抽 / 十连                        | 1 / 10 纠缠之缘                                             |
| 3★ 武器                            | +15 星尘                                                    |
| 4★ 武器                            | +2 星辉                                                     |
| 4★ 角色，首次获得                  | 无                                                          |
| 4★ 角色，重复 C1–C6 / 满命后       | +2 / +5 星辉                                                |
| 5★ 角色，首次获得                  | 无                                                          |
| 5★ 角色，重复 C1–C6 / 满命后       | +10 / +25 星辉                                              |
| 派蒙百货                           | 5 星辉 = 1 纠缠之缘；75 星尘 = 1 纠缠之缘（每自然月最多 5 个） |

角色命之座 = `min(copies − 1, 6)`。

充值：免费，`{amount}` = 1600、16000 或 160000 原石（10、100 或 1000 抽），类似 anti-gamble 的
100 / 1000 € 充值。其他数额 → 400。

### 2.7 随机性

使用 `secrets.SystemRandom()`（操作系统熵源，相当于 anti-gamble 中 `SecureRandom` 的 Python 版本），
以依赖注入方式提供，测试时可传入带种子的 `random.Random`。

## 3. 数据库设计

MySQL 8 运行在 Docker 中，`localhost:3308`（3307 是 anti-gamble 的），数据库名 `genshinwish`。
表由 Alembic 迁移在 `alembic upgrade head` 时创建。测试使用内存 SQLite，模型只使用可移植的类型。
服务器使用 `utf8mb4`。

同样的表结构和数据，以手写、分步骤的 SQL 形式（用于教学）放在 `database_design/` 中（使用单独的
`genshinwish_lesson` 数据库）。

```
users ──< refresh_tokens
  │
  ├──1 wallets ──< wallet_transactions
  ├──1 pity_states
  ├──< wishes ──< wish_results >── items
  └──< inventory >── items
banners ──< banner_items >── items
```

### 3.1 图鉴数据（从 `pull_simulator/data/*.json` 导入，运行时只读）

**items** —— 所有可能抽出的角色和武器。

| 列                     | 类型                                   | 说明                               |
|------------------------|----------------------------------------|------------------------------------|
| id                     | BIGINT PK AI                           |                                    |
| slug                   | VARCHAR(80) UNIQUE                     | `raiden-shogun`、`favonius-sword`  |
| name                   | VARCHAR(120)                           | `Raiden Shogun`                    |
| name_zh                | VARCHAR(120) NULL                      | `雷电将军`，界面为中文时显示       |
| item_type              | ENUM('CHARACTER','WEAPON')             |                                    |
| rarity                 | SMALLINT                               | 3、4、5                            |
| element                | ENUM('ANEMO','GEO','ELECTRO','DENDRO','HYDRO','PYRO','CRYO') NULL | 仅角色 |
| weapon_type            | ENUM('SWORD','CLAYMORE','POLEARM','BOW','CATALYST') | 角色使用的武器类型或武器本身的类型 |
| release_date           | DATE                                   | 仅供参考                           |
| icon_path              | VARCHAR(255) NULL                      | 相对 `/static`，例如 `items/raiden-shogun/icon.png` |
| splash_path            | VARCHAR(255) NULL                      | 祈愿立绘                           |

**banners**

| 列              | 类型                    | 说明                                          |
|-----------------|-------------------------|-----------------------------------------------|
| id              | BIGINT PK AI            |                                               |
| code            | VARCHAR(60) UNIQUE      | `2.1-1-raiden-shogun`（版本-期数-UP 角色）    |
| name            | VARCHAR(120)            | `Reign of Serenity`                           |
| name_zh         | VARCHAR(120) NULL       | `影寂天下人`                                  |
| version         | VARCHAR(8)              | `2.1`                                         |
| phase           | SMALLINT                | 1 或 2（上半 / 下半）                         |
| slot            | SMALLINT                | 1 = 角色活动祈愿，2 = 角色活动祈愿-2          |
| start_date      | DATE                    | 历史日期，用于展示和排序                      |
| end_date        | DATE                    |                                               |
| banner_image_path | VARCHAR(255) NULL     | 相对 `/static`；NULL = 用 UP 5★ 的立绘绘制    |

**banner_items** —— 每个卡池的官方物品池，与其 wiki 页面一致。

| 列        | 类型                            | 说明 |
|-----------|---------------------------------|------|
| banner_id | BIGINT FK, PK                   |      |
| item_id   | BIGINT FK, PK                   |      |
| role      | ENUM('FEATURED','STANDARD')     | FEATURED：UP 的 5★ 和 3 个 4★；STANDARD：歪了和出 3★ 时从中抽取 |

引擎的 `BannerPool`（UP 5★、UP 4★、常驻 5★、非 UP 4★ 角色、非 UP 4★ 武器、3★）由这些行构建，
按卡池缓存在内存中。

### 3.2 账号（与 anti-gamble 相同）

**users** —— `id, email UNIQUE, password_hash (bcrypt), display_name, status ENUM('ACTIVE','DISABLED'), created_at, updated_at`。无注册功能：初始化脚本创建 `traveler@test.local` / `123456`。

**refresh_tokens** —— `id, user_id FK, token_hash CHAR(64) UNIQUE (SHA-256), expires_at, revoked_at NULL, created_at`。

### 3.3 玩家状态

**wallets** —— 每个用户一行，随用户创建。

| 列                      | 类型            | 说明                                      |
|-------------------------|-----------------|-------------------------------------------|
| user_id                 | BIGINT PK FK    |                                           |
| primogems               | BIGINT UNSIGNED | 原石                                      |
| intertwined_fates       | INT UNSIGNED    | 纠缠之缘                                  |
| starglitter             | INT UNSIGNED    | 星辉                                      |
| stardust                | INT UNSIGNED    | 星尘                                      |
| total_primogems_added   | BIGINT UNSIGNED | 累计充值                                  |
| total_pulls             | INT UNSIGNED    |                                           |
| stardust_fates_month    | CHAR(7) NULL    | `2026-10`，下面计数器对应的月份           |
| stardust_fates_bought   | TINYINT         | 当月用星尘兑换的纠缠之缘数（≤ 5）         |
| updated_at              | DATETIME        |                                           |

**wallet_transactions** —— 每一笔变动，写入后不再修改。

| 列            | 类型                                                  | 说明 |
|---------------|-------------------------------------------------------|------|
| id            | BIGINT PK AI                                          |      |
| user_id       | BIGINT FK                                             |      |
| type          | ENUM('TOP_UP','BUY_FATE','WISH','REFUND','EXCHANGE')  |      |
| currency      | ENUM('PRIMOGEM','FATE','STARGLITTER','STARDUST')      |      |
| amount        | INT                                                   | 有符号 |
| balance_after | BIGINT UNSIGNED                                       | 该货币变动后的余额 |
| wish_id       | CHAR(36) NULL                                         | `WISH` 和 `REFUND` 行 |
| created_at    | DATETIME                                              |      |

索引 `(user_id, id)`。一次操作可以写多行（购买 10 个纠缠之缘：−1600 PRIMOGEM 和 +10 FATE）。

**pity_states** —— 每个用户每个池一行。

| 列               | 类型                          | 说明                           |
|------------------|-------------------------------|--------------------------------|
| user_id          | BIGINT FK, PK                 |                                |
| pool             | ENUM('CHARACTER_EVENT'), PK   | 为以后的其他卡池预留           |
| pity5            | TINYINT                       | 距上一个 5★ 的抽数（0..89）    |
| pity4            | TINYINT                       | 距上一个 4★+ 的抽数（0..9，出 5★ 后可达 10） |
| guaranteed5      | BOOLEAN                       | 5★ 大保底                      |
| guaranteed4      | BOOLEAN                       | 4★ 大保底                      |
| updated_at       | DATETIME                      |                                |

**wishes** —— 每次单抽或十连一行。

| 列           | 类型            | 说明                            |
|--------------|-----------------|---------------------------------|
| id           | CHAR(36) PK     | UUID                            |
| user_id      | BIGINT FK       |                                 |
| banner_id    | BIGINT FK       |                                 |
| count        | TINYINT         | 1 或 10                         |
| fates_spent  | TINYINT         |                                 |
| created_at   | DATETIME        |                                 |

**wish_results** —— 每个获得的物品一行，按抽取顺序。这就是祈愿记录。

| 列                  | 类型                                              | 说明 |
|---------------------|---------------------------------------------------|------|
| id                  | BIGINT PK AI                                      | 全局抽取顺序 |
| wish_id             | CHAR(36) FK                                       |      |
| user_id             | BIGINT FK                                         | 冗余字段，方便查询记录 |
| banner_id           | BIGINT FK                                         |      |
| position            | TINYINT                                           | 本次祈愿中的 1..10 |
| item_id             | BIGINT FK                                         |      |
| rarity              | TINYINT                                           | 从物品复制 |
| pity                | TINYINT                                           | 出货时的 5★ 保底数（5★）或 4★ 保底数（4★）；3★ 为 NULL |
| featured            | BOOLEAN                                           | 是否 UP |
| outcome             | ENUM('WON_50_50','LOST_50_50','GUARANTEED') NULL  | 5★ 和 4★ |
| is_new              | BOOLEAN                                           | 首次获得 |
| constellation_after | TINYINT NULL                                      | 角色 |
| starglitter         | TINYINT                                           | 本抽返还 |
| stardust            | TINYINT                                           |      |
| created_at          | DATETIME                                          |      |

索引 `(user_id, id)`、`(user_id, rarity, id)`。

**inventory** —— 玩家拥有的物品。

| 列                | 类型          |
|-------------------|---------------|
| user_id           | BIGINT FK, PK |
| item_id           | BIGINT FK, PK |
| copies            | INT UNSIGNED  |
| first_obtained_at | DATETIME      |
| last_obtained_at  | DATETIME      |

## 4. 后端设计

### 4.1 技术栈

Python 3.12+、FastAPI、Uvicorn、SQLAlchemy 2（同步，`Session`）、Alembic、PyMySQL、Pydantic v2 +
pydantic-settings、`pyjwt`、`bcrypt`、`httpx`（图片脚本）。依赖写在 `pyproject.toml` 中，用 pip 安装到
`.venv`（也可以用 `uv sync`）。
测试：pytest + FastAPI `TestClient`，跑在 SQLite 上。Lint/格式化：ruff。

### 4.2 目录结构

与 anti-gamble 相同的分层（`api` / `core` / `spi`），换成 Python 的写法：

```
pull_simulator/
  pyproject.toml
  alembic.ini
  migrations/versions/           0001_initial_schema.py
  data/
    items.json                   角色和武器（名称、中文名、稀有度、元素、武器类型）
    banners.json                 卡池及其完整物品池（取自每个卡池的 wiki 页面）
  static/                        下载的图片（git 忽略）
  start.sh                       缺少 venv 时创建 + init-db.sh + 首次下载图片 + uvicorn
  script/dev/
    docker-db.sh                 3308 端口的 MySQL 容器
    create-db.sh                 删除并重建空数据库
    migrate-db.sh                alembic upgrade head
    seed-catalog.sh              data/*.json -> items / banners / banner_items
    seed-test-users.sh/.sql      upsert 测试账号（123456 的 bcrypt 哈希，由测试校验）
    download-assets.sh           下载图片到 static/（§4.6）
    init-db.sh                   docker-db + 迁移 + 图鉴数据 + 测试用户（--reset 先删库）
  app/
    main.py                      FastAPI 应用、路由、CORS、/static 挂载、错误处理
    config.py                    配置（环境变量）
    db.py                        engine、SessionLocal、get_session 依赖
    models/                      SQLAlchemy ORM 类，每张表一个
    schemas/                     Pydantic 请求/响应模型（"views"）
    routers/                     auth、users、wallet、banners、wishes、inventory —— 只处理 HTTP
    services/                    用例，负责事务
      auth_service.py
      wallet_service.py          充值、购买纠缠之缘、兑换、返还
      wish_service.py            编排一次祈愿（§4.4）
      history_service.py         记录、统计
      inventory_service.py
    repositories/                数据库访问，每个聚合一个
    tools/                       seed_catalog.py、download_assets.py（由 script/dev 调用）
    gacha/                       纯逻辑：不依赖数据库，不依赖 FastAPI
      rules.py                   §2 的全部常量
      pool.py                    由卡池 + 物品构建 BannerPool
      engine.py                  PityState、PullResult、pull(state, pool, rng) -> (PullResult, PityState)
      refunds.py                 星辉 / 星尘 / 命之座
    shared/
      security.py                JWT、密码哈希、current_user 依赖
      errors.py                  AppError 异常体系 → HTTP 状态码
  tests/
```

### 4.3 抽卡引擎

`gacha/engine.py` 是纯函数，是 §2 规则唯一所在的地方：

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

因为是纯函数，可以单独测试，也可以不连数据库跑上百万抽做统计。

### 4.4 一次祈愿的流程

`POST /api/banners/{code}/wishes {count, auto_buy}` → `WishService.make_wish`，一个数据库事务：

1. 锁定用户的 `wallets` 和 `pity_states` 行（`SELECT … FOR UPDATE`），避免两个请求花掉同一份
   纠缠之缘或读到同一份保底。
2. 纠缠之缘不够？开启 `auto_buy` 时用原石补买（写 `BUY_FATE` 行）；仍不够 → 402
   « Not enough Intertwined Fates »，不做任何改动。
3. 扣除纠缠之缘（`WISH` 行），用已保存的保底执行 `pull_many`。
4. 对每个结果更新 `inventory`，计算命之座和返还，写入 `wish_results`。
5. 将返还加入钱包（`REFUND` 行），保存新的保底，`total_pulls += count`。
6. 提交并返回结果、祈愿后的保底和钱包。

结果一旦返回即为最终结果：与 anti-gamble 的彩票不同，没有需要「完成」的步骤，前端只负责播放揭晓效果。

### 4.5 错误

`AppError(status, message)` 的子类，统一由一个处理器处理，响应体与 anti-gamble 相同
`{timestamp, status, error, message, path}`：

| 异常                    | 状态码 |
|-------------------------|--------|
| `NotFound`              | 404（卡池不存在） |
| `BadRequest`            | 400（count 不是 1/10、充值数额无效）；FastAPI 校验错误也映射为相同响应体 |
| `InsufficientFunds`     | 402    |
| `ExchangeLimitReached`  | 409（星尘每月兑换上限） |
| `Unauthorized`          | 401    |

### 4.6 图片

只在本地运行；图片是米哈游的美术资源，按需下载，绝不提交（`pull_simulator/static/` 在 `.gitignore` 中），
与 anti-gamble-front README 中的说明相同。

`app/tools/download_assets.py`（`script/dev/download-assets.sh`）读取 `data/items.json`，从 Project Amber（gi.yatta.moe，一个整理游戏文件的
粉丝数据库）下载到 `static/items/<slug>/`。它的角色和武器列表（`/api/v2/en/avatar`、`/api/v2/en/weapon`）
把每个名称对应到游戏内的资源名：

| 文件        | 资源（`https://gi.yatta.moe/assets/UI/<asset>.png`） |
|-------------|------------------------------------------------------|
| `icon.png`  | `UI_AvatarIcon_<key>`（角色），`UI_EquipIcon_<key>`（武器） |
| `splash.png`| `UI_Gacha_AvatarImg_<key>`（角色祈愿立绘）           |

已下载的文件会跳过；缺失的在最后列出，该物品的路径保持 NULL（前端显示按稀有度着色的占位图）。
有限速，并带有说明性的 `User-Agent`。最后会重新导入图鉴数据，把图片路径填进去。

图鉴数据来自 Fandom wiki，但它的图片 CDN 会对机器人弹出 Cloudflare 验证，所以图片不从那里下载。
卡池没有单独的卡池图：前端用 UP 5★ 的立绘和元素颜色绘制卡池。

FastAPI 在 `/static` 提供该目录；API 响应中的 `icon_url` / `splash_url` 已带好前缀。

### 4.7 配置

环境变量，带本地默认值，与 anti-gamble 相同：`DB_HOST`（localhost）、`DB_PORT`（3308）、
`DB_NAME`、`DB_USERNAME`、`DB_PASSWORD`、`JWT_SECRET`、`JWT_ACCESS_TTL_MINUTES`（15）、
`JWT_REFRESH_TTL_DAYS`（7）、`JWT_COOKIE_SECURE`（false）、`CORS_ALLOWED_ORIGINS`
（`http://localhost:5173`）、`PORT`（8000）。

### 4.8 本地运行

```sh
cd pull_simulator
./start.sh                  # venv、3308 上的 MySQL、迁移、图鉴数据、测试用户、图片、API
./start.sh --reset          # 同上，先删库
.venv/bin/python -m pytest
```

## 5. API 设计

基础路径 `/api`，JSON，snake_case。需要登录的接口带 `Authorization: Bearer <token>`。
OpenAPI 文档在 `/docs`。列表接口用 `page`（从 0 开始）和 `size`（默认 20，最大 100）分页，
返回 `{items, page, size, total}`。

| 方法   | 路径                                  | 登录 | 说明 |
|--------|---------------------------------------|------|------|
| POST   | `/api/auth/login`                     | 否   | `{email, password}` → `{access_token}` + refresh cookie |
| POST   | `/api/auth/refresh`                   | 否   | 新的 access token，cookie 轮换 |
| POST   | `/api/auth/logout`                    | 否   | 吊销 refresh token，清除 cookie |
| GET    | `/api/users/me`                       | 是   | 当前登录用户 |
| POST   | `/api/users/me/reset`                 | 是   | 清空该用户的钱包、保底、记录、背包（重新开始） |
| GET    | `/api/wallet`                         | 是   | 余额和累计数据 |
| POST   | `/api/wallet/top-ups`                 | 是   | `{amount: 1600 \| 16000 \| 160000}` → 增加免费原石 |
| POST   | `/api/wallet/fates`                   | 是   | `{count, pay_with: "PRIMOGEM" \| "STARGLITTER" \| "STARDUST"}` —— 购买纠缠之缘；402、409 |
| GET    | `/api/wallet/transactions`            | 是   | 变动记录，最新在前（分页） |
| GET    | `/api/banners`                        | 否   | 所有卡池，最新在前，含 UP 物品（`?version=2.1`） |
| GET    | `/api/banners/{code}`                 | 否   | 单个卡池 |
| GET    | `/api/banners/{code}/details`         | 否   | 游戏内的「详情」：概率、规则说明、按稀有度列出的完整物品池 |
| POST   | `/api/banners/{code}/wishes`          | 是   | `{count: 1 \| 10, auto_buy: false}` → 祈愿结果；纠缠之缘不足返回 402 |
| GET    | `/api/pity`                           | 是   | 当前保底状态及下一抽的 5★ 概率 |
| GET    | `/api/wishes/history`                 | 是   | 祈愿记录，最新在前（分页；`?rarity=5`、`?banner=<code>`） |
| GET    | `/api/wishes/stats`                   | 是   | 总数、平均值、50/50 战绩、消耗 |
| GET    | `/api/inventory`                      | 是   | 已拥有物品，含数量和命之座（`?type=CHARACTER`） |

### 5.1 主要数据结构

`GET /api/banners` 中的一项：

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

`POST /api/banners/{code}/wishes` → 201：

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

`results` 按抽取顺序排列；前端展示时自行排序。`max_rarity` 决定流星颜色（蓝 / 紫 / 金）。
`fates_bought` 是 `auto_buy` 先买了多少个纠缠之缘。

`GET /api/wallet`：

```json
{ "primogems": 3200, "intertwined_fates": 0, "starglitter": 14, "stardust": 105,
  "total_primogems_added": 33600, "total_pulls": 200,
  "stardust_fates_left_this_month": 5 }
```

`GET /api/wishes/stats`：

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

`GET /api/pity`：

```json
{ "pool": "CHARACTER_EVENT", "pity5": 8, "pity4": 2, "guaranteed5": false,
  "guaranteed4": true, "next_5star_rate": 0.006,
  "pulls_to_hard_pity": 82, "pulls_to_soft_pity": 65 }
```

## 6. 前端

Vue 3 + Vite + TypeScript + ant-design-vue + Pinia + vue-router + axios，与 anti-gamble-front 相同
（复用其 `api/client.ts` 的刷新逻辑和 auth store）。只展示后端返回的内容——前端没有随机抽取，
除展示文案外不含任何规则。`VITE_API_BASE_URL=http://localhost:8000`。

页面：

- **登录** —— 与 anti-gamble 相同。
- **卡池列表**（`/`）—— 每个卡池的卡池图，按版本分组；可按版本筛选。
- **祈愿**（`/banners/:code`）—— 卡池图、UP 4★、钱包栏（原石、纠缠之缘、星辉），带纠缠之缘消耗的
  「祈愿 ×1」/「祈愿 ×10」按钮，「自动购买纠缠之缘」开关，保底面板（5★ 保底、4★ 保底、大保底标记）。
  祈愿后：短暂的金/紫/蓝闪光，然后显示结果网格（10 张卡按 5★ → 3★ 排序，卡片颜色按稀有度，NEW 标记，
  命之座，50/50 标签如「赢了 50/50」/「大保底」），然后返回。
- **祈愿记录**（`/history`）—— 类似游戏内记录的表格：时间、物品、稀有度、卡池、保底数；
  5★ 行金色、4★ 行紫色；可按稀有度筛选。
- **统计**（`/stats`）—— `GET /api/wishes/stats` 的数据，大字显示「每个 UP 5★ 平均抽数」。
- **角色**（`/inventory`）—— 已拥有角色，头像 + 命之座。
- **商店**（弹窗）—— 免费充值原石、购买纠缠之缘、派蒙百货。

国际化 zh（默认）/ en / fr，与 anti-gamble-front 相同；界面为中文时，物品和卡池名称使用 `name_zh`。

## 7. 测试

- `gacha/engine.py`，带种子的随机数，200 万抽：综合 5★ 概率 1.6 % ± 0.05 %，4★ 13 % ± 0.3 %，
  平均每个 5★ 约 62.3 抽，5★ 保底不超过 90，3★ 连续不超过 9 个，
  50/50 胜率约 50 %，不会连续两次歪常驻 5★，UP 5★ 在 180 抽内必出。
- 单元测试：每抽的软保底公式、大保底状态转换、返还和命之座（满命溢出）、按版本构建卡池
  （不会出现更晚版本的 4★）。
- SQLite 上的 API 测试：有/无纠缠之缘的祈愿（402，无任何改动）、自动购买、十连写入 10 条结果和正确的
  钱包记录、跨卡池继承保底、记录分页、充值数额、星尘每月上限、重置。

## 8. 初始图鉴数据 —— 第一版：神

每位神一个卡池，取其首次登场的卡池（首发，slot 1）。保底共享，玩家可以像游戏中一样在卡池之间切换。

已对照每个卡池的 Fandom wiki 页面核实：

| Code                 | 卡池         | UP 5★             | 版本 / 期数 | 日期                     | UP 4★ |
|----------------------|--------------|-------------------|-------------|--------------------------|-------|
| `1.0-1-venti`        | 杯装之诗     | 温迪（风）        | 1.0 / 1     | 2020-09-28 → 2020-10-18  | 芭芭拉、菲谢尔、香菱 |
| `1.1-2-zhongli`      | 陵薮市朝     | 钟离（岩）        | 1.1 / 2     | 2020-12-01 → 2020-12-22  | 辛焱、雷泽、重云 |
| `2.1-1-raiden-shogun`| 影寂天下人   | 雷电将军（雷）    | 2.1 / 1     | 2021-09-01 → 2021-09-21  | 九条裟罗、香菱、砂糖 |
| `3.2-1-nahida`       | 月草的赐慧   | 纳西妲（草）      | 3.2 / 1     | 2022-11-02 → 2022-11-18  | 雷泽、诺艾尔、班尼特 |
| `4.2-1-furina`       | 众水的颂诗   | 芙宁娜（水）      | 4.2 / 1     | 2023-11-08 → 2023-11-28  | 夏洛蒂、柯莱、北斗 |
| `5.3-1-mavuika`      | Ancient Flame Ablaze | 玛薇卡（火） | 5.3 / 1   | 2025-01-01 → 2025-01-21  | 卡齐娜、班尼特、迪奥娜 |

玛薇卡卡池的中文名没有填（不确定），界面上显示英文名。冰神（冰之女皇）在 wiki 上没有角色卡池，
还不是可玩角色，所以不加入。

每个卡池的物品池与其 wiki 页面一致（§2.2–2.4）：温迪的 1.0 卡池只会出 1.0 已有的 4★ 角色，玛薇卡的卡池
可以出 5.3 及之前的所有 4★，常驻 5★ 同理（提纳里、迪希雅只在加入常驻池之后的卡池里出现）。

因此 `data/items.json` 共 83 个物品：6 位神、7 个常驻 5★ 角色、39 个 4★ 角色、18 把常驻 4★ 武器和
13 把 3★ 武器，每个都带中文名。以后增加卡池只需在 `banners.json` 中加一项（必要时加新物品），
然后重新运行 `script/dev/seed-catalog.sh` 和 `script/dev/download-assets.sh`。

## 9. 决定

1. 登录：保留，与 anti-gamble 相同（JWT + refresh cookie，预置本地账号）。
2. 数据库：Docker 中的 MySQL 8。
3. 50/50：经典规则——歪一次，下一个 5★ 必定是 UP（§2.2）。不做捕获明光。
4. 金钱：只有免费充值原石，没有真实金钱礼包。
5. 首批卡池：6 位可玩的神（§8）。
