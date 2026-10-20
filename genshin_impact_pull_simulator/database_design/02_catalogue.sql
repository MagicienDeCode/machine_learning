-- =============================================================================================
-- Step 2 — The catalogue: items, banners, banner_items
-- =============================================================================================
-- What can be pulled, and on which banner. These tables are filled once (steps 7 and 8) and
-- only read by the game afterwards.
--
-- Concepts: many-to-many relation through a join table, composite primary key,
--           an attribute stored on the relation itself (role).

USE genshinwish_lesson;

-- ---------------------------------------------------------------------------------------------
-- items — every character and weapon that a banner can drop (83 rows in step 7).
-- ---------------------------------------------------------------------------------------------
CREATE TABLE items
(
    id           BIGINT       NOT NULL AUTO_INCREMENT,
    -- slug: a stable, readable key used in URLs and in the seed files ("raiden-shogun").
    -- The id is for joins, the slug is for humans.
    slug         VARCHAR(80)  NOT NULL,
    name         VARCHAR(120) NOT NULL,
    name_zh      VARCHAR(120) NULL,          -- 雷电将军: shown when the interface is in Chinese
    item_type    ENUM ('CHARACTER','WEAPON') NOT NULL,
    rarity       SMALLINT     NOT NULL,      -- 3, 4 or 5 stars
    -- Only characters have an element: weapons leave it NULL.
    element      ENUM ('ANEMO','GEO','ELECTRO','DENDRO','HYDRO','PYRO','CRYO') NULL,
    -- A character's weapon, or the weapon's own type.
    weapon_type  ENUM ('SWORD','CLAYMORE','POLEARM','BOW','CATALYST') NOT NULL,
    release_date DATE         NOT NULL,
    -- Picture files, relative to the backend's /static folder. NULL until downloaded.
    icon_path    VARCHAR(255) NULL,
    splash_path  VARCHAR(255) NULL,

    PRIMARY KEY (id),
    CONSTRAINT uq_items_slug UNIQUE (slug)
) ENGINE = InnoDB;

-- ---------------------------------------------------------------------------------------------
-- banners — one row per wish banner (6 rows in step 8: the Archons' first banners).
-- ---------------------------------------------------------------------------------------------
CREATE TABLE banners
(
    id                BIGINT       NOT NULL AUTO_INCREMENT,
    code              VARCHAR(60)  NOT NULL,   -- "2.1-1-raiden-shogun" = version-phase-character
    name              VARCHAR(120) NOT NULL,   -- "Reign of Serenity"
    name_zh           VARCHAR(120) NULL,       -- "影寂天下人"
    version           VARCHAR(8)   NOT NULL,   -- game version: "2.1"
    phase             SMALLINT     NOT NULL,   -- 1 = first half of the version, 2 = second half
    slot              SMALLINT     NOT NULL,   -- 1 = Character Event Wish, 2 = Event Wish-2
    start_date        DATE         NOT NULL,
    end_date          DATE         NOT NULL,
    banner_image_path VARCHAR(255) NULL,       -- NULL: the front draws it from the 5★ splash

    PRIMARY KEY (id),
    CONSTRAINT uq_banners_code UNIQUE (code)
) ENGINE = InnoDB;

-- ---------------------------------------------------------------------------------------------
-- banner_items — which items each banner can drop.
--
-- A banner drops many items, and an item (Xiangling...) appears on many banners: that is a
-- MANY-TO-MANY relation. It cannot live in either table, so it gets its own "join table" with
-- one row per (banner, item) pair.
--
-- The pair is the primary key (a COMPOSITE key): the same item cannot be listed twice on the
-- same banner. The relation also carries data of its own, its role on that banner:
--   FEATURED -> the rate-up 5★ and the three rate-up 4★
--   STANDARD -> what a lost 50/50, an off-banner 4★ or a 3★ is drawn from
-- ---------------------------------------------------------------------------------------------
CREATE TABLE banner_items
(
    banner_id BIGINT NOT NULL,
    item_id   BIGINT NOT NULL,
    role      ENUM ('FEATURED','STANDARD') NOT NULL,

    PRIMARY KEY (banner_id, item_id),
    -- Deleting a banner removes its pool. Deleting an item still used by a banner is refused
    -- (no ON DELETE: the default is RESTRICT).
    CONSTRAINT fk_banner_items_banner FOREIGN KEY (banner_id) REFERENCES banners (id) ON DELETE CASCADE,
    CONSTRAINT fk_banner_items_item FOREIGN KEY (item_id) REFERENCES items (id)
) ENGINE = InnoDB;
