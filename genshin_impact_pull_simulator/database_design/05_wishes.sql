-- =============================================================================================
-- Step 5 — Wishes: wishes and wish_results
-- =============================================================================================
-- A "header / lines" pair, like an order and its order lines:
--   * wishes       -> one row per click on "Wish ×1" or "Wish ×10"
--   * wish_results -> one row per item obtained, in pull order: this IS the wish history
--
-- Concepts: UUID primary key, one-to-many relation, several foreign keys on one table,
--           denormalised columns for fast reads, composite indexes.

USE genshinwish_lesson;

-- ---------------------------------------------------------------------------------------------
-- wishes
-- ---------------------------------------------------------------------------------------------
CREATE TABLE wishes
(
    -- CHAR(36) UUID ("6f1c2a9e-...") instead of AUTO_INCREMENT: the backend creates the id
    -- BEFORE inserting, so it can write it into wallet_transactions.wish_id in the same step.
    id          CHAR(36) NOT NULL,
    user_id     BIGINT   NOT NULL,
    banner_id   BIGINT   NOT NULL,
    count       SMALLINT NOT NULL,           -- 1 or 10
    fates_spent SMALLINT NOT NULL,
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    CONSTRAINT fk_wishes_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    CONSTRAINT fk_wishes_banner FOREIGN KEY (banner_id) REFERENCES banners (id)
) ENGINE = InnoDB;

-- ---------------------------------------------------------------------------------------------
-- wish_results — 1 or 10 rows per wish.
-- ---------------------------------------------------------------------------------------------
CREATE TABLE wish_results
(
    -- AUTO_INCREMENT here: the id gives the global pull order of the whole history.
    id                  BIGINT   NOT NULL AUTO_INCREMENT,
    wish_id             CHAR(36) NOT NULL,
    -- user_id and banner_id could be found through wishes, but the history page filters on
    -- them on every request: copying them here avoids a join (denormalisation, again).
    user_id             BIGINT   NOT NULL,
    banner_id           BIGINT   NOT NULL,
    position            SMALLINT NOT NULL,   -- 1..10 inside its wish
    item_id             BIGINT   NOT NULL,
    rarity              SMALLINT NOT NULL,   -- copied from items, same reason
    -- The pity at which it came: 5★ pity for a 5★, 4★ pity for a 4★, NULL for a 3★.
    pity                SMALLINT NULL,
    featured            BOOLEAN  NOT NULL,   -- a rate-up item of this banner?
    -- WON_50_50 / LOST_50_50 / GUARANTEED for 5★ and 4★; NULL for 3★.
    outcome             ENUM ('WON_50_50','LOST_50_50','GUARANTEED') NULL,
    is_new              BOOLEAN  NOT NULL,   -- first copy ever
    constellation_after SMALLINT NULL,       -- characters only: C0..C6 after this pull
    starglitter         SMALLINT NOT NULL,   -- refunded by this pull
    stardust            SMALLINT NOT NULL,
    created_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    CONSTRAINT fk_wish_results_wish FOREIGN KEY (wish_id) REFERENCES wishes (id) ON DELETE CASCADE,
    CONSTRAINT fk_wish_results_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    CONSTRAINT fk_wish_results_banner FOREIGN KEY (banner_id) REFERENCES banners (id),
    CONSTRAINT fk_wish_results_item FOREIGN KEY (item_id) REFERENCES items (id)
) ENGINE = InnoDB;

-- The two questions the history page asks:
--   "my pulls, newest first"            WHERE user_id = ?                ORDER BY id DESC
--   "my 5★ pulls, newest first"         WHERE user_id = ? AND rarity = 5 ORDER BY id DESC
-- One index for each. Column order matters: equality columns first, the sort column last.
CREATE INDEX idx_wish_results_user ON wish_results (user_id, id);
CREATE INDEX idx_wish_results_user_rarity ON wish_results (user_id, rarity, id);
