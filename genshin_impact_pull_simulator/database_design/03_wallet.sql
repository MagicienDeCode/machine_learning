-- =============================================================================================
-- Step 3 — Money: wallets and wallet_transactions
-- =============================================================================================
-- Two ways to keep money, used together:
--   * wallets             -> the CURRENT balances, one row per player, updated in place
--   * wallet_transactions -> the HISTORY, one row per movement, never updated (an append-only
--                            ledger, like a bank statement)
-- The balance could be recomputed from the history (SUM of amounts), but reading one row is
-- much faster: that copy is a deliberate denormalisation. balance_after lets us check them.
--
-- Concepts: one-to-one relation (primary key that is also a foreign key), signed amounts,
--           secondary index.

USE genshinwish_lesson;

-- ---------------------------------------------------------------------------------------------
-- wallets — ONE row per user: the primary key IS the foreign key (a one-to-one relation).
-- All currencies are simulated; primogems are topped up for free.
-- ---------------------------------------------------------------------------------------------
CREATE TABLE wallets
(
    user_id               BIGINT   NOT NULL,
    primogems             BIGINT   NOT NULL,   -- 原石: 160 buy one fate
    intertwined_fates     INT      NOT NULL,   -- 纠缠之缘: 1 fate = 1 pull
    starglitter           INT      NOT NULL,   -- 无主的星辉: from 4★/5★ duplicates
    stardust              INT      NOT NULL,   -- 无主的星尘: from 3★ weapons
    total_primogems_added BIGINT   NOT NULL,   -- sum of all top-ups, for the statistics
    total_pulls           INT      NOT NULL,
    -- Paimon's Bargains sells at most 5 fates for stardust per calendar month:
    -- the month the counter belongs to ("2026-10"), and the counter.
    stardust_fates_month  CHAR(7)  NULL,
    stardust_fates_bought SMALLINT NOT NULL,
    updated_at            DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (user_id),
    CONSTRAINT fk_wallets_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE = InnoDB;

-- ---------------------------------------------------------------------------------------------
-- wallet_transactions — every movement of ONE currency. One operation can write several rows:
-- buying 10 fates writes  -1600 PRIMOGEM  and  +10 FATE.
-- ---------------------------------------------------------------------------------------------
CREATE TABLE wallet_transactions
(
    id            BIGINT   NOT NULL AUTO_INCREMENT,
    user_id       BIGINT   NOT NULL,
    -- TOP_UP free primogems | BUY_FATE primogems -> fates | WISH fates spent
    -- REFUND starglitter / stardust from a pull | EXCHANGE starglitter / stardust -> fates
    type          ENUM ('TOP_UP','BUY_FATE','WISH','REFUND','EXCHANGE') NOT NULL,
    currency      ENUM ('PRIMOGEM','FATE','STARGLITTER','STARDUST') NOT NULL,
    amount        INT      NOT NULL,           -- signed: + money in, - money out
    balance_after BIGINT   NOT NULL,           -- that currency's balance once applied
    wish_id       CHAR(36) NULL,               -- the wish it belongs to (WISH and REFUND rows)
    created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    CONSTRAINT fk_wallet_transactions_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE = InnoDB;

-- An INDEX makes a search fast without reading the whole table. The history page asks
-- "the movements of user X, newest first":  WHERE user_id = ? ORDER BY id DESC
-- An index on (user_id, id) answers exactly that, already sorted.
CREATE INDEX idx_wallet_transactions_user ON wallet_transactions (user_id, id);
