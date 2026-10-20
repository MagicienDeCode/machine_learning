-- =============================================================================================
-- Step 4 — Pity: pity_states
-- =============================================================================================
-- The state the gacha needs between two pulls. The rules themselves live in the backend
-- (app/gacha/rules.py); the database only remembers where each player is.
--
--   pity5        pulls since the last 5★ (0..89). 90 pulls without a 5★ is impossible.
--   pity4        pulls since the last 4★ (0..9, rarely 10 after a 5★ took the 10th pull).
--   guaranteed5  TRUE after a lost 50/50: the next 5★ is the featured character (大保底).
--   guaranteed4  same for 4★.
--
-- Concepts: composite primary key mixing a foreign key and a fixed value, BOOLEAN.

USE genshinwish_lesson;

CREATE TABLE pity_states
(
    user_id     BIGINT   NOT NULL,
    -- Every Character Event banner shares the same pity, so there is one pool for now. Keeping
    -- it in the key leaves room for a weapon banner later, with its own pity.
    pool        ENUM ('CHARACTER_EVENT') NOT NULL,
    pity5       SMALLINT NOT NULL,
    pity4       SMALLINT NOT NULL,
    -- BOOLEAN is an alias of TINYINT(1) in MySQL: 0 = FALSE, 1 = TRUE.
    guaranteed5 BOOLEAN  NOT NULL,
    guaranteed4 BOOLEAN  NOT NULL,
    updated_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (user_id, pool),
    CONSTRAINT fk_pity_states_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE = InnoDB;
