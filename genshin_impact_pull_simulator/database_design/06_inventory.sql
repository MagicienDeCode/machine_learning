-- =============================================================================================
-- Step 6 — What a player owns: inventory
-- =============================================================================================
-- Another many-to-many join table (a user owns many items, an item is owned by many users),
-- this time carrying a count.
--
-- The constellation is NOT stored: it can always be computed, MIN(copies - 1, 6), so storing it
-- would only create a chance for the two to disagree. (Compare with the wallet balance in step
-- 3, which we DO store because recomputing it would be slow.)

USE genshinwish_lesson;

CREATE TABLE inventory
(
    user_id           BIGINT   NOT NULL,
    item_id           BIGINT   NOT NULL,
    copies            INT      NOT NULL,     -- 1 = owned once (C0), 7 = C6, more = beyond C6
    first_obtained_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_obtained_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (user_id, item_id),
    CONSTRAINT fk_inventory_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    CONSTRAINT fk_inventory_item FOREIGN KEY (item_id) REFERENCES items (id)
) ENGINE = InnoDB;

-- All 11 tables exist now:
--   SHOW TABLES;
-- And who points at whom:
--   SELECT table_name, column_name, referenced_table_name
--   FROM information_schema.key_column_usage
--   WHERE table_schema = 'genshinwish_lesson' AND referenced_table_name IS NOT NULL
--   ORDER BY table_name;
