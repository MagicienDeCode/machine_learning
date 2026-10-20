-- =============================================================================================
-- Step 10 — Writing safely: transactions
-- =============================================================================================
-- What the backend does for the player, written by hand. Each operation touches several rows
-- and several tables: either ALL the changes happen, or NONE (if anything fails, ROLLBACK).
-- That is a TRANSACTION:  START TRANSACTION; ... COMMIT;
--
-- The draw itself is random and done by the backend (app/gacha/engine.py): here we write a
-- result decided in advance, to see which rows a wish touches.
--
-- Concepts: START TRANSACTION / COMMIT / ROLLBACK, SELECT ... FOR UPDATE (row lock),
--           UUID(), the ledger pattern from step 3.

USE genshinwish_lesson;

SET @user_id = (SELECT id FROM users WHERE email = 'traveler@test.local');

-- ---------------------------------------------------------------------------------------------
-- 10.1 Top up 1600 free primogems (10 pulls' worth)
-- ---------------------------------------------------------------------------------------------
START TRANSACTION;

-- FOR UPDATE locks the wallet row until COMMIT: a second request for the same player waits
-- here, instead of reading the old balance and overwriting our change ("lost update").
SELECT primogems FROM wallets WHERE user_id = @user_id FOR UPDATE;

UPDATE wallets
SET primogems             = primogems + 1600,
    total_primogems_added = total_primogems_added + 1600
WHERE user_id = @user_id;

-- The ledger row, with the balance once applied.
INSERT INTO wallet_transactions (user_id, type, currency, amount, balance_after)
SELECT @user_id, 'TOP_UP', 'PRIMOGEM', 1600, primogems
FROM wallets
WHERE user_id = @user_id;

COMMIT;

-- ---------------------------------------------------------------------------------------------
-- 10.2 Buy 10 Intertwined Fates: -1600 primogems, +10 fates -> TWO ledger rows
-- ---------------------------------------------------------------------------------------------
START TRANSACTION;
SELECT primogems, intertwined_fates FROM wallets WHERE user_id = @user_id FOR UPDATE;
-- (The backend checks primogems >= 1600 here, and stops with "402 Insufficient funds".)

UPDATE wallets
SET primogems         = primogems - 1600,
    intertwined_fates = intertwined_fates + 10
WHERE user_id = @user_id;

INSERT INTO wallet_transactions (user_id, type, currency, amount, balance_after)
SELECT @user_id, 'BUY_FATE', 'PRIMOGEM', -1600, primogems FROM wallets WHERE user_id = @user_id;
INSERT INTO wallet_transactions (user_id, type, currency, amount, balance_after)
SELECT @user_id, 'BUY_FATE', 'FATE', 10, intertwined_fates FROM wallets WHERE user_id = @user_id;
COMMIT;

-- ---------------------------------------------------------------------------------------------
-- 10.3 One single wish on Raiden Shogun's banner, that gives... Raiden Shogun, 50/50 won.
-- (Say the player had already made 75 pulls: pity5 becomes 76 on this one.)
-- ---------------------------------------------------------------------------------------------
UPDATE pity_states SET pity5 = 75, pity4 = 3 WHERE user_id = @user_id;   -- the "before"

START TRANSACTION;

-- Lock BOTH rows the wish reads and writes.
SELECT intertwined_fates FROM wallets WHERE user_id = @user_id FOR UPDATE;
SELECT pity5, pity4, guaranteed5 FROM pity_states WHERE user_id = @user_id FOR UPDATE;

SET @wish_id = UUID();
SET @banner_id = (SELECT id FROM banners WHERE code = '2.1-1-raiden-shogun');
SET @item_id = (SELECT id FROM items WHERE slug = 'raiden-shogun');

-- 1. The wish (header).
INSERT INTO wishes (id, user_id, banner_id, count, fates_spent)
VALUES (@wish_id, @user_id, @banner_id, 1, 1);

-- 2. Spend the fate, and write it in the ledger.
UPDATE wallets
SET intertwined_fates = intertwined_fates - 1,
    total_pulls       = total_pulls + 1
WHERE user_id = @user_id;
INSERT INTO wallet_transactions (user_id, type, currency, amount, balance_after, wish_id)
SELECT @user_id, 'WISH', 'FATE', -1, intertwined_fates, @wish_id FROM wallets WHERE user_id = @user_id;

-- 3. The result (line). First copy: is_new, C0, nothing refunded.
INSERT INTO wish_results (wish_id, user_id, banner_id, position, item_id, rarity, pity, featured,
                          outcome, is_new, constellation_after, starglitter, stardust)
VALUES (@wish_id, @user_id, @banner_id, 1, @item_id, 5, 76, TRUE, 'WON_50_50', TRUE, 0, 0, 0);

-- 4. The inventory: a new row, or one more copy if already owned (upsert again).
INSERT INTO inventory (user_id, item_id, copies)
VALUES (@user_id, @item_id, 1)
ON DUPLICATE KEY UPDATE copies           = copies + 1,
                        last_obtained_at = CURRENT_TIMESTAMP;

-- 5. The pity: a 5★ resets the 5★ counter; the 4★ counter keeps running; 50/50 won -> no
--    guarantee.
UPDATE pity_states
SET pity5       = 0,
    pity4       = pity4 + 1,
    guaranteed5 = FALSE
WHERE user_id = @user_id;

COMMIT;

-- ---------------------------------------------------------------------------------------------
-- 10.4 ROLLBACK: change your mind before COMMIT and nothing happened.
-- ---------------------------------------------------------------------------------------------
START TRANSACTION;
UPDATE wallets SET primogems = 999999 WHERE user_id = @user_id;
SELECT primogems AS during_the_transaction FROM wallets WHERE user_id = @user_id;
ROLLBACK;
SELECT primogems AS after_rollback FROM wallets WHERE user_id = @user_id;

-- The ledger agrees with the wallet: summing the movements gives the balance.
SELECT w.primogems,
       (SELECT SUM(amount) FROM wallet_transactions t
        WHERE t.user_id = w.user_id AND t.currency = 'PRIMOGEM') AS sum_of_movements
FROM wallets w
WHERE w.user_id = @user_id;
