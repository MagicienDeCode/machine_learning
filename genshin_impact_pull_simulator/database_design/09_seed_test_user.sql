-- =============================================================================================
-- Step 9 — A player: user, wallet and pity
-- =============================================================================================
-- The same login as the app: traveler@test.local / 123456 (local dev only: the password is
-- public). Then the player's one-to-one rows, which the app would otherwise create on first use.
--
-- Concepts: upsert (INSERT ... ON DUPLICATE KEY UPDATE), LAST_INSERT_ID(), a session variable.

USE genshinwish_lesson;

-- Upsert: insert the row, or, when the UNIQUE email already exists, update it instead.
-- Running this file twice therefore does not fail and does not create a second account.
-- The hash is `123456` through bcrypt; the app's tests check it (tests/test_seed.py).
INSERT INTO users (email, password_hash, display_name, status)
VALUES ('traveler@test.local', '$2b$10$r0DarDL52103Bvk.Jf06uOROk5z1APlFPm/OzaCVK6.DXqmMURh9e',
        'Traveler', 'ACTIVE')
ON DUPLICATE KEY UPDATE password_hash = VALUES(password_hash),
                        display_name  = VALUES(display_name),
                        status        = VALUES(status);

-- Keep the user's id in a session variable (@name) for the next statements.
SET @user_id = (SELECT id FROM users WHERE email = 'traveler@test.local');

-- An empty wallet: everything at 0. INSERT IGNORE skips the row if the wallet already exists
-- (same primary key) instead of failing.
INSERT IGNORE INTO wallets (user_id, primogems, intertwined_fates, starglitter, stardust,
                            total_primogems_added, total_pulls, stardust_fates_month,
                            stardust_fates_bought)
VALUES (@user_id, 0, 0, 0, 0, 0, 0, NULL, 0);

-- A fresh pity: no pull yet, no guarantee.
INSERT IGNORE INTO pity_states (user_id, pool, pity5, pity4, guaranteed5, guaranteed4)
VALUES (@user_id, 'CHARACTER_EVENT', 0, 0, FALSE, FALSE);

-- Check, with a join across the three tables:
SELECT u.id, u.email, u.status, w.primogems, w.intertwined_fates, p.pity5, p.guaranteed5
FROM users u
         JOIN wallets w ON w.user_id = u.id
         JOIN pity_states p ON p.user_id = u.id;
