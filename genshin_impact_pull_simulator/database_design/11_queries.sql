-- =============================================================================================
-- Step 11 — Reading: the queries behind each page
-- =============================================================================================
-- Each query below is (a simpler version of) what the backend runs for one page or API call.
-- Run step 10 first, so there is a wish to look at.
--
-- Concepts: JOIN, WHERE, ORDER BY, LIMIT / OFFSET, GROUP BY, aggregates (COUNT, SUM, AVG, MIN,
--           MAX), CASE, conditional aggregation, subquery, EXPLAIN.

USE genshinwish_lesson;

SET @user_id = (SELECT id FROM users WHERE email = 'traveler@test.local');

-- ---------------------------------------------------------------------------------------------
-- 11.1 Banners page (GET /api/banners): each banner with its featured 5★, newest first.
-- ---------------------------------------------------------------------------------------------
SELECT b.code, b.name, b.name_zh, b.version, b.start_date, i.name AS featured_5_star
FROM banners b
         JOIN banner_items bi ON bi.banner_id = b.id AND bi.role = 'FEATURED'
         JOIN items i ON i.id = bi.item_id AND i.rarity = 5
ORDER BY b.start_date DESC;

-- ---------------------------------------------------------------------------------------------
-- 11.2 Banner details (GET /api/banners/{code}/details): the whole pool, grouped like in game.
-- CASE turns each row into a label; GROUP_CONCAT joins the names of each group.
-- ---------------------------------------------------------------------------------------------
SELECT CASE
           WHEN bi.role = 'FEATURED' THEN CONCAT('featured ', i.rarity, '★')
           WHEN i.rarity = 5 THEN 'standard 5★'
           WHEN i.rarity = 4 AND i.item_type = 'CHARACTER' THEN '4★ characters'
           WHEN i.rarity = 4 THEN '4★ weapons'
           ELSE '3★ weapons'
           END                                          AS pool_group,
       COUNT(*)                                         AS items,
       GROUP_CONCAT(i.name ORDER BY i.name SEPARATOR ', ') AS names
FROM banner_items bi
         JOIN banners b ON b.id = bi.banner_id
         JOIN items i ON i.id = bi.item_id
WHERE b.code = '2.1-1-raiden-shogun'
GROUP BY pool_group
ORDER BY MIN(i.rarity) DESC, MAX(bi.role = 'FEATURED') DESC, MIN(i.item_type);

-- ---------------------------------------------------------------------------------------------
-- 11.3 Wish history (GET /api/wishes/history?rarity=5&page=0&size=20)
-- LIMIT = page size, OFFSET = page × size.
-- ---------------------------------------------------------------------------------------------
SELECT r.created_at, i.name, i.name_zh, r.rarity, r.pity, r.outcome, b.name AS banner
FROM wish_results r
         JOIN items i ON i.id = r.item_id
         JOIN banners b ON b.id = r.banner_id
WHERE r.user_id = @user_id
  AND r.rarity = 5
ORDER BY r.id DESC
LIMIT 20 OFFSET 0;

-- EXPLAIN shows HOW MySQL runs a query. "possible_keys" lists both indexes made in step 5,
-- "key" the one it picked, "type = ref": it jumps straight to this user's rows instead of
-- reading the whole table (type = ALL). With so few rows, either index is as good.
EXPLAIN
SELECT r.id FROM wish_results r WHERE r.user_id = @user_id AND r.rarity = 5 ORDER BY r.id DESC LIMIT 20;

-- ---------------------------------------------------------------------------------------------
-- 11.4 Statistics (GET /api/wishes/stats), in ONE pass over the history.
-- Conditional aggregation: SUM(condition) counts the rows where it is true (TRUE = 1), and
-- AVG(CASE WHEN ... THEN x END) averages only those rows (the others give NULL, which AVG skips).
-- ---------------------------------------------------------------------------------------------
SELECT COUNT(*)                                          AS total_pulls,
       COUNT(*) * 160                                    AS primogems_spent,
       SUM(rarity = 5)                                   AS five_stars,
       SUM(rarity = 4)                                   AS four_stars,
       ROUND(AVG(CASE WHEN rarity = 5 THEN pity END), 1) AS average_5_star_pity,
       MIN(CASE WHEN rarity = 5 THEN pity END)           AS luckiest_5_star,
       MAX(CASE WHEN rarity = 5 THEN pity END)           AS unluckiest_5_star,
       SUM(rarity = 5 AND outcome = 'WON_50_50')         AS fifty_fifty_won,
       SUM(rarity = 5 AND outcome = 'LOST_50_50')        AS fifty_fifty_lost,
       SUM(rarity = 5 AND outcome = 'GUARANTEED')        AS guaranteed
FROM wish_results
WHERE user_id = @user_id;

-- ---------------------------------------------------------------------------------------------
-- 11.5 Characters page (GET /api/inventory?type=CHARACTER)
-- The constellation is computed, not stored (step 6): LEAST(copies - 1, 6).
-- ---------------------------------------------------------------------------------------------
SELECT i.name, i.name_zh, i.rarity, inv.copies, LEAST(inv.copies - 1, 6) AS constellation
FROM inventory inv
         JOIN items i ON i.id = inv.item_id
WHERE inv.user_id = @user_id
  AND i.item_type = 'CHARACTER'
ORDER BY i.rarity DESC, i.name;

-- ---------------------------------------------------------------------------------------------
-- 11.6 Pity panel (GET /api/pity): stored state + a value computed from it.
-- The 5★ rate of the next pull: 0.6 %, +6 % per pull from pull 74, certain on pull 90.
-- ---------------------------------------------------------------------------------------------
SELECT pity5,
       pity4,
       guaranteed5,
       90 - pity5 AS pulls_to_hard_pity,
       CASE
           WHEN pity5 + 1 >= 90 THEN 1.0
           WHEN pity5 + 1 < 74 THEN 0.006
           ELSE LEAST(1.0, 0.006 + 0.06 * (pity5 + 1 - 73))
           END    AS next_5_star_rate
FROM pity_states
WHERE user_id = @user_id;

-- ---------------------------------------------------------------------------------------------
-- 11.7 A subquery: which 4★ characters can come off-banner on Furina's banner but not on
-- Venti's? (the 4★ characters released in between)
-- ---------------------------------------------------------------------------------------------
SELECT i.name, i.release_date
FROM banner_items bi
         JOIN banners b ON b.id = bi.banner_id
         JOIN items i ON i.id = bi.item_id
WHERE b.code = '4.2-1-furina'
  AND bi.role = 'STANDARD'
  AND i.rarity = 4
  AND i.item_type = 'CHARACTER'
  AND i.id NOT IN (SELECT bi2.item_id
                   FROM banner_items bi2
                            JOIN banners b2 ON b2.id = bi2.banner_id
                   WHERE b2.code = '1.0-1-venti')
ORDER BY i.release_date;
