-- =============================================================================================
-- Cleanup — drop every table, in the REVERSE order of creation
-- =============================================================================================
-- A table cannot be dropped while another table's foreign key still points at it: drop the
-- children before the parents. (Try DROP TABLE users; first to see the error.)
-- The faster way is DROP DATABASE genshinwish_lesson; which removes everything at once.

USE genshinwish_lesson;

-- step 6
DROP TABLE IF EXISTS inventory;
-- step 5: wish_results points at wishes, so it goes first
DROP TABLE IF EXISTS wish_results;
DROP TABLE IF EXISTS wishes;
-- step 4
DROP TABLE IF EXISTS pity_states;
-- step 3
DROP TABLE IF EXISTS wallet_transactions;
DROP TABLE IF EXISTS wallets;
-- step 2: banner_items points at both banners and items
DROP TABLE IF EXISTS banner_items;
DROP TABLE IF EXISTS banners;
DROP TABLE IF EXISTS items;
-- step 1
DROP TABLE IF EXISTS refresh_tokens;
DROP TABLE IF EXISTS users;

SHOW TABLES;
