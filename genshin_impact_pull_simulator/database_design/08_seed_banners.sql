-- =============================================================================================
-- Step 8 — Fill the catalogue: banners and banner_items
-- =============================================================================================
-- First the 6 banners, then the pool of each one.
--
-- banner_items needs ids (banner_id, item_id), but we only know codes and slugs. Instead of
-- looking ids up by hand, INSERT ... SELECT inserts the rows a query returns: the query joins
-- banners and items and keeps the pairs we want.
--
-- Generated from pull_simulator/data/banners.json; each pool is the one listed on the banner's
-- page of the Genshin Impact wiki.

USE genshinwish_lesson;

INSERT INTO banners (code, name, name_zh, version, phase, slot, start_date, end_date)
VALUES
    ('1.0-1-venti', 'Ballad in Goblets', '杯装之诗', '1.0', 1, 1, '2020-09-28', '2020-10-18'),
    ('1.1-2-zhongli', 'Gentry of Hermitage', '陵薮市朝', '1.1', 2, 1, '2020-12-01', '2020-12-22'),
    ('2.1-1-raiden-shogun', 'Reign of Serenity', '影寂天下人', '2.1', 1, 1, '2021-09-01', '2021-09-21'),
    ('3.2-1-nahida', 'The Moongrass'' Enlightenment', '月草的赐慧', '3.2', 1, 1, '2022-11-02', '2022-11-18'),
    ('4.2-1-furina', 'Chanson of Many Waters', '众水的颂诗', '4.2', 1, 1, '2023-11-08', '2023-11-28'),
    ('5.3-1-mavuika', 'Ancient Flame Ablaze', NULL, '5.3', 1, 1, '2025-01-01', '2025-01-21');

-- Every banner offers the same 31 standard weapons (18 4★ + 13 3★): one statement for all
-- of them. The join condition does not mention banners at all, so EVERY banner is paired with
-- EVERY weapon (6 × 31 = 186 rows).
INSERT INTO banner_items (banner_id, item_id, role)
SELECT b.id, i.id, 'STANDARD'
FROM banners b
         JOIN items i ON i.item_type = 'WEAPON';

-- 1.0-1-venti — Ballad in Goblets / 杯装之诗 (2020-09-28 → 2020-10-18)
-- Featured: the 5★ and the three 4★ of the rate-up.
INSERT INTO banner_items (banner_id, item_id, role)
SELECT b.id, i.id, 'FEATURED'
FROM banners b
         JOIN items i ON i.slug IN (
                   'venti', 'barbara', 'fischl', 'xiangling')
WHERE b.code = '1.0-1-venti';

-- Standard characters: 5 5★ for a lost 50/50, 8 off-banner 4★.
INSERT INTO banner_items (banner_id, item_id, role)
SELECT b.id, i.id, 'STANDARD'
FROM banners b
         JOIN items i ON i.slug IN (
                   'diluc', 'jean', 'keqing', 'mona', 'qiqi', 'beidou',
                   'bennett', 'chongyun', 'ningguang', 'noelle', 'razor', 'sucrose',
                   'xingqiu')
WHERE b.code = '1.0-1-venti';

-- 1.1-2-zhongli — Gentry of Hermitage / 陵薮市朝 (2020-12-01 → 2020-12-22)
-- Featured: the 5★ and the three 4★ of the rate-up.
INSERT INTO banner_items (banner_id, item_id, role)
SELECT b.id, i.id, 'FEATURED'
FROM banners b
         JOIN items i ON i.slug IN (
                   'zhongli', 'xinyan', 'razor', 'chongyun')
WHERE b.code = '1.1-2-zhongli';

-- Standard characters: 5 5★ for a lost 50/50, 9 off-banner 4★.
INSERT INTO banner_items (banner_id, item_id, role)
SELECT b.id, i.id, 'STANDARD'
FROM banners b
         JOIN items i ON i.slug IN (
                   'diluc', 'jean', 'keqing', 'mona', 'qiqi', 'barbara',
                   'beidou', 'bennett', 'fischl', 'ningguang', 'noelle', 'sucrose',
                   'xiangling', 'xingqiu')
WHERE b.code = '1.1-2-zhongli';

-- 2.1-1-raiden-shogun — Reign of Serenity / 影寂天下人 (2021-09-01 → 2021-09-21)
-- Featured: the 5★ and the three 4★ of the rate-up.
INSERT INTO banner_items (banner_id, item_id, role)
SELECT b.id, i.id, 'FEATURED'
FROM banners b
         JOIN items i ON i.slug IN (
                   'raiden-shogun', 'kujou-sara', 'xiangling', 'sucrose')
WHERE b.code = '2.1-1-raiden-shogun';

-- Standard characters: 5 5★ for a lost 50/50, 14 off-banner 4★.
INSERT INTO banner_items (banner_id, item_id, role)
SELECT b.id, i.id, 'STANDARD'
FROM banners b
         JOIN items i ON i.slug IN (
                   'diluc', 'jean', 'keqing', 'mona', 'qiqi', 'barbara',
                   'beidou', 'bennett', 'chongyun', 'diona', 'fischl', 'ningguang',
                   'noelle', 'razor', 'rosaria', 'sayu', 'xingqiu', 'xinyan',
                   'yanfei')
WHERE b.code = '2.1-1-raiden-shogun';

-- 3.2-1-nahida — The Moongrass' Enlightenment / 月草的赐慧 (2022-11-02 → 2022-11-18)
-- Featured: the 5★ and the three 4★ of the rate-up.
INSERT INTO banner_items (banner_id, item_id, role)
SELECT b.id, i.id, 'FEATURED'
FROM banners b
         JOIN items i ON i.slug IN (
                   'nahida', 'razor', 'noelle', 'bennett')
WHERE b.code = '3.2-1-nahida';

-- Standard characters: 6 5★ for a lost 50/50, 22 off-banner 4★.
INSERT INTO banner_items (banner_id, item_id, role)
SELECT b.id, i.id, 'STANDARD'
FROM banners b
         JOIN items i ON i.slug IN (
                   'diluc', 'jean', 'keqing', 'mona', 'qiqi', 'tighnari',
                   'barbara', 'beidou', 'candace', 'chongyun', 'collei', 'diona',
                   'dori', 'fischl', 'gorou', 'kujou-sara', 'kuki-shinobu', 'ningguang',
                   'rosaria', 'sayu', 'shikanoin-heizou', 'sucrose', 'thoma', 'xiangling',
                   'xingqiu', 'xinyan', 'yanfei', 'yun-jin')
WHERE b.code = '3.2-1-nahida';

-- 4.2-1-furina — Chanson of Many Waters / 众水的颂诗 (2023-11-08 → 2023-11-28)
-- Featured: the 5★ and the three 4★ of the rate-up.
INSERT INTO banner_items (banner_id, item_id, role)
SELECT b.id, i.id, 'FEATURED'
FROM banners b
         JOIN items i ON i.slug IN (
                   'furina', 'charlotte', 'collei', 'beidou')
WHERE b.code = '4.2-1-furina';

-- Standard characters: 7 5★ for a lost 50/50, 31 off-banner 4★.
INSERT INTO banner_items (banner_id, item_id, role)
SELECT b.id, i.id, 'STANDARD'
FROM banners b
         JOIN items i ON i.slug IN (
                   'dehya', 'diluc', 'jean', 'keqing', 'mona', 'qiqi',
                   'tighnari', 'barbara', 'bennett', 'candace', 'chongyun', 'dori',
                   'diona', 'faruzan', 'fischl', 'freminet', 'gorou', 'kaveh',
                   'kirara', 'kujou-sara', 'kuki-shinobu', 'layla', 'lynette', 'mika',
                   'ningguang', 'noelle', 'razor', 'rosaria', 'sayu', 'shikanoin-heizou',
                   'sucrose', 'thoma', 'xiangling', 'xingqiu', 'xinyan', 'yanfei',
                   'yaoyao', 'yun-jin')
WHERE b.code = '4.2-1-furina';

-- 5.3-1-mavuika — Ancient Flame Ablaze (2025-01-01 → 2025-01-21)
-- Featured: the 5★ and the three 4★ of the rate-up.
INSERT INTO banner_items (banner_id, item_id, role)
SELECT b.id, i.id, 'FEATURED'
FROM banners b
         JOIN items i ON i.slug IN (
                   'mavuika', 'kachina', 'bennett', 'diona')
WHERE b.code = '5.3-1-mavuika';

-- Standard characters: 7 5★ for a lost 50/50, 36 off-banner 4★.
INSERT INTO banner_items (banner_id, item_id, role)
SELECT b.id, i.id, 'STANDARD'
FROM banners b
         JOIN items i ON i.slug IN (
                   'dehya', 'diluc', 'jean', 'keqing', 'mona', 'qiqi',
                   'tighnari', 'barbara', 'beidou', 'candace', 'charlotte', 'chevreuse',
                   'chongyun', 'collei', 'dori', 'faruzan', 'fischl', 'freminet',
                   'gaming', 'gorou', 'kaveh', 'kirara', 'kujou-sara', 'kuki-shinobu',
                   'layla', 'lynette', 'mika', 'ningguang', 'noelle', 'ororon',
                   'razor', 'rosaria', 'sayu', 'sethos', 'shikanoin-heizou', 'sucrose',
                   'thoma', 'xiangling', 'xingqiu', 'xinyan', 'yanfei', 'yaoyao',
                   'yun-jin')
WHERE b.code = '5.3-1-mavuika';

-- Check: the size of every pool. Later banners have bigger pools: each new 4★ character joins
-- the standard pool after their own debut banner.
SELECT b.code,
       SUM(bi.role = 'FEATURED')                                       AS featured,
       SUM(bi.role = 'STANDARD' AND i.rarity = 5)                      AS standard_5,
       SUM(bi.role = 'STANDARD' AND i.rarity = 4 AND i.item_type = 'CHARACTER') AS standard_4_characters,
       SUM(i.item_type = 'WEAPON')                                     AS weapons,
       COUNT(*)                                                        AS total
FROM banners b
         JOIN banner_items bi ON bi.banner_id = b.id
         JOIN items i ON i.id = bi.item_id
GROUP BY b.id, b.code, b.start_date
ORDER BY b.start_date;
