-- =============================================================================================
-- Step 7 — Fill the catalogue: items
-- =============================================================================================
-- INSERT adds rows. One INSERT can add many rows: VALUES (...), (...), ...;
-- We never give the id: AUTO_INCREMENT numbers them. The other tables will find an item by its
-- slug (step 8), so the numbers do not matter.
--
-- Generated from pull_simulator/data/items.json (the file the app loads), checked against the
-- Genshin Impact wiki. Note the escaped quote in 'Dragon''s Bane': inside a SQL string, a quote
-- is written twice.

USE genshinwish_lesson;

-- 5★ characters: the six Archons (featured) and the standard 5★ a lost 50/50 can give: 13 rows
INSERT INTO items (slug, name, name_zh, item_type, rarity, element, weapon_type, release_date)
VALUES
    ('diluc', 'Diluc', '迪卢克', 'CHARACTER', 5, 'PYRO', 'CLAYMORE', '2020-09-28'),
    ('jean', 'Jean', '琴', 'CHARACTER', 5, 'ANEMO', 'SWORD', '2020-09-28'),
    ('keqing', 'Keqing', '刻晴', 'CHARACTER', 5, 'ELECTRO', 'SWORD', '2020-09-28'),
    ('mona', 'Mona', '莫娜', 'CHARACTER', 5, 'HYDRO', 'CATALYST', '2020-09-28'),
    ('qiqi', 'Qiqi', '七七', 'CHARACTER', 5, 'CRYO', 'SWORD', '2020-09-28'),
    ('venti', 'Venti', '温迪', 'CHARACTER', 5, 'ANEMO', 'BOW', '2020-09-28'),
    ('zhongli', 'Zhongli', '钟离', 'CHARACTER', 5, 'GEO', 'POLEARM', '2020-12-01'),
    ('raiden-shogun', 'Raiden Shogun', '雷电将军', 'CHARACTER', 5, 'ELECTRO', 'POLEARM', '2021-09-01'),
    ('tighnari', 'Tighnari', '提纳里', 'CHARACTER', 5, 'DENDRO', 'BOW', '2022-08-24'),
    ('nahida', 'Nahida', '纳西妲', 'CHARACTER', 5, 'DENDRO', 'CATALYST', '2022-11-02'),
    ('dehya', 'Dehya', '迪希雅', 'CHARACTER', 5, 'PYRO', 'CLAYMORE', '2023-03-01'),
    ('furina', 'Furina', '芙宁娜', 'CHARACTER', 5, 'HYDRO', 'SWORD', '2023-11-08'),
    ('mavuika', 'Mavuika', '玛薇卡', 'CHARACTER', 5, 'PYRO', 'CLAYMORE', '2025-01-01');

-- 4★ characters: 39 rows
INSERT INTO items (slug, name, name_zh, item_type, rarity, element, weapon_type, release_date)
VALUES
    ('barbara', 'Barbara', '芭芭拉', 'CHARACTER', 4, 'HYDRO', 'CATALYST', '2020-09-28'),
    ('beidou', 'Beidou', '北斗', 'CHARACTER', 4, 'ELECTRO', 'CLAYMORE', '2020-09-28'),
    ('bennett', 'Bennett', '班尼特', 'CHARACTER', 4, 'PYRO', 'SWORD', '2020-09-28'),
    ('chongyun', 'Chongyun', '重云', 'CHARACTER', 4, 'CRYO', 'CLAYMORE', '2020-09-28'),
    ('fischl', 'Fischl', '菲谢尔', 'CHARACTER', 4, 'ELECTRO', 'BOW', '2020-09-28'),
    ('ningguang', 'Ningguang', '凝光', 'CHARACTER', 4, 'GEO', 'CATALYST', '2020-09-28'),
    ('noelle', 'Noelle', '诺艾尔', 'CHARACTER', 4, 'GEO', 'CLAYMORE', '2020-09-28'),
    ('razor', 'Razor', '雷泽', 'CHARACTER', 4, 'ELECTRO', 'CLAYMORE', '2020-09-28'),
    ('sucrose', 'Sucrose', '砂糖', 'CHARACTER', 4, 'ANEMO', 'CATALYST', '2020-09-28'),
    ('xiangling', 'Xiangling', '香菱', 'CHARACTER', 4, 'PYRO', 'POLEARM', '2020-09-28'),
    ('xingqiu', 'Xingqiu', '行秋', 'CHARACTER', 4, 'HYDRO', 'SWORD', '2020-09-28'),
    ('diona', 'Diona', '迪奥娜', 'CHARACTER', 4, 'CRYO', 'BOW', '2020-11-11'),
    ('xinyan', 'Xinyan', '辛焱', 'CHARACTER', 4, 'PYRO', 'CLAYMORE', '2020-12-01'),
    ('rosaria', 'Rosaria', '罗莎莉亚', 'CHARACTER', 4, 'CRYO', 'POLEARM', '2021-04-06'),
    ('yanfei', 'Yanfei', '烟绯', 'CHARACTER', 4, 'PYRO', 'CATALYST', '2021-04-28'),
    ('sayu', 'Sayu', '早柚', 'CHARACTER', 4, 'ANEMO', 'CLAYMORE', '2021-08-10'),
    ('kujou-sara', 'Kujou Sara', '九条裟罗', 'CHARACTER', 4, 'ELECTRO', 'BOW', '2021-09-01'),
    ('thoma', 'Thoma', '托马', 'CHARACTER', 4, 'PYRO', 'POLEARM', '2021-11-02'),
    ('gorou', 'Gorou', '五郎', 'CHARACTER', 4, 'GEO', 'BOW', '2021-12-14'),
    ('yun-jin', 'Yun Jin', '云堇', 'CHARACTER', 4, 'GEO', 'POLEARM', '2022-01-05'),
    ('kuki-shinobu', 'Kuki Shinobu', '久岐忍', 'CHARACTER', 4, 'ELECTRO', 'SWORD', '2022-06-21'),
    ('shikanoin-heizou', 'Shikanoin Heizou', '鹿野院平藏', 'CHARACTER', 4, 'ANEMO', 'CATALYST', '2022-07-13'),
    ('collei', 'Collei', '柯莱', 'CHARACTER', 4, 'DENDRO', 'BOW', '2022-08-24'),
    ('dori', 'Dori', '多莉', 'CHARACTER', 4, 'ELECTRO', 'CLAYMORE', '2022-09-09'),
    ('candace', 'Candace', '坎蒂丝', 'CHARACTER', 4, 'HYDRO', 'POLEARM', '2022-09-28'),
    ('layla', 'Layla', '莱依拉', 'CHARACTER', 4, 'CRYO', 'SWORD', '2022-11-18'),
    ('faruzan', 'Faruzan', '珐露珊', 'CHARACTER', 4, 'ANEMO', 'BOW', '2022-12-07'),
    ('yaoyao', 'Yaoyao', '瑶瑶', 'CHARACTER', 4, 'DENDRO', 'POLEARM', '2023-01-18'),
    ('mika', 'Mika', '米卡', 'CHARACTER', 4, 'CRYO', 'POLEARM', '2023-03-21'),
    ('kaveh', 'Kaveh', '卡维', 'CHARACTER', 4, 'DENDRO', 'CLAYMORE', '2023-05-02'),
    ('kirara', 'Kirara', '绮良良', 'CHARACTER', 4, 'DENDRO', 'SWORD', '2023-05-24'),
    ('lynette', 'Lynette', '琳妮特', 'CHARACTER', 4, 'ANEMO', 'SWORD', '2023-08-16'),
    ('freminet', 'Freminet', '菲米尼', 'CHARACTER', 4, 'CRYO', 'CLAYMORE', '2023-09-05'),
    ('charlotte', 'Charlotte', '夏洛蒂', 'CHARACTER', 4, 'CRYO', 'CATALYST', '2023-11-08'),
    ('chevreuse', 'Chevreuse', '夏沃蕾', 'CHARACTER', 4, 'PYRO', 'POLEARM', '2024-01-09'),
    ('gaming', 'Gaming', '嘉明', 'CHARACTER', 4, 'PYRO', 'CLAYMORE', '2024-01-31'),
    ('sethos', 'Sethos', '赛索斯', 'CHARACTER', 4, 'ELECTRO', 'BOW', '2024-06-05'),
    ('kachina', 'Kachina', '卡齐娜', 'CHARACTER', 4, 'GEO', 'POLEARM', '2024-08-28'),
    ('ororon', 'Ororon', '欧洛伦', 'CHARACTER', 4, 'ELECTRO', 'BOW', '2024-11-20');

-- 4★ weapons (the same 18 on every banner): 18 rows
INSERT INTO items (slug, name, name_zh, item_type, rarity, element, weapon_type, release_date)
VALUES
    ('dragons-bane', 'Dragon''s Bane', '匣里灭辰', 'WEAPON', 4, NULL, 'POLEARM', '2020-09-28'),
    ('eye-of-perception', 'Eye of Perception', '昭心', 'WEAPON', 4, NULL, 'CATALYST', '2020-09-28'),
    ('favonius-codex', 'Favonius Codex', '西风秘典', 'WEAPON', 4, NULL, 'CATALYST', '2020-09-28'),
    ('favonius-greatsword', 'Favonius Greatsword', '西风大剑', 'WEAPON', 4, NULL, 'CLAYMORE', '2020-09-28'),
    ('favonius-lance', 'Favonius Lance', '西风长枪', 'WEAPON', 4, NULL, 'POLEARM', '2020-09-28'),
    ('favonius-sword', 'Favonius Sword', '西风剑', 'WEAPON', 4, NULL, 'SWORD', '2020-09-28'),
    ('favonius-warbow', 'Favonius Warbow', '西风猎弓', 'WEAPON', 4, NULL, 'BOW', '2020-09-28'),
    ('lions-roar', 'Lion''s Roar', '匣里龙吟', 'WEAPON', 4, NULL, 'SWORD', '2020-09-28'),
    ('rainslasher', 'Rainslasher', '雨裁', 'WEAPON', 4, NULL, 'CLAYMORE', '2020-09-28'),
    ('rust', 'Rust', '弓藏', 'WEAPON', 4, NULL, 'BOW', '2020-09-28'),
    ('sacrificial-bow', 'Sacrificial Bow', '祭礼弓', 'WEAPON', 4, NULL, 'BOW', '2020-09-28'),
    ('sacrificial-fragments', 'Sacrificial Fragments', '祭礼残章', 'WEAPON', 4, NULL, 'CATALYST', '2020-09-28'),
    ('sacrificial-greatsword', 'Sacrificial Greatsword', '祭礼大剑', 'WEAPON', 4, NULL, 'CLAYMORE', '2020-09-28'),
    ('sacrificial-sword', 'Sacrificial Sword', '祭礼剑', 'WEAPON', 4, NULL, 'SWORD', '2020-09-28'),
    ('the-bell', 'The Bell', '钟剑', 'WEAPON', 4, NULL, 'CLAYMORE', '2020-09-28'),
    ('the-flute', 'The Flute', '笛剑', 'WEAPON', 4, NULL, 'SWORD', '2020-09-28'),
    ('the-stringless', 'The Stringless', '绝弦', 'WEAPON', 4, NULL, 'BOW', '2020-09-28'),
    ('the-widsith', 'The Widsith', '流浪乐章', 'WEAPON', 4, NULL, 'CATALYST', '2020-09-28');

-- 3★ weapons (the same 13 on every banner): 13 rows
INSERT INTO items (slug, name, name_zh, item_type, rarity, element, weapon_type, release_date)
VALUES
    ('black-tassel', 'Black Tassel', '黑缨枪', 'WEAPON', 3, NULL, 'POLEARM', '2020-09-28'),
    ('bloodtainted-greatsword', 'Bloodtainted Greatsword', '沐浴龙血的剑', 'WEAPON', 3, NULL, 'CLAYMORE', '2020-09-28'),
    ('cool-steel', 'Cool Steel', '冷刃', 'WEAPON', 3, NULL, 'SWORD', '2020-09-28'),
    ('debate-club', 'Debate Club', '以理服人', 'WEAPON', 3, NULL, 'CLAYMORE', '2020-09-28'),
    ('emerald-orb', 'Emerald Orb', '翡玉法球', 'WEAPON', 3, NULL, 'CATALYST', '2020-09-28'),
    ('ferrous-shadow', 'Ferrous Shadow', '铁影阔剑', 'WEAPON', 3, NULL, 'CLAYMORE', '2020-09-28'),
    ('harbinger-of-dawn', 'Harbinger of Dawn', '黎明神剑', 'WEAPON', 3, NULL, 'SWORD', '2020-09-28'),
    ('magic-guide', 'Magic Guide', '魔导绪论', 'WEAPON', 3, NULL, 'CATALYST', '2020-09-28'),
    ('raven-bow', 'Raven Bow', '鸦羽弓', 'WEAPON', 3, NULL, 'BOW', '2020-09-28'),
    ('sharpshooters-oath', 'Sharpshooter''s Oath', '神射手之誓', 'WEAPON', 3, NULL, 'BOW', '2020-09-28'),
    ('skyrider-sword', 'Skyrider Sword', '飞天御剑', 'WEAPON', 3, NULL, 'SWORD', '2020-09-28'),
    ('slingshot', 'Slingshot', '弹弓', 'WEAPON', 3, NULL, 'BOW', '2020-09-28'),
    ('thrilling-tales-of-dragon-slayers', 'Thrilling Tales of Dragon Slayers', '讨龙英杰谭', 'WEAPON', 3, NULL, 'CATALYST', '2020-09-28');

-- Picture paths follow a fixed pattern, so one UPDATE fills all of them at once.
-- CONCAT glues strings; the WHERE limits the splash art to characters (weapons have none).
-- (The app only fills a path once the picture is really downloaded, see app/tools.)
UPDATE items SET icon_path = CONCAT('items/', slug, '/icon.png');
UPDATE items SET splash_path = CONCAT('items/', slug, '/splash.png') WHERE item_type = 'CHARACTER';

-- Check: 83 items, split by type and rarity.
SELECT item_type, rarity, COUNT(*) AS items
FROM items
GROUP BY item_type, rarity
ORDER BY item_type, rarity DESC;
