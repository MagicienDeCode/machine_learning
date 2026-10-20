"""Regenerates 07_seed_items.sql and 08_seed_banners.sql from pull_simulator/data/*.json.

    python3 generate_seed_sql.py
"""

import json
from pathlib import Path

out = Path(__file__).resolve().parent
data = out.parent / "pull_simulator" / "data"
items = json.load(open(f"{data}/items.json")); banners = json.load(open(f"{data}/banners.json"))
q = lambda v: "NULL" if v is None else "'" + str(v).replace("'", "''") + "'"
weapons = {i["slug"] for i in items if i["item_type"] == "WEAPON"}

groups = [
    ("5★ characters: the six Archons (featured) and the standard 5★ a lost 50/50 can give",
     lambda i: i["item_type"] == "CHARACTER" and i["rarity"] == 5),
    ("4★ characters", lambda i: i["item_type"] == "CHARACTER" and i["rarity"] == 4),
    ("4★ weapons (the same 18 on every banner)", lambda i: i["item_type"] == "WEAPON" and i["rarity"] == 4),
    ("3★ weapons (the same 13 on every banner)", lambda i: i["item_type"] == "WEAPON" and i["rarity"] == 3),
]
L = ["""-- =============================================================================================
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
"""]
cols = "(slug, name, name_zh, item_type, rarity, element, weapon_type, release_date)"
for title, pred in groups:
    rows = sorted((i for i in items if pred(i)), key=lambda i: (i["release_date"], i["name"]))
    L.append(f"-- {title}: {len(rows)} rows")
    L.append(f"INSERT INTO items {cols}\nVALUES")
    vals = [f"    ({q(i['slug'])}, {q(i['name'])}, {q(i.get('name_zh'))}, {q(i['item_type'])}, "
            f"{i['rarity']}, {q(i.get('element'))}, {q(i['weapon_type'])}, {q(i['release_date'])})"
            for i in rows]
    L.append(",\n".join(vals) + ";\n")
L.append("""-- Picture paths follow a fixed pattern, so one UPDATE fills all of them at once.
-- CONCAT glues strings; the WHERE limits the splash art to characters (weapons have none).
-- (The app only fills a path once the picture is really downloaded, see app/tools.)
UPDATE items SET icon_path = CONCAT('items/', slug, '/icon.png');
UPDATE items SET splash_path = CONCAT('items/', slug, '/splash.png') WHERE item_type = 'CHARACTER';

-- Check: 83 items, split by type and rarity.
SELECT item_type, rarity, COUNT(*) AS items
FROM items
GROUP BY item_type, rarity
ORDER BY item_type, rarity DESC;
""")
open(f"{out}/07_seed_items.sql", "w").write("\n".join(L))

B = ["""-- =============================================================================================
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
VALUES"""]
B.append(",\n".join(
    f"    ({q(b['code'])}, {q(b['name'])}, {q(b.get('name_zh'))}, {q(b['version'])}, {b['phase']}, "
    f"{b['slot']}, {q(b['start_date'])}, {q(b['end_date'])})" for b in banners) + ";\n")
B.append("""-- Every banner offers the same 31 standard weapons (18 4★ + 13 3★): one statement for all
-- of them. The join condition does not mention banners at all, so EVERY banner is paired with
-- EVERY weapon (6 × 31 = 186 rows).
INSERT INTO banner_items (banner_id, item_id, role)
SELECT b.id, i.id, 'STANDARD'
FROM banners b
         JOIN items i ON i.item_type = 'WEAPON';
""")
for b in banners:
    assert set(b["off_banner4_weapons"]) | set(b["three_star_weapons"]) == weapons, b["code"]
    feat = [b["featured5"], *b["featured4"]]
    std = b["standard5"] + b["off_banner4_characters"]
    lst = lambda xs: ",\n".join(
        "                   " + ", ".join(q(x) for x in xs[k:k + 6]) for k in range(0, len(xs), 6))
    B.append(f"""-- {b['code']} — {b['name']}{' / ' + b['name_zh'] if b.get('name_zh') else ''} ({b['start_date']} → {b['end_date']})
-- Featured: the 5★ and the three 4★ of the rate-up.
INSERT INTO banner_items (banner_id, item_id, role)
SELECT b.id, i.id, 'FEATURED'
FROM banners b
         JOIN items i ON i.slug IN (
{lst(feat)})
WHERE b.code = {q(b['code'])};

-- Standard characters: {len(b['standard5'])} 5★ for a lost 50/50, {len(b['off_banner4_characters'])} off-banner 4★.
INSERT INTO banner_items (banner_id, item_id, role)
SELECT b.id, i.id, 'STANDARD'
FROM banners b
         JOIN items i ON i.slug IN (
{lst(std)})
WHERE b.code = {q(b['code'])};
""")
B.append("""-- Check: the size of every pool. Later banners have bigger pools: each new 4★ character joins
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
""")
open(f"{out}/08_seed_banners.sql", "w").write("\n".join(B))
print("07_seed_items.sql and 08_seed_banners.sql written.")
