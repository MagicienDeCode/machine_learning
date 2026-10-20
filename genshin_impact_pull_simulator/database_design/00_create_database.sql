-- =============================================================================================
-- Step 0 — Create the database
-- =============================================================================================
-- A database is a named container for tables. We pick:
--   * utf8mb4             -> real UTF-8 (4 bytes per char), needed for Chinese names (雷电将军)
--                            and symbols like ★. MySQL's old "utf8" only stores 3 bytes: avoid it.
--   * utf8mb4_0900_ai_ci  -> how text is compared and sorted: accent-insensitive (ai) and
--                            case-insensitive (ci), so 'Venti' = 'venti' in a WHERE.
--
-- The lesson uses its own database, genshinwish_lesson, so it never touches the app's data
-- (genshinwish). Run as root, see run.sh.

DROP DATABASE IF EXISTS genshinwish_lesson;

CREATE DATABASE genshinwish_lesson
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_0900_ai_ci;

-- The application connects with its own user, never root. It only needs rights on its database.
CREATE USER IF NOT EXISTS 'genshinwish'@'%' IDENTIFIED BY 'genshinwish';
GRANT ALL PRIVILEGES ON genshinwish_lesson.* TO 'genshinwish'@'%';
FLUSH PRIVILEGES;

-- Every following file starts with USE genshinwish_lesson; to work inside it.
