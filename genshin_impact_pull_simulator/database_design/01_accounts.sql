-- =============================================================================================
-- Step 1 — Accounts: users and refresh_tokens
-- =============================================================================================
-- We start with the tables that depend on nothing: everything else will point at users.
--
-- Concepts: PRIMARY KEY, AUTO_INCREMENT, NOT NULL, UNIQUE, ENUM, DEFAULT, FOREIGN KEY,
--           ON DELETE CASCADE.

USE genshinwish_lesson;

-- ---------------------------------------------------------------------------------------------
-- users — one row per login account. There is no sign-up page: rows are inserted by the seed
-- (step 9).
-- ---------------------------------------------------------------------------------------------
CREATE TABLE users
(
    -- BIGINT AUTO_INCREMENT: MySQL gives each new row the next number (1, 2, 3...).
    id            BIGINT       NOT NULL AUTO_INCREMENT,
    -- UNIQUE below: two accounts can never share an email (the login looks rows up by email).
    email         VARCHAR(255) NOT NULL,
    -- Never the password itself: its bcrypt hash ("$2b$10$..."), checked by the backend.
    password_hash VARCHAR(255) NOT NULL,
    display_name  VARCHAR(150) NOT NULL,
    -- ENUM: the column only accepts these values. Anything else is an error.
    status        ENUM ('ACTIVE','DISABLED') NOT NULL,
    -- DEFAULT: filled in by MySQL when the INSERT does not give a value.
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    CONSTRAINT uq_users_email UNIQUE (email)
) ENGINE = InnoDB;   -- InnoDB: the engine with transactions and foreign keys (MySQL's default).

-- ---------------------------------------------------------------------------------------------
-- refresh_tokens — the "stay logged in" tokens. Only a SHA-256 hash is stored (64 hex chars),
-- so a stolen copy of the database cannot be used to log in.
-- ---------------------------------------------------------------------------------------------
CREATE TABLE refresh_tokens
(
    id         BIGINT   NOT NULL AUTO_INCREMENT,
    user_id    BIGINT   NOT NULL,
    -- CHAR(64): fixed length, a SHA-256 in hex is always exactly 64 characters.
    token_hash CHAR(64) NOT NULL,
    expires_at DATETIME NOT NULL,
    -- NULL = still valid; a date = revoked (logout, or replaced by a newer token).
    revoked_at DATETIME NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    CONSTRAINT uq_refresh_tokens_hash UNIQUE (token_hash),
    -- FOREIGN KEY: user_id must be the id of an existing user.
    -- ON DELETE CASCADE: deleting the user deletes their tokens too.
    CONSTRAINT fk_refresh_tokens_user FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE = InnoDB;

-- Check what we built:
--   SHOW TABLES;
--   DESCRIBE users;
--   SHOW CREATE TABLE refresh_tokens;
