-- Dev/test seed data: there is no self-service registration, so the login account is created here.
--
-- Run against your LOCAL dev database only: the password below is public.
--
-- Usage (with the mysql container running and the tables created -- see init-db.sh):
--   ./script/dev/seed-test-users.sh
--
-- Re-running this script is safe: it upserts by email, resetting the row back to this known-good
-- state. Its wallet, pity and history are left untouched (POST /api/users/me/reset wipes them).
--
-- The password hash below is `123456` through bcrypt, as app/shared/security.py checks it; the
-- tests check it too (tests/test_seed.py).

INSERT INTO users (email, password_hash, display_name, status, created_at, updated_at)
VALUES
    ('traveler@test.local', '$2b$10$r0DarDL52103Bvk.Jf06uOROk5z1APlFPm/OzaCVK6.DXqmMURh9e', 'Traveler', 'ACTIVE', NOW(), NOW())
ON DUPLICATE KEY UPDATE
    password_hash = VALUES(password_hash),
    display_name = VALUES(display_name),
    status = VALUES(status),
    updated_at = NOW();
