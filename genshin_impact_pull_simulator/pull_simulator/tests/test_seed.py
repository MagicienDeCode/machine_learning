import re
from pathlib import Path

from app.shared.security import verify_password

SEED_SQL = Path(__file__).resolve().parent.parent / "script" / "dev" / "seed-test-users.sql"


def test_seeded_password_hash_is_123456():
    password_hash = re.search(r"'(\$2[aby]\$\d+\$[^']+)'", SEED_SQL.read_text()).group(1)
    assert verify_password("123456", password_hash)
