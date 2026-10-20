from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Environment variables. The defaults only suit local dev (see script/dev/docker-db.sh)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    db_host: str = "localhost"
    db_port: int = 3308
    db_name: str = "genshinwish"
    db_username: str = "genshinwish"
    db_password: str = "genshinwish"
    # Overrides the five above when set (the tests use an in-memory SQLite).
    database_url: str | None = None

    jwt_secret: str = "local-dev-only-secret-override-me-do-not-use-in-production"
    jwt_access_ttl_minutes: int = 15
    jwt_refresh_ttl_days: int = 7
    jwt_cookie_secure: bool = False

    cors_allowed_origins: str = "http://localhost:5173"

    static_dir: Path = BASE_DIR / "static"
    data_dir: Path = BASE_DIR / "data"

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"mysql+pymysql://{self.db_username}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}?charset=utf8mb4"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
