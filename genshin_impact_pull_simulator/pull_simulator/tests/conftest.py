import os

# Before anything imports app.db: the tests never touch MySQL.
os.environ["DATABASE_URL"] = "sqlite://"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base, get_session
from app.main import app
from app.models import User
from app.repositories import catalog_repository
from app.shared.security import hash_password
from app.tools.seed_catalog import load_catalog

EMAIL = "traveler@test.local"
PASSWORD = "123456"


@pytest.fixture
def session_factory():
    # One shared in-memory database per test.
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    with factory.begin() as session:
        load_catalog(session)
        session.add(User(email=EMAIL, password_hash=hash_password(PASSWORD), display_name="T"))
    catalog_repository.clear_pool_cache()
    yield factory
    engine.dispose()


@pytest.fixture
def client(session_factory):
    def override():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = override
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def auth(client):
    """Headers of a logged-in test user."""
    response = client.post("/api/auth/login", json={"email": EMAIL, "password": PASSWORD})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}
