from datetime import datetime

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    email: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1)


class AccessTokenView(BaseModel):
    access_token: str
    access_token_expires_at: datetime


class UserView(BaseModel):
    id: int
    email: str
    display_name: str
