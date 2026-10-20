from pydantic import BaseModel


class Page[T](BaseModel):
    items: list[T]
    page: int
    size: int
    total: int
