from datetime import UTC, datetime


def utcnow() -> datetime:
    """Naive UTC, as stored in the DATETIME columns."""
    return datetime.now(UTC).replace(tzinfo=None)
