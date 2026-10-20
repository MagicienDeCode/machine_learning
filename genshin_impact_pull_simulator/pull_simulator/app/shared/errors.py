"""Every service raises these; one handler turns them into anti-gamble's error body."""

from datetime import UTC, datetime
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class AppError(Exception):
    status = HTTPStatus.INTERNAL_SERVER_ERROR

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class BadRequest(AppError):
    status = HTTPStatus.BAD_REQUEST


class Unauthorized(AppError):
    status = HTTPStatus.UNAUTHORIZED


class InsufficientFunds(AppError):
    status = HTTPStatus.PAYMENT_REQUIRED


class NotFound(AppError):
    status = HTTPStatus.NOT_FOUND


class ExchangeLimitReached(AppError):
    status = HTTPStatus.CONFLICT


def error_body(status: HTTPStatus, message: str, request: Request) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "timestamp": datetime.now(UTC).isoformat(),
            "status": status.value,
            "error": status.phrase,
            "message": message,
            "path": request.url.path,
        },
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return error_body(exc.status, exc.message, request)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        message = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}" for e in exc.errors()
        )
        return error_body(HTTPStatus.BAD_REQUEST, message, request)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return error_body(HTTPStatus(exc.status_code), str(exc.detail), request)
