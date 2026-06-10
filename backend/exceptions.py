"""Structured exception classes for the Michi API.

Provides consistent error responses across all routers with proper
HTTP status codes and machine-readable error details.
"""


class AppException(Exception):
    """Base application exception with HTTP status code and detail."""

    def __init__(self, status_code: int, detail: str, error_code: str | None = None):
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code or f"ERR_{status_code}"
        super().__init__(detail)


class NotFoundException(AppException):
    """Resource not found (404)."""

    def __init__(self, resource: str, identifier: str = ""):
        identifier_str = f" '{identifier}'" if identifier else ""
        super().__init__(
            status_code=404,
            detail=f"{resource}{identifier_str} not found",
            error_code="NOT_FOUND",
        )


class ValidationException(AppException):
    """Request validation error (422)."""

    def __init__(self, detail: str, error_code: str = "VALIDATION_ERROR"):
        super().__init__(status_code=422, detail=detail, error_code=error_code)


class UnauthorizedException(AppException):
    """Authentication required (401)."""

    def __init__(self, detail: str = "Authentication required"):
        super().__init__(status_code=401, detail=detail, error_code="UNAUTHORIZED")


class ForbiddenException(AppException):
    """Permission denied (403)."""

    def __init__(self, detail: str = "Permission denied"):
        super().__init__(status_code=403, detail=detail, error_code="FORBIDDEN")


class RateLimitException(AppException):
    """Rate limit exceeded (429)."""

    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(status_code=429, detail=detail, error_code="RATE_LIMITED")


class PayloadTooLargeException(AppException):
    """Upload exceeds size limit (413)."""

    def __init__(self, detail: str = "File too large"):
        super().__init__(status_code=413, detail=detail, error_code="PAYLOAD_TOO_LARGE")
