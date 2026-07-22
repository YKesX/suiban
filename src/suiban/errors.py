"""Error envelope per docs/api.md: every non-2xx body is
{"error": {"type", "message", "code"}}.
"""

from __future__ import annotations

STATUS_TO_TYPE = {
    400: "invalid_request_error",
    404: "not_found_error",
    409: "conflict_error",
    429: "overloaded_error",
    500: "server_error",
}


class BonsaiError(Exception):
    """Raise anywhere in a request handler to produce the contract error envelope."""

    def __init__(
        self,
        status: int,
        message: str,
        *,
        code: str | None = None,
        error_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.error_type = error_type or STATUS_TO_TYPE.get(status, "server_error")

    def envelope(self) -> dict:
        return {"error": {"type": self.error_type, "message": self.message, "code": self.code}}
