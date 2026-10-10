"""Small stateless API safeguards; deployment proxy limits are still required."""

from urllib.parse import parse_qs, urlparse

from starlette.responses import JSONResponse


def validate_production(url: str, writer: str, reader: str, origins: list[str]):
    if not url.startswith(("postgres://", "postgresql://", "postgresql+psycopg://")):
        raise RuntimeError("Production requires persistent Postgres via DATABASE_URL")
    if parse_qs(urlparse(url).query).get("sslmode", [""])[0] not in {
        "require",
        "verify-ca",
        "verify-full",
    }:
        raise RuntimeError("Production Postgres requires explicit sslmode=require or stronger")
    if any(len(key) < 32 or "replace-with" in key for key in (writer, reader)):
        raise RuntimeError("Production requires distinct random API keys of at least 32 characters")
    if not origins or any(
        urlparse(o).scheme != "https"
        or not urlparse(o).hostname
        or urlparse(o).path
        or urlparse(o).query
        or urlparse(o).fragment
        or urlparse(o).username
        for o in origins
    ):
        raise RuntimeError(
            "Production CORS_ORIGINS must be exact HTTPS origins without trailing slash"
        )


class BodyLimitMiddleware:
    def __init__(self, app, max_bytes=32768):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.max_bytes:
                return await JSONResponse({"detail": "Request body exceeds 32 KiB"}, 413)(
                    scope, receive, send
                )
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        consumed = False

        async def replay():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
