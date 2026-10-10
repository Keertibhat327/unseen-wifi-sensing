"""Vercel ASGI entry; deploy from repository root so shared contracts are packaged."""

from backend.app import create_app

app = create_app()
