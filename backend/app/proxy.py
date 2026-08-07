"""Unit-testable preview proxy URL/path helpers."""

from __future__ import annotations


def parse_preview_path(port: str | int, path: str = "") -> tuple[int, str]:
    number = int(port)
    if number < 1 or number > 65535:
        raise ValueError("invalid preview port")
    clean = "/" + (path or "").lstrip("/")
    return number, clean


def preview_target(
    host_port: int,
    path: str = "",
    query: str = "",
    host: str = "127.0.0.1",
) -> str:
    port, clean = parse_preview_path(host_port, path)
    return f"http://{host}:{port}{clean}" + (f"?{query}" if query else "")
