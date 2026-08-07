"""Workspace path validation shared by gateway components."""

from pathlib import Path


class UnsafePath(ValueError):
    """Raised when a user supplied path escapes its workspace."""


def safe_path(root: str | Path, requested: str, *, allow_missing: bool = True) -> Path:
    """Resolve *requested* below *root*, rejecting traversal and absolute paths."""
    if not requested or "\x00" in requested:
        raise UnsafePath("path is empty or contains a NUL byte")
    candidate = Path(requested)
    if candidate.is_absolute():
        raise UnsafePath("absolute paths are not allowed")
    root_path = Path(root).resolve()
    resolved = (root_path / candidate).resolve()
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise UnsafePath("path escapes the workspace") from exc
    if not allow_missing and not resolved.exists():
        raise FileNotFoundError(requested)
    return resolved
