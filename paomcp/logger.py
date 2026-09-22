"""Central logging configuration for every pao-tools entrypoint.

Every handler writes to stderr: the stdio transport shares stdout with the
MCP wire protocol, so a single log line on stdout corrupts the session.
``configure_logging`` is idempotent and safe to call from ``__main__``,
``runtime``, ``asgi`` and client helpers.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

LOGGER_NAME = "paomcp"
DEFAULT_LOG_LEVEL = "INFO"
_LOG_LEVEL_ENV = "PAO_LOG_LEVEL"
_QUIET_LOGGERS = ("httpx", "httpcore", "urllib3")
_PAO_HANDLER_ATTR = "_pao_handler"
_PROD_VALUES = {"1", "true", "yes", "on"}


def _resolve_level(level: Optional[str]) -> int:
    raw = (level or os.environ.get(_LOG_LEVEL_ENV) or "").strip()
    if not raw:
        return logging.INFO
    if raw.isdigit():
        return int(raw)
    return logging.getLevelNamesMapping().get(raw.upper(), logging.INFO)


def _build_handler() -> logging.Handler:
    prod = os.environ.get("PROD", "").lower() in _PROD_VALUES
    if not prod:
        try:
            from rich.console import Console
            from rich.logging import RichHandler

            # RichHandler defaults to stdout; force stderr so stdio transport
            # never mixes logs into the MCP stream.
            return RichHandler(
                console=Console(stderr=True),
                rich_tracebacks=True,
                show_path=False,
            )
        except Exception:
            pass
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    return handler


def configure_logging(
    level: Optional[str] = None, *, force: bool = False
) -> logging.Logger:
    """Install the stderr-only root handler exactly once and set levels.

    ``level`` overrides ``PAO_LOG_LEVEL`` (default ``INFO``). ``force``
    rebuilds the pao-tools handler, which tests use to exercise both the
    plain and rich handler paths.
    """
    root = logging.getLogger()
    existing = [
        handler
        for handler in root.handlers
        if getattr(handler, _PAO_HANDLER_ATTR, False)
    ]
    if force:
        for handler in existing:
            root.removeHandler(handler)
            handler.close()
        existing = []
    if not existing:
        handler = _build_handler()
        setattr(handler, _PAO_HANDLER_ATTR, True)
        root.addHandler(handler)

    resolved = _resolve_level(level)
    root.setLevel(resolved)
    logging.getLogger(LOGGER_NAME).setLevel(resolved)
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger(LOGGER_NAME)
