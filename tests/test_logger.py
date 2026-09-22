"""Regression tests for the central stderr-only logging configuration."""

from __future__ import annotations

import logging
import sys

from paomcp.logger import configure_logging


def _pao_handlers():
    return [
        handler
        for handler in logging.getLogger().handlers
        if getattr(handler, "_pao_handler", False)
    ]


def _handler_stream(handler: logging.Handler):
    stream = getattr(handler, "stream", None)
    if stream is not None:
        return stream
    console = getattr(handler, "console", None)
    return getattr(console, "file", None)


def test_configure_logging_is_idempotent(monkeypatch):
    monkeypatch.delenv("PROD", raising=False)
    configure_logging(force=True)
    configure_logging()
    configure_logging()
    assert len(_pao_handlers()) == 1


def test_handler_writes_to_stderr_never_stdout(monkeypatch):
    monkeypatch.delenv("PROD", raising=False)
    logger = configure_logging(force=True)
    (handler,) = _pao_handlers()
    assert _handler_stream(handler) is sys.stderr
    assert logger.name == "paomcp"


def test_prod_mode_uses_plain_stderr_stream(monkeypatch):
    monkeypatch.setenv("PROD", "1")
    configure_logging(force=True)
    (handler,) = _pao_handlers()
    assert isinstance(handler, logging.StreamHandler)
    assert _handler_stream(handler) is sys.stderr


def test_log_level_env_and_explicit_override(monkeypatch):
    monkeypatch.setenv("PAO_LOG_LEVEL", "warning")
    configure_logging(force=True)
    assert logging.getLogger().level == logging.WARNING

    configure_logging(level="debug", force=True)
    assert logging.getLogger().level == logging.DEBUG

    monkeypatch.setenv("PAO_LOG_LEVEL", "not-a-level")
    configure_logging(force=True)
    assert logging.getLogger().level == logging.INFO
