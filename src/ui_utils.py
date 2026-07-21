"""Shared UI helpers: live log container, file-explorer launcher.

The ``StreamlitLogHandler`` pipes stdlib ``logging`` records — including the
ones emitted by ``build_index`` and ``sample_illustrations`` — into a live
``st.code`` block so the user can watch a long-running job without staring
at a blank page. Format matches the externalrisk convention:

    '%(asctime)s - %(levelname)s - %(message)s'

Call ``attach_log_handler(container)`` before invoking the job, and pair it
with ``detach_log_handler(handler)`` in a ``try/finally`` so handlers don't
accumulate across Streamlit reruns.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import streamlit as st

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DATEFMT = "%H:%M:%S"


class StreamlitLogHandler(logging.Handler):
    """Logging handler that renders formatted records to a Streamlit container.

    Rewrites the whole buffer every ``emit`` — Streamlit's container model
    doesn't support append, but for human-scale log volumes (< a few
    thousand lines) this is fast enough.
    """

    def __init__(self, container: "st.delta_generator.DeltaGenerator", max_lines: int = 500) -> None:
        super().__init__()
        self.container = container
        self.max_lines = max_lines
        self.lines: list[str] = []
        self.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(self.format(record))
            display = self.lines[-self.max_lines :]
            self.container.code("\n".join(display), language="log")
        except Exception:
            self.handleError(record)


def attach_log_handler(
    container: "st.delta_generator.DeltaGenerator",
    level: int = logging.INFO,
    logger_name: Optional[str] = None,
) -> StreamlitLogHandler:
    """Attach a StreamlitLogHandler to the root (or named) logger.

    Returns the handler so the caller can detach it afterwards.
    """
    handler = StreamlitLogHandler(container)
    handler.setLevel(level)
    logger = logging.getLogger(logger_name)
    logger.setLevel(min(logger.level or logging.WARNING, level))
    logger.addHandler(handler)
    return handler


def detach_log_handler(handler: StreamlitLogHandler, logger_name: Optional[str] = None) -> None:
    logging.getLogger(logger_name).removeHandler(handler)


def open_in_explorer(path: Path) -> None:
    """Open the given directory/file in the OS file explorer."""
    if sys.platform.startswith("win"):
        subprocess.Popen(
            ["explorer", str(path)],
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
