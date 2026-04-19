"""Tests for src.ui_utils.StreamlitLogHandler."""

import logging

from src.ui_utils import StreamlitLogHandler, attach_log_handler, detach_log_handler


class FakeContainer:
    """Stand-in for a Streamlit delta_generator: captures the last code() call."""

    def __init__(self):
        self.last_text: str = ""
        self.last_language: str | None = None
        self.calls: int = 0

    def code(self, text: str, language: str | None = None) -> None:
        self.last_text = text
        self.last_language = language
        self.calls += 1


class TestStreamlitLogHandler:
    def test_emits_formatted_line_to_container(self):
        container = FakeContainer()
        handler = StreamlitLogHandler(container)
        logger = logging.getLogger("test_streamlit_log_handler_1")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            logger.info("hello 🚀")
        finally:
            logger.removeHandler(handler)

        assert container.calls == 1
        assert "hello 🚀" in container.last_text
        assert "INFO" in container.last_text
        assert container.last_language == "log"

    def test_tails_to_max_lines(self):
        container = FakeContainer()
        handler = StreamlitLogHandler(container, max_lines=3)
        logger = logging.getLogger("test_streamlit_log_handler_2")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            for i in range(10):
                logger.info(f"line-{i}")
        finally:
            logger.removeHandler(handler)

        # Only the last 3 lines should be visible
        lines = container.last_text.splitlines()
        assert len(lines) == 3
        assert "line-7" in lines[0]
        assert "line-9" in lines[-1]

    def test_attach_and_detach_round_trip(self):
        container = FakeContainer()
        logger_name = "test_attach_detach"
        logger = logging.getLogger(logger_name)
        before = list(logger.handlers)

        handler = attach_log_handler(container, logger_name=logger_name)
        assert handler in logger.handlers
        logger.info("captured")
        assert "captured" in container.last_text

        detach_log_handler(handler, logger_name=logger_name)
        assert handler not in logger.handlers
        assert list(logger.handlers) == before
