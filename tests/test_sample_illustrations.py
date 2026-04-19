"""Tests for pure helpers in src.sample_illustrations."""

from src.sample_illustrations import extract_topic, sanitize


class TestSanitize:
    def test_strips_windows_illegal_chars(self):
        assert sanitize('foo<bar>:"/\\|?*baz') == "foobarbaz"

    def test_preserves_safe_punctuation(self):
        assert sanitize("barchart - impossible things") == "barchart - impossible things"

    def test_strips_trailing_dot_and_whitespace(self):
        assert sanitize("  hello.  ") == "hello"

    def test_empty_falls_back_to_underscore(self):
        assert sanitize("") == "_"
        assert sanitize("   ") == "_"
        assert sanitize(None) == "_"


class TestExtractTopic:
    def test_strips_visualtype_prefix(self):
        assert extract_topic("barchart - impossible things", "barchart") == "impossible things"

    def test_no_prefix_returns_full_title(self):
        assert extract_topic("some random title", "barchart") == "some random title"

    def test_empty_visualtype_returns_full_title(self):
        # With an empty prefix, the prefix check becomes " - " match — safe behavior.
        assert extract_topic("anything", "") == "anything"

    def test_case_sensitive(self):
        # Prefix match is literal; casing matters.
        assert extract_topic("Barchart - foo", "barchart") == "Barchart - foo"
