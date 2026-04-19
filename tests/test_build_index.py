"""Tests for pure helpers in src.build_index."""

from src.build_index import build_embed_text


class TestBuildEmbedText:
    def test_full_composition(self):
        text = build_embed_text(
            title="mountain climb",
            alt_text="a figure climbing a steep mountain",
            theme=["growth", "challenge"],
        )
        assert text == "mountain climb. a figure climbing a steep mountain Themes: growth, challenge."

    def test_title_only(self):
        assert build_embed_text("growth ladder", "", []) == "growth ladder."

    def test_no_title(self):
        assert (
            build_embed_text("", "a diagram of branches", ["decision"])
            == "a diagram of branches Themes: decision."
        )

    def test_strips_trailing_dot_from_title(self):
        assert build_embed_text("title.", "", []) == "title."

    def test_empty_inputs(self):
        assert build_embed_text("", "", []) == ""

    def test_enrichment_prepended(self):
        enrichment = {
            "metaphorical_meanings": ["wasted effort from misdirected focus"],
            "applicable_themes": ["productivity", "burnout"],
            "tone": "cautionary",
        }
        text = build_embed_text("bucket with holes", "a leaky bucket", ["waste"], enrichment=enrichment)
        # enrichment fields come first
        assert text.startswith("Could represent: wasted effort from misdirected focus.")
        assert "Applies to: productivity, burnout." in text
        assert "Tone: cautionary." in text
        assert "bucket with holes." in text
        assert "Themes: waste." in text

    def test_enrichment_none_matches_old_behavior(self):
        assert (
            build_embed_text("t", "a", ["x"], enrichment=None)
            == build_embed_text("t", "a", ["x"])
        )

    def test_empty_enrichment_fields_skipped(self):
        enrichment = {"metaphorical_meanings": [], "applicable_themes": [], "tone": ""}
        assert build_embed_text("t", "a", ["x"], enrichment=enrichment) == "t. a Themes: x."
