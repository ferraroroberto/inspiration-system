"""Tests for src.enrichment — pure helpers and the inject-a-caller path."""

import json

import pytest

from src import enrichment
from src.enrichment import (
    _compose_prompt,
    _load_cache,
    _parse_envelope,
    _split_batches,
    enrich_rows,
)


class TestSplitBatches:
    def test_even_split(self):
        batches = _split_batches([{"illustration_id": str(i)} for i in range(6)], 2)
        assert len(batches) == 3
        assert [len(b) for b in batches] == [2, 2, 2]

    def test_uneven_tail(self):
        batches = _split_batches([{"illustration_id": str(i)} for i in range(5)], 2)
        assert [len(b) for b in batches] == [2, 2, 1]

    def test_empty(self):
        assert _split_batches([], 15) == []

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            _split_batches([{"illustration_id": "x"}], 0)


class TestParseEnvelope:
    def test_raw_array(self):
        raw = '[{"id": "a", "tone": "hopeful"}]'
        assert _parse_envelope(raw) == [{"id": "a", "tone": "hopeful"}]

    def test_fenced_json(self):
        text = 'Sure, here you go:\n```json\n[{"id": "a"}]\n```\nHope that helps.'
        assert _parse_envelope(text) == [{"id": "a"}]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _parse_envelope("")


class TestComposePrompt:
    def test_includes_all_fields(self):
        batch = [
            {
                "illustration_id": "ill_1",
                "title": "bridge with spikes",
                "alt_text": "a bridge covered in spikes",
                "theme": ["obstacle", "change"],
            }
        ]
        prompt = _compose_prompt(batch)
        assert "ill_1" in prompt
        assert "bridge with spikes" in prompt
        assert "obstacle, change" in prompt
        assert "metaphor" in prompt.lower()

    def test_theme_as_string(self):
        batch = [
            {
                "illustration_id": "ill_2",
                "title": "t",
                "alt_text": "a",
                "theme": "obstacle;change",
            }
        ]
        prompt = _compose_prompt(batch)
        assert "obstacle;change" in prompt


class TestLoadCache:
    def test_missing_file_empty(self, tmp_path):
        assert _load_cache(tmp_path / "nope.jsonl") == {}

    def test_loads_jsonl(self, tmp_path):
        cache_path = tmp_path / "c.jsonl"
        cache_path.write_text(
            json.dumps({"id": "a", "tone": "hopeful"}) + "\n"
            + json.dumps({"id": "b", "tone": "tense"}) + "\n",
            encoding="utf-8",
        )
        loaded = _load_cache(cache_path)
        assert set(loaded.keys()) == {"a", "b"}
        assert loaded["a"]["tone"] == "hopeful"

    def test_skips_malformed_lines(self, tmp_path):
        cache_path = tmp_path / "c.jsonl"
        cache_path.write_text(
            json.dumps({"id": "a"}) + "\nnot json\n" + json.dumps({"id": "b"}) + "\n",
            encoding="utf-8",
        )
        loaded = _load_cache(cache_path)
        assert set(loaded.keys()) == {"a", "b"}


class TestEnrichRowsResume:
    def _make_caller(self, responses):
        calls = []

        def caller(prompt):
            calls.append(prompt)
            return responses.pop(0)

        caller.calls = calls  # type: ignore[attr-defined]
        return caller

    def test_full_run_writes_cache(self, tmp_path):
        rows = [
            {"illustration_id": "a", "title": "t1", "alt_text": "x", "theme": []},
            {"illustration_id": "b", "title": "t2", "alt_text": "y", "theme": []},
        ]
        cache_path = tmp_path / "e.jsonl"
        responses = [
            json.dumps([{"id": "a", "tone": "hopeful"}, {"id": "b", "tone": "tense"}])
        ]
        caller = self._make_caller(responses)

        result = enrich_rows(rows, cache_path, batch_size=10, caller=caller)
        assert set(result.keys()) == {"a", "b"}
        lines = cache_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_resume_skips_cached(self, tmp_path):
        rows = [
            {"illustration_id": "a", "title": "t1", "alt_text": "x", "theme": []},
            {"illustration_id": "b", "title": "t2", "alt_text": "y", "theme": []},
        ]
        cache_path = tmp_path / "e.jsonl"
        cache_path.write_text(json.dumps({"id": "a", "tone": "hopeful"}) + "\n", encoding="utf-8")

        responses = [json.dumps([{"id": "b", "tone": "tense"}])]
        caller = self._make_caller(responses)

        result = enrich_rows(rows, cache_path, batch_size=10, caller=caller)
        assert set(result.keys()) == {"a", "b"}
        assert len(caller.calls) == 1
        assert "b" in caller.calls[0]
        assert "t1" not in caller.calls[0]  # cached row wasn't re-sent

    def test_all_cached_no_calls(self, tmp_path):
        rows = [{"illustration_id": "a", "title": "t1", "alt_text": "x", "theme": []}]
        cache_path = tmp_path / "e.jsonl"
        cache_path.write_text(json.dumps({"id": "a", "tone": "hopeful"}) + "\n", encoding="utf-8")
        caller = self._make_caller([])
        result = enrich_rows(rows, cache_path, batch_size=10, caller=caller)
        assert result["a"]["tone"] == "hopeful"
        assert caller.calls == []
