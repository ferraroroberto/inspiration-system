#!/usr/bin/env python3
"""Generate metaphor enrichments for every illustration via the Anthropic SDK.

Two entry points:
  - ``enrich_rows(rows, cache_path, ...)`` — importable library call.
  - ``main()`` — argparse CLI. Used by ``python -m src.enrichment``.

For each illustration we ask Claude for a rich metaphor structure: visual
elements, metaphorical meanings, applicable themes, tone. Results are cached
one-object-per-line in ``index/enrichments.jsonl`` so the job is resumable
and incremental: re-running only calls the API for rows not already in the
cache.

LLM calls are routed through the local hub at http://127.0.0.1:8000 using
the standard Anthropic SDK. The hub proxies requests to Claude Code or local
models depending on the ``model`` name.

Logging follows the externalrisk convention:
    '%(asctime)s - %(levelname)s - %(message)s'
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_HUB_BASE_URL = "http://127.0.0.1:8000"
_HUB_MODEL = "claude-haiku-4-5"
_client: Optional[Anthropic] = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key="local-dummy", base_url=_HUB_BASE_URL)
    return _client


PROMPT_HEADER = """\
You are a visual metaphor librarian. For each illustration below, output the
underlying *metaphor structure*, not the literal scene.

Return a JSON array (no prose, no markdown fences) of objects with exactly
these keys:
  - id (string, echoing the illustration id)
  - visual_elements (array of short nouns describing concrete elements)
  - metaphorical_meanings (array of 3-6 short phrases — what the image
    could metaphorically represent, regardless of its literal subject)
  - applicable_themes (array of 5-10 themes this could illustrate, going
    well beyond the original theme tag — that is the point of this pass)
  - tone (one of: cautionary, hopeful, neutral, energetic, reflective,
    tense, playful)
  - abstraction_level (one of: low, medium, high)

Good metaphorical_meanings:
  "wasted effort from misdirected focus"
  "surface calm hiding internal pressure"
  "short-term comfort vs long-term cost"

Bad (too literal):
  "a bucket with holes"
  "a person sleeping"

Illustrations:
"""


def _compose_prompt(batch: List[Dict[str, Any]]) -> str:
    lines = [PROMPT_HEADER]
    for row in batch:
        theme_str = row.get("theme") or ""
        if isinstance(theme_str, list):
            theme_str = ", ".join(theme_str)
        lines.append(
            f"- id: {row['illustration_id']}\n"
            f"  title: {row.get('title', '')}\n"
            f"  alt_text: {row.get('alt_text', '')}\n"
            f"  original_theme: {theme_str}"
        )
    return "\n".join(lines)


def _split_batches(rows: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


_FENCE_RE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
_ARRAY_RE = re.compile(r"(\[.*\])", re.DOTALL)


def _parse_envelope(raw_stdout: str) -> List[Dict[str, Any]]:
    """Parse LLM output and return the inner JSON array of enrichment objects.

    Accepts a raw or fenced JSON array — ``message.content[0].text`` from the
    Anthropic SDK, as returned by ``_call_claude``.
    """
    raw_stdout = raw_stdout.strip()
    if not raw_stdout:
        raise ValueError("empty response from LLM")

    try:
        parsed = json.loads(raw_stdout)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(parsed, list):
            return parsed

    m = _FENCE_RE.search(raw_stdout)
    if m:
        return json.loads(m.group(1))
    m = _ARRAY_RE.search(raw_stdout)
    if m:
        return json.loads(m.group(1))
    raise ValueError(f"no JSON array found in assistant text: {raw_stdout[:200]!r}")


def _call_claude(prompt: str, timeout: float = 300.0) -> str:
    """Call the LLM via the local hub (Anthropic SDK). Returns the assistant text."""
    client = _get_client()
    message = client.messages.create(
        model=_HUB_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
        timeout=timeout,
    )
    return message.content[0].text


def _load_cache(cache_path: Path) -> Dict[str, Dict[str, Any]]:
    if not cache_path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"⚠️ Skipping malformed cache line in {cache_path.name}")
                continue
            if "id" in obj:
                out[obj["id"]] = obj
    return out


def _append_cache(cache_path: Path, entries: Iterable[Dict[str, Any]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def enrich_rows(
    rows: List[Dict[str, Any]],
    cache_path: Path,
    batch_size: int = 15,
    caller: Optional[Callable[[str], str]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return a dict keyed by ``illustration_id`` → enrichment object.

    Cached rows are loaded from ``cache_path``; uncached rows are batched
    and sent to the LLM via the local hub. The cache file is appended
    incrementally so interrupted runs can resume.

    ``caller(prompt) -> text`` can be injected for testing.
    """
    caller = caller or _call_claude
    cache = _load_cache(cache_path)
    logger.info(f"🗂 Cache: {len(cache)} existing enrichments in {cache_path.name}")

    todo = [r for r in rows if r["illustration_id"] not in cache]
    if not todo:
        logger.info("✅ All rows already enriched — nothing to do.")
        return cache

    logger.info(
        f"🧠 Enriching {len(todo)} rows in batches of {batch_size} "
        f"→ {(len(todo) + batch_size - 1) // batch_size} hub calls"
    )

    batches = _split_batches(todo, batch_size)
    done = 0
    t0 = time.time()
    for i, batch in enumerate(batches, 1):
        prompt = _compose_prompt(batch)
        try:
            raw = caller(prompt)
            enrichments = _parse_envelope(raw)
        except Exception as e:
            logger.error(f"❌ Batch {i}/{len(batches)} failed: {e}")
            raise

        by_id = {e["id"]: e for e in enrichments if isinstance(e, dict) and "id" in e}
        batch_entries: List[Dict[str, Any]] = []
        for row in batch:
            rid = row["illustration_id"]
            enr = by_id.get(rid)
            if not enr:
                logger.warning(f"⚠️ No enrichment returned for {rid}")
                continue
            cache[rid] = enr
            batch_entries.append(enr)

        _append_cache(cache_path, batch_entries)
        done += len(batch)
        if progress_cb is not None:
            progress_cb(done, len(todo))
        logger.info(
            f"   …batch {i}/{len(batches)} wrote {len(batch_entries)}/{len(batch)} "
            f"({done}/{len(todo)} total, {time.time() - t0:.1f}s)"
        )

    logger.info(
        f"✅ Enrichment done: {len(cache)} total in cache, "
        f"{time.time() - t0:.1f}s for this run"
    )
    return cache


def _rows_from_notion(config_path: str) -> List[Dict[str, Any]]:
    """Pull rows directly from Notion — no pre-built parquet required.

    The enrichment CLI is standalone so the user can run it live from
    a terminal without first going through build_index.
    """
    from src.notion_client import fetch_notion_maps, load_config
    from src.build_index import assemble_rows

    config = load_config(config_path)
    source_folder = Path(config["source_folder"])

    logger.info("🔎 Querying Notion for rows to enrich…")
    concepts, visual_types, illustrations = fetch_notion_maps(config)
    rows = assemble_rows(illustrations, visual_types, concepts, source_folder)
    logger.info(f"📊 Pulled {len(rows)} rows from Notion")
    return rows


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Generate metaphor enrichments via the LLM hub, cache in jsonl."
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--cache",
        default="index/enrichments.jsonl",
        help="JSONL cache file (default: %(default)s). Resumable.",
    )
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="If >0, only enrich the first N rows (prototype mode).",
    )
    args = parser.parse_args()

    try:
        rows = _rows_from_notion(args.config)
    except Exception as e:
        logger.error(f"❌ Notion pull failed: {e}")
        return 1

    if args.sample > 0:
        rows = rows[: args.sample]
        logger.info(f"🧪 Prototype mode — enriching only the first {len(rows)} rows")

    try:
        enrich_rows(rows, Path(args.cache), batch_size=args.batch_size)
        return 0
    except Exception as e:
        logger.error(f"❌ {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
