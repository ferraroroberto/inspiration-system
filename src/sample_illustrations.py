#!/usr/bin/env python3
"""Inspiration Samples — one illustration per visual type.

Picks the most-recently-created illustration per visual type from the Notion
archive and copies the matching .png into a curated 'inspiration samples'
folder, flat + nested, plus an XLS index.

Usable two ways:
  - CLI:   python -m src.sample_illustrations [--dry-run] [--config config.json]
  - Lib:   from src.sample_illustrations import plan_samples, apply_samples

Notion access + field extraction comes from src/notion_client.py (shared with
src/build_index.py); only the sample-selection + filesystem layout logic lives
here.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from src.notion_client import (
    build_concepts_map,
    build_headers,
    build_illustrations_map,
    build_visual_types_map,
    load_config,
    query_database,
    resolve_token,
)

load_dotenv()

logger = logging.getLogger(__name__)

WINDOWS_ILLEGAL = re.compile(r'[<>:"/\\|?*]')


def sanitize(segment: str) -> str:
    cleaned = WINDOWS_ILLEGAL.sub("", segment or "").strip().rstrip(".")
    return cleaned or "_"


def extract_topic(title: str, visualtype: str) -> str:
    prefix = f"{visualtype} - "
    if title.startswith(prefix):
        return title[len(prefix):].strip()
    return title


def wipe_contents(folder: Path) -> None:
    """Remove everything inside folder but keep the folder itself (iCloud-safe)."""
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        return
    for entry in folder.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


@dataclass
class SamplePlan:
    rows: List[Dict[str, Any]]
    skipped_no_illustrations: int
    missing_count: int

    @property
    def planned(self) -> int:
        return len(self.rows)


def plan_samples(config: Dict[str, Any]) -> SamplePlan:
    """Query Notion, join the three DBs, and build the per-visual-type sample rows.

    Pure planning — no filesystem writes, no copies.
    """
    token = resolve_token(config)
    headers = build_headers(token)

    source_folder = Path(config["source_folder"])
    dest_folder = Path(config["samples_dest_folder"])

    logger.info("🔎 Querying Notion…")
    concepts = build_concepts_map(query_database(config["concepts_db_id"], headers))
    visual_types = build_visual_types_map(query_database(config["visual_types_db_id"], headers))
    illustrations = build_illustrations_map(query_database(config["illustrations_db_id"], headers))

    rows: List[Dict[str, Any]] = []
    skipped_no_illustrations = 0

    for vt_id, vt in visual_types.items():
        valid_ids = [i for i in vt.get("illustration_ids", []) if i in illustrations]
        if not valid_ids:
            skipped_no_illustrations += 1
            continue

        valid_ids.sort(key=lambda i: illustrations[i].get("created") or "", reverse=True)
        sample_id = valid_ids[0]
        sample = illustrations[sample_id]

        concept_id = vt.get("concept_id")
        concept_info = concepts.get(concept_id, {}) if concept_id else {}
        concept_name = concept_info.get("concept", "")
        concept_type = concept_info.get("concept_type", "untyped")

        visualtype_name = vt["visualtype"]
        title = sample["title"]
        topic = extract_topic(title, visualtype_name)

        ct_s = sanitize(concept_type)
        c_s = sanitize(concept_name or "_no_concept_")
        vt_s = sanitize(visualtype_name)
        topic_s = sanitize(topic)

        flat_name = f"{ct_s} - {c_s} - {vt_s} - {topic_s}.png"
        nested_name = f"{vt_s} - {topic_s}.png"

        source_path = source_folder / f"{title}.png"
        dest_flat = dest_folder / "all" / flat_name
        dest_nested = dest_folder / ct_s / c_s / nested_name

        rows.append({
            "concept_type": concept_type,
            "concept": concept_name,
            "concept_id": concept_id,
            "visual_type": visualtype_name,
            "visual_type_id": vt_id,
            "illustration_title": title,
            "illustration_id": sample_id,
            "topic": topic,
            "theme": ";".join(sample.get("theme") or []),
            "tags": sample.get("tags") or "",
            "created": sample.get("created") or "",
            "source_path": str(source_path),
            "dest_path_flat": str(dest_flat),
            "dest_path_nested": str(dest_nested),
            "n_total_in_visual_type": len(vt.get("illustration_ids", [])),
            "missing_source_file": not source_path.exists(),
        })

    missing_count = sum(1 for r in rows if r["missing_source_file"])
    return SamplePlan(rows=rows, skipped_no_illustrations=skipped_no_illustrations, missing_count=missing_count)


def apply_samples(
    plan: SamplePlan,
    config: Dict[str, Any],
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Execute the plan: wipe dest, copy flat + nested, write XLS.

    Returns a summary dict suitable for display.
    """
    dest_folder = Path(config["samples_dest_folder"])
    xls_output = Path(config["samples_xls_output"])
    source_folder = Path(config["source_folder"])

    if not source_folder.exists():
        raise FileNotFoundError(f"Source folder not found: {source_folder}")

    logger.info(f"🧹 Wiping {dest_folder} …")
    wipe_contents(dest_folder)
    (dest_folder / "all").mkdir(parents=True, exist_ok=True)

    copyable = [r for r in plan.rows if not r["missing_source_file"]]
    copied = 0
    for i, r in enumerate(copyable, 1):
        src = Path(r["source_path"])
        flat = Path(r["dest_path_flat"])
        nested = Path(r["dest_path_nested"])
        flat.parent.mkdir(parents=True, exist_ok=True)
        nested.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, flat)
        shutil.copy2(src, nested)
        copied += 1
        if progress_cb is not None:
            progress_cb(i, len(copyable))

    xls_output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(plan.rows)
    df.to_excel(xls_output, index=False, engine="openpyxl")

    logger.info(f"✅ Copied {copied} illustration(s); wrote XLS: {xls_output} ({len(df)} rows)")
    return {
        "planned": plan.planned,
        "copied": copied,
        "missing": plan.missing_count,
        "skipped_no_illustrations": plan.skipped_no_illustrations,
        "xls_output": str(xls_output),
        "dest_folder": str(dest_folder),
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Sample one illustration per visual type to a curated folder + XLS.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; no copies, no XLS.")
    args = parser.parse_args()

    config = load_config(args.config)
    plan = plan_samples(config)

    logger.info(
        f"📊 Planned samples: {plan.planned} | "
        f"visual types with no illustrations: {plan.skipped_no_illustrations}"
    )
    if plan.missing_count:
        logger.warning(f"⚠️ {plan.missing_count} sample(s) have no matching .png in source")

    if args.dry_run:
        logger.info("🧪 Dry run — no files copied, no XLS written.")
        for r in plan.rows[:5]:
            logger.info(f"   → {r['dest_path_flat']}")
        if plan.planned > 5:
            logger.info(f"   … and {plan.planned - 5} more")
        return 0

    if not plan.rows:
        logger.error("❌ No samples to write. Aborting.")
        return 1

    summary = apply_samples(plan, config)
    logger.info(
        f"✅ Done. Copied {summary['copied']} / planned {summary['planned']} "
        f"(missing: {summary['missing']}) → {summary['dest_folder']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
