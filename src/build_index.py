#!/usr/bin/env python3
"""Build the local embedding index for the Inspiration Illustration Finder.

Two entry points:
  - ``build_index(config, dry_run=False, progress_cb=None)`` — importable.
    Used by the Streamlit "Build" page so we can capture its log stream.
  - ``main()`` — argparse CLI. Used by ``python -m src.build_index``.

Logging follows the externalrisk convention:
    '%(asctime)s - %(levelname)s - %(message)s'
with emoji-prefixed messages so they read well both in the console and in
the in-app log viewer.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
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


def build_embed_text(
    title: str,
    alt_text: str,
    theme: List[str],
    enrichment: Optional[Dict[str, Any]] = None,
) -> str:
    """Compose the per-row text that gets embedded.

    When ``enrichment`` is provided (from the metaphor enrichment pass),
    its fields are placed *first* — early tokens carry slightly more weight
    in sentence-transformer models, and the whole point of enrichment is to
    shift the embedding away from literal-scene matching.
    """
    parts: List[str] = []
    if enrichment:
        meanings = enrichment.get("metaphorical_meanings") or []
        themes = enrichment.get("applicable_themes") or []
        tone = enrichment.get("tone") or ""
        if meanings:
            parts.append("Could represent: " + "; ".join(meanings) + ".")
        if themes:
            parts.append("Applies to: " + ", ".join(themes) + ".")
        if tone:
            parts.append(f"Tone: {tone}.")
    if title:
        parts.append(title.strip().rstrip(".") + ".")
    if alt_text:
        parts.append(alt_text.strip())
    if theme:
        parts.append(f"Themes: {', '.join(theme)}.")
    return " ".join(parts).strip()


def assemble_rows(
    illustrations: Dict[str, Dict[str, Any]],
    visual_types: Dict[str, Dict[str, Any]],
    concepts: Dict[str, Dict[str, Any]],
    source_folder: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ill_id, ill in illustrations.items():
        vt_id = ill.get("visual_type_id")
        if not vt_id or vt_id not in visual_types:
            continue
        vt = visual_types[vt_id]
        concept_id = vt.get("concept_id")
        concept = concepts.get(concept_id, {}) if concept_id else {}

        title = ill.get("title", "")
        alt_text = ill.get("alt_text", "")
        theme = ill.get("theme", []) or []
        png_path = source_folder / f"{title}.png"

        rows.append({
            "illustration_id": ill_id,
            "title": title,
            "alt_text": alt_text,
            "theme": ";".join(theme),
            "tags": ill.get("tags") or "",
            "visual_type_id": vt_id,
            "visual_type_name": vt.get("visualtype", ""),
            "concept_id": concept_id or "",
            "concept_name": concept.get("concept", ""),
            "concept_type": concept.get("concept_type", "untyped"),
            "png_path": str(png_path),
            "missing_png": not png_path.exists(),
            "embed_text": build_embed_text(title, alt_text, theme),
            "created": ill.get("created") or "",
        })
    return rows


def build_index(
    config: Dict[str, Any],
    dry_run: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    skip_enrichment: bool = False,
) -> Dict[str, Any]:
    """Query Notion, enrich metaphor metadata, embed, persist.

    ``progress_cb(done, total)`` is called after each embedding batch.

    ``skip_enrichment=True`` bypasses the ``claude -p`` enrichment pass and
    falls back to the pre-enrichment ``embed_text`` composition. Handy for
    fast dev iterations or when Claude Code is unavailable.
    """
    token = resolve_token(config)
    headers = build_headers(token)

    source_folder = Path(config["source_folder"])
    index_folder = Path(config["index_folder"])
    model_name = config.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")

    if not source_folder.exists():
        logger.warning(f"⚠️ Source PNG folder not found: {source_folder} (rows will be flagged missing_png)")

    logger.info("🔎 Querying Notion…")
    concepts = build_concepts_map(query_database(config["concepts_db_id"], headers))
    visual_types = build_visual_types_map(query_database(config["visual_types_db_id"], headers))
    illustrations = build_illustrations_map(query_database(config["illustrations_db_id"], headers))

    rows = assemble_rows(illustrations, visual_types, concepts, source_folder)
    missing_png_count = sum(1 for r in rows if r["missing_png"])
    dropped = len(illustrations) - len(rows)

    logger.info(
        f"📊 Illustrations: {len(illustrations)} | joined rows: {len(rows)} "
        f"| dropped (no visual_type): {dropped} | missing PNG: {missing_png_count}"
    )

    if dry_run:
        logger.info("🧪 Dry run — no embedding, no files written.")
        if rows:
            sample = rows[0]
            logger.info(f"🔍 Sample row 0: {sample['title']!r}")
            logger.info(f"   embed_text: {sample['embed_text']!r}")
            logger.info(f"   visual_type: {sample['visual_type_name']} | concept: {sample['concept_name']}")
        return {
            "planned": len(rows),
            "dropped": dropped,
            "missing_png": missing_png_count,
            "dry_run": True,
        }

    if not rows:
        raise RuntimeError("No rows to embed. Aborting.")

    enriched_count = 0
    if skip_enrichment:
        logger.info("⏭️  Skipping metaphor enrichment (skip_enrichment=True)")
    else:
        from src.enrichment import enrich_rows

        cache_path = index_folder / "enrichments.jsonl"
        theme_rows = [
            {
                "illustration_id": r["illustration_id"],
                "title": r["title"],
                "alt_text": r["alt_text"],
                "theme": r["theme"].split(";") if r["theme"] else [],
            }
            for r in rows
        ]
        enrichments = enrich_rows(theme_rows, cache_path)
        for r in rows:
            enr = enrichments.get(r["illustration_id"])
            if enr:
                enriched_count += 1
                r["metaphorical_meanings"] = "; ".join(enr.get("metaphorical_meanings") or [])
                r["applicable_themes"] = "; ".join(enr.get("applicable_themes") or [])
                r["tone"] = enr.get("tone") or ""
                r["abstraction_level"] = enr.get("abstraction_level") or ""
                r["embed_text"] = build_embed_text(
                    r["title"],
                    r["alt_text"],
                    r["theme"].split(";") if r["theme"] else [],
                    enrichment=enr,
                )
            else:
                r["metaphorical_meanings"] = ""
                r["applicable_themes"] = ""
                r["tone"] = ""
                r["abstraction_level"] = ""
        logger.info(f"🎨 Enriched embed_text for {enriched_count}/{len(rows)} rows")

    logger.info(f"🧠 Loading model: {model_name} (first run may download a large blob)")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [r["embed_text"] for r in rows]

    batch_size = 64
    total = len(texts)
    all_embeddings: List[np.ndarray] = []
    logger.info(f"⚙️ Embedding {total} rows in batches of {batch_size}…")

    t0 = time.time()
    for i in range(0, total, batch_size):
        chunk = texts[i : i + batch_size]
        vecs = model.encode(
            chunk,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        all_embeddings.append(vecs)
        done = min(i + batch_size, total)
        if progress_cb is not None:
            progress_cb(done, total)
        if done % (batch_size * 4) == 0 or done == total:
            logger.info(f"   …{done}/{total}")

    embeddings = np.vstack(all_embeddings)
    elapsed = time.time() - t0

    index_folder.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    parquet_path = index_folder / "illustrations.parquet"
    npy_path = index_folder / "embeddings.npy"
    meta_path = index_folder / "index_meta.json"

    df.to_parquet(parquet_path, index=False)
    np.save(npy_path, embeddings)
    meta = {
        "model": model_name,
        "dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "missing_png_count": int(missing_png_count),
        "enriched_count": int(enriched_count),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info(f"✅ Indexed {len(rows)} illustrations in {elapsed:.1f}s (missing PNG: {missing_png_count})")
    logger.info(f"   → {parquet_path}")
    logger.info(f"   → {npy_path}  shape={embeddings.shape}")
    logger.info(f"   → {meta_path}")

    return {
        "planned": len(rows),
        "dropped": dropped,
        "missing_png": missing_png_count,
        "dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "elapsed_s": elapsed,
        "parquet_path": str(parquet_path),
        "npy_path": str(npy_path),
        "meta_path": str(meta_path),
        "dry_run": False,
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Build the inspiration embedding index.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--dry-run", action="store_true", help="Don't embed or write; print counts + sample.")
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Don't call claude -p for metaphor enrichment; use pre-enrichment embed_text.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        build_index(config, dry_run=args.dry_run, skip_enrichment=args.skip_enrichment)
        return 0
    except RuntimeError as e:
        logger.error(f"❌ {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
