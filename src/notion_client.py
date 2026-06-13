"""Notion pull + field extraction helpers for the inspiration-system project.

Ported/extended from automation/notion/sample_illustrations.py to keep this
project self-contained (no cross-project imports).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

NOTION_VERSION = "2022-06-28"


def load_config(config_path: str) -> Dict[str, Any]:
    paths_to_try = [
        config_path,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(config_path)),
    ]
    for path in paths_to_try:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"Config not found at {' or '.join(paths_to_try)}")


def resolve_token(config: Dict[str, Any]) -> str:
    raw = config.get("notion_api_key", "")
    if isinstance(raw, str) and raw.startswith("${") and raw.endswith("}"):
        token = os.getenv(raw[2:-1])
    else:
        token = raw or os.getenv("NOTION_API_TOKEN")
    if not token:
        raise ValueError("NOTION_API_TOKEN not available (env var or config)")
    return token


def build_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def query_database(db_id: str, headers: Dict[str, str]) -> List[Dict[str, Any]]:
    url = f"https://api.notion.com/v1/databases/{db_id}/query"
    pages: List[Dict[str, Any]] = []
    body: Dict[str, Any] = {"page_size": 100}
    while True:
        r = requests.post(url, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
        pages.extend(data.get("results", []))
        if not data.get("has_more"):
            break
        body["start_cursor"] = data["next_cursor"]
    logging.info(f"Fetched {len(pages)} pages from db {db_id[:8]}…")
    return pages


def extract_title(prop: Dict[str, Any]) -> str:
    return "".join(s.get("plain_text", "") for s in prop.get("title", [])).strip()


def extract_rich_text(prop: Dict[str, Any]) -> str:
    return "".join(s.get("plain_text", "") for s in prop.get("rich_text", [])).strip()


def extract_relation_ids(prop: Dict[str, Any]) -> List[str]:
    return [r["id"] for r in prop.get("relation", [])]


def extract_multi_select(prop: Dict[str, Any]) -> List[str]:
    return [x["name"] for x in prop.get("multi_select", [])]


def extract_select(prop: Dict[str, Any]) -> Optional[str]:
    sel = prop.get("select")
    return sel["name"] if sel else None


def build_concepts_map(pages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for p in pages:
        props = p.get("properties", {})
        types = extract_multi_select(props.get("type", {}))
        out[p["id"]] = {
            "concept": extract_title(props.get("concept", {})),
            "types": types,
            "concept_type": "+".join(types) if types else "untyped",
        }
    return out


def build_visual_types_map(pages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for p in pages:
        props = p.get("properties", {})
        concept_ids = extract_relation_ids(props.get("concept", {}))
        out[p["id"]] = {
            "visualtype": extract_title(props.get("visualtype", {})),
            "concept_id": concept_ids[0] if concept_ids else None,
            "illustration_ids": extract_relation_ids(props.get("illustrations", {})),
        }
    return out


def build_illustrations_map(pages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Reads the fields needed for semantic search + UI rendering.

    Reads more fields than automation/notion/sample_illustrations.py — notably
    `ALT text` (rich_text) and the `type` relation to visual_types.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for p in pages:
        props = p.get("properties", {})
        visual_type_ids = extract_relation_ids(props.get("type", {}))
        out[p["id"]] = {
            "title": extract_title(props.get("illustration", {})),
            "alt_text": extract_rich_text(props.get("ALT text", {})),
            "theme": extract_multi_select(props.get("theme", {})),
            "tags": extract_select(props.get("Tags", {})),
            "visual_type_id": visual_type_ids[0] if visual_type_ids else None,
            "created": p.get("created_time"),
        }
    return out


def fetch_notion_maps(
    config: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Resolve the Notion token, build headers, and query the three databases.

    Returns ``(concepts_map, visual_types_map, illustrations_map)`` in that
    order — the canonical triple used by ``build_index``, ``enrichment``, and
    ``sample_illustrations``.  Callers that previously duplicated the
    ``resolve_token + build_headers + three-query`` sequence should call this
    instead.
    """
    token = resolve_token(config)
    headers = build_headers(token)
    concepts = build_concepts_map(query_database(config["concepts_db_id"], headers))
    visual_types = build_visual_types_map(query_database(config["visual_types_db_id"], headers))
    illustrations = build_illustrations_map(query_database(config["illustrations_db_id"], headers))
    return concepts, visual_types, illustrations
