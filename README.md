# Inspiration Illustration Finder

Paste an idea, quote, or paragraph — get 10 ranked illustration suggestions from your Notion archive, each from a different *visual metaphor family*, so you see a palette of framings instead of ten variations of the same barchart.

## Why this exists

~1,500 illustrations live in Notion, each tagged with a narrow theme ("psychological safety", "burnout", "decision-making"). When drafting a post or a slide, the hard question is never "find me the illustration filed under X" — it's "find me an illustration whose *underlying metaphor* fits what I'm saying, even if it was filed under something else entirely."

A naive embedding index clusters by surface description. Search for "psychological safety" and you get ten illustrations already tagged with psychological safety — useful but obvious. The system misses a *cauldron* illustration that frames safety as "complex outcomes require specific ingredients", or a *traffic signposts* illustration that frames it as "clarity of direction as a form of safety". Those lateral matches are where the idea-to-visual leap happens, and they were invisible.

This project fixes that by running every illustration through a **metaphor enrichment** pass before embedding — an LLM extracts the *underlying structure* of each image (what it could represent, regardless of what it's literally showing), and those abstractions become first-class citizens in the embedding text. Cosine similarity then operates in metaphor-space, not description-space.

### Concrete effect

For the query "burning out from overcommitment", top-10 results now include:

- **batteries spent unused** — *depletion from overuse* (direct)
- **tetris full - urgent what really matters** — *overload from filling every gap with the urgent* (lateral)
- **raw overcooked chicken - sweet spot stress** — *optimal point between two extremes* (far lateral)
- **coal diamond dust - pressure** — *optimal stress produces transformation* (far lateral)
- **hands up none - commit to nothing distracted by everything** — *extremes both lead to paralysis* (lateral — frames the opposite of overcommitment)

Pre-enrichment, only the direct match would have surfaced. The cooking, geology, and game-mechanics framings were filed under their original themes and would never have come up for a "burnout" query. Now they do, because the enrichment pass captured that they're *about* overload and optimal-range thinking, regardless of their literal subject.

## What's inside

Three linked Streamlit pages, one library of logic:

| Page | What it does |
|---|---|
| 🔍 **Search** | Paste text → embed with bge-large query prefix → cosine-rank the 1,024-dim index → dedup by visual type → render top N as cards with thumbnails, breadcrumbs, match bars, ALT text, enrichment fields, and an Open folder button. |
| 🗂 **Samples** | Rebuild the `inspiration samples/` folder on disk — one PNG per visual type, flat + nested layouts, with an XLS index. Plan (dry-run) or apply. Live log. |
| 🧠 **Build** | Full rebuild: query Notion → enrich via Anthropic SDK → local hub at `127.0.0.1:8000`, model `claude-haiku-4-5` (cached, incremental) → compose enriched embed text → embed with `BAAI/bge-large-en-v1.5` → persist parquet + npy + meta. Live log + progress bar. |

## How it works

Three phases. The first two are offline and incremental; the third is live per-query.

### Phase 1 — Metaphor enrichment

*Where:* `src/enrichment.py` · *Trigger:* Build page, or `python -m src.enrichment`

For each illustration, the Anthropic SDK routes a request to the local hub at `http://127.0.0.1:8000` (model `claude-haiku-4-5`) to return a rich JSON object:

```json
{
  "id": "<illustration_id>",
  "visual_elements": ["coal", "diamond", "dust"],
  "metaphorical_meanings": [
    "optimal stress produces transformation",
    "pressure as creative force",
    "pressure without limits destroys"
  ],
  "applicable_themes": [
    "deliberate practice", "burnout", "transformation",
    "resilience", "creativity under constraint", "grit",
    "feedback loops", "managed stress"
  ],
  "tone": "reflective",
  "abstraction_level": "high"
}
```

The prompt is explicit about the goal: "output the underlying metaphor structure, not the literal scene" — with good/bad examples inline. The LLM is instructed to brainstorm 5–10 applicable themes *beyond* the original theme tag; that's the whole point of the pass.

**Batching**: 15 rows per hub call (~100 calls for 1,500 rows). Smaller than the obvious 20 to leave context headroom for the richer output schema.

**Caching**: results appended incrementally to `index/enrichments.jsonl`, one object per line, keyed by `illustration_id`. Re-runs skip anything already cached — adding 50 new Notion rows costs ~4 calls, not the whole 100. If a call fails mid-run, rerunning picks up from the last batch.

**Prerequisite**: the local LLM hub must be running at `127.0.0.1:8000`. Enrichment calls are routed through the hub, which proxies to Claude Code using your existing subscription — no separate API key or per-call billing, but the hub process must be up.

### Phase 2 — Offline indexing

*Where:* `src/build_index.py` · *Trigger:* Build page, or `python -m src.build_index`

1. Pull the three Notion databases (concepts / visual types / illustrations), join them.
2. Call `enrich_rows(...)` — reads the cache, calls the hub only for uncached rows.
3. Compose the embed text, putting enriched fields *first*:
   ```
   Could represent: optimal stress produces transformation; pressure as creative force.
   Applies to: deliberate practice, burnout, transformation, resilience, …
   Tone: reflective.
   coal diamond dust - pressure. <ALT text>. Themes: growth, pressure.
   ```
   Early tokens carry slightly more weight in sentence-transformer models — so putting the enriched abstractions first shifts the embedding away from literal-scene matching and toward metaphor-space.
4. Embed with `BAAI/bge-large-en-v1.5` (1024-dim unit vectors).
5. Persist to `index/`:
   - `illustrations.parquet` — all row metadata + enriched columns (`metaphorical_meanings`, `applicable_themes`, `tone`, `abstraction_level`)
   - `embeddings.npy` — float32 matrix, shape `(1507, 1024)` (~6 MB)
   - `index_meta.json` — model name, dim, count, `enriched_count`, `built_at`

### Phase 3 — Interactive search

*Where:* `src/pages/search.py` · *Trigger:* paste text, click Find illustrations

1. Prefix the query with bge's retrieval instruction (`"Represent this sentence for searching relevant passages: …"`) — applied only on the query side, per the model card. Document embeddings are stored without a prefix.
2. Embed the query (sub-100ms on CPU after the first warm-up).
3. Dot product against the matrix = cosine similarity (both sides are L2-normalised). Sub-millisecond on 1,507 rows.
4. Walk the ranked list keeping only the *highest-scoring* illustration per `visual_type_id`. Without this dedup, the top 10 for "reaching a goal" would be ten near-identical mountains. The enrichment pass handles *metaphor* diversity; this walk handles *visual-rendering* diversity.
5. Render the top N as cards.

Once built, the index is self-contained. The Notion API and Claude Code are not touched again until you rebuild.

## Project layout

```
inspiration-system/
├── app/
│   └── app.py                Streamlit entry — st.navigation + home page
├── src/
│   ├── menu.py               Page registry + render_home()
│   ├── ui_utils.py           StreamlitLogHandler + open_in_explorer
│   ├── notion_client.py      Notion pull + field extraction
│   ├── build_index.py        Notion → join → enrich → embed → persist (lib + CLI)
│   ├── enrichment.py         Metaphor enrichment via local hub (lib + CLI)
│   ├── sample_illustrations.py  Plan + apply the curated-samples folder (lib + CLI)
│   └── pages/
│       ├── search.py
│       ├── curated_samples.py
│       └── rebuild_index.py
├── tests/                    pytest suite for pure logic + the log handler
├── config.json               DB IDs, source/dest paths, model
├── requirements.txt
├── .streamlit/config.toml    disables the module-introspecting file watcher
├── .env                      NOTION_API_TOKEN (gitignored)
├── run_app.bat               launches Streamlit
└── LICENSE                   MIT
```

Streamlit page scripts live under `src/pages/` and are registered explicitly
via `st.navigation(st.Page(...))` in `app/app.py`. All shared logic stays under
`src/`.

## Live-log pattern

`src/ui_utils.StreamlitLogHandler` is a stdlib `logging.Handler` that rewrites
an `st.empty()` → `st.code()` container on every log record. Modules use the
standard `logger.info()` / `logger.warning()` API; the handler is attached by
the pages that run long jobs (Build, Samples) in a `try/finally`
so handlers don't accumulate across reruns.

Log format follows the externalrisk convention:

```
%(asctime)s - %(levelname)s - %(message)s
```

with emoji-prefixed messages (🔎 🧠 ⚙️ 📊 ✅ ⚠️ ❌) that read cleanly both in
the terminal and in the in-app viewer.

## First-time setup

```
py -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then copy `.env.example` → `.env` and paste your Notion integration token:

```
NOTION_API_TOKEN=secret_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Copy `config.example.json` → `config.json` and fill in your Notion DB IDs and local folder paths:

```
cp config.example.json config.json
# then open config.json and set:
#   illustrations_db_id, visual_types_db_id, concepts_db_id  — from your Notion DB URLs
#   source_folder       — absolute path to your local PNG archive
#   index_folder        — where the built index files will live
#   samples_dest_folder / samples_xls_output  — destination for the Samples page
```

The app will fail with "Missing config.json" on first launch if this step is skipped.

**Enrichment prerequisite:** the **Build** page runs metaphor enrichment through the local LLM hub at `127.0.0.1:8000`. The hub must be running before starting a full build — see the [`claude-local-calls`](../claude-local-calls) repo. (`--skip-enrichment` bypasses this when you only need to re-embed an already-enriched index.)

## Usage

Launch the app:

```
run_app.bat
```

Opens `http://localhost:8501`. From the sidebar:

- **Build** — start here the first time. The **Dry run** tab verifies the Notion connection without writing anything; **Full build** pulls Notion, runs metaphor enrichment via the local hub at `127.0.0.1:8000` (cached — only new rows incur hub calls), and embeds with `BAAI/bge-large-en-v1.5` (first run downloads ~1.3GB). **Requires the local hub to be running.**
- **Search** — paste text, hit the button, browse results.
- **Samples** — **Plan** to preview what would be written, then **Rebuild** to wipe the destination folder and write the flat + nested PNGs plus `samples_index.xlsx`.

Or use the CLIs directly:

```
python -m src.build_index                          Full index build (enrichment + embed)
python -m src.build_index --dry-run                Planning only
python -m src.build_index --skip-enrichment        Skip enrichment hub calls (dev iteration)

python -m src.enrichment                           Run only the enrichment pass, cache to jsonl
python -m src.enrichment --sample 20               Prototype on the first 20 Notion rows
python -m src.enrichment --batch-size 15           Tune batch size (default 15)

python -m src.sample_illustrations                 Full sample rebuild
python -m src.sample_illustrations --dry-run       Planning only
```

## Config

[config.json](config.json) — Notion DB IDs, source PNG folder, model name,
index folder, sample-folder paths. Defaults point at the iCloud Affinity
Designer archive.

To swap the embedding model (e.g. a multilingual variant), change `embed_model`
and rerun the index build.

## Tests

```
& .\.venv\Scripts\python.exe -m pytest
```

The suite covers:

- `src.build_index.build_embed_text` — composition edge cases (title only, ALT only, empty inputs, trailing-dot stripping, enrichment prepended, enrichment=None back-compat).
- `src.enrichment._split_batches` / `_parse_envelope` / `_load_cache` — batch splitting, LLM response parsing (raw array, plain text, fenced JSON, is_error), cache round-trip and malformed-line tolerance.
- `src.enrichment.enrich_rows` — end-to-end with an injected caller: full run writes cache, resume skips cached rows, all-cached makes zero calls.
- `src.sample_illustrations.sanitize` / `extract_topic` — Windows-illegal-char removal, empty-fallback, visualtype-prefix stripping.
- `src.ui_utils.StreamlitLogHandler` — formatting, tail-to-max-lines, attach/detach round-trip with a fake container.

36 tests, ~1s. No Notion, no hub calls, no model downloads.

## Notes

- **Two diversity mechanisms, one at each end.** Enrichment gives the embedding space a richer *metaphor* axis so lateral matches appear at all; the visual-type dedup in the search walk prevents the final top-N from collapsing into ten near-identical renderings of the same idea. You need both: enrichment without dedup surfaces lateral metaphors but still lets barcharts dominate; dedup without enrichment gives you diverse renderings of the *obvious* interpretation only.
- **Enrichment is resumable and idempotent.** `index/enrichments.jsonl` is append-only, one object per line, keyed by `illustration_id`. `enrich_rows(...)` loads the cache, filters to uncached, and appends new entries after each batch. Interrupt at any point, rerun, it picks up where it left off. Adding 50 new Notion rows makes ~4 hub calls, not ~100.
- **bge query prefix is conditional.** `src/pages/search.py` prefixes the query with `"Represent this sentence for searching relevant passages: "` only when the indexed model name contains `bge`. Swap to a different model via `config.json` and the prefix is skipped automatically.
- **No torchvision required.** We use sentence-transformers for text embeddings only. Streamlit's default file watcher used to introspect every `transformers.models.*.image_processing_*` module at startup, which tried to `import torchvision` and flooded the console with tracebacks. `.streamlit/config.toml` sets `fileWatcherType = "none"` to skip that introspection.
- **Missing PNGs aren't fatal.** Each indexed row has a `missing_png` flag; the Search page filters them out at ranking time so broken results never reach the UI. The sample rebuild logs how many planned samples have no matching `.png` in the source folder (rows are still written to the XLS index for debugging).

## License

[MIT](LICENSE)
