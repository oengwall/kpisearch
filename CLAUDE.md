# KPI Search

Semantic search for Swedish municipal KPIs from Kolada.

## Stack

- **Backend**: FastAPI (Python 3.14)
- **Embeddings**: sentence-transformers (HuggingFace models)
- **Data**: Polars DataFrames, Parquet files
- **Frontend**: Vanilla HTML/JS (no build step)
- **Package manager**: uv

## Key Files

```
kpisearch/
├── app.py           # FastAPI routes (search API + admin API)
├── search.py        # KpiSearcher class, embedding creation
├── admin_config.py  # Model enum, config persistence
├── auth.py          # Admin password auth (HTTP Basic)
├── download_kpis.py # Fetch KPIs from Kolada API
├── sync.py          # Unified incremental sync command
└── static/
    ├── index.html   # Search frontend
    └── admin.html   # Admin panel (model switching, settings)

data/
├── kpis.parquet              # KPI data (id, title, description, content_hash)
├── embeddings_*.parquet      # Per-model embeddings (title + description separate)
├── admin_config.json         # Current model, title_weight
└── admin_password.txt        # Hashed admin password (gitignored)
```

## Commands

```bash
uv sync                                        # Install dependencies
uv run uvicorn kpisearch.app:app --reload      # Run dev server
uv run python -m kpisearch.sync                # Incremental sync (KPIs + embeddings)
uv run python -m kpisearch.search build-all    # Full rebuild all model embeddings
uv run python -m kpisearch.search build        # Full rebuild current model embeddings
uv run python -m kpisearch.auth set <pw>       # Set admin password
```

## Models

Three embedding models available (switchable at runtime via admin):

| Model | Dim | Notes |
|-------|-----|-------|
| KBLab/sentence-bert-swedish-cased | 768 | Swedish-specific (default) |
| intfloat/multilingual-e5-small | 384 | High-quality multilingual, uses query:/passage: prefixes |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 384 | Lightweight multilingual |

## Search Tuning

- **Title/description weighting**: Separate embeddings, combined with configurable weight (default 60% title)
- **E5 prefix**: Automatically adds `query:` / `passage:` prefixes for E5 model
- **API params**: `title_weight` (0-1) can override default per request

## Admin

- URL: `/admin`
- Default password: `change_this_now_really!` (forced change on first login)
- Features: model switching, title weight slider, password change

## Data Pipeline

1. `download_kpis.py` fetches from Kolada API → `data/kpis.parquet`
2. `search.py build` creates embeddings → `data/embeddings_*.parquet`
3. Embeddings auto-rebuild when switching to a model without cached embeddings

## Incremental Sync

The `sync.py` module provides incremental updates:

1. Fetches KPIs from Kolada API
2. Compares with existing data using `content_hash` (MD5 of title+description)
3. Identifies added, changed, and deleted KPIs
4. Updates embeddings only for changed KPIs (using locally cached models only)

Use `uv run python -m kpisearch.sync` for production updates.
