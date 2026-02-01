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
└── static/
    ├── index.html   # Search frontend
    └── admin.html   # Admin panel (model switching, settings)

data/
├── kpis.parquet              # KPI data (id, title, description)
├── embeddings_*.parquet      # Per-model embeddings (title + description separate)
├── admin_config.json         # Current model, title_weight
└── admin_password.txt        # Hashed admin password (gitignored)
```

## Commands

```bash
uv sync                                    # Install dependencies
uv run uvicorn kpisearch.app:app --reload  # Run dev server
uv run python -m kpisearch.search build    # Rebuild embeddings
uv run python -m kpisearch.auth set <pw>   # Set admin password
```

## Models

Three embedding models available (switchable at runtime via admin):

| Model | Notes |
|-------|-------|
| KBLab/sentence-bert-swedish-cased | Swedish-specific (default) |
| intfloat/multilingual-e5-large | Best quality, uses query:/passage: prefixes |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Lightweight multilingual |

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
