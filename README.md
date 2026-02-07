# KPI Search

Semantic search for Swedish KPIs from [Kolada](https://www.kolada.se/), using vector embeddings.

## Quick Start

Requires [uv](https://docs.astral.sh/uv/) package manager.

```bash
uv sync                                       # Install dependencies
uv run python -m kpisearch.sync               # Download KPIs + build embeddings
uv run python -m kpisearch.search build-all   # Build embeddings for all models
uv run uvicorn kpisearch.app:app              # Start server
```

Open http://localhost:8000 in your browser.

## Data Pipeline

The sync command is the easiest way to get started. It fetches KPIs from the Kolada API, detects changes using content hashes, and only recomputes embeddings for what changed.

```bash
uv run python -m kpisearch.sync
```

For a full rebuild instead (e.g. after switching models):

```bash
uv run python -m kpisearch.download_kpis     # Re-download all KPIs
uv run python -m kpisearch.search build       # Rebuild embeddings for current model
uv run python -m kpisearch.search build-all   # Rebuild embeddings for all models
```

## Search Methods

The frontend offers three search methods (toggleable via checkboxes):

| Method | How it works |
|--------|-------------|
| **Semantisk** | Pure vector similarity search |
| **Hybrid** | Semantic search with additive keyword boost for title matches |
| **Kolada API** | Proxied title search via Kolada's own API |

## API

### GET /api/search

Semantic search. Parameters: `q` (required), `limit` (default 10), `min_score` (default 0.4), `title_weight` (0-1).

### GET /api/hybrid-search

Hybrid search. Parameters: `q` (required), `limit` (default 10), `title_weight` (0-1).

### GET /api/kolada-search

Proxied Kolada API search. Parameters: `q` (required), `limit` (default 15).

```bash
curl "http://localhost:8000/api/search?q=skolresultat&limit=5"
```

## Models

Three embedding models, switchable at runtime via the admin panel:

| Model | Notes |
|-------|-------|
| KBLab/sentence-bert-swedish-cased | Swedish-specific (default) |
| intfloat/multilingual-e5-small | High-quality multilingual |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Lightweight multilingual |

## Admin

Admin panel at `/admin`, protected with HTTP Basic Auth.

Default password: `change_this_now_really!` (forced change on first login).

```bash
uv run python -m kpisearch.auth set <new-password>   # Set password
uv run python -m kpisearch.auth reset                 # Reset to default
```

## Development

```bash
uv run uvicorn kpisearch.app:app --reload   # Dev server with auto-reload
uv run ruff check kpisearch/                # Lint
uv run ruff format kpisearch/               # Format
uv run ty check                             # Type check
```
