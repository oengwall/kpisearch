# KPI Search

Semantic search for Swedish KPIs from [Kolada](https://www.kolada.se/), using vector embeddings.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
uv sync
```

This will install dependencies and download the required Python version if not already available.

## Running the application

Start the FastAPI server:

```bash
uv run uvicorn kpisearch.app:app --reload
```

Open http://localhost:8000 in your browser.

## Data pipeline

### Incremental sync (recommended)

Sync KPIs from Kolada API and update embeddings for changed items only:

```bash
uv run python -m kpisearch.sync
```

This detects added, changed, and deleted KPIs using content hashes, and only recomputes embeddings for what changed. Uses locally cached models only (no downloads).

### Full rebuild

Download all KPIs from Kolada API:

```bash
uv run python -m kpisearch.download_kpis
```

Build the embeddings index for current model:

```bash
uv run python -m kpisearch.search build
```

Build embeddings for all models:

```bash
uv run python -m kpisearch.search build-all
```

## Development

Format code:

```bash
uv run ruff format kpisearch/
```

Lint and auto-fix:

```bash
uv run ruff check --fix kpisearch/
```

Type-check:

```bash
uv run ty check kpisearch/
```

## API

### GET /api/search

Search for KPIs matching a query.

**Parameters:**
- `q` (required) - Search query
- `limit` (default: 10) - Max results (1-50)
- `min_score` (default: 0.4) - Minimum similarity score (0-1)

**Example:**

```bash
curl "http://localhost:8000/api/search?q=skolresultat&limit=5"
```

## Models

The application supports multiple embedding models that can be switched at runtime:

| Model | Description |
|-------|-------------|
| KBLab/sentence-bert-swedish-cased | Swedish-specific, optimized for Swedish text (default) |
| intfloat/multilingual-e5-small | High-quality multilingual, supports 100+ languages |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Lightweight multilingual, good speed/quality balance |

## Admin

Admin page available at `/admin`. Protected with HTTP Basic Auth.

Default password: `change_this_now_really!`

### Change admin password

```bash
# Set new password
uv run python -m kpisearch.auth set <new-password>

# Reset to default
uv run python -m kpisearch.auth reset
```

### Admin API

```bash
# List models
curl -u admin:password http://localhost:8000/admin/models

# Switch model
curl -u admin:password -X POST http://localhost:8000/admin/models \
  -H "Content-Type: application/json" \
  -d '{"model_id": "intfloat/multilingual-e5-large"}'
```
