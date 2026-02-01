# KPI Search

Semantic search for Swedish KPIs from [Kolada](https://www.kolada.se/), using vector embeddings.

## Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
uv sync
```

## Running the application

Start the FastAPI server:

```bash
uv run uvicorn kpisearch.app:app --reload
```

Open http://localhost:8000 in your browser.

## Data pipeline

Download KPIs from Kolada API:

```bash
uv run python -m kpisearch.download_kpis
```

Build the embeddings index:

```bash
uv run python -m kpisearch.search build
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

## Model

Uses [KBLab/sentence-bert-swedish-cased](https://huggingface.co/KBLab/sentence-bert-swedish-cased) for Swedish text embeddings.
