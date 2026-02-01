from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from kpisearch.search import KpiSearcher

STATIC_DIR = Path(__file__).parent / 'static'

searcher: KpiSearcher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global searcher
    print('Loading search index...')
    searcher = KpiSearcher()
    print('Search index loaded.')
    yield
    searcher = None


app = FastAPI(title='KPI Search', lifespan=lifespan)


class KpiResult(BaseModel):
    id: str
    score: float
    title: str
    description: str


class SearchResponse(BaseModel):
    query: str
    results: list[KpiResult]


@app.get('/api/search', response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description='Search query'),
    limit: int = Query(10, ge=1, le=50, description='Max results'),
    min_score: float = Query(0.4, ge=0.0, le=1.0, description='Minimum score'),
) -> SearchResponse:
    """Search for KPIs matching the query."""
    assert searcher is not None
    results_df = searcher.search(q, top_k=limit, min_score=min_score)

    results = [
        KpiResult(
            id=row['id'],
            score=row['score'],
            title=row['title'],
            description=row['description'],
        )
        for row in results_df.iter_rows(named=True)
    ]

    return SearchResponse(query=q, results=results)


@app.get('/')
def index() -> FileResponse:
    """Serve the frontend."""
    return FileResponse(STATIC_DIR / 'index.html')


app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
