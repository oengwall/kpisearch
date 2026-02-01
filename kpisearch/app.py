from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from kpisearch.admin_config import (
    EmbeddingModel,
    get_current_model,
    get_embeddings_path,
    get_title_weight,
    set_current_model,
    set_title_weight,
)
from kpisearch.auth import CurrentAdmin, is_default_password, set_admin_password, verify_admin_password
from kpisearch.search import KpiSearcher

STATIC_DIR = Path(__file__).parent / 'static'

searcher: KpiSearcher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global searcher
    model = get_current_model()
    print(f'Loading search index with model: {model.value}...')
    searcher = KpiSearcher(model)
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
    title_weight: float | None = Query(None, ge=0.0, le=1.0, description='Title weight (0-1), uses default if not set'),
) -> SearchResponse:
    """Search for KPIs matching the query."""
    assert searcher is not None
    results_df = searcher.search(q, top_k=limit, min_score=min_score, title_weight=title_weight)

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


# Admin API models


class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    is_current: bool
    has_embeddings: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    current_model: str


class SwitchModelRequest(BaseModel):
    model_id: str


class SwitchModelResponse(BaseModel):
    success: bool
    message: str
    current_model: str


class AdminStatusResponse(BaseModel):
    must_change_password: bool


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class ChangePasswordResponse(BaseModel):
    success: bool
    message: str


class TitleWeightResponse(BaseModel):
    title_weight: float


class SetTitleWeightRequest(BaseModel):
    title_weight: float


# Admin endpoints (basic auth protected)


@app.get('/admin/status', response_model=AdminStatusResponse)
def admin_status(_: CurrentAdmin) -> AdminStatusResponse:
    """Check admin status, including if password must be changed."""
    return AdminStatusResponse(must_change_password=is_default_password())


@app.post('/admin/password', response_model=ChangePasswordResponse)
def change_password(request: ChangePasswordRequest, _: CurrentAdmin) -> ChangePasswordResponse:
    """Change the admin password."""
    if not verify_admin_password(request.current_password):
        return ChangePasswordResponse(success=False, message='Nuvarande lösenord är felaktigt')

    if len(request.new_password) < 8:
        return ChangePasswordResponse(success=False, message='Nytt lösenord måste vara minst 8 tecken')

    set_admin_password(request.new_password)
    return ChangePasswordResponse(success=True, message='Lösenordet har ändrats')


@app.get('/admin/title-weight', response_model=TitleWeightResponse)
def get_title_weight_endpoint(_: CurrentAdmin) -> TitleWeightResponse:
    """Get the current title weight setting."""
    return TitleWeightResponse(title_weight=get_title_weight())


@app.post('/admin/title-weight', response_model=TitleWeightResponse)
def set_title_weight_endpoint(request: SetTitleWeightRequest, _: CurrentAdmin) -> TitleWeightResponse:
    """Set the title weight for search."""
    weight = max(0.0, min(1.0, request.title_weight))
    set_title_weight(weight)
    return TitleWeightResponse(title_weight=weight)


@app.get('/admin/models', response_model=ModelsResponse)
def list_models(_: CurrentAdmin) -> ModelsResponse:
    """List available embedding models."""
    current = get_current_model()
    models = []
    for model in EmbeddingModel:
        models.append(
            ModelInfo(
                id=model.value,
                name=model.display_name,
                description=model.description,
                is_current=(model == current),
                has_embeddings=get_embeddings_path(model).exists(),
            )
        )
    return ModelsResponse(models=models, current_model=current.value)


@app.post('/admin/models', response_model=SwitchModelResponse)
def switch_model(request: SwitchModelRequest, _: CurrentAdmin) -> SwitchModelResponse:
    """Switch to a different embedding model."""
    global searcher

    # Validate model ID
    try:
        new_model = EmbeddingModel(request.model_id)
    except ValueError:
        valid_ids = [m.value for m in EmbeddingModel]
        return SwitchModelResponse(
            success=False,
            message=f'Invalid model ID. Valid options: {valid_ids}',
            current_model=get_current_model().value,
        )

    current = get_current_model()
    if new_model == current:
        return SwitchModelResponse(
            success=True,
            message='Model already active',
            current_model=current.value,
        )

    # Switch model
    set_current_model(new_model)

    # Reload searcher with new model (will build embeddings if needed)
    print(f'Switching to model: {new_model.value}...')
    searcher = KpiSearcher(new_model)
    print('Model switch complete.')

    return SwitchModelResponse(
        success=True,
        message=f'Switched to {new_model.display_name}',
        current_model=new_model.value,
    )


@app.get('/')
def index() -> FileResponse:
    """Serve the frontend."""
    return FileResponse(STATIC_DIR / 'index.html')


@app.get('/admin')
def admin_page() -> FileResponse:
    """Serve the admin frontend."""
    return FileResponse(STATIC_DIR / 'admin.html')


app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
