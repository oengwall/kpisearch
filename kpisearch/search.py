import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer

from kpisearch import KPI_PARQUET_PATH
from kpisearch.admin_config import EmbeddingModel, get_current_model, get_embeddings_path, get_title_weight


def load_model(model_name: str, local_only: bool = True) -> SentenceTransformer:
    """Load a sentence transformer model."""
    return SentenceTransformer(model_name, local_files_only=local_only)


def create_embeddings(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Create embeddings for a list of texts."""
    return model.encode(texts, show_progress_bar=True)


def build_embeddings_index(model: EmbeddingModel | None = None) -> None:
    """Build embeddings for all KPIs and save to parquet."""
    if model is None:
        model = get_current_model()

    model_name = model.value
    embeddings_path = get_embeddings_path(model)

    print(f'Loading model: {model_name}')
    transformer = load_model(model_name, local_only=False)

    print(f'Loading KPIs from {KPI_PARQUET_PATH}')
    kpis = pl.read_parquet(KPI_PARQUET_PATH)

    # Prepare texts - add E5 prefix if needed
    prefix = 'passage: ' if model.uses_e5_prefix else ''

    titles = [f'{prefix}{row["title"]}' for row in kpis.select('title').iter_rows(named=True)]
    descriptions = [f'{prefix}{row["description"]}' for row in kpis.select('description').iter_rows(named=True)]

    print(f'Creating title embeddings for {len(titles)} KPIs...')
    title_embeddings = create_embeddings(transformer, titles).astype(np.float32)

    print(f'Creating description embeddings for {len(descriptions)} KPIs...')
    description_embeddings = create_embeddings(transformer, descriptions).astype(np.float32)

    # Store embeddings alongside KPI ids (using pl.Array to preserve float32 dtype)
    embedding_dim = title_embeddings.shape[1]
    embeddings_df = pl.DataFrame(
        {
            'id': kpis['id'],
            'title_embedding': pl.Series(title_embeddings.tolist(), dtype=pl.Array(pl.Float32, embedding_dim)),
            'description_embedding': pl.Series(
                description_embeddings.tolist(), dtype=pl.Array(pl.Float32, embedding_dim)
            ),
        }
    )

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_df.write_parquet(embeddings_path)
    print(f'Saved embeddings to {embeddings_path}')


class KpiSearcher:
    """Vector similarity search for KPIs."""

    model: SentenceTransformer
    model_enum: EmbeddingModel
    kpis: pl.DataFrame
    title_embeddings: np.ndarray
    description_embeddings: np.ndarray
    kpi_ids: list[str]

    def __init__(self, model: EmbeddingModel | None = None) -> None:
        if model is None:
            model = get_current_model()

        self.model_enum = model
        embeddings_path = get_embeddings_path(model)

        # Build embeddings if they don't exist
        if not embeddings_path.exists():
            print(f'Embeddings not found for {model.value}. Building...')
            build_embeddings_index(model)

        self.model = load_model(model.value)
        self.kpis = pl.read_parquet(KPI_PARQUET_PATH)
        embeddings_df = pl.read_parquet(embeddings_path)

        # Check if we have the new format (separate title/description) or old format
        if 'title_embedding' in embeddings_df.columns:
            self.title_embeddings = np.array(embeddings_df['title_embedding'].to_list())
            self.description_embeddings = np.array(embeddings_df['description_embedding'].to_list())
        else:
            # Old format - rebuild with new format
            print('Old embedding format detected. Rebuilding with new format...')
            build_embeddings_index(model)
            embeddings_df = pl.read_parquet(embeddings_path)
            self.title_embeddings = np.array(embeddings_df['title_embedding'].to_list())
            self.description_embeddings = np.array(embeddings_df['description_embedding'].to_list())

        self.kpi_ids = embeddings_df['id'].to_list()

    def search(
        self, query: str, top_k: int = 10, min_score: float = 0.4, title_weight: float | None = None
    ) -> pl.DataFrame:
        """Search for KPIs matching the query.

        Args:
            query: Search query text
            top_k: Maximum number of results
            min_score: Minimum similarity score threshold
            title_weight: Weight for title similarity (0-1). Description weight is 1 - title_weight.
                         If None, uses the configured default.
        """
        if title_weight is None:
            title_weight = get_title_weight()

        # Add E5 prefix if needed
        if self.model_enum.uses_e5_prefix:
            query = f'query: {query}'

        query_embedding = self.model.encode([query])[0]

        # Compute cosine similarities for both title and description
        title_similarities = self._cosine_similarity(query_embedding, self.title_embeddings)
        description_similarities = self._cosine_similarity(query_embedding, self.description_embeddings)

        # Combine with weighting
        similarities = title_weight * title_similarities + (1 - title_weight) * description_similarities

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Get matching KPI ids and scores, filtered by min_score
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score < min_score:
                continue
            kpi_id = self.kpi_ids[idx]
            results.append({'id': kpi_id, 'score': float(score)})

        if not results:
            return pl.DataFrame(schema={'id': pl.Utf8, 'score': pl.Float64, 'title': pl.Utf8, 'description': pl.Utf8})

        results_df = pl.DataFrame(results)

        # Join with KPI data to get title and description
        return results_df.join(
            self.kpis.select('id', 'title', 'description'),
            on='id',
        ).sort('score', descending=True)

    def _cosine_similarity(self, query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all embeddings."""
        query_norm = query / np.linalg.norm(query)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings_norm @ query_norm


def build_all_embeddings() -> None:
    """Build embeddings for all available models."""
    for model in EmbeddingModel:
        print(f'\n{"="*60}')
        print(f'Building embeddings for: {model.display_name}')
        print(f'{"="*60}\n')
        build_embeddings_index(model)
    print('\nAll embeddings built successfully.')


def update_embeddings(
    model: EmbeddingModel,
    added_ids: list[str],
    changed_ids: list[str],
    deleted_ids: list[str],
) -> bool:
    """Incrementally update embeddings for a model.

    Args:
        model: The embedding model to update
        added_ids: KPI IDs that were added
        changed_ids: KPI IDs that were changed
        deleted_ids: KPI IDs that were deleted

    Returns:
        True if update succeeded, False if model not cached locally
    """
    embeddings_path = get_embeddings_path(model)

    # If no embeddings file exists, skip (user needs to run full build first)
    if not embeddings_path.exists():
        print(f'No embeddings found for {model.display_name}, skipping')
        return False

    # Try to load model locally (no download)
    try:
        transformer = load_model(model.value, local_only=True)
    except OSError as e:
        if 'local_files_only' in str(e) or 'does not appear to have' in str(e):
            print(f'Model {model.display_name} not cached locally, skipping')
            return False
        raise

    ids_to_update = set(added_ids) | set(changed_ids)
    ids_to_remove = set(deleted_ids) | set(changed_ids)

    if not ids_to_update and not ids_to_remove:
        print(f'{model.display_name}: No changes to apply')
        return True

    # Load existing embeddings
    embeddings_df = pl.read_parquet(embeddings_path)

    # Remove deleted and changed rows
    if ids_to_remove:
        embeddings_df = embeddings_df.filter(~pl.col('id').is_in(list(ids_to_remove)))
        print(f'{model.display_name}: Removed {len(ids_to_remove)} embeddings')

    # Compute new embeddings for added and changed KPIs
    if ids_to_update:
        kpis = pl.read_parquet(KPI_PARQUET_PATH)
        kpis_to_embed = kpis.filter(pl.col('id').is_in(list(ids_to_update)))

        prefix = 'passage: ' if model.uses_e5_prefix else ''

        titles = [f'{prefix}{row["title"]}' for row in kpis_to_embed.select('title').iter_rows(named=True)]
        descriptions = [
            f'{prefix}{row["description"]}' for row in kpis_to_embed.select('description').iter_rows(named=True)
        ]

        print(f'{model.display_name}: Computing embeddings for {len(titles)} KPIs...')
        title_embeddings = create_embeddings(transformer, titles).astype(np.float32)
        description_embeddings = create_embeddings(transformer, descriptions).astype(np.float32)

        # Create DataFrame for new embeddings
        embedding_dim = title_embeddings.shape[1]
        new_embeddings_df = pl.DataFrame(
            {
                'id': kpis_to_embed['id'],
                'title_embedding': pl.Series(title_embeddings.tolist(), dtype=pl.Array(pl.Float32, embedding_dim)),
                'description_embedding': pl.Series(
                    description_embeddings.tolist(), dtype=pl.Array(pl.Float32, embedding_dim)
                ),
            }
        )

        # Append new embeddings
        embeddings_df = pl.concat([embeddings_df, new_embeddings_df])
        print(f'{model.display_name}: Added {len(new_embeddings_df)} embeddings')

    # Save updated embeddings
    embeddings_df.write_parquet(embeddings_path)
    print(f'{model.display_name}: Saved {len(embeddings_df)} total embeddings')

    return True


def sync_embeddings(added_ids: list[str], changed_ids: list[str], deleted_ids: list[str]) -> None:
    """Update embeddings for all models that have cached embeddings."""
    for model in EmbeddingModel:
        print(f'\n--- {model.display_name} ---')
        update_embeddings(model, added_ids, changed_ids, deleted_ids)


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'build-all':
        build_all_embeddings()
    elif len(sys.argv) > 1 and sys.argv[1] == 'build':
        # Optional: specify model as second argument
        if len(sys.argv) > 2:
            model = EmbeddingModel(sys.argv[2])
        else:
            model = get_current_model()
        build_embeddings_index(model)
    elif len(sys.argv) > 1 and sys.argv[1] == 'sync':
        # Incremental sync - requires IDs as JSON on stdin or run via kpisearch.sync
        print('Use `python -m kpisearch.sync` for full incremental sync.')
        print('This command is meant to be called programmatically.')
    else:
        model = get_current_model()
        embeddings_path = get_embeddings_path(model)
        if not embeddings_path.exists():
            print('Embeddings not found. Building index first...')
            build_embeddings_index(model)

        searcher = KpiSearcher(model)

        query = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else 'skolresultat'
        print(f'\nSearching for: {query}\n')

        results = searcher.search(query, top_k=5)
        for row in results.iter_rows(named=True):
            print(f'[{row["score"]:.3f}] {row["id"]}: {row["title"]}')
            print(f'         {row["description"][:100]}...')
            print()


if __name__ == '__main__':
    main()
