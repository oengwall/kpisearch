import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer

from kpisearch import KPI_EMBEDDINGS_PATH, KPI_PARQUET_PATH

MODEL_NAME = 'KBLab/sentence-bert-swedish-cased'


def load_model(local_only: bool = True) -> SentenceTransformer:
    """Load the Swedish sentence transformer model."""
    return SentenceTransformer(MODEL_NAME, local_files_only=local_only)


def create_embeddings(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Create embeddings for a list of texts."""
    return model.encode(texts, show_progress_bar=True)


def build_embeddings_index() -> None:
    """Build embeddings for all KPIs and save to parquet."""
    print(f'Loading model: {MODEL_NAME}')
    model = load_model(local_only=False)  # Allow download when building index

    print(f'Loading KPIs from {KPI_PARQUET_PATH}')
    kpis = pl.read_parquet(KPI_PARQUET_PATH)

    # Combine title and description for embedding
    texts = [
        f'{row["title"]}. {row["description"]}' for row in kpis.select('title', 'description').iter_rows(named=True)
    ]

    print(f'Creating embeddings for {len(texts)} KPIs...')
    embeddings = create_embeddings(model, texts)

    # Store embeddings alongside KPI ids
    embeddings_df = pl.DataFrame(
        {
            'id': kpis['id'],
            'embedding': embeddings.tolist(),
        }
    )

    KPI_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    embeddings_df.write_parquet(KPI_EMBEDDINGS_PATH)
    print(f'Saved embeddings to {KPI_EMBEDDINGS_PATH}')


class KpiSearcher:
    """Vector similarity search for KPIs."""

    model: SentenceTransformer
    kpis: pl.DataFrame
    embeddings: np.ndarray
    kpi_ids: list[str]

    def __init__(self) -> None:
        self.model = load_model()
        self.kpis = pl.read_parquet(KPI_PARQUET_PATH)
        embeddings_df = pl.read_parquet(KPI_EMBEDDINGS_PATH)
        self.embeddings = np.array(embeddings_df['embedding'].to_list())
        self.kpi_ids = embeddings_df['id'].to_list()

    def search(self, query: str, top_k: int = 10, min_score: float = 0.4) -> pl.DataFrame:
        """Search for KPIs matching the query."""
        query_embedding = self.model.encode([query])[0]

        # Compute cosine similarities
        similarities = self._cosine_similarity(query_embedding, self.embeddings)

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


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'build':
        build_embeddings_index()
    else:
        if not KPI_EMBEDDINGS_PATH.exists():
            print('Embeddings not found. Building index first...')
            build_embeddings_index()

        searcher = KpiSearcher()

        query = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else 'skolresultat'
        print(f'\nSearching for: {query}\n')

        results = searcher.search(query, top_k=5)
        for row in results.iter_rows(named=True):
            print(f'[{row["score"]:.3f}] {row["id"]}: {row["title"]}')
            print(f'         {row["description"][:100]}...')
            print()


if __name__ == '__main__':
    main()
