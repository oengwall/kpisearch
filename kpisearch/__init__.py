from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'
KPI_PARQUET_PATH = DATA_DIR / 'kpis.parquet'
KPI_EMBEDDINGS_PATH = DATA_DIR / 'kpi_embeddings.parquet'
