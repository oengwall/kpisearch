from pathlib import Path

import httpx
import polars as pl

from kpisearch import KPI_PARQUET_PATH

KOLADA_API_BASE = 'https://api.kolada.se/v3'


def fetch_all_kpis() -> list[dict]:
    """Fetch all KPIs from the Kolada API with pagination."""
    kpis = []
    page = 1
    per_page = 5000

    with httpx.Client(timeout=30.0) as client:
        while True:
            response = client.get(
                f'{KOLADA_API_BASE}/kpi',
                params={'page': page, 'per_page': per_page},
            )
            response.raise_for_status()
            data = response.json()

            kpis.extend(data['values'])

            if len(data['values']) < per_page:
                break
            page += 1

    return kpis


def kpis_to_dataframe(kpis: list[dict]) -> pl.DataFrame:
    """Convert KPI list to a polars DataFrame with relevant columns."""
    records = []
    for kpi in kpis:
        records.append(
            {
                'id': kpi.get('id'),
                'title': kpi.get('title'),
                'description': kpi.get('description'),
                'operating_area': kpi.get('operating_area'),
                'municipality_type': kpi.get('municipality_type'),
                'publication_date': kpi.get('publication_date'),
                'prel_publication_date': kpi.get('prel_publication_date'),
                'publ_period': kpi.get('publ_period'),
                'is_divided_by_gender': kpi.get('is_divided_by_gender'),
                'has_ou_data': kpi.get('has_ou_data'),
                'auspices': kpi.get('auspices'),
                'perspective': kpi.get('perspective'),
            }
        )
    return pl.DataFrame(records)


def save_kpis_to_parquet(df: pl.DataFrame, path: Path) -> None:
    """Save DataFrame to parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    print(f'Saved {len(df)} KPIs to {path}')


def main() -> None:
    print('Fetching KPIs from Kolada API...')
    kpis = fetch_all_kpis()
    print(f'Fetched {len(kpis)} KPIs')

    df = kpis_to_dataframe(kpis)
    save_kpis_to_parquet(df, KPI_PARQUET_PATH)

    print('\nSample of KPIs:')
    print(df.select('id', 'title').head(10))


if __name__ == '__main__':
    main()
