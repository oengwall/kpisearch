import hashlib
from dataclasses import dataclass
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


def compute_content_hash(title: str, description: str) -> str:
    """Compute SHA256 hash of title + description for change detection."""
    content = f'{title or ""}{description or ""}'
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def kpis_to_dataframe(kpis: list[dict]) -> pl.DataFrame:
    """Convert KPI list to a polars DataFrame with relevant columns."""
    records = []
    for kpi in kpis:
        title = kpi.get('title') or ''
        description = kpi.get('description') or ''
        records.append(
            {
                'id': kpi.get('id'),
                'title': title,
                'description': description,
                'content_hash': compute_content_hash(title, description),
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


@dataclass
class SyncResult:
    """Result of syncing KPIs with the Kolada API."""

    added: list[str]
    changed: list[str]
    deleted: list[str]

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.changed or self.deleted)

    def __str__(self) -> str:
        return f'{len(self.added)} added, {len(self.changed)} changed, {len(self.deleted)} deleted'


def sync_kpis() -> SyncResult:
    """Incrementally sync KPIs from Kolada API.

    Returns a SyncResult with lists of added, changed, and deleted KPI IDs.
    """
    print('Fetching KPIs from Kolada API...')
    kpis = fetch_all_kpis()
    print(f'Fetched {len(kpis)} KPIs')

    new_df = kpis_to_dataframe(kpis)

    # Load existing data if it exists
    if KPI_PARQUET_PATH.exists():
        existing_df = pl.read_parquet(KPI_PARQUET_PATH)

        # Ensure existing data has content_hash column (for migration)
        if 'content_hash' not in existing_df.columns:
            print('Adding content_hash to existing data...')
            existing_df = existing_df.with_columns(
                pl.struct(['title', 'description'])
                .map_elements(
                    lambda row: compute_content_hash(row['title'] or '', row['description'] or ''),
                    return_dtype=pl.Utf8,
                )
                .alias('content_hash')
            )

        existing_ids = set(existing_df['id'].to_list())
        new_ids = set(new_df['id'].to_list())

        # Find added and deleted
        added = list(new_ids - existing_ids)
        deleted = list(existing_ids - new_ids)

        # Find changed (same ID, different hash)
        common_ids = existing_ids & new_ids
        existing_hashes = dict(
            zip(
                existing_df.filter(pl.col('id').is_in(list(common_ids)))['id'].to_list(),
                existing_df.filter(pl.col('id').is_in(list(common_ids)))['content_hash'].to_list(),
            )
        )
        new_hashes = dict(
            zip(
                new_df.filter(pl.col('id').is_in(list(common_ids)))['id'].to_list(),
                new_df.filter(pl.col('id').is_in(list(common_ids)))['content_hash'].to_list(),
            )
        )
        changed = [kpi_id for kpi_id in common_ids if existing_hashes.get(kpi_id) != new_hashes.get(kpi_id)]
    else:
        # No existing data - everything is new
        added = new_df['id'].to_list()
        changed = []
        deleted = []

    result = SyncResult(added=added, changed=changed, deleted=deleted)
    print(f'Sync result: {result}')

    # Save the new data
    save_kpis_to_parquet(new_df, KPI_PARQUET_PATH)

    return result


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'sync':
        sync_kpis()
    else:
        # Full download (default behavior)
        print('Fetching KPIs from Kolada API...')
        kpis = fetch_all_kpis()
        print(f'Fetched {len(kpis)} KPIs')

        df = kpis_to_dataframe(kpis)
        save_kpis_to_parquet(df, KPI_PARQUET_PATH)

        print('\nSample of KPIs:')
        print(df.select('id', 'title').head(10))


if __name__ == '__main__':
    main()
