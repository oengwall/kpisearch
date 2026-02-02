"""Unified sync command for incremental KPI and embedding updates.

Usage:
    uv run python -m kpisearch.sync
"""

from kpisearch.download_kpis import sync_kpis
from kpisearch.search import sync_embeddings


def main() -> None:
    """Sync KPIs from Kolada API and update embeddings incrementally."""
    print('=' * 60)
    print('KPI Sync - Incremental Update')
    print('=' * 60)

    # Step 1: Sync KPIs
    print('\n[1/2] Syncing KPIs from Kolada API...\n')
    result = sync_kpis()

    if not result.has_changes:
        print('\nNo changes detected. Embeddings are up to date.')
        return

    # Step 2: Update embeddings for all cached models
    print('\n[2/2] Updating embeddings...\n')
    sync_embeddings(result.added, result.changed, result.deleted)

    print('\n' + '=' * 60)
    print(f'Sync complete: {result}')
    print('=' * 60)


if __name__ == '__main__':
    main()
