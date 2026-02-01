import json
from enum import Enum
from pathlib import Path

from kpisearch import DATA_DIR

ADMIN_CONFIG_PATH = DATA_DIR / 'admin_config.json'


class EmbeddingModel(str, Enum):
    """Available embedding models for KPI search."""

    KBLAB_SWEDISH = 'KBLab/sentence-bert-swedish-cased'
    MULTILINGUAL_E5_LARGE = 'intfloat/multilingual-e5-large'
    MULTILINGUAL_MINILM = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

    @property
    def key(self) -> str:
        """Get a safe key for filenames."""
        return self.value.replace('/', '_')

    @property
    def display_name(self) -> str:
        """Get a human-readable name."""
        names = {
            self.KBLAB_SWEDISH: 'KBLab Swedish BERT',
            self.MULTILINGUAL_E5_LARGE: 'Multilingual E5 Large',
            self.MULTILINGUAL_MINILM: 'Multilingual MiniLM',
        }
        return names[self]

    @property
    def description(self) -> str:
        """Get a description of the model."""
        descriptions = {
            self.KBLAB_SWEDISH: 'Swedish-specific, optimized for Swedish text',
            self.MULTILINGUAL_E5_LARGE: 'High-quality multilingual embeddings, supports 100+ languages',
            self.MULTILINGUAL_MINILM: 'Lightweight multilingual, good balance of speed/quality',
        }
        return descriptions[self]

    @property
    def uses_e5_prefix(self) -> bool:
        """Check if this model uses E5-style query/passage prefixes."""
        return self == EmbeddingModel.MULTILINGUAL_E5_LARGE


def get_embeddings_path(model: EmbeddingModel) -> Path:
    """Get the embeddings file path for a specific model."""
    return DATA_DIR / f'embeddings_{model.key}.parquet'


def load_admin_config() -> dict:
    """Load admin config from JSON file."""
    if not ADMIN_CONFIG_PATH.exists():
        return {'current_model': EmbeddingModel.KBLAB_SWEDISH.value}
    with open(ADMIN_CONFIG_PATH) as f:
        return json.load(f)


def save_admin_config(config: dict) -> None:
    """Save admin config to JSON file."""
    ADMIN_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ADMIN_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)


def get_current_model() -> EmbeddingModel:
    """Get the currently configured embedding model."""
    config = load_admin_config()
    return EmbeddingModel(config.get('current_model', EmbeddingModel.KBLAB_SWEDISH.value))


def set_current_model(model: EmbeddingModel) -> None:
    """Set the current embedding model."""
    config = load_admin_config()
    config['current_model'] = model.value
    save_admin_config(config)


def get_title_weight() -> float:
    """Get the title weight for search (0.0-1.0). Description weight is 1 - title_weight."""
    config = load_admin_config()
    return config.get('title_weight', 0.6)


def set_title_weight(weight: float) -> None:
    """Set the title weight for search."""
    config = load_admin_config()
    config['title_weight'] = max(0.0, min(1.0, weight))
    save_admin_config(config)
