from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
SENTENCES_DIR = PROJECT_ROOT / 'sentences'
CACHE_DIR = PROJECT_ROOT / 'sentences'

from clustering import Clustering