from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
SENTENCES_DIR = PROJECT_ROOT / '_sentences'
CACHE_DIR = PROJECT_ROOT / '_cache'

# from .clustering import Clustering
from .pattern_extractor import PatternExtractor