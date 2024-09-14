from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent

SENTENCES_DIR = PROJECT_ROOT / '_sentences'
CACHE_DIR = PROJECT_ROOT / '_cache'
PLOT_DIR = PROJECT_ROOT / '_plots'

SENTENCES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# from .clustering import Clustering
from .pattern_extractor import PatternExtractor
from .pattern_extractor_gnn import PatternExtractorWithGNN
from .load_sentences import Loader