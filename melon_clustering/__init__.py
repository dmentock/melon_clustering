from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent

SENTENCES_DIR = PROJECT_ROOT / '_sentences'
CACHE_DIR = PROJECT_ROOT / '_cache'
PLOT_DIR = PROJECT_ROOT / '_plots'

SENTENCES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

import yaml
with open(PROJECT_ROOT / 'reference_clusters_de.yaml', 'r') as f:
    reference_clusters_de = yaml.safe_load(f)

# from .clustering import Clustering
from .pattern_extractor_one_hot import PatternExtractorOneHot
from .pattern_extractor_graph import PatternExtractorGraph, Node
# from .pattern_extractor_stanza import PatternExtractor
# from .pattern_extractor_gnn import PatternExtractorWithGNN
from .cluster_evaluator import ClusterEvaluator
from .cluster_manager import ClusterManager
from .load_sentences import Loader