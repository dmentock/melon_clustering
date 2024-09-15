import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from collections import defaultdict
import numpy as np
from typing import List, Tuple, Dict

class ClusterManager:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def dimensionality_reduction(self, vectors: np.ndarray, method: str = 'LSA', **kwargs) -> np.ndarray:
        self.logger.debug(f"Starting dimensionality reduction using {method} with params {kwargs}")
        if method == 'LSA':
            reduced_vectors = TruncatedSVD(n_components=kwargs.get('n_components', 2), random_state=42).fit_transform(vectors)
        elif method == 'PCA':
            reduced_vectors = PCA(n_components=kwargs.get('n_components', 2), random_state=42).fit_transform(vectors)
        elif method == 't-SNE':
            from sklearn.manifold import TSNE
            reduced_vectors = TSNE(n_components=kwargs.get('n_components', 2),
                                   perplexity=kwargs.get('perplexity', 5),
                                   random_state=42).fit_transform(vectors)
        elif method == 'MDS':
            from sklearn.manifold import MDS
            reduced_vectors = MDS(n_components=kwargs.get('n_components', 2),
                                   random_state=42).fit_transform(vectors)
        else:
            self.logger.warning(f"Unsupported dimensionality reduction method: {method}")
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        self.logger.debug(f"Completed dimensionality reduction using {method}")
        return reduced_vectors

    def cluster_sentences(self, reduced_vectors: np.ndarray, method: str = 'KMeans', **kwargs) -> np.ndarray:
        self.logger.debug(f"Starting clustering using {method} with params {kwargs}")
        if method == 'KMeans':
            labels = KMeans(n_clusters=kwargs.get('n_clusters', 3), n_init=10, random_state=42).fit_predict(reduced_vectors)
        elif method == 'Agglomerative':
            labels = AgglomerativeClustering(n_clusters=kwargs.get('n_clusters', 3), linkage='ward').fit_predict(reduced_vectors)
        elif method == 'DBSCAN':
            labels = DBSCAN(eps=kwargs.get('eps', 0.5), min_samples=kwargs.get('min_samples', 2)).fit_predict(reduced_vectors)
        else:
            self.logger.warning(f"Unsupported clustering method: {method}")
            raise ValueError(f"Unsupported clustering method: {method}")
        self.logger.debug(f"Completed clustering using {method}")
        return labels

    def plot_clusters(self, reduced_vectors: np.ndarray, labels: List[int], title: str, sentences: List[str] = None, annotate: bool = False):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=reduced_vectors[:, 0], y=reduced_vectors[:, 1], hue=labels, palette='viridis')
        plt.title(title)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        if annotate and sentences:
            for i, sentence in enumerate(sentences):
                plt.annotate(sentence, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=8, alpha=0.7)
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.show()
        self.logger.debug(f"Plotted clusters for {title}")

    def plot_all_configurations_grid(self, configurations: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]):
        dim_methods = sorted(list(set([dim for dim, _ in configurations.keys()])))
        cluster_methods = sorted(list(set([cluster for _, cluster in configurations.keys()])))
        num_dim = len(dim_methods)
        num_cluster = len(cluster_methods)

        fig, axes = plt.subplots(num_dim, num_cluster, figsize=(5 * num_cluster, 4 * num_dim))
        if num_dim == 1 and num_cluster == 1:
            axes = np.array([[axes]])
        elif num_dim == 1 or num_cluster == 1:
            axes = axes.reshape(num_dim, num_cluster)

        for i, dim_method in enumerate(dim_methods):
            for j, cluster_method in enumerate(cluster_methods):
                key = (dim_method, cluster_method)
                ax = axes[i, j]
                if key in configurations:
                    reduced_vectors, labels = configurations[key]
                    sns.scatterplot(x=reduced_vectors[:, 0], y=reduced_vectors[:, 1], hue=labels, palette='viridis', legend=False, ax=ax)
                    ax.set_title(f"{dim_method} + {cluster_method}")
                    ax.set_xlabel('Dimension 1')
                    ax.set_ylabel('Dimension 2')
                else:
                    ax.axis('off')  # Hide axes without data
        plt.tight_layout()
        plt.show()
        self.logger.debug("Plotted all configurations grid.")

    def run_all_configurations(self,
                               vectors: np.ndarray,
                               dim_methods: List[str],
                               cluster_methods: List[str],
                               sentence_paths: List[Tuple[int, str]],
                               plot_seaborn: bool = False,
                               plot_plt: bool = False,
                               annotate_plt: bool = False) -> Dict[Tuple[str, str], Dict]:
        configurations = {}

        for dim_method in dim_methods:
            for cluster_method in cluster_methods:
                try:
                    if dim_method == 'LSA':
                        reduced_vectors = self.dimensionality_reduction(vectors, method='LSA', n_components=2)
                    elif dim_method == 'PCA':
                        reduced_vectors = self.dimensionality_reduction(vectors, method='PCA', n_components=2)
                    elif dim_method == 't-SNE':
                        reduced_vectors = self.dimensionality_reduction(vectors, method='t-SNE', n_components=2, perplexity=5)
                    elif dim_method == 'MDS':
                        reduced_vectors = self.dimensionality_reduction(vectors, method='MDS', n_components=2)
                    else:
                        self.logger.warning(f"Unsupported dimensionality reduction method: {dim_method}")
                        continue
                except Exception as e:
                    self.logger.error(f"Error in dimensionality reduction {dim_method}: {e}")
                    continue

                try:
                    if cluster_method in ['KMeans', 'Agglomerative']:
                        labels = self.cluster_sentences(reduced_vectors, method=cluster_method, n_clusters=3)
                    elif cluster_method == 'DBSCAN':
                        labels = self.cluster_sentences(reduced_vectors, method=cluster_method, eps=0.5, min_samples=2)
                    else:
                        self.logger.warning(f"Unsupported clustering method: {cluster_method}")
                        continue
                except Exception as e:
                    self.logger.error(f"Error in clustering {cluster_method}: {e}")
                    continue

                configurations[(dim_method, cluster_method)] = {
                    'labels': labels,
                    'reduced_vectors': reduced_vectors
                }

                if plot_plt:
                    if annotate_plt:
                        sentences_for_plot = [sentence for _, sentence in sentence_paths]
                        title = f"{dim_method} + {cluster_method}"
                        self.plot_clusters(reduced_vectors, labels, title=title, sentences=sentences_for_plot, annotate=True)

        if plot_seaborn:
            configurations_plot = {}
            for key, value in configurations.items():
                configurations_plot[key] = (value['reduced_vectors'], value['labels'])
            self.plot_all_configurations_grid(configurations_plot)

        self.logger.debug("Ran all clustering configurations.")
        return configurations
