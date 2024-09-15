import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import numpy as np
import re
import logging
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

from melon_clustering import PLOT_DIR

class ClusterEvaluator:
    def __init__(self, reference_clusters: List[List[Tuple[str, str]]]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sentence_id_to_original = {}
        self.reference_clusters = reference_clusters
        self.reference_sentence_ids = set()
        self.sentences_dict = defaultdict(list)

    def add_sentences(self, sentence_clusters: List[List[Tuple[str, str]]], is_reference: bool = False) -> Dict[str, List[Tuple[int, str]]]:
        sentence_id = max(self.sentence_id_to_original.keys()) + 1 if self.sentence_id_to_original else 0
        for cluster in sentence_clusters:
            for sentence, word in cluster:
                processed_sentence = self.replace_with_root(sentence, word)
                self.sentences_dict[word.lower()].append((sentence_id, processed_sentence))
                self.sentence_id_to_original[sentence_id] = sentence
                if is_reference:
                    self.reference_sentence_ids.add(sentence_id)
                sentence_id += 1
        return self.sentences_dict

    def add_reference_sentences(self, reference_clusters: List[List[Tuple[str, str]]]) -> Dict[str, List[Tuple[int, str]]]:
        return self.add_sentences(reference_clusters, is_reference=True)

    def add_additional_sentences(self, additional_sentences: Dict[str, List[str]]) -> Dict[str, List[Tuple[int, str]]]:
        additional_clusters = [[(sentence, word)] for word, sentences in additional_sentences.items() for sentence in sentences]
        return self.add_sentences(additional_clusters)

    def normalize_sentence(self, sentence: str) -> str:
        sentence = sentence.replace('<START>', '').replace('<END>', '').replace('<ROOT>', '').strip()
        sentence = re.sub(r'\s+', ' ', sentence)
        sentence = sentence.lower()
        return sentence

    def replace_with_root(self, sentence: str, target_word: str) -> str:
        processed_sentence = re.sub(rf'\b{re.escape(target_word)}\b', '<ROOT>', sentence, flags=re.IGNORECASE)
        return processed_sentence
        # return self.normalize_sentence(processed_sentence)

    def prepare_reference_clusters_ids(self) -> List[Set[int]]:
        ref_clusters_ids = []
        for cluster in self.reference_clusters:
            cluster_ids = set()
            for sentence, word in cluster:
                processed_sentence = self.replace_with_root(sentence, word)
                for sid in self.reference_sentence_ids:
                    original_sentence = self.sentence_id_to_original[sid]
                    match_sentence = self.replace_with_root(original_sentence, word)
                    if processed_sentence == match_sentence:
                        cluster_ids.add(sid)
                        break
            ref_clusters_ids.append(cluster_ids)
        return ref_clusters_ids

    def jaccard_similarity(self, set1: Set[int], set2: Set[int]) -> float:
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        if not union:
            return 0.0
        return len(intersection) / len(union)

    def compare_clusters(self, generated_clusters: List[List[int]], reference_clusters_ids: List[Set[int]]) -> List[Tuple[int, int, float]]:
        similarity_scores = []
        for ref_idx, ref_cluster in enumerate(reference_clusters_ids):
            best_similarity = 0.0
            best_gen_idx = -1
            for gen_idx, gen_cluster in enumerate(generated_clusters):
                sim = self.jaccard_similarity(ref_cluster, set(gen_cluster))
                if sim > best_similarity:
                    best_similarity = sim
                    best_gen_idx = gen_idx
            similarity_scores.append((ref_idx, best_gen_idx, best_similarity))
        return similarity_scores

    def evaluate_clusters(self, generated_clusters: List[List[int]]) -> List[Tuple[str, str, float]]:
        reference_clusters_ids = self.prepare_reference_clusters_ids()
        similarity_scores = self.compare_clusters(generated_clusters, reference_clusters_ids)
        results = []
        for ref_idx, gen_idx, sim in similarity_scores:
            ref_method = f"Ref {ref_idx}"
            gen_method = f"Gen {gen_idx}" if gen_idx != -1 else "None"
            results.append((ref_method, gen_method, sim))
        return results

    def evaluate_multiple_configurations(self, clustering_results: Dict[Tuple[str, str], Dict], ref_word: str, plot=True, save=False, **extra_info) -> pd.DataFrame:
        results = []
        reference_clusters_ids = self.prepare_reference_clusters_ids()
        for (dim_method, cluster_method), result in clustering_results.items():
            labels = result['labels']
            generated_clusters = defaultdict(list)
            for sentence_id, cluster_id in zip(sorted(self.reference_sentence_ids), labels):
                generated_clusters[cluster_id].append(sentence_id)
            generated_clusters_list = [cluster for cluster in generated_clusters.values()]
            similarity_scores = self.compare_clusters(generated_clusters_list, reference_clusters_ids)
            avg_similarity = np.mean([score for _, _, score in similarity_scores])
            results.append({
                'Dimensionality Reduction': dim_method,
                'Clustering Method': cluster_method,
                'Average Jaccard Similarity': avg_similarity
            })
        results_df = pd.DataFrame(results)
        self.plot_average_similarity_heatmap([results_df], f'Clustering accuracy for word {ref_word}',plot=plot, save=save, **extra_info)
        return results_df


    def isolate_reference_embeddings(self, vectors_combined: np.ndarray, num_additional_sentences: int) -> np.ndarray:
        num_reference_sentences = len(self.reference_sentence_ids)
        vectors_reference = vectors_combined[:num_reference_sentences]
        return vectors_reference

    def plot_average_similarity_heatmap(self, list_of_results_df: List[pd.DataFrame], title: str, plot=True, save=False, **extra_info):
        if not list_of_results_df:
            return

        combined_df = pd.concat(list_of_results_df)
        avg_combined_df = combined_df.groupby(['Dimensionality Reduction', 'Clustering Method'])['Average Jaccard Similarity'].mean().reset_index()
        pivot_df = avg_combined_df.pivot(index='Dimensionality Reduction', columns='Clustering Method', values='Average Jaccard Similarity')

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', fmt=".4f", linewidths=.5)
        if extra_info:
            title += f" ({', '.join([f'{k} {v}' for k, v in extra_info.items()])})"
        plt.title(title)
        plt.tight_layout()
        if save:
            plt.savefig(PLOT_DIR / f'heatmap_{title}.png', format='png', dpi=300, bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()

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
