import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import numpy as np
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class ClusterEvaluator:
    def __init__(self, reference_clusters: List[List[Tuple[str, str]]]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sentence_id_to_original = {}
        self.reference_clusters = reference_clusters
        self.reference_sentence_ids = set()

    def add_reference_sentences(self, reference_clusters: List[List[Tuple[str, str]]]) -> Dict[str, List[Tuple[int, str]]]:
        sentences_dict = defaultdict(list)
        sentence_id = max(self.sentence_id_to_original.keys()) + 1 if self.sentence_id_to_original else 0

        for cluster in reference_clusters:
            for sentence, word in cluster:
                processed_sentence = re.sub(rf'\b{re.escape(word)}\b', '<ROOT>', sentence, flags=re.IGNORECASE)
                sentences_dict[word.lower()].append((sentence_id, processed_sentence))
                self.sentence_id_to_original[sentence_id] = sentence
                self.reference_sentence_ids.add(sentence_id)
                sentence_id += 1

        self.logger.debug("Added reference sentences.")
        return sentences_dict

    def add_additional_sentences(self, additional_sentences: Dict[str, List[str]]) -> Dict[str, List[Tuple[int, str]]]:
        sentences_dict = defaultdict(list)
        sentence_id = max(self.sentence_id_to_original.keys()) + 1 if self.sentence_id_to_original else 0

        for word, sentences in additional_sentences.items():
            for sentence in sentences:
                processed_sentence = re.sub(rf'\b{re.escape(word)}\b', '<ROOT>', sentence, flags=re.IGNORECASE)
                sentences_dict[word.lower()].append((sentence_id, processed_sentence))
                self.sentence_id_to_original[sentence_id] = sentence
                sentence_id += 1

        self.logger.debug("Added additional sentences.")
        return sentences_dict

    def normalize_sentence(self, sentence: str) -> str:
        sentence = sentence.replace('<START>', '').replace('<END>', '').replace('<ROOT>', '').strip()
        sentence = re.sub(r'\s+', ' ', sentence)
        sentence = sentence.lower()
        return sentence

    def replace_with_root(self, sentence: str, target_word: str) -> str:
        processed_sentence = re.sub(rf'\b{re.escape(target_word)}\b', '<ROOT>', sentence, flags=re.IGNORECASE)
        return self.normalize_sentence(processed_sentence)

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
        self.logger.debug("Prepared reference cluster IDs.")
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
        self.logger.debug("Compared generated clusters with reference clusters.")
        return similarity_scores

    def evaluate_clusters(self, generated_clusters: List[List[int]]) -> List[Tuple[str, str, float]]:
        reference_clusters_ids = self.prepare_reference_clusters_ids()
        similarity_scores = self.compare_clusters(generated_clusters, reference_clusters_ids)
        results = []
        for ref_idx, gen_idx, sim in similarity_scores:
            ref_method = f"Ref {ref_idx}"
            gen_method = f"Gen {gen_idx}" if gen_idx != -1 else "None"
            results.append((ref_method, gen_method, sim))
        self.logger.debug("Evaluated clusters.")
        return results

    def evaluate_multiple_configurations(self,
                                         clustering_results: Dict[Tuple[str, str], Dict],
                                         ref_word: str,
                                         plot: bool = True,
                                         **extra_info) -> pd.DataFrame:
        results = []
        reference_clusters_ids = self.prepare_reference_clusters_ids()

        for (dim_method, cluster_method), result in clustering_results.items():
            labels = result['labels']
            reduced_vectors = result['reduced_vectors']

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

            self.logger.debug(f"Configuration: {dim_method} + {cluster_method}")
            for ref_method, gen_method, sim in similarity_scores:
                self.logger.debug(f"{ref_method} -> {gen_method}: {sim:.4f}")

        results_df = pd.DataFrame(results)
        if plot:
            pivot_df = results_df.pivot(index='Dimensionality Reduction', columns='Clustering Method', values='Average Jaccard Similarity')

            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', fmt=".4f", linewidths=.5)
            title = f'Clustering accuracy for word "{ref_word}"'
            if extra_info:
                title += f" ({', '.join([f'{k}: {v}' for k, v in extra_info.items()])})"
            plt.title(title)
            plt.tight_layout()
            plt.show()

        self.logger.debug("Evaluated multiple configurations and generated heatmap.")
        return results_df

    def isolate_reference_embeddings(self, vectors_combined: np.ndarray, num_additional_sentences: int) -> np.ndarray:
        num_reference_sentences = len(self.reference_sentence_ids)
        vectors_reference = vectors_combined[:num_reference_sentences]
        self.logger.debug("Isolated reference sentence embeddings.")
        return vectors_reference

    def plot_average_similarity_per_dim_method(self, results_df: pd.DataFrame):
        avg_per_dim = results_df.groupby('Dimensionality Reduction')['Average Jaccard Similarity'].mean().reset_index()

        plt.figure(figsize=(8,6))
        sns.barplot(data=avg_per_dim, x='Dimensionality Reduction', y='Average Jaccard Similarity', palette='Blues_d')
        plt.title('Average Jaccard Similarity per Dimensionality Reduction Method')
        plt.xlabel('Dimensionality Reduction Method')
        plt.ylabel('Average Jaccard Similarity')
        plt.ylim(0, 1)  # Assuming similarity scores are between 0 and 1
        plt.tight_layout()
        plt.show()

        self.logger.debug("Plotted average similarity per dimensionality reduction method.")

    def plot_average_similarity_per_cluster_method(self, results_df: pd.DataFrame):
        avg_per_cluster = results_df.groupby('Clustering Method')['Average Jaccard Similarity'].mean().reset_index()

        plt.figure(figsize=(8,6))
        sns.barplot(data=avg_per_cluster, x='Clustering Method', y='Average Jaccard Similarity', palette='Greens_d')
        plt.title('Average Jaccard Similarity per Clustering Method')
        plt.xlabel('Clustering Method')
        plt.ylabel('Average Jaccard Similarity')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

        self.logger.debug("Plotted average similarity per clustering method.")

    def plot_average_similarity_heatmap(self, list_of_results_df: List[pd.DataFrame], ref_word: str, **extra_info):
        if not list_of_results_df:
            return

        combined_df = pd.concat(list_of_results_df)
        avg_combined_df = combined_df.groupby(['Dimensionality Reduction', 'Clustering Method'])['Average Jaccard Similarity'].mean().reset_index()
        pivot_df = avg_combined_df.pivot(index='Dimensionality Reduction', columns='Clustering Method', values='Average Jaccard Similarity')

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt=".4f", linewidths=.5)
        title = f'Clustering accuracy for word {ref_word}'
        if extra_info:
            title += f" ({', '.join([f'{k}: {v}' for k, v in extra_info.items()])})"
        plt.title(title)
        plt.tight_layout()
        plt.show()
