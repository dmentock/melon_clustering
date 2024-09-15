import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import numpy as np
import re
from melon_clustering import PLOT_DIR, CACHE_DIR

class ClusterEvaluator:
    def __init__(self, reference_clusters: List[List[Tuple[str, str]]]):
        self.sentence_id_to_original = {}
        self.reference_clusters = reference_clusters
        self.reference_sentence_ids = set()
        self.sentences_dict = defaultdict(list)
        self.results_collection = []  # To store results for multiple configurations

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

    def replace_with_root(self, sentence: str, target_word: str) -> str:
        processed_sentence = re.sub(rf'\b{re.escape(target_word)}\b', '<ROOT>', sentence, flags=re.IGNORECASE)
        return processed_sentence

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
        self.results_collection.append(results_df)  # Store the result for future analysis
        self.plot_average_similarity_heatmap([results_df], f'Clustering accuracy for word {ref_word}', plot=plot, save=save, **extra_info)
        return results_df

    def export_results(self):
        combined_results_df = pd.concat(self.results_collection)
        combined_results_df.to_csv(CACHE_DIR / 'clustering_results.csv', index=False)
        return combined_results_df

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
