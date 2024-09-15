from collections import defaultdict
from typing import List, Dict, Tuple, Set
import numpy as np
import re

class ClusterEvaluator:
    def __init__(self, reference_clusters: List[List[Tuple[str, str]]]):
        self.sentence_id_to_original = {}
        self.reference_clusters = reference_clusters

    def add_reference_sentences(self, reference_clusters: List[List[Tuple[str, str]]]) -> Tuple[Dict[str, List[Tuple[int, str]]], Dict[int, str]]:
        sentences_dict = defaultdict(list)
        sentence_id = max(self.sentence_id_to_original.keys()) if self.sentence_id_to_original else 0
        for cluster in reference_clusters:
            for sentence, word in tuple(cluster):
                processed_sentence = re.sub(rf'\b{re.escape(word)}\b', '<ROOT>', sentence, flags=re.IGNORECASE)
                sentences_dict[word.lower()].append((sentence_id, processed_sentence))
                self.sentence_id_to_original[sentence_id] = sentence
                sentence_id += 1
        return sentences_dict

    def add_db_sentences(self, sentences_dict):
        return self.add_reference_sentences([[[sentence, word] for sentence in sentences] for word, sentences in sentences_dict.items()])

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
                for sid, original_sentence in self.sentence_id_to_original.items():
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
        # Optionally, print the similarity scores
        print("\nSimilarity Scores between Generated Clusters and Reference Clusters:")
        for ref_method, gen_method, sim in results:
            if gen_method != "None" and sim > 0:
                print(f"{ref_method} matches {gen_method} with Jaccard similarity: {sim:.4f}")
            else:
                print(f"{ref_method} has no matching Generated Cluster. Jaccard similarity: {sim:.4f}")
        return results
