from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import List, Tuple, Dict, Set

class PatternExtractor:
    """
    A class to extract patterns around a target character in sentences,
    compute the Jaccard distance between them, and project the sentences
    into a 2D space based on pattern similarity.
    """

    def __init__(self, language, radius: int = 5, filter_threshold: int = 3):
        """
        Initialize the PatternExtractor class.

        :param radius: The size of the window around the target character to consider.
        :param filter_threshold: The minimum number of occurrences a pattern must have to be kept.
        """
        self.language = language
        self.radius = radius
        self.filter_threshold = filter_threshold
        self.pattern_list = []
        self.mask_patterns = {}

    def _build_pattern_flat(self, sentence_tokens: List[str], idx: int, pattern_dict: Dict) -> Dict:
        """
        Build patterns around the target character and store them in the pattern dictionary.

        :param sentence_tokens: Tokens of the sentence being processed.
        :param idx: Index of the target character in the sentence.
        :param pattern_dict: A dictionary storing pattern counts.
        :return: Updated pattern dictionary with counts.
        """
        window_start = max(0, idx - self.radius)
        window_end = min(len(sentence_tokens), idx + self.radius + 1)

        window = sentence_tokens[window_start:window_end]

        # Replace the target character with <MASK>
        window[idx - window_start] = '<MASK>'

        # Build patterns
        pattern_length = len(window)
        for n in range(1, pattern_length + 1):
            for i in range(pattern_length - n + 1):
                pattern = tuple(window[i:i+n])
                if pattern not in pattern_dict:
                    pattern_dict[pattern] = 0
                pattern_dict[pattern] += 1

        return pattern_dict

    def get_token_list(self, sentences):
        if self.language == 'jp':
            return [list(sentence) for sentence in sentences]
        return [sentence.split(' ') for sentence in sentences]

    def extract_patterns_flat(self, sentences: List[str], target_char: str) -> Dict:
        """
        Extract flat patterns from the sentences based on the target character.

        :param sentences: A list of sentences.
        :param target_char: The target character to find patterns around.
        :return: A dictionary of patterns and their counts.
        """
        pattern_dict = {}

        for sentence in self.get_token_list(sentences):
            tokens = ['<START>'] + list(sentence) + ['END']
            for idx, token in enumerate(tokens):
                if token == target_char:
                    pattern_dict = self._build_pattern_flat(tokens, idx, pattern_dict)

        # Filter patterns based on threshold
        filtered_pattern_dict = OrderedDict(
            {pattern: count for pattern, count in sorted(pattern_dict.items(), key=lambda x: x[1], reverse=True)
             if count >= self.filter_threshold}
        )

        return filtered_pattern_dict

    def filter_mask_patterns(self, pattern_dict: Dict) -> Dict:
        """
        Filter only patterns that contain <MASK>.

        :param pattern_dict: A dictionary of patterns and their counts.
        :return: A dictionary of filtered patterns.
        """
        self.mask_patterns = {pattern: count for pattern, count in pattern_dict.items() if '<MASK>' in pattern}

    def map_sentences_to_patterns(self, sentences: List[str], target_char: str) -> List[Tuple[str, List[int]]]:
        """
        Map each sentence to the pattern indices it matches.

        :param sentences: A list of sentences.
        :param target_char: The target character to find patterns around.
        :return: A list of tuples, where each tuple contains a sentence and the list of matched pattern indices.
        """
        sentence_pattern_mapping = []
        self.pattern_list = list(self.mask_patterns.keys())  # Store patterns for indexing

        for sentence in self.get_token_list(sentences):
            tokens = ['<START>'] + list(sentence) + ['END']
            matched_indices = set()

            for idx, token in enumerate(tokens):
                if token == target_char:
                    window_start = max(0, idx - self.radius)
                    window_end = min(len(tokens), idx + self.radius + 1)
                    window = tokens[window_start:window_end]
                    window[idx - window_start] = '<MASK>'

                    # Ensure core pattern (<MASK>,) is always included
                    if ('<MASK>',) in self.pattern_list:
                        matched_indices.add(self.pattern_list.index(('<MASK>',)))

                    # Check which patterns match the current window
                    for i, pattern in enumerate(self.pattern_list):
                        pattern_length = len(pattern)
                        for j in range(len(window) - pattern_length + 1):
                            if tuple(window[j:j + pattern_length]) == pattern:
                                matched_indices.add(i)

            sentence_pattern_mapping.append((sentence, sorted(matched_indices)))

        return sentence_pattern_mapping

    def jaccard_distance(self, set1: Set[int], set2: Set[int]) -> float:
        """
        Compute the Jaccard distance between two sets of pattern indices.

        :param set1: The first set of pattern indices.
        :param set2: The second set of pattern indices.
        :return: The Jaccard distance between the two sets.
        """
        intersection = len(set(set1).intersection(set(set2)))
        union = len(set(set1).union(set(set2)))
        return 1 - intersection / union if union != 0 else 1

    def create_distance_matrix(self, sentence_pattern_mapping: List[Tuple[str, List[int]]]) -> np.ndarray:
        """
        Create a distance matrix using the Jaccard distance between sentences.

        :param sentence_pattern_mapping: A list of tuples mapping sentences to pattern indices.
        :return: A symmetric distance matrix.
        """
        num_sentences = len(sentence_pattern_mapping)
        distance_matrix = np.zeros((num_sentences, num_sentences))

        for i in range(num_sentences):
            for j in range(i, num_sentences):
                set1 = sentence_pattern_mapping[i][1]
                set2 = sentence_pattern_mapping[j][1]
                dist = self.jaccard_distance(set1, set2)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist  # Symmetry

        return distance_matrix

    def compute_pcoa(self, sentence_pattern_mapping: List[Tuple[str, List[int]]]) -> np.ndarray:
        """
        Perform Principal Coordinates Analysis (PCoA) using PCA on the distance matrix.

        :param sentence_pattern_mapping: A list of tuples mapping sentences to pattern indices.
        :return: A 2D array representing the coordinates of each sentence in 2D space.
        """
        distance_matrix = self.create_distance_matrix(sentence_pattern_mapping)

        # Perform PCA on the distance matrix
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(distance_matrix)

        return points_2d

    def plot_pcoa(self, sentence_pattern_mapping: List[Tuple[str, List[int]]]) -> None:
        """
        Plot the sentences in 2D space based on their pattern similarity.

        :param sentence_pattern_mapping: A list of tuples mapping sentences to pattern indices.
        """
        points_2d = self.compute_pcoa(sentence_pattern_mapping)

        # Plot the results
        plt.figure(figsize=(8, 6))
        for i, (sentence, _) in enumerate(sentence_pattern_mapping):
            plt.scatter(points_2d[i, 0], points_2d[i, 1])
            plt.text(points_2d[i, 0] + 0.02, points_2d[i, 1] + 0.02, f'Sentence {i+1}', fontsize=9)

        plt.title('2D Projection of Sentences Based on Pattern Similarity (PCoA)')
        plt.xlabel('PCoA Dimension 1')
        plt.ylabel('PCoA Dimension 2')
        plt.grid(True)
        plt.show()