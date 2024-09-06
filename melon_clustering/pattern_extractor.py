import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from collections import OrderedDict
from typing import List, Tuple, Dict

class PatternExtractor:
    def __init__(self, language: str, radius: int = 5, filter_threshold: int = 3):
        self.language = language
        self.radius = radius
        self.filter_threshold = filter_threshold
        self.pattern_list = []
        self.mask_patterns = {}

    def _build_pattern_flat(self, sentence_tokens: List[str], idx: int, pattern_dict: Dict) -> Dict:
        window_start = max(0, idx - self.radius)
        window_end = min(len(sentence_tokens), idx + self.radius + 1)
        window = sentence_tokens[window_start:window_end]
        window[idx - window_start] = '<MASK>'
        sentence_patterns = set()
        pattern_length = len(window)
        for n in range(1, pattern_length + 1):
            for i in range(pattern_length - n + 1):
                pattern = tuple(window[i:i+n])
                if '<MASK>' in pattern:
                    sentence_patterns.add(pattern)
        for pattern in sentence_patterns:
            if pattern not in pattern_dict:
                pattern_dict[pattern] = 0
            pattern_dict[pattern] += 1
        return pattern_dict

    def get_token_list(self, sentences: List[str]) -> List[List[str]]:
        if self.language == 'jp':
            return [['<START>'] + list(sentence.lower()) + ['<END>'] for sentence in sentences]
        return [['<START>'] + sentence.lower().split() + ['<END>'] for sentence in sentences]

    def extract_patterns_flat(self, sentences_by_morphology: Dict[str, List[str]]) -> Dict:
        pattern_dict = {}
        for morphology, sentences in sentences_by_morphology.items():
            for tokens in self.get_token_list(sentences):
                for idx, token in enumerate(tokens):
                    if token == morphology.lower():
                        pattern_dict = self._build_pattern_flat(tokens, idx, pattern_dict)
        filtered_pattern_dict = OrderedDict(
            {pattern: count for pattern, count in sorted(pattern_dict.items(), key=lambda x: x[1], reverse=True)
             if count >= self.filter_threshold}
        )
        self.mask_patterns = filtered_pattern_dict

    def map_sentences_to_patterns(self, sentences_by_morphology: Dict[str, List[str]]) -> List[Tuple[str, List[int]]]:
        sentence_pattern_mapping = []
        self.pattern_list = list(self.mask_patterns.keys())
        for morphology, sentences in sentences_by_morphology.items():
            for sentence in self.get_token_list(sentences):
                tokens = ['<START>'] + [token.lower() for token in sentence] + ['END']
                matched_indices = set()
                for idx, token in enumerate(tokens):
                    if token == morphology.lower():
                        window_start = max(0, idx - self.radius)
                        window_end = min(len(tokens), idx + self.radius + 1)
                        window = tokens[window_start:window_end]
                        window[idx - window_start] = '<MASK>'
                        if ('<MASK>',) in self.pattern_list:
                            matched_indices.add(self.pattern_list.index(('<MASK>',)))
                        for i, pattern in enumerate(self.pattern_list):
                            pattern_length = len(pattern)
                            for j in range(len(window) - pattern_length + 1):
                                if tuple(window[j:j + pattern_length]) == pattern:
                                    matched_indices.add(i)
                sentence_pattern_mapping.append((sentence, sorted(matched_indices)))
        return sentence_pattern_mapping