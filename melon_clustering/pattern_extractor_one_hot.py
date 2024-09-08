import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
from collections import OrderedDict
from typing import List, Tuple, Dict
import re
import yaml

class PatternExtractor:
    def __init__(self, language: str, radius: int = 5, filter_threshold: int = 3):
        self.language = language
        self.radius = radius
        self.filter_threshold = filter_threshold
        self.pattern_list = []
        self.mask_patterns = {}

    def load_sentences(self, path):
        with open(path, 'r', encoding='utf-8') as f:
        # with open(SENTENCES_DIR / 'erinnern_shortened.yaml', 'r', encoding='utf-8') as f:
            sentences_dict = yaml.safe_load(f)

        for morphology, sentences in sentences_dict.items():
            for i, sentence in enumerate(sentences):
                if self.language == 'jp':
                    sentences_dict[morphology][i] = sentence.replace(' ', '')
        return sentences_dict

    def _build_pattern_flat(self, sentence_tokens: List[str], idx: int, pattern_dict: Dict) -> Dict:
        window_start = max(0, idx - self.radius)
        window_end = min(len(sentence_tokens), idx + self.radius + 1)
        window = sentence_tokens[window_start:window_end]
        window[idx - window_start] = '<MASK>'
        sentence_patterns = set()
        for n in range(1, len(window) + 1):
            for i in range(len(window) - n + 1):
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

    def morphology_in_token(self, morphology, token):
        if morphology.lower() in token.lower():
            trailing_part = re.sub(re.escape(morphology.lower()), '', token.lower(), count=1)
            if not bool(re.search(r'[a-z]', trailing_part)):
                return True
        return False

    def extract_patterns_flat(self, sentences_by_morphology: Dict[str, List[str]]) -> Dict:
        pattern_dict = {}
        for morphology, sentences in sentences_by_morphology.items():
            for tokens in self.get_token_list(sentences):
                for idx, token in enumerate(tokens):
                    if self.morphology_in_token(morphology, token):
                        pattern_dict = self._build_pattern_flat(tokens, idx, pattern_dict)

        filtered_pattern_dict = OrderedDict(
            {pattern: count for pattern, count in sorted(pattern_dict.items(), key=lambda x: x[1], reverse=True)
             if count >= self.filter_threshold}
        )
        self.mask_patterns = filtered_pattern_dict
        self.pattern_list = list(self.mask_patterns.keys())

    def map_sentences_to_patterns(self, sentences_by_morphology: Dict[str, List[str]]) -> List[Tuple[str, List[int]]]:
        sentence_pattern_mapping = []
        for morphology, sentences in sentences_by_morphology.items():
            for sentence_tokens in self.get_token_list(sentences):
                matched_indices = set()
                for idx, token in enumerate(sentence_tokens):
                    if self.morphology_in_token(morphology, token):
                        window_start = max(0, idx - self.radius)
                        window_end = min(len(sentence_tokens), idx + self.radius + 1)
                        window = sentence_tokens[window_start:window_end]
                        window[idx - window_start] = '<MASK>'

                        # Check if this window matches any patterns
                        for pattern_idx, pattern in enumerate(self.pattern_list):
                            pattern_length = len(pattern)
                            for j in range(len(window) - pattern_length + 1):
                                if tuple(window[j:j + pattern_length]) == pattern:
                                    matched_indices.add(pattern_idx)
                sentence_pattern_mapping.append((sentence_tokens, sorted(matched_indices)))

        return sentence_pattern_mapping