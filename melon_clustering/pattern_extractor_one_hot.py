import numpy as np
import logging

from typing import List, Tuple, Dict, Set
import re
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class PatternExtractorOneHot:
    def __init__(self, language: str, radius: int = 5, filter_threshold: int = 2):
        self.language = language
        self.radius = radius
        self.filter_threshold = filter_threshold
        self.pattern_list = []
        self.mask_patterns = {}
        self.sentence_paths = []  # stores (sentence_id, sentence)
        self.logger = logging.getLogger(self.__class__.__name__)

    def is_root_token(self, token: str) -> bool:
        return token == '<ROOT>'

    def _build_pattern_flat(self, sentence_tokens: List[str], idx: int, pattern_dict: Dict) -> Dict:
        window_start = max(0, idx - self.radius)
        window_end = min(len(sentence_tokens), idx + self.radius + 1)
        window = sentence_tokens[window_start:window_end]
        sentence_patterns = set()

        for n in range(1, len(window) + 1):
            for i in range(len(window) - n + 1):
                pattern = tuple(window[i:i + n])
                if '<ROOT>' in pattern:
                    sentence_patterns.add(pattern)

        for pattern in sentence_patterns:
            pattern_dict[pattern] = pattern_dict.get(pattern, 0) + 1

        return pattern_dict

    def get_token_list(self, sentences: List[str]) -> List[List[str]]:
        tokenized_sentences = []
        for sentence in sentences:
            tokens = re.findall(r'<\w+>|[\w]+|[^\w\s]', sentence)
            tokenized_sentences.append(['<START>'] + tokens + ['<END>'])
        return tokenized_sentences

    def extract_patterns_flat(self, sentences_dict: Dict[str, List[Tuple[int, str]]], min_occurrences: int = 2) -> None:
        pattern_dict = {}
        self.sentence_paths = []

        for word, sentences in sentences_dict.items():
            for sentence_id, sentence in sentences:
                tokens = self.get_token_list([sentence])[0]
                for idx, token in enumerate(tokens):
                    if self.is_root_token(token):
                        pattern_dict = self._build_pattern_flat(tokens, idx, pattern_dict)
                        self.sentence_paths.append((sentence_id, ' '.join(tokens)))

        filtered_pattern_dict = OrderedDict(
            (pattern, count) for pattern, count in sorted(pattern_dict.items(), key=lambda x: x[1], reverse=True)
            if count >= max(min_occurrences, self.filter_threshold)
        )

        self.mask_patterns = filtered_pattern_dict
        self.pattern_list = list(self.mask_patterns.keys())
        self.logger.debug(f"Extracted {len(self.pattern_list)} patterns.")


    def map_sentences_to_patterns(self, sentences_dict: Dict[str, List[Tuple[int, str]]]) -> List[Tuple[int, List[int]]]:
        sentence_pattern_mapping = []
        for word, sentences in sentences_dict.items():
            for sentence_id, sentence in sentences:
                matched_indices = set()
                tokens = self.get_token_list([sentence])[0]
                for idx, token in enumerate(tokens):
                    if self.is_root_token(token):
                        window_start = max(0, idx - self.radius)
                        window_end = min(len(tokens), idx + self.radius + 1)
                        window = tokens[window_start:window_end]
                        for pattern_idx, pattern in enumerate(self.pattern_list):
                            pattern_length = len(pattern)
                            for j in range(len(window) - pattern_length + 1):
                                if tuple(window[j:j + pattern_length]) == pattern:
                                    matched_indices.add(pattern_idx)
                sentence_pattern_mapping.append((sentence_id, sorted(matched_indices)))
        self.logger.debug(f"Mapped patterns to sentences: {sentence_pattern_mapping}")
        return sentence_pattern_mapping

    def generate_embeddings(self, sentence_pattern_mapping: List[Tuple[int, List[int]]]) -> np.ndarray:
        num_patterns = len(self.pattern_list)
        num_sentences = len(sentence_pattern_mapping)
        one_hot_vectors = np.zeros((num_sentences, num_patterns))
        for i, (_, pattern_indices) in enumerate(sentence_pattern_mapping):
            for index in pattern_indices:
                one_hot_vectors[i, index] = 1
        self.logger.debug(f"Encoded patterns into one-hot vectors: {one_hot_vectors}")
        return one_hot_vectors
