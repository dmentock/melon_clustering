import numpy as np
from collections import defaultdict, deque
import random
import sys
sys.setrecursionlimit(1500)

class Node:
    def __init__(self, word, id):
        self.word = word
        self.id = id
        self.count = 0
        self.children = {}


class PatternExtractorGraph:
    def __init__(self, seed = 1):
        np.random.seed(seed)
        random.seed(seed)
        self.preceding_tree = Node('<ROOT>', 0)
        self.following_tree = Node('<ROOT>', 1)
        self.node_counter = 2
        self.word_to_preceding_ids = defaultdict(list)
        self.word_to_following_ids = defaultdict(list)
        self.id_to_node = {0: self.preceding_tree, 1: self.following_tree}
        self.child_to_parents = defaultdict(list)
        self.node_embeddings = {}  # Store embeddings for each node

    def add_start_end_flags_lower(self, sentences):
        return [f"<START> {sentence} <END>" for sentence in sentences]

    def get_or_create_node(self, current_tree, word, parent_id, direction):
        word_to_ids = self.word_to_preceding_ids if direction == 'preceding' else self.word_to_following_ids
        if word not in current_tree.children:
            new_node = Node(word, self.node_counter)
            current_tree.children[word] = new_node
            word_to_ids[word].append(self.node_counter)
            self.id_to_node[self.node_counter] = new_node
            if parent_id is not None:
                self.child_to_parents[self.node_counter].append(parent_id)
            self.node_counter += 1
        else:
            child_id = current_tree.children[word].id
            if parent_id is not None:
                self.child_to_parents[child_id].append(parent_id)
        return current_tree.children[word]

    def add_to_tree(self, words, direction, count=1):
        if direction == 'preceding':
            current_tree = self.preceding_tree
            word_range = range(len(words) - 1, -1, -1)
        else:
            current_tree = self.following_tree
            word_range = range(len(words))

        parent_id = current_tree.id

        for i in word_range:
            current_word = words[i]
            current_tree = self.get_or_create_node(current_tree, current_word, parent_id, direction)
            current_tree.count += count
            parent_id = current_tree.id

    def create_tree_mask_as_root(self, sentences_dict):
        for morphology, sentences_and_ids in sentences_dict.items():
            sentences_with_flags = self.add_start_end_flags_lower([sentence for id, sentence in sentences_and_ids])
            for sentence in sentences_with_flags:
                words = sentence.split()
                key_word_index = words.index('<ROOT>')
                words_before = words[:key_word_index]
                words_after = words[key_word_index + 1:]

                self.add_to_tree(words_before, 'preceding')
                self.add_to_tree(words_after, 'following')

    def print_tree(self, node, level=0):
        print('  ' * level + f"{node.word} (count: {node.count}, id: {node.id})")
        for child in node.children.values():
            self.print_tree(child, level + 1)

    def print_trees(self):
        print("Preceding Tree (before <ROOT>):")
        self.print_tree(self.preceding_tree)
        print("\nFollowing Tree (after <ROOT>):")
        self.print_tree(self.following_tree)

    def get_nodes_by_word(self, word, direction):
        word_to_ids = self.word_to_preceding_ids if direction == 'preceding' else self.word_to_following_ids
        ids = word_to_ids.get(word, [])
        return {self.id_to_node[id] for id in ids}

    def get_parents_by_id(self, id):
        parent_ids = self.child_to_parents.get(id, [])
        return [self.id_to_node[parent_id] for parent_id in parent_ids]

    def optimize_tree(self, word, direction, overlap_threshold=1, visited=None, revisit_queue=None, processing_stack=None):
        if visited is None:
            visited = set()
        if revisit_queue is None:
            revisit_queue = set()
        if processing_stack is None:
            processing_stack = set()

        word_to_ids = self.word_to_preceding_ids if direction == 'preceding' else self.word_to_following_ids

        all_nodes = self.get_nodes_by_word(word, direction)
        groups_with_overlap_children = []
        parent_node_ids = []

        node_changed = False

        for node in all_nodes:
            # Detect cycle: if node is already being processed, break the cycle
            if node.id in processing_stack:
                print(f"Cycle detected at node {node.id}!")
                continue

            # Skip already visited nodes unless they're in the revisit queue
            if node.id in visited and node.id not in revisit_queue:
                continue

            # Mark node as visited and currently being processed
            visited.add(node.id)
            processing_stack.add(node.id)  # Add to processing stack to track recursion depth

            parent_node_ids.extend(self.child_to_parents[node.id])

            if word in ['<START>', '<END>']:
                if not groups_with_overlap_children:
                    groups_with_overlap_children.append({
                        'combined_structure': None,
                        'nodes': [node]
                    })
                else:
                    groups_with_overlap_children[-1]['nodes'].append(node)
                # Remove from processing_stack when done
                if node.id in processing_stack:
                    processing_stack.remove(node.id)
                continue

            # Extract child and parent structures for comparison
            child_structure = frozenset(child.word for child in node.children.values())  # Use frozenset for comparison
            parent_structure = set(parent.word for parent in self.get_parents_by_id(node.id))
            combined_structure = child_structure.union(parent_structure)

            added_to_group = False
            for group in groups_with_overlap_children:
                group_structure = group['combined_structure']
                intersection = combined_structure.intersection(group_structure)
                if len(combined_structure) > 0 and len(group_structure) > 0:
                    overlap_ratio_combined_structure = len(intersection) / len(combined_structure)
                    overlap_ratio_group_structure = len(intersection) / len(group_structure)
                    if overlap_ratio_combined_structure >= overlap_threshold or overlap_ratio_group_structure >= overlap_threshold:
                        group['nodes'].append(node)
                        group['combined_structure'] = group['combined_structure'].union(combined_structure)
                        added_to_group = True
                        break

            if not added_to_group:
                groups_with_overlap_children.append({
                    'combined_structure': combined_structure,
                    'nodes': [node]
                })

            # Process the parent nodes and optimize their trees
            for node_id in set(parent_node_ids):
                if node_id in self.id_to_node:
                    self.optimize_tree(self.id_to_node[node_id].word, direction, overlap_threshold=overlap_threshold,
                                    visited=visited, revisit_queue=revisit_queue, processing_stack=processing_stack)

            # Remove the node from the processing stack after recursion completes
            if node.id in processing_stack:
                processing_stack.remove(node.id)  # Safely remove the node after it has been processed

        # Ensure all nodes are removed from processing stack if needed
        if all_nodes:
            for node in all_nodes:
                if node.id in processing_stack:  # Check before removing
                    processing_stack.remove(node.id)


    def initialize_node_embeddings(self, embedding_size=100):
        """
        Initialize random embeddings for each node.
        """
        for node_id in self.id_to_node.keys():
            self.node_embeddings[node_id] = np.random.rand(embedding_size)

    def compute_weighted_sentence_embedding(self, sentence_path, steepness = None):
        """
        Compute a weighted sentence embedding by traversing the sentence path.
        Nodes closer to the root are weighted more heavily.
        """
        if not steepness:
            steepness = 1
        weighted_embedding = np.zeros(len(self.node_embeddings[0]))  # Assuming all embeddings have the same size
        for i, node_id in enumerate(sentence_path):
            distance_from_root = i + 1  # Assuming sentence_path is ordered from ROOT
            weight = 1 - (1 / (1 + np.exp(-steepness * distance_from_root)))
            weighted_embedding += weight * self.node_embeddings[node_id]
        return weighted_embedding / len(sentence_path)

    def get_sentence_embedding(self, sentence, morphology, steepness=None):
        """
        Given a sentence, return its corresponding embedding.
        """

        words = sentence.split()
        node_path = deque([0])
        key_word_index = words.index('<ROOT>')
        words_before = words[:key_word_index]
        words_after = words[key_word_index + 1:]

        current_node = self.preceding_tree
        for i in range(len(words_before) - 1, -1, -1):
            word = words_before[i]
            if word in current_node.children:
                current_node = current_node.children[word]
                node_path.appendleft(current_node.id)
            else:
                continue

        current_node = self.following_tree
        for word in words_after:
            if word in current_node.children:
                current_node = current_node.children[word]
                node_path.append(current_node.id)
            else:
                continue
        return self.compute_weighted_sentence_embedding(node_path, steepness=steepness)

    def get_all_sentence_embeddings(self, sentences_dict, steepness=None):
        """
        Generate embeddings for all sentences in the dataset.
        """
        sentence_embeddings = []
        sentence_list = []
        for morphology, sentences_and_ids in sentences_dict.items():
            sentences_with_flags = self.add_start_end_flags_lower([sentence for id, sentence in sentences_and_ids])
            for sentence in sentences_with_flags:
                embedding = self.get_sentence_embedding(sentence, morphology, steepness=steepness)
                sentence_embeddings.append(embedding)
                sentence_list.append(sentence)
        return np.array(sentence_embeddings), sentence_list

    def initialize(self, sentences_dict, overlap_threshold = 1):
        self.create_tree_mask_as_root(sentences_dict)
        self.optimize_tree('<START>', 'preceding', overlap_threshold=overlap_threshold)
        self.optimize_tree('<END>', 'forward', overlap_threshold=overlap_threshold)