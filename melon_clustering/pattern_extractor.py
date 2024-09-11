from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque

class Node:
    def __init__(self, word, id):
        self.word = word
        self.id = id
        self.count = 0
        self.children = {}


class PatternExtractor:
    def __init__(self):
        self.preceding_tree = Node('<ROOT>', 0)
        self.following_tree = Node('<ROOT>', 1)
        self.node_counter = 2
        self.word_to_ids = defaultdict(list)
        self.id_to_node = {0: self.preceding_tree, 1: self.following_tree}
        self.child_to_parents = defaultdict(list)
        self.node_embeddings = {}  # Store embeddings for each node

    def add_start_end_flags_lower(self, sentences):
        return [f"<START> {sentence.lower()} <END>" for sentence in sentences]

    def get_or_create_node(self, current_tree, word, parent_id=None):
        if word not in current_tree.children:
            new_node = Node(word, self.node_counter)
            current_tree.children[word] = new_node
            self.word_to_ids[word].append(self.node_counter)
            self.id_to_node[self.node_counter] = new_node
            if parent_id is not None and parent_id not in self.child_to_parents[self.node_counter]:
                self.child_to_parents[self.node_counter].append(parent_id)
            self.node_counter += 1
        else:
            child_id = current_tree.children[word].id
            if parent_id is not None and parent_id not in self.child_to_parents[child_id]:
                self.child_to_parents[child_id].append(parent_id)
        return current_tree.children[word]

    def add_to_tree(self, words, direction, count=1):
        current_tree = self.preceding_tree if direction == 'preceding' else self.following_tree
        if direction == 'preceding':
            word_range = range(len(words) - 1, -1, -1)
        else:
            word_range = range(len(words))

        parent_id = current_tree.id

        for i in word_range:
            current_word = words[i]
            current_tree = self.get_or_create_node(current_tree, current_word, parent_id)
            current_tree.count += count
            parent_id = current_tree.id

    def create_tree_mask_as_root(self, sentences_dict):
        for morphology, sentences in sentences_dict.items():
            sentences_with_flags = self.add_start_end_flags_lower(sentences)

            for sentence in sentences_with_flags:
                words = sentence.split()
                key_word_index = words.index(morphology.lower())
                words_before = words[:key_word_index]
                words_after = words[key_word_index + 1:]

                self.add_to_tree(words_before, 'preceding')
                self.add_to_tree(words_after, 'following')

    def print_tree(self, node, level=0):
        print('  ' * level + f"{node.word} (count: {node.count}, id: {node.id})")
        for child in node.children.values():
            self.print_tree(child, level + 1)

    def print_trees(self):
        print("Preceding Tree (before <MASK>):")
        self.print_tree(self.preceding_tree)
        print("\nFollowing Tree (after <MASK>):")
        self.print_tree(self.following_tree)

    def get_nodes_by_word(self, word):
        ids = self.word_to_ids.get(word, [])
        return {self.id_to_node[id] for id in ids}

    def get_node_by_id(self, id):
        return self.id_to_node.get(id)

    def get_parents_by_id(self, id):
        parent_ids = self.child_to_parents.get(id, [])
        return [self.id_to_node[parent_id] for parent_id in parent_ids]

    def optimize_tree(self, word, overlap_threshold=1):
        all_nodes = self.get_nodes_by_word(word)
        groups_with_overlap_children = []
        parent_node_ids = []

        for node in all_nodes:
            parent_node_ids.extend(self.child_to_parents[node.id])
            child_structure = set(child.word for child in node.children.values())

            parent_structure = set(parent.word for parent in self.get_parents_by_id(node.id))

            combined_structure = child_structure.union(parent_structure)
            added_to_group = False

            for group in groups_with_overlap_children:
                group_structure = group['combined_structure']
                intersection = combined_structure.intersection(group_structure)
                overlap_ratio = 0
                if len(combined_structure) > 0 and len(group_structure) > 0:
                    overlap_ratio = len(intersection) / max(len(combined_structure), len(group_structure))

                if overlap_ratio >= overlap_threshold:
                    group['nodes'].append(node)
                    group['combined_structure'] = group['combined_structure'].union(combined_structure)
                    added_to_group = True
                    break

            if not added_to_group:
                groups_with_overlap_children.append({
                    'combined_structure': combined_structure,
                    'nodes': [node]
                })

        new_nodes = []
        for node_group in [group['nodes'] for group in groups_with_overlap_children if len(group['nodes']) > 1]:
            node_with_smallest_id = min(node_group, key=lambda node: node.id)
            new_nodes.append(node_with_smallest_id)
            for node in node_group:
                if node != node_with_smallest_id:
                    node_with_smallest_id.children.update(node.children)
                    for child_node in node.children.values():
                        if node.id in self.child_to_parents[child_node.id]:
                            self.child_to_parents[child_node.id].remove(node.id)
                        if node_with_smallest_id.id not in self.child_to_parents[child_node.id]:
                            self.child_to_parents[child_node.id].append(node_with_smallest_id.id)

                    parent_ids = self.child_to_parents.pop(node.id)
                    for parent_id in parent_ids:
                        if parent_id not in self.child_to_parents[node_with_smallest_id.id]:
                            self.child_to_parents[node_with_smallest_id.id].append(parent_id)
                        self.get_node_by_id(parent_id).children[word] = node_with_smallest_id
                    self.word_to_ids[word].remove(node.id)


        for node_id in set(parent_node_ids):
            self.optimize_tree(self.id_to_node[node_id].word, overlap_threshold=overlap_threshold)

    def initialize_node_embeddings(self, embedding_size=100):
        """
        Initialize random embeddings for each node.
        """
        for node_id in self.id_to_node.keys():
            self.node_embeddings[node_id] = np.random.rand(embedding_size)

    def compute_weighted_sentence_embedding(self, sentence_path, weighting_scheme = None):
        """
        Compute a weighted sentence embedding by traversing the sentence path.
        Nodes closer to the root are weighted more heavily.
        """
        weighted_embedding = np.zeros(len(self.node_embeddings[0]))  # Assuming all embeddings have the same size
        for i, node_id in enumerate(sentence_path):
            if weighting_scheme == 'linear':
                distance_from_root = i + 1
                weight = 1 / distance_from_root
            elif weighting_scheme == 'inverse_sigmoid':
                distance_from_root = i + 1  # Assuming sentence_path is ordered from ROOT
                steepness = 1
                weight = 1 - (1 / (1 + np.exp(-steepness * distance_from_root)))
            else:
                weight = 1
            weighted_embedding += weight * self.node_embeddings[node_id]
        return weighted_embedding / len(sentence_path)

    def get_sentence_embedding(self, sentence, morphology, weighting_scheme = None):
        """
        Given a sentence, return its corresponding embedding.
        """

        words = sentence.lower().split()
        node_path = deque([0])
        key_word_index = words.index(morphology.lower())
        words_before = words[:key_word_index]
        words_after = words[key_word_index + 1:]

        current_node = self.preceding_tree
        for i in range(len(words_before) - 1, -1, -1):
            word = words_before[i]
            if word in current_node.children:
                current_node = current_node.children[word]
                node_path.appendleft(current_node.id)
            else:
                raise

        current_node = self.following_tree
        for word in words_after:
            if word in current_node.children:
                current_node = current_node.children[word]
                node_path.append(current_node.id)
            else:
                raise
        return self.compute_weighted_sentence_embedding(node_path, weighting_scheme=weighting_scheme)

    def get_all_sentence_embeddings(self, sentences_dict, weighting_scheme=None):
        """
        Generate embeddings for all sentences in the dataset.
        """
        sentence_embeddings = []
        sentence_list = []
        for morphology, sentences in sentences_dict.items():
            sentences_with_flags = self.add_start_end_flags_lower(sentences)
            for sentence in sentences_with_flags:
                embedding = self.get_sentence_embedding(sentence, morphology, weighting_scheme=weighting_scheme)
                sentence_embeddings.append(embedding)
                sentence_list.append(sentence)
        return np.array(sentence_embeddings), sentence_list

    def cluster_sentences(self, embeddings, n_clusters=3):
        """
        Apply K-Means clustering to sentence embeddings.
        """
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(embeddings)
        return clusters

    def reduce_dimensionality(self, embeddings, n_components=2):
        """
        Reduce dimensionality of embeddings for visualization.
        """
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
        return reduced_embeddings

    def visualize_clusters(self, reduced_embeddings, clusters, sentence_list, title_appendix=None):
        """
        Visualize clusters using a 2D plot.
        """
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis')
        # plt.colorbar(scatter)

        # Annotate sentences for better interpretability
        for i, sentence in enumerate(sentence_list):
            plt.annotate(sentence, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8, alpha=0.7)

        plt.title(f"Sentence Clustering Visualization{', '+title_appendix if title_appendix else ''}")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.show()

    def initialize(self, sentences_dict, overlap_threshold = 1):
        self.create_tree_mask_as_root(sentences_dict)
        self.optimize_tree('<START>', overlap_threshold=overlap_threshold)
        self.optimize_tree('<END>', overlap_threshold=overlap_threshold)