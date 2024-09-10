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

    def optimize_tree(self, word, overlap = 1):
        print("optimize", word)
        all_nodes = self.get_nodes_by_word(word)
        groups = []
        print("all_nodes",all_nodes)
        for node in all_nodes:
            # Get the current node's child structure as a set of (word, id) tuples
            child_structure = set((child.word, child.id) for child in node.children.values())
            print("child_structure",child_structure)
            # Flag to check if the node is added to an existing group
            added_to_group = False

            # Iterate over existing groups to find a match based on the 80% overlap rule
            for group in groups:
                group_child_structure = group['child_structure']

                # Calculate the intersection between the current node's child structure and the group's structure
                intersection = child_structure.intersection(group_child_structure)

                # Calculate the percentage overlap for both the node and the group
                overlap_node = len(intersection) / len(child_structure) if child_structure else 0
                overlap_group = len(intersection) / len(group_child_structure) if group_child_structure else 0

                # If both overlaps are at least 80%, group the node with this group
                print("overlap_node",overlap_node)
                print("overlap_group",overlap_group)
                if overlap_node >= overlap and overlap_group >= overlap:
                    group['nodes'].append(node)
                    added_to_group = True
                    break

            # If no group satisfies the overlap condition, create a new group
            if not added_to_group:
                groups.append({
                    'child_structure': child_structure,
                    'nodes': [node]
                })
        print("groups",groups)
        for node_group in groups:
            node_with_smallest_id = min(node_group['nodes'], key=lambda node: node.id)
            parents = []
            for node in node_group['nodes']:
                if node != node_with_smallest_id:
                    parents.extend(self.get_parents_by_id(node.id))
                    node_with_smallest_id.children.update(node.children)
            for parent_node in parents:
                parent_node.children[word] = node_with_smallest_id
            for parent in parents:
                self.optimize_tree(parent.word, overlap=overlap)

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

    def initialize(self, sentences_dict, overlap = 1):
        self.create_tree_mask_as_root(sentences_dict)
        self.optimize_tree('<START>', overlap=overlap)
        self.optimize_tree('<END>', overlap=overlap)