import matplotlib.pyplot as plt
import networkx as nx
import re
from collections import defaultdict

class Node:
    def __init__(self, label, id):
        self.label = label  # This can be POS tag, deprel, or a combination
        self.id = id
        self.count = 0
        self.children = {}  # key: child label, value: Node

# PatternExtractor class
class PatternExtractor:
    def __init__(self):
        self.graph = {}
        self.node_counter = 0
        self.label_to_nodes = defaultdict(list)  # Use defaultdict(list)
        self.id_to_node = {}
        self.root_nodes = []
        self.sentence_paths = []

    # Function to extract tags
    def extract_tags(self, sentence_info):
        print("sentence_info",sentence_info)
        sentence = sentence_info['sentence']
        dependencies = sentence_info['dependencies']
        # Regex to match <id:pos/deprel>(word)
        pattern = r'<(\d+):([^/]+)/([^>]+)>\(([^)]+)\)'

        pos_list = []
        deprel_list = []
        word_list = []

        for match in re.finditer(pattern, sentence):
            word_id, pos, deprel, word = match.groups()
            pos_list.append(f"<{pos}>")
            deprel_list.append(f"<{deprel}>")
            word_list.append(word)

        # Join the lists back into strings
        pos_string = " ".join(pos_list)
        deprel_string = " ".join(deprel_list)
        word_string = " ".join(word_list)

        return pos_string, deprel_string, word_list, dependencies

    def add_to_graph(self, sentence_info):
        pos_string, deprel_string, word_list, dependencies = self.extract_tags(sentence_info)
        node_map = {}  # Map from token id to node
        path = []      # List to store the path of node IDs for this sentence

        # Build nodes for each token
        tokens = re.finditer(r'<(\d+):([^/]+)/([^>]+)>\(([^)]+)\)', sentence_info['sentence'])
        for match in tokens:
            word_id, pos, deprel, word = match.groups()
            deprel = deprel.split(':')[0]  # Normalize dependency relation if needed
            node_label = f"<{pos}/{deprel}>"
            # Check if a node with this label already exists
            if self.label_to_nodes[node_label]:
                node = self.label_to_nodes[node_label][0]  # Reuse existing node
            else:
                node = Node(node_label, self.node_counter)
                self.id_to_node[self.node_counter] = node
                self.label_to_nodes[node_label].append(node)
                self.node_counter += 1
            node.count += 1
            node_map[int(word_id)] = node
            path.append(node.id)  # Record the node ID in the path

        # Build edges based on dependencies
        for dep_id, head_id, deprel in dependencies:
            dep_node = node_map[int(dep_id)]
            if int(head_id) == 0:
                # This is the root node
                if dep_node not in self.root_nodes:
                    self.root_nodes.append(dep_node)
                continue
            head_node = node_map[int(head_id)]
            # Use labels to avoid duplicate nodes
            if dep_node.label not in head_node.children:
                head_node.children[dep_node.label] = dep_node

        # After processing the sentence, store the sentence and its path
        self.sentence_paths.append((sentence_info['sentence'], path))


    def print_graph(self):
        for root in self.root_nodes:
            self.print_node(root)

    def print_node(self, node, level=0, visited=None):
        if visited is None:
            visited = set()
        if node.id in visited:
            return
        visited.add(node.id)
        print('  ' * level + f"{node.label} (count: {node.count}, id: {node.id})")
        for child in node.children.values():
            self.print_node(child, level + 1, visited)

    def visualize_graph(self):
        G = nx.DiGraph()
        visited = set()
        for root in self.root_nodes:
            self.add_nodes_to_graph(G, root, visited)

        pos = nx.spring_layout(G)
        labels = {node_id: self.id_to_node[node_id].label for node_id in G.nodes()}
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=1500, node_color='lightblue', arrowsize=20)
        plt.show()

    def add_nodes_to_graph(self, G, node, visited):
        if node.id in visited:
            return
        visited.add(node.id)
        G.add_node(node.id)
        for child in node.children.values():
            G.add_edge(node.id, child.id)
            self.add_nodes_to_graph(G, child, visited)
