import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque

import torch
import torch.nn as nn
import random

from melon_clustering import PatternExtractor, PLOT_DIR

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class Node:
    def __init__(self, word, id):
        self.word = word
        self.id = id
        self.count = 0
        self.children = {}

class PatternExtractorWithGNN(PatternExtractor):
    def __init__(self, seed=1):
        super().__init__()
        self.graph = nx.DiGraph()  # store graph edges
        self.node_embeddings = {}  # store GCN embeddings

        torch.manual_seed(seed) # provide seed for reproducability
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def set_up_digraph(self):
        for node in self.id_to_node.values():
            for child_node in node.children.values():
                self.graph.add_edge(node.id, child_node.id)

    def build_graph(self):
        edge_index = torch.tensor(list(self.graph.edges)).t().contiguous()
        return edge_index

    def initialize_node_features(self, feature_dim=100):
        num_nodes = self.node_counter
        node_features = torch.randn((num_nodes, feature_dim), requires_grad=True)
        return node_features

    def train_gnn(self, edge_index, node_features, hidden_dim=64, output_dim=100, epochs=200):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GCN(input_dim=node_features.shape[1], hidden_dim=hidden_dim, output_dim=output_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        edge_index = edge_index.to(device)
        node_features = node_features.to(device)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(node_features, edge_index)
            loss = F.mse_loss(out, node_features)  # Unsupervised loss
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

        return out.detach().cpu().numpy()

    def generate_sentence_embeddings(self, gnn_node_embeddings, sentence, morphology, steepness=1.0):
        words = sentence.split()
        node_path = deque([0])
        key_word_index = words.index(morphology.lower())
        words_before = words[:key_word_index]
        words_after = words[key_word_index + 1:]

        current_node = self.preceding_tree
        for word in words_before[::-1]:
            if word in current_node.children:
                current_node = current_node.children[word]
                node_path.appendleft(current_node.id)

        current_node = self.following_tree
        for word in words_after:
            if word in current_node.children:
                current_node = current_node.children[word]
                node_path.append(current_node.id)

        return self.compute_weighted_sentence_embedding(gnn_node_embeddings, node_path, steepness=steepness)

    def compute_weighted_sentence_embedding(self, gnn_node_embeddings, sentence_path, steepness=1.0):
        weighted_embedding = np.zeros(gnn_node_embeddings.shape[1])
        for i, node_id in enumerate(sentence_path):
            weight = 1 - (1 / (1 + np.exp(-steepness * (i + 1))))
            weighted_embedding += weight * gnn_node_embeddings[node_id]
        return weighted_embedding / len(sentence_path)
