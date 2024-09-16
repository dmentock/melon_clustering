import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
from melon_clustering import PatternExtractorGraph

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

class PatternExtractorGNN(PatternExtractorGraph):
    def __init__(self, seed = 1):
        super().__init__(seed = seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.graph = nx.DiGraph()
        self.node_embeddings = {}

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

    def initialize_node_embeddings(self, hidden_dim=64, output_dim=100, epochs=200):
        self.set_up_digraph()
        edge_index = self.build_graph()
        node_features = self.initialize_node_features(feature_dim=100)
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
        self.node_embeddings = out.detach().cpu().numpy()

    def compute_weighted_sentence_embedding(self, sentence_path, steepness=1.0):
        weighted_embedding = np.zeros(self.node_embeddings.shape[1])
        for i, node_id in enumerate(sentence_path):
            weight = 1 - (1 / (1 + np.exp(-steepness * (i + 1))))
            weighted_embedding += weight * self.node_embeddings[node_id]
        return weighted_embedding / len(sentence_path)
