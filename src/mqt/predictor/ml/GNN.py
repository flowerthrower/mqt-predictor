from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import (
    GlobalAttention,
    TransformerConv,
)

# import gymnasium
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Net(torch.nn.Module):  # type: ignore[misc]
    def __init__(
        self,
        num_node_categories: int,
        num_edge_categories: int,
        node_embedding_dim: int,
        edge_embedding_dim: int,
        num_layers: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.node_embedding = nn.Embedding(num_node_categories, node_embedding_dim)
        self.edge_embedding = nn.Embedding(num_edge_categories, edge_embedding_dim)

        self.layers = []
        for _ in range(num_layers):
            self.layers.append(TransformerConv(-1, hidden_dim, edge_dim=edge_embedding_dim + 2))

        self.gate_nn = torch.nn.Linear(hidden_dim, 1)
        self.nn = torch.nn.Linear(hidden_dim, output_dim)
        self.global_attention = GlobalAttention(self.gate_nn, self.nn)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Apply the node and edge embeddings
        x = self.node_embedding(x).squeeze()
        embedding = self.edge_embedding(edge_attr[:, 0])
        edge_attr = torch.cat([embedding, edge_attr[:, 1:]], dim=1)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = torch.nn.functional.relu(x)

        # Apply a readout layer to get a single vector that represents the entire graph
        x = self.global_attention(x, batch)

        return F.log_softmax(x, dim=1)


#
# class GraphFeaturesExtractor(BaseFeaturesExtractor):
#    def __init__(self, observation_space: gymnasium.spaces.Graph, features_dim: int = 64):
#        super(GraphFeaturesExtractor, self).__init__(observation_space, features_dim)
#
#        num_node_categories = 10 # distinct gate types (incl. 'id' and 'meas')
#        num_edge_categories = 10 # distinct wires (quantum + classical)
#        node_embedding_dim = 4 # dimension of the node embedding
#        edge_embedding_dim = 4 # dimension of the edge embedding
#        num_layers = 2 # number of neighbor aggregations
#        hidden_dim = 16 # dimension of the hidden layers
#        output_dim = 3 # dimension of the output vector
#
#        self.node_embedding = nn.Embedding(num_node_categories, node_embedding_dim)
#        self.edge_embedding = nn.Embedding(num_edge_categories, edge_embedding_dim)
#
#        self.layers = []
#        for _ in range(num_layers):
#            self.layers.append(TransformerConv(-1, hidden_dim, edge_dim=edge_embedding_dim+2))
#
#        self.gate_nn = torch.nn.Linear(hidden_dim, 1)
#        self.nn = torch.nn.Linear(hidden_dim, output_dim)
#        self.global_attention = GlobalAttention(self.gate_nn, self.nn)
#
#
#    def forward(self, data: torch.Tensor) -> torch.Tensor:
#        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
#
#        # Apply the node and edge embeddings
#        x = self.node_embedding(x).squeeze()
#        embedding = self.edge_embedding(edge_attr[:, 0])
#        edge_attr = torch.cat([embedding, edge_attr[:, 1:]], dim=1)
#
#        for layer in self.layers:
#            x = layer(x, edge_index, edge_attr)
#            x = torch.nn.functional.relu(x)
#
#        # Apply a readout layer to get a single vector that represents the entire graph
#        x = self.global_attention(x, batch)
#
#        return x
#
# class GraphMaskableActorCriticPolicy(MaskableActorCriticPolicy):
#    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
#        super(GraphMaskableActorCriticPolicy, self).__init__(observation_space, action_space, lr_schedule,
#                                                             features_extractor_class=GraphFeaturesExtractor, **kwargs)
#
#
