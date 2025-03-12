import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HeteroConv, global_mean_pool
from torch_geometric.data import HeteroData
from theseus.models.GraphDQN.graph_rep import (
    GunGraph,
    HeroGraph,
)  # Import the relevant classes


class GunGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=4):
        super(GunGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def preprocess_state(self, state):
        graph = GunGraph(state)
        return graph.data

    def forward(self, data):
        x, edge_index = data["enemy"].x, data["gun", "shoots", "enemy"].edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long))
        x = self.fc(x)
        return x  # Output raw Q-values


class HeroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels=4):
        super(HeroGNN, self).__init__()
        self.convs = HeteroConv(
            {
                ("hero", "defeats", "enemy"): GCNConv(4, hidden_channels),
                ("hero", "dodges", "bullet"): GCNConv(3, hidden_channels),
                ("hero", "to_go_to", "door"): GCNConv(4, hidden_channels),
                ("hero", "sees_block", "wall"): GCNConv(4, hidden_channels),
            }
        )
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def preprocess_state(self, state):
        graph = HeroGraph(state)
        return graph.data

    def forward(self, data):
        x_dict = {key: data[key].x for key in data.node_types}
        edge_index_dict = data.edge_index_dict
        x_dict = self.convs(x_dict, edge_index_dict)
        hero_x = x_dict["hero"].mean(dim=0, keepdim=True)
        x = self.fc(hero_x)
        return x  # Output raw Q-values
