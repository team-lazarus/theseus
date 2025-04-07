import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HeteroConv, SAGEConv, global_mean_pool
from torch_geometric.data import HeteroData
from theseus.models.GraphDQN.graph_rep import (
    GunGraph,
    HeroGraph,
    State,
)
# from theseus.utils import State # If needed


class GunGNN(torch.nn.Module):
    # Output channels correspond to the 8 shooting directions
    GUN_OUT_CHANNELS = 8

    def __init__(self, in_channels: int = 4, hidden_channels: int = 4, out_channels: int = GUN_OUT_CHANNELS):
        super(GunGNN, self).__init__()
        # Using GCNConv based on previous versions, ensure graph structure is compatible
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # Final layer outputs Q-values for the 8 shooting directions
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def preprocess_state(self, state: State) -> HeteroData:
        graph = GunGraph(state)
        return graph.data

    def forward(self, data: HeteroData) -> torch.Tensor:
        # Assuming edge represents potential targets the gun can shoot
        x, edge_index = data["enemy"].x, data["gun", "shoots", "enemy"].edge_index

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()

        batch = data['enemy'].batch if 'batch' in data['enemy'] else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch) # Aggregate information about potential targets
        q_values_shoot_direction = self.fc(x) # Get Q-values for directions
        return q_values_shoot_direction


class HeroGNN(torch.nn.Module):
    # Output channels correspond to the 9 movement actions (8 directions + stay)
    HERO_OUT_CHANNELS = 9

    def __init__(self, hidden_channels: int, out_channels: int = HERO_OUT_CHANNELS):
        super(HeroGNN, self).__init__()

        hero_in = 6
        enemy_in = 4
        bullet_in = 3
        door_in = 4
        wall_in = 4

        self.convs = HeteroConv(
            {
                # Edges from hero outward
                ("hero", "defeats", "enemy"): SAGEConv((hero_in, enemy_in), hidden_channels),
                ("hero", "dodges", "bullet"): SAGEConv((hero_in, bullet_in), hidden_channels),
                ("hero", "to_go_to", "door"): SAGEConv((hero_in, door_in), hidden_channels),
                ("hero", "sees_block", "wall"): SAGEConv((hero_in, wall_in), hidden_channels),
                # Reverse Edges back to hero
                ("enemy", "rev_defeats", "hero"): SAGEConv((enemy_in, hero_in), hidden_channels),
                ("bullet", "rev_dodges", "hero"): SAGEConv((bullet_in, hero_in), hidden_channels),
                ("door", "rev_to_go_to", "hero"): SAGEConv((door_in, hero_in), hidden_channels),
                ("wall", "rev_sees_block", "hero"): SAGEConv((wall_in, hero_in), hidden_channels),
            },
            aggr='sum'
        )

        # Final layer outputs Q-values for the 9 movement actions
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def preprocess_state(self, state: State) -> HeteroData:
        graph = HeroGraph(state) # Includes reverse edges now
        return graph.data

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = {key: data[key].x for key in data.node_types}
        edge_index_dict = data.edge_index_dict

        x_dict = self.convs(x_dict, edge_index_dict) # Updates nodes, including 'hero'

        hero_x = x_dict["hero"] # Use updated hero features

        batch = data['hero'].batch if 'batch' in data['hero'] else torch.zeros(hero_x.size(0), dtype=torch.long, device=hero_x.device)
        hero_pooled = global_mean_pool(hero_x, batch)

        q_values_movement = self.fc(hero_pooled) # Get Q-values for movement

        return q_values_movement
