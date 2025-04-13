import torch
import torch.nn.functional as F
import logging
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
from torch_geometric.data import HeteroData

from theseus.models.GraphDQN.graph_rep import GunGraph, HeroGraph, State


class GunGNN(torch.nn.Module):
    """GNN focusing on gun actions, using gun's state informed by enemies."""

    GUN_OUT_CHANNELS = 8

    def __init__(
        self, hidden_channels: int, out_channels: int = GUN_OUT_CHANNELS
    ) -> None:
        """Initializes the GunGNN with forward and reverse convolutions."""
        super().__init__()
        gun_in = 1
        enemy_in = 4

        self.conv1 = HeteroConv(
            {
                ("enemy", "rev_shoots", "gun"): SAGEConv(
                    (enemy_in, gun_in), hidden_channels
                ),
                # Optionally include ("gun", "shoots", "enemy") if needed
            },
            aggr="mean",
        )  # Use mean aggregation

        self.fc = torch.nn.Linear(hidden_channels, out_channels)
        self.logger = logging.getLogger(f"model-{self.__class__.__name__}")

    def preprocess_state(self, state: State) -> HeteroData:
        """Creates the GunGraph data object including reverse edges."""
        graph = GunGraph(state)
        return graph.data

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Forward pass using HeteroConv to update gun node."""
        x_dict = {ntype: data[ntype].x for ntype in data.node_types}
        edge_index_dict = data.edge_index_dict

        x_dict_updated = self.conv1(x_dict, edge_index_dict)

        if "gun" not in x_dict_updated:
            self.logger.warning("Gun node features not updated (check enemies/edges).")
            batch_size = data.num_graphs if hasattr(data, "num_graphs") else 1
            return torch.zeros(
                (batch_size, self.fc.out_features), device=self.fc.weight.device
            )

        gun_x_updated = x_dict_updated["gun"]

        # Check for empty tensor after potential update (e.g., if input features were empty)
        if gun_x_updated.numel() == 0:
            self.logger.warning("Updated gun features tensor is empty.")
            batch_size = data.num_graphs if hasattr(data, "num_graphs") else 1
            return torch.zeros(
                (batch_size, self.fc.out_features), device=self.fc.weight.device
            )

        # Input to fc layer should have shape [batch_size, hidden_channels]
        q_values = self.fc(gun_x_updated)
        return q_values


class HeroGNN(torch.nn.Module):
    """GNN focusing on hero actions based on environment state."""

    HERO_OUT_CHANNELS = 9

    def __init__(
        self, hidden_channels: int, out_channels: int = HERO_OUT_CHANNELS
    ) -> None:
        """Initializes the HeroGNN."""
        super().__init__()
        hero_in = 6
        enemy_in = 4
        bullet_in = 3
        door_in = 4
        wall_in = 4

        self.convs = HeteroConv(
            {
                ("hero", "defeats", "enemy"): SAGEConv(
                    (hero_in, enemy_in), hidden_channels
                ),
                ("hero", "dodges", "bullet"): SAGEConv(
                    (hero_in, bullet_in), hidden_channels
                ),
                ("hero", "to_go_to", "door"): SAGEConv(
                    (hero_in, door_in), hidden_channels
                ),
                ("hero", "sees_block", "wall"): SAGEConv(
                    (hero_in, wall_in), hidden_channels
                ),
                ("enemy", "rev_defeats", "hero"): SAGEConv(
                    (enemy_in, hero_in), hidden_channels
                ),
                ("bullet", "rev_dodges", "hero"): SAGEConv(
                    (bullet_in, hero_in), hidden_channels
                ),
                ("door", "rev_to_go_to", "hero"): SAGEConv(
                    (door_in, hero_in), hidden_channels
                ),
                ("wall", "rev_sees_block", "hero"): SAGEConv(
                    (wall_in, hero_in), hidden_channels
                ),
            },
            aggr="sum",
        )

        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def preprocess_state(self, state: State) -> HeteroData:
        """Creates the HeroGraph data object."""
        graph = HeroGraph(state)
        return graph.data

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Forward pass using HeteroConv."""
        x_dict = {key: data[key].x for key in data.node_types}
        edge_index_dict = data.edge_index_dict

        x_dict_updated = self.convs(x_dict, edge_index_dict)

        if "hero" not in x_dict_updated:
            raise RuntimeError("Hero features not updated by HeteroConv.")

        hero_x = x_dict_updated["hero"]

        batch = (
            data["hero"].batch
            if "batch" in data["hero"]
            else torch.zeros(hero_x.size(0), dtype=torch.long, device=hero_x.device)
        )
        hero_pooled = global_mean_pool(hero_x, batch)

        q_values = self.fc(hero_pooled)
        return q_values
