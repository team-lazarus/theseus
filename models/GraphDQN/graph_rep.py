import torch
from theseus.utils import State
from torch_geometric.data import HeteroData
import random


def normalize(value: float, min_val: float, max_val: float):
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0


def random_state(
    num_enemies: int = 3, num_bullets: int = 2, num_doors: int = 2, num_walls: int = 3
):
    hero = {
        "position": [
            normalize(random.uniform(0, 10), 0, 10),
            normalize(random.uniform(0, 10), 0, 10),
        ],
        "health": normalize(random.randint(0, 10), 0, 10),
        "phase_cooldown": normalize(random.uniform(0, 5), 0, 5),
        "ability_cooldown": normalize(random.uniform(0, 5), 0, 5),
        "shoot_cooldown": normalize(random.uniform(0, 5), 0, 5),
    }

    bullets = [
        {
            "position": [
                normalize(random.uniform(0, 10), 0, 10),
                normalize(random.uniform(0, 10), 0, 10),
            ],
            "direction": normalize(random.uniform(0, 3.14), 0, 3.14),
            "type": normalize(random.randint(0, 2), 0, 2),
        }
        for _ in range(num_bullets)
    ]

    enemies = [
        {
            "position": [
                normalize(random.uniform(0, 10), 0, 10),
                normalize(random.uniform(0, 10), 0, 10),
            ],
            "health": normalize(random.randint(0, 10), 0, 10),
            "direction": normalize(random.uniform(0, 3.14), 0, 3.14),
            "type": normalize(random.randint(0, 3), 0, 3),
        }
        for _ in range(num_enemies)
    ]

    doors = [
        [normalize(random.uniform(0, 10), 0, 10) for _ in range(4)]
        for _ in range(num_doors)
    ]
    backdoor = [normalize(random.uniform(0, 10), 0, 10) for _ in range(4)]
    walls = [
        [normalize(random.uniform(0, 10), 0, 10) for _ in range(4)]
        for _ in range(num_walls)
    ]

    return State(hero, bullets, enemies, doors, backdoor, walls)


class GunGraph:
    """Represents the state focused on gun-enemy interactions."""

    def __init__(self, state: State) -> None:
        """Initializes the GunGraph HeteroData object."""
        self.data = HeteroData()

        self.data["gun"].num_nodes = 1
        self.data["gun"].x = torch.ones((1, 1), dtype=torch.float)

        num_enemies = len(state.enemies)
        self.data["enemy"].num_nodes = num_enemies
        enemy_features = [
            [state.hero.x - e.x, state.hero.y - e.y, e.type_, e.health]
            for e in state.enemies
        ]
        self.data["enemy"].x = torch.tensor(enemy_features, dtype=torch.float).view(
            -1, 4
        )

        # Forward edge
        if num_enemies > 0:
            forward_edge_index = torch.tensor(
                [[0] * num_enemies, list(range(num_enemies))], dtype=torch.long
            )
            self.data["gun", "shoots", "enemy"].edge_index = forward_edge_index
            # Reverse edge
            self.data["enemy", "rev_shoots", "gun"].edge_index = forward_edge_index[
                [1, 0]
            ]
        else:
            self.data["gun", "shoots", "enemy"].edge_index = torch.empty(
                (2, 0), dtype=torch.long
            )
            self.data["enemy", "rev_shoots", "gun"].edge_index = torch.empty(
                (2, 0), dtype=torch.long
            )

    def show_details(self) -> None:
        """Prints details about the graph structure and tensors."""
        print("GunGraph Details:")
        print(
            f"- Gun nodes: {self.data['gun'].num_nodes}, Features Shape: {self.data['gun'].x.shape}"
        )
        print(
            f"- Enemy nodes: {self.data['enemy'].num_nodes}, Features Shape: {self.data['enemy'].x.shape}"
        )
        for edge_type in self.data.edge_types:
            print(
                f"- {edge_type} Edge Index Shape: {self.data[edge_type].edge_index.shape}"
            )


class HeroGraph:
    """Represents the state focused on hero-centric interactions."""

    def __init__(self, state: State) -> None:
        """Initializes the HeroGraph HeteroData object."""
        self.data = HeteroData()
        self.data["hero"].num_nodes = 1
        self.data["hero"].x = torch.tensor(
            [
                [
                    state.hero.x,
                    state.hero.y,
                    state.hero.health,
                    state.hero.phase_cooldown,
                    state.hero.ability_cooldown,
                    state.hero.shoot_cooldown,  # Check attribute name correctness
                ]
            ],
            dtype=torch.float,
        )  # Shape [1, 6]

        num_enemies = len(state.enemies)
        num_bullets = len(state.bullets)
        num_doors = len(state.doors)
        num_walls = len(state.walls)

        self.data["enemy"].num_nodes = num_enemies
        self.data["bullet"].num_nodes = num_bullets
        self.data["door"].num_nodes = num_doors
        self.data["wall"].num_nodes = num_walls

        enemy_features = [
            [e.x, e.y, e.type_, e.health]
            for e in state.enemies
        ]
        self.data["enemy"].x = torch.tensor(enemy_features, dtype=torch.float).view(
            -1, 4
        )

        bullet_features = [[b.x, b.y, b.direction] for b in state.bullets]
        self.data["bullet"].x = torch.tensor(bullet_features, dtype=torch.float).view(
            -1, 3
        )

        self.data["door"].x = torch.tensor(state.doors, dtype=torch.float).view(-1, 4)
        self.data["wall"].x = torch.tensor(state.walls, dtype=torch.float).view(-1, 4)

        # Forward Edges
        edge_types_forward = [
            ("hero", "defeats", "enemy", num_enemies),
            ("hero", "dodges", "bullet", num_bullets),
            ("hero", "to_go_to", "door", num_doors),
            ("hero", "sees_block", "wall", num_walls),
        ]
        for src, rel, dst, num_dst in edge_types_forward:
            if num_dst > 0:
                self.data[src, rel, dst].edge_index = torch.tensor(
                    [[0] * num_dst, list(range(num_dst))], dtype=torch.long
                )
            else:
                self.data[src, rel, dst].edge_index = torch.empty(
                    (2, 0), dtype=torch.long
                )

        # Reverse Edges
        edge_types_reverse = [
            ("enemy", "rev_defeats", "hero", num_enemies),
            ("bullet", "rev_dodges", "hero", num_bullets),
            ("door", "rev_to_go_to", "hero", num_doors),
            ("wall", "rev_sees_block", "hero", num_walls),
        ]
        for dst, rel_rev, src, num_dst in edge_types_reverse:
            rel = rel_rev.replace("rev_", "")  # Find original relation name
            if num_dst > 0:
                original_edge_index = self.data[src, rel, dst].edge_index
                self.data[dst, rel_rev, src].edge_index = original_edge_index[[1, 0]]
            else:
                self.data[dst, rel_rev, src].edge_index = torch.empty(
                    (2, 0), dtype=torch.long
                )

    def show_details(self) -> None:
        """Prints details about the graph structure and tensors."""
        print("HeroGraph Details:")
        for node_type in self.data.node_types:
            print(
                f"- {node_type} nodes: {self.data[node_type].num_nodes}, Features Shape: {self.data[node_type].x.shape}"
            )
        for edge_type in self.data.edge_types:
            print(
                f"- {edge_type} Edge Index Shape: {self.data[edge_type].edge_index.shape}"
            )
