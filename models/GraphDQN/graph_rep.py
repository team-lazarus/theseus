import torch
from torch_geometric.data import HeteroData
import random


def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0


def random_state(num_enemies=3, num_bullets=2, num_doors=2, num_walls=3):
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
    def __init__(self, state):
        self.data = HeteroData()
        self.data["gun"].num_nodes = 1
        num_enemies = len(state.enemies)
        self.data["enemy"].num_nodes = num_enemies

        self.data["enemy"].x = torch.tensor(
            [
                [e["position"][0], e["position"][1], e["type"], e["health"]]
                for e in state.enemies
            ],
            dtype=torch.float,
        )

        self.data["gun", "shoots", "enemy"].edge_index = torch.tensor(
            [[0] * num_enemies, list(range(num_enemies))], dtype=torch.long
        )

    def show_details(self):
        print("GunGraph Details:")
        print("- Gun nodes:", self.data["gun"].num_nodes)
        print("- Enemies:", self.data["enemy"].num_nodes)
        print("- Enemy Attributes (x, y, type, health):\n", self.data["enemy"].x)


class HeroGraph:
    def __init__(self, state):
        self.data = HeteroData()
        self.data["hero"].num_nodes = 1

        self.data["hero"].x = torch.tensor(
            [
                state.hero["position"][0],
                state.hero["position"][1],
                state.hero["health"],
                state.hero["phase_cooldown"],
                state.hero["ability_cooldown"],
                state.hero["shoot_cooldown"],
            ],
            dtype=torch.float,
        )

        num_enemies = len(state.enemies)
        num_bullets = len(state.bullets)
        num_doors = len(state.doors)
        num_walls = len(state.walls)

        self.data["enemy"].num_nodes = num_enemies
        self.data["bullet"].num_nodes = num_bullets
        self.data["door"].num_nodes = num_doors
        self.data["wall"].num_nodes = num_walls

        self.data["enemy"].x = torch.tensor(
            [
                [e["position"][0], e["position"][1], e["type"], e["health"]]
                for e in state.enemies
            ],
            dtype=torch.float,
        )

        self.data["bullet"].x = torch.tensor(
            [
                [b["position"][0], b["position"][1], b["direction"]]
                for b in state.bullets
            ],
            dtype=torch.float,
        )

        self.data["door"].x = torch.tensor(state.doors, dtype=torch.float)
        self.data["wall"].x = torch.tensor(state.walls, dtype=torch.float)

        self.data["hero", "defeats", "enemy"].edge_index = torch.tensor(
            [[0] * num_enemies, list(range(num_enemies))], dtype=torch.long
        )

        self.data["hero", "dodges", "bullet"].edge_index = torch.tensor(
            [[0] * num_bullets, list(range(num_bullets))], dtype=torch.long
        )

        self.data["hero", "to_go_to", "door"].edge_index = torch.tensor(
            [[0] * num_doors, list(range(num_doors))], dtype=torch.long
        )

        self.data["hero", "sees_block", "wall"].edge_index = torch.tensor(
            [[0] * num_walls, list(range(num_walls))], dtype=torch.long
        )

    def show_details(self):
        print("HeroGraph Details:")
        print("- Hero nodes:", self.data["hero"].num_nodes)
        print("- Enemies:", self.data["enemy"].num_nodes)
        print("- Bullets:", self.data["bullet"].num_nodes)
        print("- Doors:", self.data["door"].num_nodes)
        print("- Walls:", self.data["wall"].num_nodes)
