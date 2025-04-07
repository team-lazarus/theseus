import torch
from theseus.utils import State
from torch_geometric.data import HeteroData
import random


def normalize(value: float, min_val: float, max_val: float):
    return (value - min_val) / (max_val - min_val) if max_val > min_val else 0.0


def random_state(num_enemies: int=3, num_bullets: int=2, num_doors: int=2, num_walls: int=3):
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
    def __init__(self, state: State):
        self.data = HeteroData()
        self.data["gun"].num_nodes = 1 # Gun node exists
        num_enemies = len(state.enemies)
        self.data["enemy"].num_nodes = num_enemies

        # --- FIX: Ensure enemy features are always 2D ---
        enemy_features = [
            [e.x, e.y, e.type_, e.health]
            for e in state.enemies
        ]
        # Use .view() to ensure shape is [num_enemies, 4], even [0, 4] if empty
        self.data["enemy"].x = torch.tensor(
            enemy_features, dtype=torch.float
        ).view(-1, 4)

        # --- FIX: Handle empty edge index case ---
        if num_enemies > 0:
            self.data["gun", "shoots", "enemy"].edge_index = torch.tensor(
                [[0] * num_enemies, list(range(num_enemies))], dtype=torch.long
            )
        else:
            # Create an empty edge_index with correct shape [2, 0]
            self.data["gun", "shoots", "enemy"].edge_index = torch.empty((2, 0), dtype=torch.long)

    def show_details(self):
        print("GunGraph Details:")
        print("- Gun nodes:", self.data["gun"].num_nodes)
        print("- Enemies:", self.data["enemy"].num_nodes)
        # Add shape check for debugging
        print("- Enemy Features Shape:", self.data["enemy"].x.shape)
        print("- Shoots Edge Index Shape:", self.data["gun", "shoots", "enemy"].edge_index.shape)


class HeroGraph:
    def __init__(self, state: State):
        self.data = HeteroData()
        self.data["hero"].num_nodes = 1

        # --- Define Hero Features ---
        self.data["hero"].x = torch.tensor(
            [[ # Add outer brackets
                state.hero.x,
                state.hero.y,
                state.hero.health,
                state.hero.phase_cooldown,
                state.hero.ability_cooldown,
                state.hero.shoot_cooldown, # Ensure correct attribute name
            ]], # Add outer brackets
            dtype=torch.float,
        )

        # --- Define Other Node Types and Features ---
        num_enemies = len(state.enemies)
        num_bullets = len(state.bullets)
        num_doors = len(state.doors)
        num_walls = len(state.walls)

        self.data["enemy"].num_nodes = num_enemies
        self.data["bullet"].num_nodes = num_bullets
        self.data["door"].num_nodes = num_doors
        self.data["wall"].num_nodes = num_walls

        enemy_features = [[e.x, e.y, e.type_, e.health] for e in state.enemies]
        self.data["enemy"].x = torch.tensor(enemy_features, dtype=torch.float).view(-1, 4)

        bullet_features = [[b.x, b.y, b.direction] for b in state.bullets]
        self.data["bullet"].x = torch.tensor(bullet_features, dtype=torch.float).view(-1, 3)

        self.data["door"].x = torch.tensor(state.doors, dtype=torch.float).view(-1, 4)
        self.data["wall"].x = torch.tensor(state.walls, dtype=torch.float).view(-1, 4)

        # --- Define Forward Edge Indices ---
        if num_enemies > 0:
            self.data["hero", "defeats", "enemy"].edge_index = torch.tensor(
                [[0] * num_enemies, list(range(num_enemies))], dtype=torch.long
            )
        else:
             self.data["hero", "defeats", "enemy"].edge_index = torch.empty((2, 0), dtype=torch.long)

        if num_bullets > 0:
            self.data["hero", "dodges", "bullet"].edge_index = torch.tensor(
                [[0] * num_bullets, list(range(num_bullets))], dtype=torch.long
            )
        else:
             self.data["hero", "dodges", "bullet"].edge_index = torch.empty((2, 0), dtype=torch.long)

        if num_doors > 0:
            self.data["hero", "to_go_to", "door"].edge_index = torch.tensor(
                [[0] * num_doors, list(range(num_doors))], dtype=torch.long
            )
        else:
             self.data["hero", "to_go_to", "door"].edge_index = torch.empty((2, 0), dtype=torch.long)

        if num_walls > 0:
            self.data["hero", "sees_block", "wall"].edge_index = torch.tensor(
                [[0] * num_walls, list(range(num_walls))], dtype=torch.long
            )
        else:
             self.data["hero", "sees_block", "wall"].edge_index = torch.empty((2, 0), dtype=torch.long)

        # --- ADD THE REVERSE EDGE CODE SNIPPET HERE ---
        if num_enemies > 0:
            # Original edge index for ("hero", "defeats", "enemy")
            hero_defeats_enemy_edge_index = self.data["hero", "defeats", "enemy"].edge_index
            # Reverse edge index: Swap rows 0 and 1
            self.data["enemy", "rev_defeats", "hero"].edge_index = hero_defeats_enemy_edge_index[[1, 0]]
        else:
             self.data["enemy", "rev_defeats", "hero"].edge_index = torch.empty((2, 0), dtype=torch.long)

        if num_bullets > 0:
            hero_dodges_bullet_edge_index = self.data["hero", "dodges", "bullet"].edge_index
            self.data["bullet", "rev_dodges", "hero"].edge_index = hero_dodges_bullet_edge_index[[1, 0]]
        else:
            self.data["bullet", "rev_dodges", "hero"].edge_index = torch.empty((2, 0), dtype=torch.long)

        if num_doors > 0:
            hero_to_go_to_door_edge_index = self.data["hero", "to_go_to", "door"].edge_index
            self.data["door", "rev_to_go_to", "hero"].edge_index = hero_to_go_to_door_edge_index[[1, 0]]
        else:
            self.data["door", "rev_to_go_to", "hero"].edge_index = torch.empty((2, 0), dtype=torch.long)

        if num_walls > 0:
            hero_sees_block_wall_edge_index = self.data["hero", "sees_block", "wall"].edge_index
            self.data["wall", "rev_sees_block", "hero"].edge_index = hero_sees_block_wall_edge_index[[1, 0]]
        else:
             self.data["wall", "rev_sees_block", "hero"].edge_index = torch.empty((2, 0), dtype=torch.long)

    def show_details(self):
        print("HeroGraph Details:")
        print("- Hero nodes:", self.data["hero"].num_nodes)
        print("- Enemies:", self.data["enemy"].num_nodes)
        print("- Bullets:", self.data["bullet"].num_nodes)
        print("- Doors:", self.data["door"].num_nodes)
        print("- Walls:", self.data["wall"].num_nodes)
