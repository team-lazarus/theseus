from math import tan, isclose
from typing import Dict, Any
from dataclasses import dataclass

"""
Simple classes to help store the game state.

Player: Class containing hero information
Attributes:
    x, y: int, int          # Hero's coordinates
    health: int             # Health value from 0-10
    phase_cooldown: float   # Cooldown time for phase ability
    ability_cooldown: float # Cooldown time for special ability
    shoot_cooldown: float   # Cooldown time between shots
    

Enemy: Class containing enemy information
Attributes:
    x, y: int, int          # Enemy's coordinates
    health: int             # Health value from 0-10
    direction: float        # Facing direction in radians
    type_: str              # Enemy type identifier

Bullet: Class containing bullet information
Attributes:
    x, y: int, int          # Bullet's coordinates
    direction: float        # Direction in radians
    type_: str              # Bullet type identifier

These classes are designed to wrap the dictionary data structures provided by the pipeline:
- Player wraps: {"position": [x, y], "health": int, "phase_cooldown": float, 
                "ability_cooldown": float, "shoot_cooldown": float}
- Enemy wraps: {"position": [x, y], "health": int, "direction": float, "type": str}
- Bullet wraps: {"position": [x, y], "direction": float, "type": str}

"""


@dataclass
class Player(object):
    def __init__(self, status: Dict[str, Any]) -> None:
        self.update(status)

    def update(self, status: Dict[str, Any]) -> None:
        self.x, self.y = status["position"]
        self.health = status["health"]
        self.phase_cooldown = status["phase_cooldown"]
        self.ability_cooldown = status["ability_cooldown"]
        self.shoot_cooldown = status["shoot_cooldown"]


@dataclass
class Enemy(object):
    def __init__(self, status: Dict[str, Any]) -> None:
        self.update(status)

    def update(self, status: Dict[str, Any]) -> None:
        self.x, self.y = status["position"]
        self.health = status["health"]
        self.direction = status["direction"]
        self.type_ = status["type"]


@dataclass
class Bullet(object):
    def __init__(self, status: Dict[str, Any]) -> None:
        self.update(status)

    def update(self, status: Dict[str, Any]) -> None:
        self.x, self.y = status["position"]
        self.direction = status["direction"]
        self.type_ = status["type"]

    def on_target(self, target_coords) -> bool:
        tx, ty = target_coords

        return isclose(tan(self.direction), (self.y - ty) / (self.x - tx))
