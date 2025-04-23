from typing import Dict, Any, List
from theseus.utils.actors import Player, Enemy, Bullet


class State(object):
    """Stores the game state.

    Attributes:
        hero (Dict[str, Any]): Dictionary containing hero information with structure:
            {
                "position": [x, y],      # Hero's coordinates
                "health": int,           # Health value from 0-10
                "phase_cooldown": float, # Cooldown time for phase ability
                "ability_cooldown": float,  # Cooldown time for special ability
                "shoot_cooldown": float  # Cooldown time between shots
            }

        bullets (List[Dict[str, Any]]): List of active bullets. Each bullet is a dictionary:
            {
                "position": [x, y],     # Bullet's coordinates
                "direction": float,      # Direction in radians
                "type": str             # Bullet type identifier }

        enemies (List[Dict[str, Any]]): List of enemies. Each enemy is a dictionary:
            {
                "position": [x, y],     # Enemy's coordinates
                "health": int,          # Health value from 0-10
                "direction": float,      # Facing direction in radians
                "type": str             # Enemy type identifier
            }

        doors (List[List[float]]): List of door bounding boxes.
            Each door is defined by [x_1, y_1, x_2, y_2] coordinates.

        backdoor (List[float]): Bounding box coordinates for the door to previous room.
            Defined as [x_1, y_1, x_2, y_2].

        walls (List[List[float]]): List of wall bounding boxes.
            Each wall is defined by [x_1, y_1, x_2, y_2] coordinates.

    TODO:
        * Implement state (Mukundan Gurumurthy)
    """

    def __init__(
        self,
        hero: Dict[str, Any],
        bullets: List[Dict[str, Any]],
        enemies: List[Dict[str, Any]],
        doors: List[List[float]],
        backdoor: List[float],
        walls: List[List[float]],
    ) -> None:
        
        self.hero = Player(hero)
        self.bullets = list(map(Bullet, bullets))
        self.enemies = list(map(Enemy, enemies))
        self.doors = doors
        self.backdoor = backdoor
        self.walls = walls

    # def __hash__(self) -> int:
    #
    #     return 0
