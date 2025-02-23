from . import TCPClient
from enum import Enum
from typing import List
from random import randrange

from theseus.utils import State

class Environment(object):
    """
    Environment
    -----------
    Encapsulates the environment our bot will be interacting with

    Attributes
    ----------
    tcp_client: TCPClient => communicates with the game
    action_space: ActionSpace => All the actions the player can take
    """

    def __init__(self):
        self.tcp_client = TCPClient()

        self.action_space = ActionSpace()

    def initialise_environment(self):
        """
        Initialises environment
        """
        self.tcp_client.connect()
        game_response = self.tcp_client.read()
        state, _, _ = self.parse_game_response(game_response)

        return state

    def step(self, action: int | str | List[int]):
        """
        Executes an action in the environment
        Attributes:
        -----------
        action: int|str|List[int]
            The action can be an integer from our action space,
            the string value of the parsed action vector
            or a list of integer values, for our action vector
        returns next_state, reward, terminated (is the episode over)
        """
        if type(action) == type(0):
            action = str(self.action_space.interpret_action(action))
        elif type(action) == type([]) and type(action[0]) == type(0):
            action = str(action)
        else:
            raise TypeError("Unsupported type as argument")

        self.tcp_client.write(action)
        game_response = self.tcp_client.read()
        next_state, terminated, reward = self.parse_game_response(game_response)

        return next_state, reward, terminated

    def parse_game_response(self, game_response):
        """
        Parses the game response JSON object
        returns next_state, reward, terminated
        
        TODO: Chaitanya Modi
        """
        return State(), False, 0


class ActionSpace(object):
    """
    Defines the action space for an agent in a grid-based environment.
    The agent can perform:
    - Movement in 8 directions
    - Attacking in 8 directions
    - Phasing (on/off)
    - Dropping a bomb (on/off)

    Attributes:
    -----------
    MOVE_DIMENSION: int
        Number of movement directions (5: up, down, left, right, stay)
    ATTACK_DIMENSION: int
        Number of attack directions (5: same as movement)
    PHASE_DIMENSION: int
        Binary decision for phasing (1 means available, 0 means not used)
    BOMB_DIMENSION: int
        Binary decision for dropping a bomb (1 means available, 0 means not used)
    """

    MOVE_DIMENSION: int = 9
    ATTACK_DIMENSION: int = 9
    PHASE_DIMENSION: int = 1
    BOMB_DIMENSION: int = 1

    @property
    def total_actions(self) -> int:
        """
        Computes the total number of possible actions an agent can take.
        """
        return (
            self.MOVE_DIMENSION
            * self.ATTACK_DIMENSION
            * self.PHASE_DIMENSION
            * self.BOMB_DIMENSION
        )

    def sample(self) -> int:
        """
        Samples a random action from our action space
        """
        return randrange(0, self.total_actions)

    @classmethod
    def interpret_action(cls, action: int) -> List[int]:
        """
        Decodes an integer action into its components:
        - Move direction
        - Attack direction
        - Phase on/off
        - Bomb on/off

        Parameters:
        -----------
        action : int
            The encoded action as an integer.

        Returns:
        --------
        List[int]: [move direction, attack direction, phase decision, bomb decision]
        """
        if action < 0:
            raise ValueError("Action cannot be negative.")

        move_direction: int = action % cls.MOVE_DIMENSION
        action //= cls.MOVE_DIMENSION

        attack_direction: int = action % cls.ATTACK_DIMENSION
        action //= cls.ATTACK_DIMENSION

        phase_decision: int = action % cls.PHASE_DIMENSION
        action //= cls.PHASE_DIMENSION

        bomb_decision: int = action % cls.BOMB_DIMENSION
        action //= cls.BOMB_DIMENSION

        return [move_direction, attack_direction, phase_decision, bomb_decision]

    @classmethod
    def encode_action(cls, actions: List[int]) -> int:
        """
        Encodes a list of action components into a single integer.
        The order of components is:
        - Move direction
        - Attack direction
        - Phase on/off
        - Bomb on/off

        Parameters:
        -----------
        actions : List[int]
            A list containing the four action components.

        Returns:
        --------
        int: Encoded integer representation of the action.
        """
        if len(actions) != 4:
            raise ValueError("Actions list must contain exactly 4 elements.")

        move_dir, attack_dir, phase_on, bomb_on = actions

        # Validate inputs
        if not (0 <= move_dir < cls.MOVE_DIMENSION):
            raise ValueError(f"Invalid move direction: {move_dir}")
        if not (0 <= attack_dir < cls.ATTACK_DIMENSION):
            raise ValueError(f"Invalid attack direction: {attack_dir}")
        if not (0 <= phase_on < cls.PHASE_DIMENSION + 1):  # Binary decision
            raise ValueError(f"Invalid phase decision: {phase_on}")
        if not (0 <= bomb_on < cls.BOMB_DIMENSION + 1):  # Binary decision
            raise ValueError(f"Invalid bomb decision: {bomb_on}")

        # Encode action using positional values
        action: int = move_dir
        action += attack_dir * cls.MOVE_DIMENSION
        action += phase_on * cls.MOVE_DIMENSION * cls.ATTACK_DIMENSION
        action += (
            bomb_on * cls.MOVE_DIMENSION * cls.ATTACK_DIMENSION * cls.PHASE_DIMENSION
        )

        return action
