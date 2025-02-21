import pytest
from typing import List
from theseus.utils.network import ActionSpace

import pytest
from typing import List


def test_total_actions_calculation():
    """Test that total_actions property calculates correctly"""
    action_space = ActionSpace()
    expected = (
        ActionSpace.MOVE_DIMENSION
        * ActionSpace.ATTACK_DIMENSION
        * ActionSpace.PHASE_DIMENSION
        * ActionSpace.BOMB_DIMENSION
    )
    assert action_space.total_actions == expected


def test_total_actions_with_different_dimensions():
    """Test total_actions with modified dimensions"""
    # Save original dimensions
    original_move = ActionSpace.MOVE_DIMENSION
    original_attack = ActionSpace.ATTACK_DIMENSION
    original_phase = ActionSpace.PHASE_DIMENSION
    original_bomb = ActionSpace.BOMB_DIMENSION

    try:
        # Modify dimensions
        ActionSpace.MOVE_DIMENSION = 3
        ActionSpace.ATTACK_DIMENSION = 4
        ActionSpace.PHASE_DIMENSION = 2
        ActionSpace.BOMB_DIMENSION = 2

        action_space = ActionSpace()
        assert action_space.total_actions == 48  # 3 * 4 * 2 * 2
    finally:
        # Restore original dimensions
        ActionSpace.MOVE_DIMENSION = original_move
        ActionSpace.ATTACK_DIMENSION = original_attack
        ActionSpace.PHASE_DIMENSION = original_phase
        ActionSpace.BOMB_DIMENSION = original_bomb


@pytest.mark.parametrize(
    "action,expected",
    [
        (0, [0, 0, 0, 0]),  # Minimum values
        (24, [4, 4, 0, 0]),  # Maximum values within default dimensions
        (12, [2, 2, 0, 0]),  # Middle values
        (1, [1, 0, 0, 0]),  # Only move direction
        (5, [0, 1, 0, 0]),  # Only attack direction
    ],
)
def test_interpret_action_valid_inputs(action, expected):
    """Test interpret_action with various valid inputs"""
    result = ActionSpace.interpret_action(action)
    assert result == expected


def test_interpret_action_negative():
    """Test interpret_action with negative input"""
    with pytest.raises(
        ValueError
    ):
        ActionSpace.interpret_action(-1)


@pytest.mark.parametrize(
    "actions,expected",
    [
        ([0, 0, 0, 0], 0),  # Minimum values
        ([4, 4, 0, 0], 24),  # Maximum values within default dimensions
        ([2, 2, 0, 0], 12),  # Middle values
        ([1, 0, 0, 0], 1),  # Only move direction
        ([0, 1, 0, 0], 5),  # Only attack direction
    ],
)
def test_encode_action_valid_inputs(actions, expected):
    """Test encode_action with various valid inputs"""
    result = ActionSpace.encode_action(actions)
    assert result == expected


@pytest.mark.parametrize(
    "invalid_actions",
    [
        [6, 0, 0, 0],  # Invalid move direction
        [0, 6, 0, 0],  # Invalid attack direction
        [0, 0, 2, 0],  # Invalid phase decision
        [0, 0, 0, 2],  # Invalid bomb decision
        [1, 1, 1],  # Too few elements
        [1, 1, 1, 1, 1],  # Too many elements
    ],
)
def test_encode_action_invalid_inputs(invalid_actions):
    """Test encode_action with invalid inputs"""
    with pytest.raises(ValueError):
        ActionSpace.encode_action(invalid_actions)


def test_encode_decode_symmetry():
    """Test that encoding and then decoding returns the original action"""
    original_actions = [2, 3, 0, 0]
    encoded = ActionSpace.encode_action(original_actions)
    decoded = ActionSpace.interpret_action(encoded)
    assert decoded == original_actions


def test_with_modified_dimensions():
    """Test with modified dimensions"""
    # Save original dimensions
    original_move = ActionSpace.MOVE_DIMENSION
    original_attack = ActionSpace.ATTACK_DIMENSION
    original_phase = ActionSpace.PHASE_DIMENSION
    original_bomb = ActionSpace.BOMB_DIMENSION

    try:
        # Modify dimensions
        ActionSpace.MOVE_DIMENSION = 3
        ActionSpace.ATTACK_DIMENSION = 3
        ActionSpace.PHASE_DIMENSION = 2
        ActionSpace.BOMB_DIMENSION = 2

        # Test encode_action with new dimensions
        actions = [2, 2, 1, 1]
        encoded = ActionSpace.encode_action(actions)
        decoded = ActionSpace.interpret_action(encoded)
        assert decoded == actions

        # Test invalid actions with new dimensions
        with pytest.raises(ValueError):
            ActionSpace.encode_action(
                [3, 0, 0, 0]
            )  # Invalid move direction for new dimensions

    finally:
        # Restore original dimensions
        ActionSpace.MOVE_DIMENSION = original_move
        ActionSpace.ATTACK_DIMENSION = original_attack
        ActionSpace.PHASE_DIMENSION = original_phase
        ActionSpace.BOMB_DIMENSION = original_bomb


def test_boundary_conditions():
    """Test boundary conditions for different dimensions"""
    # Save original dimensions
    original_move = ActionSpace.MOVE_DIMENSION
    original_attack = ActionSpace.ATTACK_DIMENSION
    original_phase = ActionSpace.PHASE_DIMENSION
    original_bomb = ActionSpace.BOMB_DIMENSION

    try:
        # Test with minimum possible dimensions
        ActionSpace.MOVE_DIMENSION = 1
        ActionSpace.ATTACK_DIMENSION = 1
        ActionSpace.PHASE_DIMENSION = 1
        ActionSpace.BOMB_DIMENSION = 1

        # Test minimum case
        assert ActionSpace.interpret_action(0) == [0, 0, 0, 0]
        assert ActionSpace.encode_action([0, 0, 0, 0]) == 0

        # Test with larger dimensions
        ActionSpace.MOVE_DIMENSION = 10
        ActionSpace.ATTACK_DIMENSION = 10
        ActionSpace.PHASE_DIMENSION = 2
        ActionSpace.BOMB_DIMENSION = 2

        # Test a large action number
        large_action = [9, 9, 1, 1]
        encoded = ActionSpace.encode_action(large_action)
        decoded = ActionSpace.interpret_action(encoded)
        assert decoded == large_action

    finally:
        # Restore original dimensions
        ActionSpace.MOVE_DIMENSION = original_move
        ActionSpace.ATTACK_DIMENSION = original_attack
        ActionSpace.PHASE_DIMENSION = original_phase
        ActionSpace.BOMB_DIMENSION = original_bomb


def test_action_space_consistency():
    """Test that all possible actions are unique and reversible"""
    action_space = ActionSpace()
    seen_actions = set()

    # Test all possible combinations within small dimensions
    original_dims = (
        ActionSpace.MOVE_DIMENSION,
        ActionSpace.ATTACK_DIMENSION,
        ActionSpace.PHASE_DIMENSION,
        ActionSpace.BOMB_DIMENSION,
    )

    try:
        # Use smaller dimensions to make testing feasible
        ActionSpace.MOVE_DIMENSION = 3
        ActionSpace.ATTACK_DIMENSION = 3
        ActionSpace.PHASE_DIMENSION = 2
        ActionSpace.BOMB_DIMENSION = 2

        total_actions = (
            ActionSpace.MOVE_DIMENSION
            * ActionSpace.ATTACK_DIMENSION
            * ActionSpace.PHASE_DIMENSION
            * ActionSpace.BOMB_DIMENSION
        )

        for action in range(total_actions):
            decoded = ActionSpace.interpret_action(action)
            encoded = ActionSpace.encode_action(decoded)

            # Check that each action is unique
            action_tuple = tuple(decoded)
            assert action_tuple not in seen_actions
            seen_actions.add(action_tuple)

            # Check that encoding and decoding are consistent
            assert encoded == action
            assert ActionSpace.interpret_action(encoded) == decoded

    finally:
        # Restore original dimensions
        (
            ActionSpace.MOVE_DIMENSION,
            ActionSpace.ATTACK_DIMENSION,
            ActionSpace.PHASE_DIMENSION,
            ActionSpace.BOMB_DIMENSION,
        ) = original_dims
