import torch
import random
import logging
import os
import yaml
import numpy as np
import importlib
import pandas as pd
from collections import deque
from datetime import datetime
from itertools import count
from typing import List, Tuple, Optional, Type, Self, Deque, Dict, Any, Union
from torch import nn, optim
from torch_geometric.data import Batch, HeteroData
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TaskID,
)
from rich.table import Table
from rich.console import Console

# Assuming these local imports exist and are correct
from theseus.utils import State, ExperienceReplayMemory
from theseus.utils.network import Environment
import theseus.constants as c
from theseus.models.GraphDQN.ActionGNN import HeroGNN, GunGNN



HERO_ACTION_SPACE_SIZE: int = 9
GUN_ACTION_SPACE_SIZE: int = 8
LOGGING_WINDOW: int = 100
SAVE_INTERVAL: int = 500


HERO_ACTION_SPACE_SIZE: int = 9
GUN_ACTION_SPACE_SIZE: int = 8
LOGGING_WINDOW: int = 100
SAVE_INTERVAL: int = 500


class AgentTheseusGNN:
    """
    Agent managing simultaneous training of Hero and Gun GNNs using DQN.

    This agent uses two separate GNNs (one for hero movement, one for gun
    actions) and trains them concurrently. It employs an Experience Replay
    Memory and epsilon-greedy action selection. Target networks are used
    for stability and synced periodically. Progress is visualized using
    rich.progress. Tracks agent survival time and provides a summary table.

    Args:
        hero_policy_net: The policy network for hero actions.
        hero_target_net: The target network for hero actions.
        gun_policy_net: The policy network for gun actions.
        gun_target_net: The target network for gun actions.
        env: The environment instance.
        loss_fn_class: The class type for the loss function (default: nn.MSELoss).
        optimizer_class: The class type for the optimizer (default: optim.AdamW).
        learning_rate: Learning rate for the optimizers.
        discount_factor: Discount factor (gamma) for future rewards.
        epsilon_init: Initial epsilon value for exploration.
        epsilon_decay: Decay rate for epsilon after each episode.
        epsilon_min: Minimum value for epsilon.
        mini_batch_size: Size of the mini-batch sampled from replay memory.
        target_sync_rate: Frequency (in steps) for syncing target networks.
        replay_memory_size: Capacity of the experience replay memory.
        log_window_size: Number of episodes for calculating rolling averages.
        save_interval: Frequency (in episodes) for saving model checkpoints.

    Attributes:
        logger: Logger instance for logging agent activities.
        device: Computation device ('cuda' or 'cpu').
        env: Environment instance.
        hero_policy_net: Hero policy network instance.
        hero_target_net: Hero target network instance.
        gun_policy_net: Gun policy network instance.
        gun_target_net: Gun target network instance.
        discount_factor: Gamma value.
        mini_batch_size: Batch size for learning.
        target_sync_rate: Target network update frequency.
        sync_steps_taken: Steps counter for target network sync.
        log_window_size: Window size for metric averaging.
        save_interval: Checkpoint save frequency.
        epsilon_init: Initial epsilon value.
        epsilon_decay: Epsilon decay rate.
        epsilon_min: Minimum epsilon value.
        epsilon: Current epsilon value for exploration.
        loss_fn: Loss function instance.
        hero_optimizer: Optimizer for the hero policy network.
        gun_optimizer: Optimizer for the gun policy network.
        memory: Experience replay memory instance.
        episode_rewards_hero_deque: Deque for rolling hero rewards.
        episode_rewards_gun_deque: Deque for rolling gun rewards.
        episode_time_alive_deque: Deque for rolling hero survival time.
        total_reward_hero: Cumulative hero reward.
        total_reward_gun: Cumulative gun reward.
        hero_loss_deque: Deque for rolling hero loss.
        gun_loss_deque: Deque for rolling gun loss.
        training_summary_data: List to store episode metrics for final summary.
    """

    def __init__(
        self,
        hero_policy_net: HeroGNN,
        hero_target_net: HeroGNN,
        gun_policy_net: GunGNN,
        gun_target_net: GunGNN,
        env: Environment,
        *,
        loss_fn_class: Type[nn.Module] = nn.MSELoss,
        optimizer_class: Type[optim.Optimizer] = optim.AdamW,
        learning_rate: float = 1e-4,
        discount_factor: float = 0.99,
        epsilon_init: float = 1,
        epsilon_decay: float = 0.9999,
        epsilon_min: float = 0.05,
        mini_batch_size: int = 64,
        target_sync_rate: int = 500,
        replay_memory_size: int = c.REPLAY_MEMORY_SIZE,
        log_window_size: int = LOGGING_WINDOW,
        save_interval: int = SAVE_INTERVAL,
    ) -> None:
        """Initializes the dual-GNN agent with metrics tracking."""
        self.logger: logging.Logger = logging.getLogger("agent-theseus-gnn")
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.env: Environment = env

        self._validate_network(hero_policy_net, "Hero Policy")
        self._validate_network(hero_target_net, "Hero Target")
        self._validate_network(gun_policy_net, "Gun Policy")
        self._validate_network(gun_target_net, "Gun Target")
        self.hero_policy_net: HeroGNN = hero_policy_net.to(self.device)
        self.hero_target_net: HeroGNN = hero_target_net.to(self.device)
        self.gun_policy_net: GunGNN = gun_policy_net.to(self.device)
        self.gun_target_net: GunGNN = gun_target_net.to(self.device)

        self.discount_factor: float = discount_factor
        self.mini_batch_size: int = mini_batch_size
        self.target_sync_rate: int = target_sync_rate
        self.sync_steps_taken: int = 0
        self.log_window_size: int = log_window_size
        self.save_interval: int = save_interval

        self.epsilon_init: float = epsilon_init
        self.epsilon_decay: float = epsilon_decay
        self.epsilon_min: float = epsilon_min
        self.epsilon: float = epsilon_init

        self.loss_fn: nn.Module = loss_fn_class()
        self.hero_optimizer: optim.Optimizer = optimizer_class(
            self.hero_policy_net.parameters(), lr=learning_rate, amsgrad=True
        )
        self.gun_optimizer: optim.Optimizer = optimizer_class(
            self.gun_policy_net.parameters(), lr=learning_rate, amsgrad=True
        )

        self.memory: ExperienceReplayMemory = ExperienceReplayMemory(replay_memory_size)

        self.episode_rewards_hero_deque: Deque[float] = deque(
            maxlen=self.log_window_size
        )
        self.episode_rewards_gun_deque: Deque[float] = deque(
            maxlen=self.log_window_size
        )
        self.episode_time_alive_deque: Deque[int] = deque(
            maxlen=self.log_window_size
        )  # New deque for time alive
        self.total_reward_hero: float = 0.0
        self.total_reward_gun: float = 0.0
        self.hero_loss_deque: Deque[float] = deque(maxlen=self.log_window_size * 10)
        self.gun_loss_deque: Deque[float] = deque(maxlen=self.log_window_size * 10)

        self.training_summary_data: List[Dict[str, Union[int, float]]] = (
            []
        )  # For summary table

        self.hero_target_net.load_state_dict(self.hero_policy_net.state_dict())
        self.gun_target_net.load_state_dict(self.gun_policy_net.state_dict())
        self.hero_target_net.eval()
        self.gun_target_net.eval()
        self.console = Console()  # For printing the final table

    def _validate_network(self, network: nn.Module, name: str) -> None:
        """
        Checks if a network has the required 'preprocess_state' method.

        Args:
            network: The network module to validate.
            name: The name of the network for error messages.

        Raises:
            AttributeError: If the network lacks a callable 'preprocess_state'.
        """
        if not hasattr(network, "preprocess_state") or not callable(
            network.preprocess_state
        ):
            raise AttributeError(
                f"{name} network must have a 'preprocess_state' method."
            )

    def _update_metrics(
        self, ep_reward_hero: float, ep_reward_gun: float, time_alive: int
    ) -> None:
        """
        Updates rolling and cumulative reward metrics after an episode.

        Also updates the rolling average for time alive.

        Args:
            ep_reward_hero: The total hero reward collected in the episode.
            ep_reward_gun: The total gun reward collected in the episode.
            time_alive: The number of steps the hero survived in the episode.
        """
        self.episode_rewards_hero_deque.append(ep_reward_hero)
        self.episode_rewards_gun_deque.append(ep_reward_gun)
        self.episode_time_alive_deque.append(time_alive)  # Update time alive deque
        self.total_reward_hero += ep_reward_hero
        self.total_reward_gun += ep_reward_gun

    def _log_episode_metrics(self, episode: int, steps: int) -> None:
        """
        Logs key performance metrics for the completed episode.

        Calculates and logs rolling averages for rewards, time alive, and losses,
        cumulative rewards, current epsilon, steps taken, and memory size.

        Args:
            episode: The index of the completed episode.
            steps: The number of steps taken in the completed episode (same as time_alive).
        """
        avg_rew_hero = (
            np.mean(self.episode_rewards_hero_deque)
            if self.episode_rewards_hero_deque
            else 0.0
        )
        avg_rew_gun = (
            np.mean(self.episode_rewards_gun_deque)
            if self.episode_rewards_gun_deque
            else 0.0
        )
        avg_time_alive = (
            np.mean(self.episode_time_alive_deque)
            if self.episode_time_alive_deque
            else 0.0
        )
        avg_loss_hero = (
            np.mean(self.hero_loss_deque) if self.hero_loss_deque else float("nan")
        )
        avg_loss_gun = (
            np.mean(self.gun_loss_deque) if self.gun_loss_deque else float("nan")
        )

        metrics_list = [
            f"TimeAlive={steps}",
            f"AvgTimeAlive={avg_time_alive:.2f}",
            f"Epsilon={self.epsilon:.4f}",
            f"AvgR_Hero={avg_rew_hero:.3f}",
            f"AvgR_Gun={avg_rew_gun:.3f}",
            f"CumR_Hero={self.total_reward_hero:.2f}",
            f"CumR_Gun={self.total_reward_gun:.2f}",
            f"AvgL_Hero={avg_loss_hero:.4f}",
            f"AvgL_Gun={avg_loss_gun:.4f}",
            f"Memory={len(self.memory)}",
        ]
        log_str = f"Ep {episode} Summary | " + " | ".join(metrics_list)
        self.logger.info(log_str)

    def train(self, num_episodes: Optional[int] = None) -> None:
        """
        Runs the main training loop for a specified number of episodes.

        Uses rich.progress to display progress including live metrics.
        Logs metrics periodically. Handles potential runtime errors.
        Displays a summary table at the end of training if num_episodes is set.

        Args:
            num_episodes: The total number of episodes to train for.
                          If None, training continues indefinitely.
        """
        self.logger.info(
            f"Starting training on {self.device} for {num_episodes or 'infinite'} episodes..."
        )
        self.training_summary_data = []  # Reset summary data at the start of training

        progress_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(),
        ]
        total_episodes_for_progress = num_episodes

        if num_episodes is None:
            progress_columns = [
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("Episode {task.completed}"),
            ]
            total_episodes_for_progress = None
            self.logger.warning(
                "Training infinitely (num_episodes=None). Progress bar will not show total or ETA. Summary table disabled."
            )

        with Progress(*progress_columns, transient=False) as progress:
            episode_task: TaskID = progress.add_task(
                "[cyan]Training Episodes...", total=total_episodes_for_progress
            )

            episode_iterator = (
                range(num_episodes) if num_episodes is not None else count()
            )
            completed_episodes = 0

            try:
                for episode in episode_iterator:
                    try:
                        ep_reward_hero, ep_reward_gun, time_alive = self._run_episode(
                            episode, progress, episode_task
                        )
                        self._update_metrics(ep_reward_hero, ep_reward_gun, time_alive)
                        self._log_episode_metrics(episode, time_alive)

                        # Store data for final summary
                        self.training_summary_data.append(
                            {
                                "Episode": episode + 1,
                                "Reward_Hero": ep_reward_hero,
                                "Reward_Gun": ep_reward_gun,
                                "Time_Alive": time_alive,
                            }
                        )

                        self._learn()
                        self._decay_epsilon()
                        self._save_checkpoint_if_needed(episode)

                        progress.update(episode_task, advance=1)
                        completed_episodes += 1

                    except RuntimeError as e:
                        self.logger.critical(
                            f"Stopping training due to runtime error in episode {episode}: {e}",
                            exc_info=True,
                        )
                        progress.stop()
                        break
                    except Exception as e:
                        self.logger.critical(
                            f"Unexpected error during episode {episode}: {e}",
                            exc_info=True,
                        )
                        progress.stop()
                        break

            finally:
                if num_episodes is not None:
                    final_desc = (
                        "[green]Training Finished"
                        if completed_episodes == num_episodes
                        else "[yellow]Training Stopped Early"
                    )
                    progress.update(
                        episode_task,
                        description=final_desc,
                        completed=completed_episodes,
                    )
                    # Display summary only if training ran for a defined number of episodes
                    self._display_training_summary(completed_episodes)
                else:
                    progress.update(
                        episode_task,
                        description="[yellow]Training Stopped (Infinite Mode)",
                    )

        self.logger.info("Training finished.")

    def _run_episode(
        self, episode_num: int, progress: Progress, task_id: TaskID
    ) -> Tuple[float, float, int]:
        """
        Runs a single episode of interaction with the environment.

        Selects actions, steps the environment, stores experience,
        accumulates rewards, and tracks survival time (`time_alive`).
        Updates the progress bar description with live metrics.

        Args:
            episode_num: The current episode index.
            progress: The rich Progress instance for updating display.
            task_id: The TaskID for the main episode progress bar.

        Returns:
            A tuple containing:
            - Total hero reward for the episode.
            - Total gun reward for the episode.
            - Number of steps the hero was alive (time_alive).

        Raises:
            RuntimeError: If environment initialization fails.
        """
        state: State = self._initialize_episode()
        self.logger.debug(f"Episode {episode_num}: Initialized environment.")
        terminated: bool = False
        truncated: bool = False
        episode_reward_hero: float = 0.0
        episode_reward_gun: float = 0.0
        time_alive: int = 0  # Initialize time alive counter

        while not terminated and not truncated:
            time_alive += 1  # Increment time alive counter each step

            # Calculate current rolling averages for display
            avg_r_hero_disp = (
                np.mean(self.episode_rewards_hero_deque)
                if self.episode_rewards_hero_deque
                else 0.0
            )
            avg_r_gun_disp = (
                np.mean(self.episode_rewards_gun_deque)
                if self.episode_rewards_gun_deque
                else 0.0
            )

            # Update progress description with live metrics
            progress.update(
                task_id,
                description=(
                    f"[cyan]Ep. {episode_num}[/cyan] [yellow]Step {time_alive}[/yellow] "
                    f"| Epsilon: [b]{self.epsilon:.4f}[/b] "
                    f"| AvgR Gun: [b]{avg_r_gun_disp:.2f}[/b] "
                    f"| AvgR Hero: [b]{avg_r_hero_disp:.2f}[/b]"
                ),
            )

            move_action, shoot_action = self._select_actions(state)
            step_result: Tuple[Optional[State], float, float, bool, bool] = (
                self._step_environment(move_action, shoot_action)
            )
            next_state, reward_hero, reward_gun, terminated, truncated = step_result

            if next_state is not None:
                experience: Tuple[State, int, int, State, float, float, bool] = (
                    state,
                    move_action,
                    shoot_action,
                    next_state,
                    reward_hero,
                    reward_gun,
                    terminated,
                )
                self.memory.append(experience)
                self.sync_steps_taken += 1
                state = next_state
            elif not terminated and not truncated:
                self.logger.warning(
                    f"Episode {episode_num}: Env step returned None state, terminating episode at step {time_alive}."
                )
                terminated = True

            episode_reward_hero += reward_hero
            episode_reward_gun += reward_gun

        self.logger.debug(f"Episode {episode_num}: Finished in {time_alive} steps.")
        return episode_reward_hero, episode_reward_gun, time_alive

    def _initialize_episode(self) -> State:
        """
        Resets the environment to get the initial state for a new episode.

        Returns:
            The initial state of the environment.

        Raises:
            RuntimeError: If the environment fails to initialize or returns
                          an invalid state type.
        """
        try:
            initial_state: State = self.env.initialise_environment()
            if not isinstance(initial_state, State):
                raise TypeError(
                    f"Environment did not return State object, got {type(initial_state)}"
                )
            return initial_state
        except Exception as e:
            self.logger.error(f"Failed to initialize environment: {e}", exc_info=True)
            raise RuntimeError("Environment initialization failed.") from e

    def _select_actions(self, state: State) -> Tuple[int, int]:
        """
        Selects hero movement and gun shooting actions using epsilon-greedy.

        With probability epsilon, selects random actions. Otherwise, selects
        actions greedily based on the policy networks' predictions.

        Args:
            state: The current environment state.

        Returns:
            A tuple containing the selected move action and shoot action.
        """
        if random.random() < self.epsilon:
            move_action: int = random.randrange(HERO_ACTION_SPACE_SIZE)
            shoot_action: int = random.randrange(GUN_ACTION_SPACE_SIZE)
            self.logger.debug(
                f"Actions (Random): Move={move_action}, Shoot={shoot_action}"
            )
        else:
            move_action_pred: Optional[int] = self._predict_action(
                self.hero_policy_net, state
            )
            shoot_action_pred: Optional[int] = self._predict_action(
                self.gun_policy_net, state
            )

            move_action = (
                move_action_pred
                if move_action_pred is not None
                else random.randrange(HERO_ACTION_SPACE_SIZE)
            )
            shoot_action = (
                shoot_action_pred
                if shoot_action_pred is not None
                else random.randrange(GUN_ACTION_SPACE_SIZE)
            )
            if move_action_pred is None or shoot_action_pred is None:
                self.logger.warning(
                    f"Action prediction failed, using random fallback. Move: {move_action} Shoot: {shoot_action}"
                )
            else:
                self.logger.debug(
                    f"Actions (Predicted): Move={move_action}, Shoot={shoot_action}"
                )

        return move_action, shoot_action

    def _predict_action(self, policy_net: nn.Module, state: State) -> Optional[int]:
        """
        Predicts the best action for a given state using a policy network.

        Performs preprocessing, forward pass, and selects the action with the
        highest Q-value. Handles potential errors during prediction.

        Args:
            policy_net: The policy network (HeroGNN or GunGNN) to use.
            state: The current environment state.

        Returns:
            The index of the predicted best action, or None if prediction fails.
        """
        try:
            graph_data: Optional[Union[HeteroData, Batch]] = (
                policy_net.preprocess_state(state)
            )
            if graph_data is None:
                self.logger.warning(
                    f"Preprocessing failed for {type(policy_net).__name__}."
                )
                return None

            graph_data = graph_data.to(self.device)
            policy_net.eval()
            with torch.no_grad():
                q_values: torch.Tensor = policy_net(graph_data)
            policy_net.train()

            self.logger.debug(f"{type(policy_net).__name__} Q-values: {q_values}")

            if q_values.numel() == 0:
                self.logger.warning(
                    f"{type(policy_net).__name__} produced empty Q-values."
                )
                return None

            if q_values.ndim > 1:
                if q_values.shape[0] == 1:
                    q_values = q_values.squeeze(0)
                else:
                    self.logger.error(
                        f"Unexpected Q-value shape from {type(policy_net).__name__}: {q_values.shape}"
                    )
                    return None

            if q_values.numel() == 0:
                self.logger.warning(
                    f"{type(policy_net).__name__} Q-values became empty after processing."
                )
                return None

            action: int = q_values.argmax().item()
            return action
        except Exception as e:
            self.logger.error(
                f"Error predicting action with {type(policy_net).__name__}: {e}",
                exc_info=True,
            )
            return None

    def _step_environment(
        self, move_action: int, shoot_action: int
    ) -> Tuple[Optional[State], float, float, bool, bool]:
        """
        Sends combined actions to the environment and receives the outcome.

        Formats actions, calls env.step(), validates the result, and extracts
        next state, rewards, terminated, and truncated flags.

        Args:
            move_action: The selected hero movement action.
            shoot_action: The selected gun shooting action.

        Returns:
            A tuple containing:
            - The next state (Optional[State]).
            - Hero reward (float).
            - Gun reward (float).
            - Terminated flag (bool).
            - Truncated flag (bool).
        """
        try:
            combined_action_list: List[int] = [move_action, shoot_action, 0, 0]
            step_result: Tuple = self.env.step(combined_action_list)

            if not isinstance(step_result, tuple) or len(step_result) < 3:
                raise TypeError(
                    f"Environment step returned unexpected result format: {type(step_result)}"
                )

            next_s, reward_tuple, terminated_flag = step_result[:3]
            truncated_flag = (
                step_result[3]
                if len(step_result) > 3 and isinstance(step_result[3], bool)
                else False
            )

            next_state: Optional[State] = None
            if isinstance(next_s, State):
                next_state = next_s
            elif (
                next_s is not None
                and not bool(terminated_flag)
                and not bool(truncated_flag)
            ):
                self.logger.error(
                    f"Environment step returned invalid non-terminal state type: {type(next_s)}"
                )
                return None, 0.0, 0.0, True, True  # Indicate critical failure

            if not isinstance(reward_tuple, (tuple, list)) or len(reward_tuple) < 2:
                self.logger.error(
                    f"Environment step returned invalid reward format: {reward_tuple}"
                )
                return next_state, 0.0, 0.0, True, True  # Signal error

            reward_hero: float = float(
                0 if reward_tuple[0] == None else reward_tuple[0]
            )
            reward_gun: float = float(0 if reward_tuple[1] == None else reward_tuple[1])
            terminated: bool = bool(terminated_flag)
            truncated: bool = bool(truncated_flag)

            return next_state, reward_hero, reward_gun, terminated, truncated

        except Exception as e:
            self.logger.error(f"Error during environment step: {e}", exc_info=True)
            return None, 0.0, 0.0, True, True

    def _learn(self) -> None:
        """
        Performs a learning step if enough experience is available.

        Samples a mini-batch from memory, performs optimization steps for
        both networks, and checks if target networks need synchronization.
        """
        if len(self.memory) < self.mini_batch_size:
            return

        self.logger.debug(
            f"Performing learning step with batch size {self.mini_batch_size}"
        )
        mini_batch: List[Tuple] = self.memory.sample(self.mini_batch_size)
        self._optimize_step(mini_batch)
        self._sync_target_networks_if_needed()

    def _optimize_step(self, mini_batch: List[Tuple]) -> None:
        """
        Performs optimization using a mini-batch, handling non-terminal states correctly.
        """
        try:
            states, move_actions, shoot_actions, next_states, hero_rewards, gun_rewards, terminations = zip(*mini_batch)
        except ValueError as e:
            self.logger.error(f"Error unpacking minibatch: {e}")
            return

        # --- Preprocess current states ---
        try:
            # Preprocess ALL current states
            h_graph_s, g_graph_s = self._preprocess_batch(
                states, self.hero_policy_net, self.gun_policy_net, "policy", terminations=None
            )
            if h_graph_s is None or g_graph_s is None:
                self.logger.warning("Skipping opt step: Current state preprocessing failed.")
                return
        except Exception as e: # Catch potential errors from Batching
            self.logger.error(f"Error preprocessing current states: {e}", exc_info=True)
            return

        # --- Preprocess ONLY non-terminal next states ---
        try:
            # Pass terminations to filter next_states
            h_graph_ns_nonterm, g_graph_ns_nonterm = self._preprocess_batch(
                next_states, self.hero_target_net, self.gun_target_net, "target", terminations=terminations
            )
            # These batches (if not None) now ONLY contain graphs for non-terminal next states
        except Exception as e:
             self.logger.error(f"Error preprocessing next states: {e}", exc_info=True)
             return


        # --- Convert base components to tensors ---
        try:
            move_actions_t = torch.tensor(move_actions, dtype=torch.long, device=self.device)
            shoot_actions_t = torch.tensor(shoot_actions, dtype=torch.long, device=self.device)
            hero_rewards_t = torch.tensor(hero_rewards, dtype=torch.float, device=self.device)
            gun_rewards_t = torch.tensor(gun_rewards, dtype=torch.float, device=self.device)
            # Non-terminal mask is crucial for indexing
            non_terminal_mask = torch.tensor([not t for t in terminations], dtype=torch.bool, device=self.device)
        except (TypeError, ValueError) as e:
            self.logger.error(f"Error converting batch data to tensors: {e}")
            return

        # --- Optimize Hero Network ---
        self._calculate_and_apply_loss(
            policy_net=self.hero_policy_net,
            target_net=self.hero_target_net,
            optimizer=self.hero_optimizer,
            batch_states=h_graph_s, # All current states
            actions_t=move_actions_t,
            # Pass ONLY the batch of non-terminal next states
            batch_next_states_nonterm=h_graph_ns_nonterm,
            rewards_t=hero_rewards_t,
            non_terminal_mask=non_terminal_mask, # Mask for original batch size
            loss_deque=self.hero_loss_deque,
        )

        # --- Optimize Gun Network ---
        self._calculate_and_apply_loss(
            policy_net=self.gun_policy_net,
            target_net=self.gun_target_net,
            optimizer=self.gun_optimizer,
            batch_states=g_graph_s, # All current states
            actions_t=shoot_actions_t,
             # Pass ONLY the batch of non-terminal next states
            batch_next_states_nonterm=g_graph_ns_nonterm,
            rewards_t=gun_rewards_t,
            non_terminal_mask=non_terminal_mask, # Mask for original batch size
            loss_deque=self.gun_loss_deque,
        )


    def _preprocess_batch(
        self,
        states: Tuple[Optional[State], ...],
        net1: nn.Module,
        net2: nn.Module,
        net_type: str,
        terminations: Optional[Tuple[bool, ...]] = None # Add terminations flag
    ) -> Tuple[Optional[Batch], Optional[Batch]]:
        """
        Preprocesses states, optionally filtering for non-terminal states.
        """
        graphs1_list = []
        graphs2_list = []
        # We don't necessarily need valid_indices if we filter lists directly

        for i, state in enumerate(states):
            # --- Filtering logic for next_states ---
            is_terminal = terminations[i] if terminations is not None else False
            # If processing next_states (terminations provided) AND state is terminal, SKIP
            if terminations is not None and is_terminal:
                continue
            # If state is None (can happen in next_states if env returns None after terminal), SKIP
            if state is None:
                continue
            # --- End Filtering ---

            try:
                graph1 = net1.preprocess_state(state)
                graph2 = net2.preprocess_state(state)
                # Append only if BOTH preprocessing steps succeed for this state
                if graph1 is not None and graph2 is not None:
                    graphs1_list.append(graph1)
                    graphs2_list.append(graph2)
                # else: Optionally log preprocess failure for individual state
            except Exception as e:
                self.logger.warning(
                    f"Error preprocessing state index {i} ({net_type}): {e}", exc_info=False
                )

        # If the filtered list is empty, return None for the batches
        if not graphs1_list:
            self.logger.debug(
                f"Preprocessing yielded no valid graphs for the {net_type} batch "
                f"{'(filtered)' if terminations is not None else ''}."
            )
            return None, None

        # Create batches from the (potentially filtered) lists
        try:
            batch1 = Batch.from_data_list(graphs1_list).to(self.device)
            batch2 = Batch.from_data_list(graphs2_list).to(self.device)
            return batch1, batch2
        except Exception as e:
            self.logger.error(
                f"Error creating Batch for {net_type} nets: {e}", exc_info=True
            )
            return None, None


    def _calculate_and_apply_loss(
        self,
        policy_net: nn.Module,
        target_net: nn.Module,
        optimizer: optim.Optimizer,
        batch_states: Batch, # Batch of ALL current states
        actions_t: torch.Tensor, # Actions for ALL original states
        # Batch containing ONLY non-terminal next states
        batch_next_states_nonterm: Optional[Batch],
        rewards_t: torch.Tensor, # Rewards for ALL original states
        # Mask indicating non-terminal states in the original batch
        non_terminal_mask: torch.Tensor,
        loss_deque: Deque[float],
    ) -> None:
        """Calculates DQN loss and performs optimization, handling non-terminal states."""
        try:
            # --- Current Q Calculation (uses all states) ---
            policy_net.train()
            current_q_all = policy_net(batch_states)
            current_q = current_q_all.gather(
                dim=1, index=actions_t.unsqueeze(dim=1)
            ).squeeze(dim=1)
        except Exception as e:
            self.logger.error(f"Error getting current Q-values: {e}", exc_info=True)
            return

        # --- Target Q Calculation ---
        # Initialize next_q_values for the whole original batch size
        next_q_values = torch.zeros_like(rewards_t, device=self.device)

        # Calculate target Q ONLY for non-terminal states
        # Check if there are any non-terminal states AND if the corresponding batch exists
        if non_terminal_mask.any() and batch_next_states_nonterm is not None:
            target_net.eval()
            with torch.no_grad():
                try:
                    # Target net processes ONLY the non-terminal graphs
                    target_next_q_all = target_net(batch_next_states_nonterm)
                    # Get max Q value for these non-terminal states
                    max_target_next_q = target_next_q_all.max(dim=1)[0]

                    # Use the mask to place these values into the correct indices
                    # The size of max_target_next_q should match the number of True values in non_terminal_mask
                    if len(max_target_next_q) == non_terminal_mask.sum():
                        next_q_values[non_terminal_mask] = max_target_next_q
                    else:
                        # This case should be less likely now with filtering in _preprocess_batch
                        self.logger.error(
                            f"CRITICAL ALIGNMENT ERROR for {type(target_net).__name__}: "
                            f"Target net output size {len(max_target_next_q)} does not match "
                            f"non-terminal mask count {non_terminal_mask.sum()}. Check filtering logic."
                        )
                        # Cannot safely calculate target Q, skipping loss calculation might be best
                        return

                except Exception as e:
                    self.logger.error(f"Error getting target Q-values: {e}", exc_info=True)
                    # Proceed with zeros for next_q_values if target calc fails

        # Calculate final target (zeros for terminal states implicitly handled)
        target_q = rewards_t + (self.discount_factor * next_q_values)
        target_q = target_q.detach() # Detach target Q values

        # --- Loss Calculation and Optimization ---
        try:
            loss = self.loss_fn(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(policy_net.parameters(), clip_value=1.0)
            optimizer.step()

            loss_value = loss.item()
            loss_deque.append(loss_value)
            self.logger.debug(f"Loss ({type(policy_net).__name__}): {loss_value:.4f}")
        except Exception as e:
            self.logger.error(f"Error during loss/optimization: {e}", exc_info=True)

    def _sync_target_networks_if_needed(self) -> None:
        """
        Copies weights from policy networks to target networks periodically.

        Sync occurs when the number of steps taken (`sync_steps_taken`)
        reaches the `target_sync_rate`. Resets the counter after syncing.
        """
        if self.sync_steps_taken >= self.target_sync_rate:
            self.logger.info(
                f"Syncing target networks (triggered after {self.sync_steps_taken} steps)"
            )
            try:
                self.hero_target_net.load_state_dict(self.hero_policy_net.state_dict())
                self.gun_target_net.load_state_dict(self.gun_policy_net.state_dict())
                self.hero_target_net.eval()
                self.gun_target_net.eval()
                self.sync_steps_taken = 0
                self.logger.info("Target networks synced successfully.")
            except Exception as e:
                self.logger.error(f"Failed to sync target networks: {e}", exc_info=True)
                self.sync_steps_taken = 0

    def _decay_epsilon(self) -> None:
        """Decays the exploration rate (epsilon) according to the decay factor."""
        old_epsilon = self.epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        if old_epsilon != self.epsilon:
            self.logger.debug(
                f"Epsilon decayed from {old_epsilon:.4f} to {self.epsilon:.4f}"
            )

    def _save_checkpoint_if_needed(self, episode: int) -> None:
        """
        Saves a checkpoint of the agent's state if the save interval is reached.

        Args:
            episode: The current episode index.
        """
        if (
            self.save_interval > 0
            and episode > 0
            and (episode + 1) % self.save_interval == 0
        ):
            self.logger.info(
                f"Reached save interval at episode {episode}. Saving checkpoint..."
            )
            save_path = self.dump()
            if save_path:
                self.logger.info(f"Checkpoint saved successfully to: {save_path}")
            else:
                self.logger.error(f"Failed to save checkpoint for episode {episode}.")

    def _display_training_summary(self, total_episodes_completed: int) -> None:
        """
        Displays a summary table of training performance over episode blocks.

        Calculates average rewards and time alive for each 10% chunk of
        the completed episodes and prints them using rich.table.

        Args:
            total_episodes_completed: The total number of episodes that were run.
        """
        if not self.training_summary_data or total_episodes_completed == 0:
            self.logger.info(
                "No training data recorded or no episodes completed, skipping summary."
            )
            return

        self.logger.info("Generating Training Summary Table...")
        df = pd.DataFrame(self.training_summary_data)

        # Define blocks (e.g., 10% of total completed episodes)
        block_size = max(
            1, total_episodes_completed // 10
        )  # Ensure block_size is at least 1
        num_blocks = (
            total_episodes_completed + block_size - 1
        ) // block_size  # Ceiling division

        summary_rows = []
        for i in range(num_blocks):
            start_episode = i * block_size + 1
            end_episode = min((i + 1) * block_size, total_episodes_completed)
            block_data = df[
                (df["Episode"] >= start_episode) & (df["Episode"] <= end_episode)
            ]

            if block_data.empty:
                continue

            avg_reward_hero = block_data["Reward_Hero"].mean()
            avg_reward_gun = block_data["Reward_Gun"].mean()
            avg_time_alive = block_data["Time_Alive"].mean()
            episode_range = f"{start_episode}-{end_episode}"
            summary_rows.append(
                (
                    episode_range,
                    f"{avg_reward_hero:.3f}",
                    f"{avg_reward_gun:.3f}",
                    f"{avg_time_alive:.2f}",
                )
            )

        # Create and print the table using rich
        table = Table(
            title=f"Training Summary (Completed {total_episodes_completed} Episodes)"
        )
        table.add_column("Episode Block", justify="center", style="cyan", no_wrap=True)
        table.add_column("Avg Hero Reward", justify="right", style="magenta")
        table.add_column("Avg Gun Reward", justify="right", style="green")
        table.add_column("Avg Time Alive", justify="right", style="yellow")

        for row in summary_rows:
            table.add_row(*row)

        self.console.print(table)

    def dump(self, save_dir: str = "model_saves") -> Optional[str]:
        """
        Saves the complete agent state to a timestamped directory.

        Includes network state dicts, optimizer states, and hyperparameters
        in a YAML configuration file.

        Args:
            save_dir: The base directory to save the checkpoint subdirectory in.

        Returns:
            The path to the saved checkpoint directory, or None on failure.
        """
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name: str = f"theseus_gnn_{timestamp}"
        dpath: str = os.path.join(save_dir, base_name)

        try:
            os.makedirs(dpath, exist_ok=True)
            self.logger.info(f"Saving agent state to directory: {dpath}")

            filenames: Dict[str, str] = {
                "hero_policy": "hero_policy_state.pth",
                "hero_target": "hero_target_state.pth",
                "gun_policy": "gun_policy_state.pth",
                "gun_target": "gun_target_state.pth",
                "hero_optim": "hero_optimizer.pth",
                "gun_optim": "gun_optimizer.pth",
                "config": f"{base_name}_config.yaml",
            }

            torch.save(
                self.hero_policy_net.state_dict(),
                os.path.join(dpath, filenames["hero_policy"]),
            )
            torch.save(
                self.hero_target_net.state_dict(),
                os.path.join(dpath, filenames["hero_target"]),
            )
            torch.save(
                self.gun_policy_net.state_dict(),
                os.path.join(dpath, filenames["gun_policy"]),
            )
            torch.save(
                self.gun_target_net.state_dict(),
                os.path.join(dpath, filenames["gun_target"]),
            )

            torch.save(
                self.hero_optimizer.state_dict(),
                os.path.join(dpath, filenames["hero_optim"]),
            )
            torch.save(
                self.gun_optimizer.state_dict(),
                os.path.join(dpath, filenames["gun_optim"]),
            )

            state_info: Dict[str, Any] = {
                "hero_policy_file": filenames["hero_policy"],
                "hero_target_file": filenames["hero_target"],
                "gun_policy_file": filenames["gun_policy"],
                "gun_target_file": filenames["gun_target"],
                "hero_optim_file": filenames["hero_optim"],
                "gun_optim_file": filenames["gun_optim"],
                "hero_policy_class": f"{type(self.hero_policy_net).__module__}.{type(self.hero_policy_net).__name__}",
                "gun_policy_class": f"{type(self.gun_policy_net).__module__}.{type(self.gun_policy_net).__name__}",
                "epsilon": self.epsilon,
                "sync_steps_taken": self.sync_steps_taken,
                "learning_rate": self.hero_optimizer.param_groups[0]["lr"],
                "discount_factor": self.discount_factor,
                "mini_batch_size": self.mini_batch_size,
                "target_sync_rate": self.target_sync_rate,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_min": self.epsilon_min,
                "log_window_size": self.log_window_size,
                "save_interval": self.save_interval,
                "replay_memory_size": len(self.memory),
                "total_reward_hero": self.total_reward_hero,
                "total_reward_gun": self.total_reward_gun,
                "loss_fn_class": f"{type(self.loss_fn).__module__}.{type(self.loss_fn).__name__}",
                "optimizer_class": f"{type(self.hero_optimizer).__module__}.{type(self.hero_optimizer).__name__}",
            }
            yaml_path: str = os.path.join(dpath, filenames["config"])
            with open(yaml_path, "w") as f:
                yaml.dump(state_info, f, default_flow_style=False, sort_keys=False)

            self.logger.info("Agent state saved successfully.")
            return dpath

        except Exception as e:
            self.logger.error(
                f"Failed to dump agent state to {dpath}: {e}", exc_info=True
            )
            return None

    @classmethod
    def load(cls, load_path: Union[str, os.PathLike]) -> Self | None:
        """
        Loads agent state from a specified checkpoint directory.

        Reconstructs networks, optimizers, and internal state based on the
        saved configuration file and state dicts.

        Args:
            load_path: The path to the checkpoint directory to load from.

        Returns:
            An instance of AgentTheseusGNN with loaded state, or None on failure.
        """
        logger = logging.getLogger("agent-theseus-gnn-load")
        logger.info(f"Attempting to load agent state from directory: {load_path}")

        load_path_str: str = str(load_path)
        if not os.path.isdir(load_path_str):
            logger.error(f"Load path is not a valid directory: {load_path_str}")
            return None

        yaml_files = [
            f for f in os.listdir(load_path_str) if f.endswith("_config.yaml")
        ]
        if not yaml_files:
            logger.error(f"No '_config.yaml' file found in: {load_path_str}")
            return None
        if len(yaml_files) > 1:
            logger.warning(
                f"Multiple config files found, using the first one: {yaml_files[0]}"
            )
        yaml_path: str = os.path.join(load_path_str, yaml_files[0])

        try:
            with open(yaml_path, "r") as f:
                state_info: Dict[str, Any] = yaml.safe_load(f)
        except Exception as e:
            logger.error(
                f"Error reading YAML configuration {yaml_path}: {e}", exc_info=True
            )
            return None

        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading models onto device: {device}")

        try:

            def get_class(class_path: str) -> Type:
                module_path, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                return getattr(module, class_name)

            HeroPolicyClass: Type[HeroGNN] = get_class(state_info["hero_policy_class"])
            GunPolicyClass: Type[GunGNN] = get_class(state_info["gun_policy_class"])

            # TODO: Add network argument loading from state_info if constructors require them
            hero_policy_net = HeroPolicyClass()
            hero_target_net = HeroPolicyClass()
            gun_policy_net = GunPolicyClass()
            gun_target_net = GunPolicyClass()

            hp_path = os.path.join(load_path_str, state_info["hero_policy_file"])
            ht_path = os.path.join(load_path_str, state_info["hero_target_file"])
            gp_path = os.path.join(load_path_str, state_info["gun_policy_file"])
            gt_path = os.path.join(load_path_str, state_info["gun_target_file"])

            hero_policy_net.load_state_dict(torch.load(hp_path, map_location=device))
            hero_target_net.load_state_dict(torch.load(ht_path, map_location=device))
            gun_policy_net.load_state_dict(torch.load(gp_path, map_location=device))
            gun_target_net.load_state_dict(torch.load(gt_path, map_location=device))
            logger.info("Network state dicts loaded successfully.")

        except (
            ImportError,
            AttributeError,
            KeyError,
            FileNotFoundError,
            Exception,
        ) as e:
            logger.error(
                f"Error reconstructing or loading network models: {e}", exc_info=True
            )
            return None

        try:
            learning_rate: float = state_info.get("learning_rate", 1e-4)
            OptimizerClass: Type[optim.Optimizer] = get_class(
                state_info.get("optimizer_class", "torch.optim.AdamW")
            )
            LossFnClass: Type[nn.Module] = get_class(
                state_info.get("loss_fn_class", "torch.nn.MSELoss")
            )

            # Networks must be moved to device *before* optimizer instantiation
            hero_policy_net.to(device)
            gun_policy_net.to(device)
            hero_target_net.to(device)  # Target nets also need to be on device
            gun_target_net.to(device)

            hero_optimizer = OptimizerClass(
                hero_policy_net.parameters(), lr=learning_rate
            )
            gun_optimizer = OptimizerClass(
                gun_policy_net.parameters(), lr=learning_rate
            )

            ho_path = os.path.join(load_path_str, state_info["hero_optim_file"])
            go_path = os.path.join(load_path_str, state_info["gun_optim_file"])
            if os.path.exists(ho_path):
                hero_optimizer.load_state_dict(torch.load(ho_path, map_location=device))
                logger.info("Hero optimizer state loaded.")
            else:
                logger.warning(
                    f"Hero optimizer state file not found: {ho_path}. Initializing fresh."
                )
            if os.path.exists(go_path):
                gun_optimizer.load_state_dict(torch.load(go_path, map_location=device))
                logger.info("Gun optimizer state loaded.")
            else:
                logger.warning(
                    f"Gun optimizer state file not found: {go_path}. Initializing fresh."
                )

            logger.info("Optimizers and Loss function reconstructed.")

        except (
            ImportError,
            AttributeError,
            KeyError,
            FileNotFoundError,
            Exception,
        ) as e:
            logger.error(
                f"Error reconstructing or loading optimizers/loss: {e}", exc_info=True
            )
            return None

        try:
            env = Environment()  # Assuming default constructor
        except Exception as e:
            logger.error(f"Failed to instantiate Environment: {e}", exc_info=True)
            return None

        try:
            agent = cls(
                hero_policy_net=hero_policy_net,
                hero_target_net=hero_target_net,
                gun_policy_net=gun_policy_net,
                gun_target_net=gun_target_net,
                env=env,
                loss_fn_class=LossFnClass,
                optimizer_class=OptimizerClass,
                learning_rate=learning_rate,  # Loaded LR used for optimizers
                discount_factor=state_info.get("discount_factor", 0.99),
                epsilon_init=state_info.get("epsilon", 0.05),  # Use saved epsilon
                epsilon_decay=state_info.get("epsilon_decay", 0.9995),
                epsilon_min=state_info.get("epsilon_min", 0.05),
                mini_batch_size=state_info.get("mini_batch_size", 64),
                target_sync_rate=state_info.get("target_sync_rate", 500),
                replay_memory_size=state_info.get(
                    "replay_memory_size", c.REPLAY_MEMORY_SIZE
                ),
                log_window_size=state_info.get("log_window_size", LOGGING_WINDOW),
                save_interval=state_info.get("save_interval", SAVE_INTERVAL),
            )

            # Restore specific state variables
            agent.hero_optimizer = hero_optimizer
            agent.gun_optimizer = gun_optimizer
            agent.epsilon = state_info.get(
                "epsilon", agent.epsilon_min
            )  # Ensure current epsilon is loaded
            agent.sync_steps_taken = state_info.get("sync_steps_taken", 0)
            agent.total_reward_hero = state_info.get("total_reward_hero", 0.0)
            agent.total_reward_gun = state_info.get("total_reward_gun", 0.0)

            # Ensure target networks are in eval mode after loading
            agent.hero_target_net.eval()
            agent.gun_target_net.eval()

            logger.info(f"Agent loaded successfully from {load_path_str}")
            return agent

        except Exception as e:
            logger.error(
                f"Error instantiating AgentTheseusGNN during final load step: {e}",
                exc_info=True,
            )
            return None
