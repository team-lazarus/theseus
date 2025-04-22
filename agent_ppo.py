import torch
import random
import logging
import os
import yaml
import numpy as np
import importlib
import pandas as pd
from collections import deque, namedtuple
from datetime import datetime
from itertools import count
from typing import List, Tuple, Optional, Type, Self, Deque, Dict, Any, Union

# Assuming necessary imports exist and are correctly handled
try:
    from torch import nn, optim
    from torch.distributions import Categorical
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
    from theseus.utils import State  # Assuming State is defined here

    # NOTE: ExperienceReplayMemory is not used in PPO's standard form
    # from theseus.utils import ExperienceReplayMemory
    from theseus.utils.network import Environment
    import theseus.constants as c

    # --- IMPORTANT: Assuming these GNNs can act as Actors/Critics ---
    # You might need distinct classes or modifications
    from theseus.models.GraphDQN.ActionGNN import HeroGNN as HeroActorGNN
    from theseus.models.GraphDQN.ActionGNN import (
        HeroGNN as HeroCriticGNN,
    )  # Placeholder
    from theseus.models.GraphDQN.ActionGNN import GunGNN as GunActorGNN
    from theseus.models.GraphDQN.ActionGNN import GunGNN as GunCriticGNN  # Placeholder

    # --------------------------------------------------------------

except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.error(
        f"Failed to import necessary libraries: {e}. Please ensure all dependencies are installed."
    )
    raise ImportError(f"Critical import failed: {e}") from e

# Define constants (can be moved to a config file)
HERO_ACTION_SPACE_SIZE: int = 9
GUN_ACTION_SPACE_SIZE: int = 8
LOGGING_WINDOW: int = 50
SAVE_INTERVAL: int = 5  # Episodes
DEFAULT_HORIZON: int = 2048  # Steps per rollout collection
DEFAULT_EPOCHS_PER_UPDATE: int = 10
DEFAULT_MINI_BATCH_SIZE_PPO: int = 64
DEFAULT_CLIP_EPSILON: float = 0.2
DEFAULT_GAE_LAMBDA: float = 0.95
DEFAULT_ENTROPY_COEFF: float = 0.01
DEFAULT_VF_COEFF: float = 0.5
DEFAULT_LEARNING_RATE_PPO: float = 3e-4
DEFAULT_DISCOUNT_FACTOR_PPO: float = 0.99

# Structure to hold trajectory data
TrajectoryStep = namedtuple(
    "TrajectoryStep",
    [
        "state_graph_hero",
        "state_graph_gun",  # Preprocessed graphs
        "move_action",
        "shoot_action",
        "move_log_prob",
        "shoot_log_prob",
        "hero_value",
        "gun_value",
        "hero_reward",
        "gun_reward",
        "terminated",
    ],
)


class AgentTheseusPPO:
    """
    Agent managing simultaneous training of Hero and Gun GNNs using PPO.

    This agent uses separate Actor and Critic GNNs for hero movement and gun
    actions. It collects trajectories on-policy, calculates advantages using GAE,
    and updates networks using the PPO clipped surrogate objective and value loss.

    Args:
        hero_actor_net: The actor network for hero actions.
        hero_critic_net: The critic network for hero state value.
        gun_actor_net: The actor network for gun actions.
        gun_critic_net: The critic network for gun state value.
        env: The environment instance.
        optimizer_class: The class type for the optimizer (default: optim.AdamW).
        learning_rate: Learning rate for the optimizers.
        discount_factor: Discount factor (gamma) for future rewards.
        horizon: Number of steps to collect in each rollout.
        epochs_per_update: Number of optimization epochs per rollout.
        mini_batch_size: Size of mini-batches during optimization epochs.
        clip_epsilon: PPO clipping parameter.
        gae_lambda: Lambda parameter for Generalized Advantage Estimation.
        entropy_coeff: Coefficient for the entropy bonus in the actor loss.
        vf_coeff: Coefficient for the value function loss in the total loss.
        log_window_size: Number of episodes for calculating rolling averages.
        save_interval: Frequency (in episodes) for saving model checkpoints.

    Attributes:
        (Many similar attributes to DQN version, but adapted for PPO)
        trajectory_buffer: Stores the steps collected during a rollout.
        total_steps: Counter for steps within the current rollout horizon.
        ... (other PPO specific attributes)
    """

    def __init__(
        self,
        hero_actor_net: HeroActorGNN,
        hero_critic_net: HeroCriticGNN,
        gun_actor_net: GunActorGNN,
        gun_critic_net: GunCriticGNN,
        env: Environment,
        *,
        optimizer_class: Type[optim.Optimizer] = optim.AdamW,
        learning_rate: float = DEFAULT_LEARNING_RATE_PPO,
        discount_factor: float = DEFAULT_DISCOUNT_FACTOR_PPO,
        horizon: int = DEFAULT_HORIZON,
        epochs_per_update: int = DEFAULT_EPOCHS_PER_UPDATE,
        mini_batch_size: int = DEFAULT_MINI_BATCH_SIZE_PPO,
        clip_epsilon: float = DEFAULT_CLIP_EPSILON,
        gae_lambda: float = DEFAULT_GAE_LAMBDA,
        entropy_coeff: float = DEFAULT_ENTROPY_COEFF,
        vf_coeff: float = DEFAULT_VF_COEFF,
        log_window_size: int = LOGGING_WINDOW,
        save_interval: int = SAVE_INTERVAL,
    ) -> None:
        """Initializes the PPO dual-GNN agent."""
        self.logger: logging.Logger = logging.getLogger("agent-theseus-ppo")
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.env: Environment = env

        # --- Network Setup ---
        self._validate_network(hero_actor_net, "Hero Actor")
        self._validate_network(hero_critic_net, "Hero Critic")
        self._validate_network(gun_actor_net, "Gun Actor")
        self._validate_network(gun_critic_net, "Gun Critic")
        self.hero_actor_net: HeroActorGNN = hero_actor_net.to(self.device)
        self.hero_critic_net: HeroCriticGNN = hero_critic_net.to(self.device)
        self.gun_actor_net: GunActorGNN = gun_actor_net.to(self.device)
        self.gun_critic_net: GunCriticGNN = gun_critic_net.to(self.device)

        # --- PPO Hyperparameters ---
        self.discount_factor: float = discount_factor
        self.horizon: int = horizon
        self.epochs_per_update: int = epochs_per_update
        self.mini_batch_size: int = mini_batch_size
        self.clip_epsilon: float = clip_epsilon
        self.gae_lambda: float = gae_lambda
        self.entropy_coeff: float = entropy_coeff
        self.vf_coeff: float = vf_coeff

        # --- Optimizers ---
        # Combine params if sharing backbones, otherwise separate optimizers
        # Assuming separate networks for now
        self.hero_actor_optimizer: optim.Optimizer = optimizer_class(
            self.hero_actor_net.parameters(),
            lr=learning_rate,
            eps=1e-5,  # Adam typically uses eps
        )
        self.hero_critic_optimizer: optim.Optimizer = optimizer_class(
            self.hero_critic_net.parameters(), lr=learning_rate, eps=1e-5
        )
        self.gun_actor_optimizer: optim.Optimizer = optimizer_class(
            self.gun_actor_net.parameters(), lr=learning_rate, eps=1e-5
        )
        self.gun_critic_optimizer: optim.Optimizer = optimizer_class(
            self.gun_critic_net.parameters(), lr=learning_rate, eps=1e-5
        )
        # Critic loss function
        self.critic_loss_fn: nn.Module = nn.MSELoss()

        # --- Data Collection ---
        self.trajectory_buffer: List[TrajectoryStep] = []
        self.total_steps: int = 0  # Steps collected in the current rollout

        # --- Metrics Tracking ---
        self.log_window_size: int = log_window_size
        self.save_interval: int = save_interval
        self.episode_rewards_hero_deque: Deque[float] = deque(
            maxlen=self.log_window_size
        )
        self.episode_rewards_gun_deque: Deque[float] = deque(
            maxlen=self.log_window_size
        )
        self.episode_time_alive_deque: Deque[int] = deque(maxlen=self.log_window_size)
        self.total_reward_hero: float = 0.0
        self.total_reward_gun: float = 0.0
        # PPO specific metrics
        self.hero_actor_loss_deque: Deque[float] = deque(
            maxlen=self.log_window_size * epochs_per_update
        )
        self.hero_critic_loss_deque: Deque[float] = deque(
            maxlen=self.log_window_size * epochs_per_update
        )
        self.gun_actor_loss_deque: Deque[float] = deque(
            maxlen=self.log_window_size * epochs_per_update
        )
        self.gun_critic_loss_deque: Deque[float] = deque(
            maxlen=self.log_window_size * epochs_per_update
        )
        self.entropy_hero_deque: Deque[float] = deque(
            maxlen=self.log_window_size * epochs_per_update
        )
        self.entropy_gun_deque: Deque[float] = deque(
            maxlen=self.log_window_size * epochs_per_update
        )

        self.training_summary_data: List[Dict[str, Union[int, float]]] = (
            []
        )  # For summary table
        self.console = Console()

        self.current_state: Optional[State] = (
            None  # Track current state across steps/episodes
        )

    def _validate_network(self, network: nn.Module, name: str) -> None:
        """Checks if a network has the required 'preprocess_state' method."""
        if not hasattr(network, "preprocess_state") or not callable(
            getattr(network, "preprocess_state", None)  # Safer check
        ):
            raise AttributeError(
                f"{name} network must have a callable 'preprocess_state' method."
            )
        # Optional: Add checks for output shapes if possible/needed

    def _update_metrics(
        self, ep_reward_hero: float, ep_reward_gun: float, time_alive: int
    ) -> None:
        """Updates rolling and cumulative reward metrics after an episode."""
        self.episode_rewards_hero_deque.append(ep_reward_hero)
        self.episode_rewards_gun_deque.append(ep_reward_gun)
        self.episode_time_alive_deque.append(time_alive)
        self.total_reward_hero += ep_reward_hero
        self.total_reward_gun += ep_reward_gun

    def _log_episode_metrics(self, episode: int, steps: int) -> None:
        """Logs key performance metrics for the completed episode."""
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

        avg_loss_actor_h = (
            np.mean(self.hero_actor_loss_deque)
            if self.hero_actor_loss_deque
            else float("nan")
        )
        avg_loss_critic_h = (
            np.mean(self.hero_critic_loss_deque)
            if self.hero_critic_loss_deque
            else float("nan")
        )
        avg_loss_actor_g = (
            np.mean(self.gun_actor_loss_deque)
            if self.gun_actor_loss_deque
            else float("nan")
        )
        avg_loss_critic_g = (
            np.mean(self.gun_critic_loss_deque)
            if self.gun_critic_loss_deque
            else float("nan")
        )
        avg_entropy_h = (
            np.mean(self.entropy_hero_deque)
            if self.entropy_hero_deque
            else float("nan")
        )
        avg_entropy_g = (
            np.mean(self.entropy_gun_deque) if self.entropy_gun_deque else float("nan")
        )

        metrics_list = [
            f"TimeAlive={steps}",
            f"AvgTimeAlive={avg_time_alive:.2f}",
            f"AvgR_Hero={avg_rew_hero:.3f}",
            f"AvgR_Gun={avg_rew_gun:.3f}",
            f"CumR_Hero={self.total_reward_hero:.2f}",
            f"CumR_Gun={self.total_reward_gun:.2f}",
            f"ALoss_H={avg_loss_actor_h:.4f}",
            f"CLoss_H={avg_loss_critic_h:.4f}",
            f"ALoss_G={avg_loss_actor_g:.4f}",
            f"CLoss_G={avg_loss_critic_g:.4f}",
            f"Entropy_H={avg_entropy_h:.3f}",
            f"Entropy_G={avg_entropy_g:.3f}",
            f"RolloutProg={self.total_steps}/{self.horizon}",
        ]
        log_str = f"Ep {episode} Summary | " + " | ".join(metrics_list)
        self.logger.info(log_str)

    def train(self, num_episodes: Optional[int] = None) -> None:
        """Runs the main PPO training loop."""
        self.logger.info(
            f"Starting PPO training on {self.device} for {num_episodes or 'infinite'} episodes..."
        )
        self.logger.info(f"Collect Horizon: {self.horizon} steps")
        self.training_summary_data = []  # Reset summary data

        # Reset environment state at the beginning of training
        self.current_state = self._initialize_episode()
        if self.current_state is None:
            self.logger.critical("Initial environment reset failed. Stopping training.")
            return

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
                "Training infinitely. Progress bar will not show total or ETA."
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
                    ep_reward_hero, ep_reward_gun, time_alive, terminated, truncated = (
                        self._run_episode_or_rollout(episode, progress, episode_task)
                    )

                    # --- Logging and Metrics ---
                    if (
                        terminated or truncated
                    ):  # Only log full episode stats when an episode actually ends
                        self._update_metrics(ep_reward_hero, ep_reward_gun, time_alive)
                        self._log_episode_metrics(episode, time_alive)
                        # Store data for final summary table
                        self.training_summary_data.append(
                            {
                                "Episode": episode + 1,
                                "Reward_Hero": ep_reward_hero,
                                "Reward_Gun": ep_reward_gun,
                                "Time_Alive": time_alive,
                            }
                        )
                        self._save_checkpoint_if_needed(episode)
                        progress.update(episode_task, advance=1)
                        completed_episodes += 1

                    # --- Learning Step Trigger ---
                    if self.total_steps >= self.horizon:
                        self.logger.info(
                            f"Horizon {self.horizon} reached. Starting PPO update."
                        )
                        self._update_policy()  # Perform PPO update
                        # Update happens independent of episode boundaries
                        progress.update(
                            episode_task,
                            description=f"[cyan]Ep. {episode} (Updating Policy...)",
                        )

            except RuntimeError as e:
                self.logger.critical(
                    f"Stopping training due to runtime error in episode {episode}: {e}",
                    exc_info=True,
                )
            except KeyboardInterrupt:
                self.logger.warning("Training interrupted by user.")
            except Exception as e:
                self.logger.critical(
                    f"Unexpected error during episode {episode}: {e}", exc_info=True
                )
            finally:
                progress.stop()
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
                    if completed_episodes > 0:
                        self._display_training_summary(completed_episodes)
                else:
                    progress.update(
                        episode_task,
                        description="[yellow]Training Stopped (Infinite Mode)",
                    )
                    if completed_episodes > 0:
                        self._display_training_summary(completed_episodes)

        self.logger.info("Training finished.")

    def _run_episode_or_rollout(
        self, episode_num: int, progress: Progress, task_id: TaskID
    ) -> Tuple[float, float, int, bool, bool]:
        """
        Runs steps until the horizon is met or an episode terminates/truncates.

        Collects data (state, action, probs, values, rewards, dones) for the trajectory buffer.
        Resets the environment if needed.

        Returns:
            Tuple: (episode_reward_hero, episode_reward_gun, episode_steps, terminated, truncated)
                   These values are only fully meaningful if an episode completed within this call.
                   If the horizon was reached mid-episode, rewards/steps are partial for that ep.
        """
        episode_reward_hero: float = 0.0
        episode_reward_gun: float = 0.0
        episode_steps: int = 0
        terminated: bool = False
        truncated: bool = False

        # Start from the current state (might be from a previous step/episode)
        if self.current_state is None:
            self.current_state = self._initialize_episode()
            if self.current_state is None:
                raise RuntimeError("Failed to get initial state for episode/rollout.")
            self.logger.debug(f"Episode {episode_num} started.")

        # Loop until horizon is met or episode ends
        for step_in_rollout in range(self.horizon - self.total_steps):
            if self.current_state is None:  # Should not happen if initialized correctly
                self.logger.error("Critical: current_state became None during rollout.")
                terminated = True  # Force stop
                break

            # --- Preprocess State for GNNs ---
            # This might be expensive, do it once per step
            try:
                with (
                    torch.no_grad()
                ):  # No gradients needed for data collection forward passes
                    graph_hero_actor = self.hero_actor_net.preprocess_state(
                        self.current_state
                    )
                    graph_gun_actor = self.gun_actor_net.preprocess_state(
                        self.current_state
                    )
                    # Assume critics use the same preprocessing for simplicity
                    graph_hero_critic = self.hero_critic_net.preprocess_state(
                        self.current_state
                    )
                    graph_gun_critic = self.gun_critic_net.preprocess_state(
                        self.current_state
                    )

                if (
                    graph_hero_actor is None
                    or graph_gun_actor is None
                    or graph_hero_critic is None
                    or graph_gun_critic is None
                ):
                    self.logger.warning(
                        f"Preprocessing failed at step {episode_steps} in Ep {episode_num}. Ending episode."
                    )
                    terminated = True  # Treat as failure
                    break

                # Move graphs to device once
                graph_hero_actor = graph_hero_actor.to(self.device)
                graph_gun_actor = graph_gun_actor.to(self.device)
                graph_hero_critic = graph_hero_critic.to(self.device)
                graph_gun_critic = graph_gun_critic.to(self.device)

            except Exception as e:
                self.logger.error(
                    f"Error preprocessing state in Ep {episode_num}, Step {episode_steps}: {e}",
                    exc_info=True,
                )
                terminated = True
                break

            # --- Select Actions and Get Values ---
            with torch.no_grad():
                move_action, move_log_prob = self._sample_action(
                    self.hero_actor_net, graph_hero_actor
                )
                shoot_action, shoot_log_prob = self._sample_action(
                    self.gun_actor_net, graph_gun_actor
                )
                hero_value = self._get_value(self.hero_critic_net, graph_hero_critic)
                gun_value = self._get_value(self.gun_critic_net, graph_gun_critic)

            if (
                move_action is None
                or shoot_action is None
                or hero_value is None
                or gun_value is None
            ):
                self.logger.warning(
                    f"Action sampling or value estimation failed in Ep {episode_num}. Ending episode."
                )
                terminated = True
                break

            # --- Step Environment ---
            step_result: Tuple[Optional[State], float, float, bool, bool] = (
                self._step_environment(move_action, shoot_action)
            )
            next_state, reward_hero, reward_gun, terminated, truncated = step_result

            # --- Store Transition ---
            step_data = TrajectoryStep(
                state_graph_hero=graph_hero_actor.cpu(),  # Store graphs on CPU to save GPU memory
                state_graph_gun=graph_gun_actor.cpu(),
                move_action=move_action,
                shoot_action=shoot_action,
                move_log_prob=move_log_prob.cpu(),
                shoot_log_prob=shoot_log_prob.cpu(),
                hero_value=hero_value.cpu(),
                gun_value=gun_value.cpu(),
                hero_reward=reward_hero,
                gun_reward=reward_gun,
                terminated=terminated,  # Store terminated flag for GAE
            )
            self.trajectory_buffer.append(step_data)

            # --- Update State and Counters ---
            self.current_state = next_state
            self.total_steps += 1
            episode_steps += 1
            episode_reward_hero += reward_hero
            episode_reward_gun += reward_gun

            # Update progress bar description less frequently if needed
            if episode_steps % 20 == 0:
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
                progress.update(
                    task_id,
                    description=(
                        f"[cyan]Ep. {episode_num}[/cyan] [yellow]Step {episode_steps}[/yellow] "
                        f"| Rollout: [b]{self.total_steps}/{self.horizon}[/b] "
                        f"| AvgR Gun: [b]{avg_r_gun_disp:.2f}[/b] "
                        f"| AvgR Hero: [b]{avg_r_hero_disp:.2f}[/b]"
                    ),
                )

            # --- Check for Episode End or Horizon ---
            if terminated or truncated:
                self.logger.debug(
                    f"Episode {episode_num} ended at step {episode_steps} ({'Terminated' if terminated else 'Truncated'}). Rollout steps: {self.total_steps}/{self.horizon}"
                )
                self.current_state = (
                    self._initialize_episode()
                )  # Reset for next rollout/episode
                if self.current_state is None:
                    self.logger.error("Failed to reset env after episode end.")
                    # Training loop will likely stop due to subsequent None state check
                break  # Exit the inner loop (rollout step loop)

            if self.total_steps >= self.horizon:
                self.logger.debug(
                    f"Horizon {self.horizon} reached during Ep {episode_num} at step {episode_steps}."
                )
                break  # Exit the inner loop (rollout step loop)

        return (
            episode_reward_hero,
            episode_reward_gun,
            episode_steps,
            terminated,
            truncated,
        )

    def _initialize_episode(self) -> Optional[State]:
        """Resets the environment and returns the initial state."""
        try:
            initial_state: State = self.env.initialise_environment()
            if not isinstance(initial_state, State):
                raise TypeError(f"Env did not return State, got {type(initial_state)}")
            return initial_state
        except Exception as e:
            self.logger.error(f"Failed to initialize environment: {e}", exc_info=True)
            return None

    def _sample_action(
        self, actor_net: nn.Module, state_graph: Union[HeteroData, Batch]
    ) -> Tuple[Optional[int], Optional[torch.Tensor]]:
        """Samples an action from the actor network's output distribution."""
        try:
            actor_net.eval()  # Set to eval mode for sampling
            logits: torch.Tensor = actor_net(state_graph)
            actor_net.train()  # Set back to train mode

            if logits.numel() == 0:
                self.logger.warning(
                    f"{type(actor_net).__name__} produced empty logits."
                )
                return None, None
            if logits.ndim > 1:  # Handle batch dim if present (should be 1 sample here)
                logits = logits.squeeze(0)

            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # self.logger.debug(f"{type(actor_net).__name__} sampled action: {action.item()}, log_prob: {log_prob.item():.4f}")
            return action.item(), log_prob

        except Exception as e:
            self.logger.error(
                f"Error sampling action with {type(actor_net).__name__}: {e}",
                exc_info=True,
            )
            return None, None

    def _get_value(
        self, critic_net: nn.Module, state_graph: Union[HeteroData, Batch]
    ) -> Optional[torch.Tensor]:
        """Gets the value estimate from the critic network."""
        try:
            critic_net.eval()  # Set to eval mode for value estimation
            value: torch.Tensor = critic_net(state_graph)
            critic_net.train()  # Set back to train mode

            if value.numel() != 1:
                self.logger.warning(
                    f"{type(critic_net).__name__} produced non-scalar value: shape {value.shape}. Taking first element."
                )
                value = value.flatten()[0]
            # self.logger.debug(f"{type(critic_net).__name__} estimated value: {value.item():.4f}")
            return value.squeeze()  # Ensure scalar tensor

        except Exception as e:
            self.logger.error(
                f"Error getting value with {type(critic_net).__name__}: {e}",
                exc_info=True,
            )
            return None

    def _step_environment(
        self, move_action: int, shoot_action: int
    ) -> Tuple[Optional[State], float, float, bool, bool]:
        """Steps the environment with the chosen actions."""
        # This function remains largely the same as the DQN version
        try:
            combined_action_list: List[int] = [move_action, shoot_action, 0, 0]
            step_result: Tuple = self.env.step(combined_action_list)

            if not isinstance(step_result, tuple) or len(step_result) < 3:
                raise TypeError(
                    f"Env step returned unexpected format: {type(step_result)}"
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
                    f"Env step returned invalid non-terminal state type: {type(next_s)}"
                )
                return None, 0.0, 0.0, True, True  # Critical failure

            if not isinstance(reward_tuple, (tuple, list)) or len(reward_tuple) < 2:
                self.logger.error(
                    f"Env step returned invalid reward format: {reward_tuple}"
                )
                return next_state, 0.0, 0.0, True, True

            reward_hero: float = float(
                0 if reward_tuple[0] is None else reward_tuple[0]
            )
            reward_gun: float = float(0 if reward_tuple[1] is None else reward_tuple[1])
            terminated: bool = bool(terminated_flag)
            truncated: bool = bool(truncated_flag)

            return next_state, reward_hero, reward_gun, terminated, truncated

        except Exception as e:
            self.logger.error(f"Error during environment step: {e}", exc_info=True)
            return None, 0.0, 0.0, True, True

    def _calculate_gae_and_returns(
        self, last_state: Optional[State], last_terminated: bool, last_truncated: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates Generalized Advantage Estimation (GAE) and returns (value targets).
        Processes the stored trajectory_buffer.

        Args:
             last_state: The state *after* the last step in the buffer. Needed for bootstrapping if not terminated.
             last_terminated: Whether the episode terminated at the end of the buffer.
             last_truncated: Whether the episode truncated at the end of the buffer.

        Returns:
            Tuple: (hero_advantages, hero_returns, gun_advantages, gun_returns)
        """
        hero_advantages = []
        gun_advantages = []
        hero_last_gae_lam = 0.0
        gun_last_gae_lam = 0.0

        # --- Get value of the last state for bootstrapping ---
        last_hero_value = 0.0
        last_gun_value = 0.0
        if (
            not last_terminated and last_state is not None
        ):  # Bootstrap if not terminal state
            try:
                with torch.no_grad():
                    graph_h_last = self.hero_critic_net.preprocess_state(last_state)
                    graph_g_last = self.gun_critic_net.preprocess_state(last_state)
                    if graph_h_last is not None and graph_g_last is not None:
                        graph_h_last = graph_h_last.to(self.device)
                        graph_g_last = graph_g_last.to(self.device)
                        val_h = self._get_value(self.hero_critic_net, graph_h_last)
                        val_g = self._get_value(self.gun_critic_net, graph_g_last)
                        if val_h is not None:
                            last_hero_value = val_h.item()
                        if val_g is not None:
                            last_gun_value = val_g.item()
                    else:
                        self.logger.warning(
                            "Preprocessing failed for GAE bootstrap state. Using value 0."
                        )

            except Exception as e:
                self.logger.warning(
                    f"Error getting value for GAE bootstrap state: {e}. Using value 0."
                )

        # --- Iterate backwards through the trajectory ---
        for step in reversed(self.trajectory_buffer):
            current_hero_value = step.hero_value.item()
            current_gun_value = step.gun_value.item()
            reward_hero = step.hero_reward
            reward_gun = step.gun_reward
            terminated_mask = 1.0 - float(
                step.terminated
            )  # 0 if terminated, 1 otherwise

            # Calculate delta (TD error)
            delta_hero = (
                reward_hero
                + self.discount_factor * last_hero_value * terminated_mask
                - current_hero_value
            )
            delta_gun = (
                reward_gun
                + self.discount_factor * last_gun_value * terminated_mask
                - current_gun_value
            )

            # Calculate GAE advantage for this step
            adv_hero = (
                delta_hero
                + self.discount_factor
                * self.gae_lambda
                * hero_last_gae_lam
                * terminated_mask
            )
            adv_gun = (
                delta_gun
                + self.discount_factor
                * self.gae_lambda
                * gun_last_gae_lam
                * terminated_mask
            )

            hero_advantages.append(adv_hero)
            gun_advantages.append(adv_gun)

            # Update for next iteration
            hero_last_gae_lam = adv_hero
            gun_last_gae_lam = adv_gun
            last_hero_value = current_hero_value
            last_gun_value = current_gun_value

        # Reverse advantages to match trajectory order
        hero_advantages.reverse()
        gun_advantages.reverse()

        hero_adv_tensor = torch.tensor(
            hero_advantages, dtype=torch.float32, device=self.device
        )
        gun_adv_tensor = torch.tensor(
            gun_advantages, dtype=torch.float32, device=self.device
        )

        # --- Calculate returns (value targets) ---
        # Returns = Advantages + Values
        values_hero = torch.stack(
            [step.hero_value for step in self.trajectory_buffer]
        ).to(self.device)
        values_gun = torch.stack(
            [step.gun_value for step in self.trajectory_buffer]
        ).to(self.device)
        hero_returns = hero_adv_tensor + values_hero
        gun_returns = gun_adv_tensor + values_gun

        # --- Normalize Advantages (optional but recommended) ---
        hero_adv_tensor = (hero_adv_tensor - hero_adv_tensor.mean()) / (
            hero_adv_tensor.std() + 1e-8
        )
        gun_adv_tensor = (gun_adv_tensor - gun_adv_tensor.mean()) / (
            gun_adv_tensor.std() + 1e-8
        )

        return hero_adv_tensor, hero_returns, gun_adv_tensor, gun_returns

    def _update_policy(self) -> None:
        """Performs the PPO learning update using the collected trajectory."""
        if not self.trajectory_buffer:
            self.logger.warning("Attempted PPO update with empty trajectory buffer.")
            return

        # Need the state *after* the last step for GAE calculation
        # self.current_state holds this if the horizon ended mid-episode
        last_state = self.current_state
        # We also need to know if the trajectory ended due to termination/truncation
        # The _run_episode_or_rollout function returned these flags, but they are tied
        # to the *episode*, not necessarily the *end of the buffer*.
        # A simpler approach for GAE: check the 'terminated' flag of the *last* step in the buffer.
        last_step_info = self.trajectory_buffer[-1]
        last_terminated = last_step_info.terminated
        # We don't explicitly store 'truncated' in buffer, assume bootstrap if not terminated
        last_truncated = not last_terminated

        # --- 1. Calculate Advantages and Returns ---
        try:
            hero_advantages, hero_returns, gun_advantages, gun_returns = (
                self._calculate_gae_and_returns(
                    last_state, last_terminated, last_truncated
                )
            )
        except Exception as e:
            self.logger.error(f"Error calculating GAE/Returns: {e}", exc_info=True)
            self.trajectory_buffer.clear()  # Clear buffer on critical error
            self.total_steps = 0
            return

        # --- 2. Prepare Data for Optimization ---
        # Extract data from buffer
        hero_graphs = [step.state_graph_hero for step in self.trajectory_buffer]
        gun_graphs = [step.state_graph_gun for step in self.trajectory_buffer]
        move_actions = torch.tensor(
            [step.move_action for step in self.trajectory_buffer],
            dtype=torch.long,
            device=self.device,
        )
        shoot_actions = torch.tensor(
            [step.shoot_action for step in self.trajectory_buffer],
            dtype=torch.long,
            device=self.device,
        )
        old_move_log_probs = torch.stack(
            [step.move_log_prob for step in self.trajectory_buffer]
        ).to(self.device)
        old_shoot_log_probs = torch.stack(
            [step.shoot_log_prob for step in self.trajectory_buffer]
        ).to(self.device)

        # Batch the graphs (needs PyG Batch)
        try:
            batch_hero_graphs = Batch.from_data_list(hero_graphs).to(self.device)
            batch_gun_graphs = Batch.from_data_list(gun_graphs).to(self.device)
        except Exception as e:
            self.logger.error(f"Error creating PyG Batch objects: {e}", exc_info=True)
            self.trajectory_buffer.clear()
            self.total_steps = 0
            return

        data_size = len(self.trajectory_buffer)
        indices = np.arange(data_size)

        # --- 3. Optimization Loop ---
        self.logger.debug(
            f"Starting PPO optimization ({self.epochs_per_update} epochs, {data_size} samples, {self.mini_batch_size} batch size)"
        )
        for epoch in range(self.epochs_per_update):
            np.random.shuffle(indices)  # Shuffle data each epoch
            for start in range(0, data_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]

                # Get minibatch data
                # Note: Slicing PyG Batch objects directly is complex. Instead, we'll pass the
                # full batch and use the indices inside the loss calculation where needed,
                # or reconstruct mini-batches if absolutely necessary (more overhead).
                # Let's try using indices on the tensors directly.
                mb_hero_graphs_batch = Batch.from_data_list(
                    [hero_graphs[i] for i in mb_indices]
                ).to(self.device)
                mb_gun_graphs_batch = Batch.from_data_list(
                    [gun_graphs[i] for i in mb_indices]
                ).to(self.device)

                mb_move_actions = move_actions[mb_indices]
                mb_shoot_actions = shoot_actions[mb_indices]
                mb_old_move_log_probs = old_move_log_probs[mb_indices]
                mb_old_shoot_log_probs = old_shoot_log_probs[mb_indices]
                mb_hero_advantages = hero_advantages[mb_indices]
                mb_gun_advantages = gun_advantages[mb_indices]
                mb_hero_returns = hero_returns[mb_indices]
                mb_gun_returns = gun_returns[mb_indices]

                try:
                    # --- Forward pass for current policy/value estimates ---
                    # Hero
                    hero_logits = self.hero_actor_net(mb_hero_graphs_batch)
                    hero_dist = Categorical(logits=hero_logits)
                    new_move_log_probs = hero_dist.log_prob(mb_move_actions)
                    hero_entropy = hero_dist.entropy().mean()
                    hero_values = self.hero_critic_net(
                        mb_hero_graphs_batch
                    ).squeeze()  # Assume critic outputs [B, 1]

                    # Gun
                    gun_logits = self.gun_actor_net(mb_gun_graphs_batch)
                    gun_dist = Categorical(logits=gun_logits)
                    new_shoot_log_probs = gun_dist.log_prob(mb_shoot_actions)
                    gun_entropy = gun_dist.entropy().mean()
                    gun_values = self.gun_critic_net(
                        mb_gun_graphs_batch
                    ).squeeze()  # Assume critic outputs [B, 1]

                    # --- Calculate Hero Losses ---
                    # Actor Loss (PPO Clipped Objective)
                    ratio_hero = torch.exp(new_move_log_probs - mb_old_move_log_probs)
                    surr1_hero = ratio_hero * mb_hero_advantages
                    surr2_hero = (
                        torch.clamp(
                            ratio_hero, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                        )
                        * mb_hero_advantages
                    )
                    actor_loss_hero = -torch.min(surr1_hero, surr2_hero).mean()

                    # Critic Loss (Value Loss)
                    critic_loss_hero = self.critic_loss_fn(hero_values, mb_hero_returns)

                    # Total Hero Loss
                    total_loss_hero = (
                        actor_loss_hero
                        + self.vf_coeff * critic_loss_hero
                        - self.entropy_coeff * hero_entropy
                    )

                    # --- Calculate Gun Losses ---
                    # Actor Loss
                    ratio_gun = torch.exp(new_shoot_log_probs - mb_old_shoot_log_probs)
                    surr1_gun = ratio_gun * mb_gun_advantages
                    surr2_gun = (
                        torch.clamp(
                            ratio_gun, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                        )
                        * mb_gun_advantages
                    )
                    actor_loss_gun = -torch.min(surr1_gun, surr2_gun).mean()

                    # Critic Loss
                    critic_loss_gun = self.critic_loss_fn(gun_values, mb_gun_returns)

                    # Total Gun Loss
                    total_loss_gun = (
                        actor_loss_gun
                        + self.vf_coeff * critic_loss_gun
                        - self.entropy_coeff * gun_entropy
                    )

                    # --- Optimization Steps ---
                    # Hero
                    self.hero_actor_optimizer.zero_grad()
                    self.hero_critic_optimizer.zero_grad()
                    total_loss_hero.backward()
                    # Optional: Gradient clipping
                    # nn.utils.clip_grad_norm_(self.hero_actor_net.parameters(), max_norm=0.5)
                    # nn.utils.clip_grad_norm_(self.hero_critic_net.parameters(), max_norm=0.5)
                    self.hero_actor_optimizer.step()
                    self.hero_critic_optimizer.step()

                    # Gun
                    self.gun_actor_optimizer.zero_grad()
                    self.gun_critic_optimizer.zero_grad()
                    total_loss_gun.backward()
                    # Optional: Gradient clipping
                    # nn.utils.clip_grad_norm_(self.gun_actor_net.parameters(), max_norm=0.5)
                    # nn.utils.clip_grad_norm_(self.gun_critic_net.parameters(), max_norm=0.5)
                    self.gun_actor_optimizer.step()
                    self.gun_critic_optimizer.step()

                    # --- Store Loss Metrics ---
                    self.hero_actor_loss_deque.append(actor_loss_hero.item())
                    self.hero_critic_loss_deque.append(critic_loss_hero.item())
                    self.entropy_hero_deque.append(hero_entropy.item())
                    self.gun_actor_loss_deque.append(actor_loss_gun.item())
                    self.gun_critic_loss_deque.append(critic_loss_gun.item())
                    self.entropy_gun_deque.append(gun_entropy.item())

                except Exception as e:
                    self.logger.error(
                        f"Error during PPO optimization step (Epoch {epoch}, Batch {start // self.mini_batch_size}): {e}",
                        exc_info=True,
                    )
                    # Decide whether to continue or break epoch/update
                    break  # Break inner loop for safety

        # --- 4. Clear Trajectory Buffer ---
        self.trajectory_buffer.clear()
        self.total_steps = 0  # Reset step counter for next rollout
        self.logger.debug("PPO update finished. Trajectory buffer cleared.")

    def _save_checkpoint_if_needed(self, episode: int) -> None:
        """Saves a checkpoint of the agent's state if the save interval is reached."""
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
        """Displays a summary table of training performance."""
        # This function remains the same as the DQN version
        if not self.training_summary_data or total_episodes_completed == 0:
            self.logger.info(
                "No training data recorded or no episodes completed, skipping summary."
            )
            return

        self.logger.info("Generating Training Summary Table...")
        df = pd.DataFrame(self.training_summary_data)
        block_size = max(1, total_episodes_completed // 10)
        num_blocks = (total_episodes_completed + block_size - 1) // block_size

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
            summary_rows.append(
                (
                    f"{start_episode}-{end_episode}",
                    f"{avg_reward_hero:.3f}",
                    f"{avg_reward_gun:.3f}",
                    f"{avg_time_alive:.2f}",
                )
            )

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

    def dump(self, save_dir: str = "model_saves_ppo") -> Optional[str]:
        """Saves the PPO agent state."""
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name: str = f"theseus_ppo_{timestamp}"
        dpath: str = os.path.join(save_dir, base_name)

        try:
            os.makedirs(dpath, exist_ok=True)
            self.logger.info(f"Saving PPO agent state to directory: {dpath}")

            filenames: Dict[str, str] = {
                "hero_actor": "hero_actor_state.pth",
                "hero_critic": "hero_critic_state.pth",
                "gun_actor": "gun_actor_state.pth",
                "gun_critic": "gun_critic_state.pth",
                "hero_actor_optim": "hero_actor_optimizer.pth",
                "hero_critic_optim": "hero_critic_optimizer.pth",
                "gun_actor_optim": "gun_actor_optimizer.pth",
                "gun_critic_optim": "gun_critic_optimizer.pth",
                "config": f"{base_name}_config.yaml",
            }

            # Save network states
            torch.save(
                self.hero_actor_net.state_dict(),
                os.path.join(dpath, filenames["hero_actor"]),
            )
            torch.save(
                self.hero_critic_net.state_dict(),
                os.path.join(dpath, filenames["hero_critic"]),
            )
            torch.save(
                self.gun_actor_net.state_dict(),
                os.path.join(dpath, filenames["gun_actor"]),
            )
            torch.save(
                self.gun_critic_net.state_dict(),
                os.path.join(dpath, filenames["gun_critic"]),
            )

            # Save optimizer states
            torch.save(
                self.hero_actor_optimizer.state_dict(),
                os.path.join(dpath, filenames["hero_actor_optim"]),
            )
            torch.save(
                self.hero_critic_optimizer.state_dict(),
                os.path.join(dpath, filenames["hero_critic_optim"]),
            )
            torch.save(
                self.gun_actor_optimizer.state_dict(),
                os.path.join(dpath, filenames["gun_actor_optim"]),
            )
            torch.save(
                self.gun_critic_optimizer.state_dict(),
                os.path.join(dpath, filenames["gun_critic_optim"]),
            )

            state_info: Dict[str, Any] = {
                "agent_type": "PPO",
                "hero_actor_file": filenames["hero_actor"],
                "hero_critic_file": filenames["hero_critic"],
                "gun_actor_file": filenames["gun_actor"],
                "gun_critic_file": filenames["gun_critic"],
                "hero_actor_optim_file": filenames["hero_actor_optim"],
                "hero_critic_optim_file": filenames["hero_critic_optim"],
                "gun_actor_optim_file": filenames["gun_actor_optim"],
                "gun_critic_optim_file": filenames["gun_critic_optim"],
                # Store class paths assuming they are needed for reconstruction
                "hero_actor_class": f"{type(self.hero_actor_net).__module__}.{type(self.hero_actor_net).__name__}",
                "hero_critic_class": f"{type(self.hero_critic_net).__module__}.{type(self.hero_critic_net).__name__}",
                "gun_actor_class": f"{type(self.gun_actor_net).__module__}.{type(self.gun_actor_net).__name__}",
                "gun_critic_class": f"{type(self.gun_critic_net).__module__}.{type(self.gun_critic_net).__name__}",
                # PPO Hyperparameters
                "learning_rate": self.hero_actor_optimizer.param_groups[0][
                    "lr"
                ],  # Assume same LR for all
                "discount_factor": self.discount_factor,
                "horizon": self.horizon,
                "epochs_per_update": self.epochs_per_update,
                "mini_batch_size": self.mini_batch_size,
                "clip_epsilon": self.clip_epsilon,
                "gae_lambda": self.gae_lambda,
                "entropy_coeff": self.entropy_coeff,
                "vf_coeff": self.vf_coeff,
                # Other relevant state
                "log_window_size": self.log_window_size,
                "save_interval": self.save_interval,
                "total_reward_hero": self.total_reward_hero,
                "total_reward_gun": self.total_reward_gun,
                "optimizer_class": f"{type(self.hero_actor_optimizer).__module__}.{type(self.hero_actor_optimizer).__name__}",
                # "critic_loss_fn_class": f"{type(self.critic_loss_fn).__module__}.{type(self.critic_loss_fn).__name__}", # Usually just MSELoss
            }
            yaml_path: str = os.path.join(dpath, filenames["config"])
            with open(yaml_path, "w") as f:
                yaml.dump(state_info, f, default_flow_style=False, sort_keys=False)

            self.logger.info("PPO Agent state saved successfully.")
            return dpath

        except Exception as e:
            self.logger.error(
                f"Failed to dump PPO agent state to {dpath}: {e}", exc_info=True
            )
            return None

    @classmethod
    def load(cls, load_path: Union[str, os.PathLike]) -> Optional[Self]:
        """Loads a PPO agent state from a checkpoint directory."""
        logger = logging.getLogger("agent-theseus-ppo-load")
        logger.info(f"Attempting to load PPO agent state from: {load_path}")

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
        yaml_path: str = os.path.join(load_path_str, yaml_files[0])

        try:
            with open(yaml_path, "r") as f:
                state_info: Dict[str, Any] = yaml.safe_load(f)
        except Exception as e:
            logger.error(
                f"Error reading YAML configuration {yaml_path}: {e}", exc_info=True
            )
            return None

        # --- Verify Agent Type ---
        if state_info.get("agent_type") != "PPO":
            logger.error(
                f"Config file indicates agent type is {state_info.get('agent_type')}, not PPO."
            )
            return None

        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading models onto device: {device}")

        try:

            def get_class(class_path: str) -> Type:
                module_path, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                return getattr(module, class_name)

            # --- Reconstruct Networks ---
            HeroActorClass = get_class(state_info["hero_actor_class"])
            HeroCriticClass = get_class(state_info["hero_critic_class"])
            GunActorClass = get_class(state_info["gun_actor_class"])
            GunCriticClass = get_class(state_info["gun_critic_class"])

            # Instantiate networks (assuming default constructors or loading args from state_info if needed)
            hero_actor_net = HeroActorClass()
            hero_critic_net = HeroCriticClass()
            gun_actor_net = GunActorClass()
            gun_critic_net = GunCriticClass()

            # Load state dicts
            ha_path = os.path.join(load_path_str, state_info["hero_actor_file"])
            hc_path = os.path.join(load_path_str, state_info["hero_critic_file"])
            ga_path = os.path.join(load_path_str, state_info["gun_actor_file"])
            gc_path = os.path.join(load_path_str, state_info["gun_critic_file"])
            hero_actor_net.load_state_dict(torch.load(ha_path, map_location=device))
            hero_critic_net.load_state_dict(torch.load(hc_path, map_location=device))
            gun_actor_net.load_state_dict(torch.load(ga_path, map_location=device))
            gun_critic_net.load_state_dict(torch.load(gc_path, map_location=device))
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
            # --- Reconstruct Optimizers ---
            learning_rate = state_info.get("learning_rate", DEFAULT_LEARNING_RATE_PPO)
            OptimizerClass = get_class(
                state_info.get("optimizer_class", "torch.optim.AdamW")
            )

            # Move networks to device BEFORE creating optimizers
            hero_actor_net.to(device)
            hero_critic_net.to(device)
            gun_actor_net.to(device)
            gun_critic_net.to(device)

            hero_actor_optimizer = OptimizerClass(
                hero_actor_net.parameters(), lr=learning_rate
            )
            hero_critic_optimizer = OptimizerClass(
                hero_critic_net.parameters(), lr=learning_rate
            )
            gun_actor_optimizer = OptimizerClass(
                gun_actor_net.parameters(), lr=learning_rate
            )
            gun_critic_optimizer = OptimizerClass(
                gun_critic_net.parameters(), lr=learning_rate
            )

            # Load optimizer states if files exist
            optim_files = [
                "hero_actor_optim",
                "hero_critic_optim",
                "gun_actor_optim",
                "gun_critic_optim",
            ]
            optimizers = [
                hero_actor_optimizer,
                hero_critic_optimizer,
                gun_actor_optimizer,
                gun_critic_optimizer,
            ]
            for name, optim in zip(optim_files, optimizers):
                optim_path = os.path.join(load_path_str, state_info[f"{name}_file"])
                if os.path.exists(optim_path):
                    optim.load_state_dict(torch.load(optim_path, map_location=device))
                    logger.info(f"{name.replace('_', ' ').title()} state loaded.")
                else:
                    logger.warning(
                        f"{name.replace('_', ' ').title()} state file not found: {optim_path}. Initializing fresh."
                    )

            logger.info("Optimizers reconstructed.")

        except (
            ImportError,
            AttributeError,
            KeyError,
            FileNotFoundError,
            Exception,
        ) as e:
            logger.error(
                f"Error reconstructing or loading optimizers: {e}", exc_info=True
            )
            return None

        try:
            env = Environment()  # Assuming default constructor works
        except Exception as e:
            logger.error(f"Failed to instantiate Environment: {e}", exc_info=True)
            return None

        try:
            # --- Instantiate Agent with loaded state ---
            agent = cls(
                hero_actor_net=hero_actor_net,
                hero_critic_net=hero_critic_net,
                gun_actor_net=gun_actor_net,
                gun_critic_net=gun_critic_net,
                env=env,
                optimizer_class=OptimizerClass,
                learning_rate=learning_rate,  # Already applied to loaded optims
                discount_factor=state_info.get(
                    "discount_factor", DEFAULT_DISCOUNT_FACTOR_PPO
                ),
                horizon=state_info.get("horizon", DEFAULT_HORIZON),
                epochs_per_update=state_info.get(
                    "epochs_per_update", DEFAULT_EPOCHS_PER_UPDATE
                ),
                mini_batch_size=state_info.get(
                    "mini_batch_size", DEFAULT_MINI_BATCH_SIZE_PPO
                ),
                clip_epsilon=state_info.get("clip_epsilon", DEFAULT_CLIP_EPSILON),
                gae_lambda=state_info.get("gae_lambda", DEFAULT_GAE_LAMBDA),
                entropy_coeff=state_info.get("entropy_coeff", DEFAULT_ENTROPY_COEFF),
                vf_coeff=state_info.get("vf_coeff", DEFAULT_VF_COEFF),
                log_window_size=state_info.get("log_window_size", LOGGING_WINDOW),
                save_interval=state_info.get("save_interval", SAVE_INTERVAL),
            )

            # --- Restore specific state variables ---
            agent.hero_actor_optimizer = hero_actor_optimizer
            agent.hero_critic_optimizer = hero_critic_optimizer
            agent.gun_actor_optimizer = gun_actor_optimizer
            agent.gun_critic_optimizer = gun_critic_optimizer
            agent.total_reward_hero = state_info.get("total_reward_hero", 0.0)
            agent.total_reward_gun = state_info.get("total_reward_gun", 0.0)
            # PPO doesn't have epsilon or sync steps state to restore

            # Ensure networks are in train mode after loading (PPO alternates eval/train)
            agent.hero_actor_net.train()
            agent.hero_critic_net.train()
            agent.gun_actor_net.train()
            agent.gun_critic_net.train()

            logger.info(f"PPO Agent loaded successfully from {load_path_str}")
            return agent

        except Exception as e:
            logger.error(
                f"Error instantiating AgentTheseusPPO during final load step: {e}",
                exc_info=True,
            )
            return None

    def dump(self, save_dir: str = "model_saves_ppo") -> Optional[str]:
        """Saves the PPO agent state, including network architecture details."""
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name: str = f"theseus_ppo_{timestamp}"
        dpath: str = os.path.join(save_dir, base_name)

        try:
            os.makedirs(dpath, exist_ok=True)
            self.logger.info(f"Saving PPO agent state to directory: {dpath}")

            filenames: Dict[str, str] = {
                "hero_actor": "hero_actor_state.pth",
                "hero_critic": "hero_critic_state.pth",
                "gun_actor": "gun_actor_state.pth",
                "gun_critic": "gun_critic_state.pth",
                "hero_actor_optim": "hero_actor_optimizer.pth",
                "hero_critic_optim": "hero_critic_optimizer.pth",
                "gun_actor_optim": "gun_actor_optimizer.pth",
                "gun_critic_optim": "gun_critic_optimizer.pth",
                "config": f"{base_name}_config.yaml",
            }

            # Save network states
            torch.save(
                self.hero_actor_net.state_dict(),
                os.path.join(dpath, filenames["hero_actor"]),
            )
            torch.save(
                self.hero_critic_net.state_dict(),
                os.path.join(dpath, filenames["hero_critic"]),
            )
            torch.save(
                self.gun_actor_net.state_dict(),
                os.path.join(dpath, filenames["gun_actor"]),
            )
            torch.save(
                self.gun_critic_net.state_dict(),
                os.path.join(dpath, filenames["gun_critic"]),
            )

            # Save optimizer states
            torch.save(
                self.hero_actor_optimizer.state_dict(),
                os.path.join(dpath, filenames["hero_actor_optim"]),
            )
            torch.save(
                self.hero_critic_optimizer.state_dict(),
                os.path.join(dpath, filenames["hero_critic_optim"]),
            )
            torch.save(
                self.gun_actor_optimizer.state_dict(),
                os.path.join(dpath, filenames["gun_actor_optim"]),
            )
            torch.save(
                self.gun_critic_optimizer.state_dict(),
                os.path.join(dpath, filenames["gun_critic_optim"]),
            )

            state_info: Dict[str, Any] = {
                "agent_type": "PPO",
                "hero_actor_file": filenames["hero_actor"],
                "hero_critic_file": filenames["hero_critic"],
                "gun_actor_file": filenames["gun_actor"],
                "gun_critic_file": filenames["gun_critic"],
                "hero_actor_optim_file": filenames["hero_actor_optim"],
                "hero_critic_optim_file": filenames["hero_critic_optim"],
                "gun_actor_optim_file": filenames["gun_actor_optim"],
                "gun_critic_optim_file": filenames["gun_critic_optim"],
                # Store class paths assuming they are needed for reconstruction
                "hero_actor_class": f"{type(self.hero_actor_net).__module__}.{type(self.hero_actor_net).__name__}",
                "hero_critic_class": f"{type(self.hero_critic_net).__module__}.{type(self.hero_critic_net).__name__}",
                "gun_actor_class": f"{type(self.gun_actor_net).__module__}.{type(self.gun_actor_net).__name__}",
                "gun_critic_class": f"{type(self.gun_critic_net).__module__}.{type(self.gun_critic_net).__name__}",
                # --- Store Network Architecture Details ---
                "hero_hidden_channels": self.hero_hidden_channels,
                "gun_hidden_channels": self.gun_hidden_channels,
                # Optional: store out_channels if they might differ from defaults
                # "hero_out_channels": self.hero_actor_net.fc.out_features,
                # "gun_out_channels": self.gun_actor_net.fc.out_features,
                # ------------------------------------------
                # PPO Hyperparameters
                "learning_rate": self.hero_actor_optimizer.param_groups[0][
                    "lr"
                ],  # Assume same LR for all
                "discount_factor": self.discount_factor,
                "horizon": self.horizon,
                "epochs_per_update": self.epochs_per_update,
                "mini_batch_size": self.mini_batch_size,
                "clip_epsilon": self.clip_epsilon,
                "gae_lambda": self.gae_lambda,
                "entropy_coeff": self.entropy_coeff,
                "vf_coeff": self.vf_coeff,
                # Other relevant state
                "log_window_size": self.log_window_size,
                "save_interval": self.save_interval,
                "total_reward_hero": self.total_reward_hero,
                "total_reward_gun": self.total_reward_gun,
                "optimizer_class": f"{type(self.hero_actor_optimizer).__module__}.{type(self.hero_actor_optimizer).__name__}",
            }
            yaml_path: str = os.path.join(dpath, filenames["config"])
            with open(yaml_path, "w") as f:
                yaml.dump(state_info, f, default_flow_style=False, sort_keys=False)

            self.logger.info("PPO Agent state saved successfully.")
            return dpath

        except AttributeError as e:
            self.logger.error(
                f"Failed to dump agent state due to missing attribute (likely hidden_channels): {e}",
                exc_info=True,
            )
            return None
        except Exception as e:
            self.logger.error(
                f"Failed to dump PPO agent state to {dpath}: {e}", exc_info=True
            )
            return None

    @classmethod
    def load(cls, load_path: Union[str, os.PathLike]) -> Optional[Self]:
        """Loads a PPO agent state from a checkpoint directory."""
        logger = logging.getLogger("agent-theseus-ppo-load")
        logger.info(f"Attempting to load PPO agent state from: {load_path}")

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
        # Take the first one if multiple exist (e.g., if saving failed mid-way previously)
        yaml_path: str = os.path.join(load_path_str, yaml_files[0])
        logger.info(f"Using config file: {yaml_path}")

        try:
            with open(yaml_path, "r") as f:
                state_info: Dict[str, Any] = yaml.safe_load(f)
        except Exception as e:
            logger.error(
                f"Error reading YAML configuration {yaml_path}: {e}", exc_info=True
            )
            return None

        # --- Verify Agent Type ---
        if state_info.get("agent_type") != "PPO":
            logger.error(
                f"Config file indicates agent type is {state_info.get('agent_type')}, not PPO."
            )
            return None

        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading models onto device: {device}")

        try:

            def get_class(class_path: str) -> Type:
                module_path, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                return getattr(module, class_name)

            # --- Get Network Architecture Details ---
            try:
                hero_hidden = state_info["hero_hidden_channels"]
                gun_hidden = state_info["gun_hidden_channels"]
                logger.info(f"Loaded Hero Hidden Channels: {hero_hidden}")
                logger.info(f"Loaded Gun Hidden Channels: {gun_hidden}")
                # Optional: Load out_channels if saved
                # hero_out = state_info.get('hero_out_channels', HERO_ACTION_SPACE_SIZE)
                # gun_out = state_info.get('gun_out_channels', GUN_ACTION_SPACE_SIZE)
            except KeyError as e:
                logger.error(
                    f"Missing required network architecture key '{e}' in config file {yaml_path}. "
                    f"Cannot reconstruct networks."
                )
                return None

            # --- Reconstruct Networks ---
            HeroActorClass = get_class(state_info["hero_actor_class"])
            HeroCriticClass = get_class(state_info["hero_critic_class"])
            GunActorClass = get_class(state_info["gun_actor_class"])
            GunCriticClass = get_class(state_info["gun_critic_class"])

            # Instantiate networks WITH hidden_channels (and optionally out_channels)
            # Assuming actor/critic pairs share the same architecture params
            hero_actor_net = HeroActorClass(
                hidden_channels=hero_hidden
            )  # , out_channels=hero_out)
            hero_critic_net = HeroCriticClass(
                hidden_channels=hero_hidden
            )  # , out_channels=1) # Critic outputs single value
            gun_actor_net = GunActorClass(
                hidden_channels=gun_hidden
            )  # , out_channels=gun_out)
            gun_critic_net = GunCriticClass(
                hidden_channels=gun_hidden
            )  # , out_channels=1) # Critic outputs single value

            # Load state dicts
            def _load_state_dict(net: nn.Module, file_key: str):
                file_path = os.path.join(load_path_str, state_info[file_key])
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"Network state file not found: {file_path}"
                    )
                net.load_state_dict(torch.load(file_path, map_location=device))
                logger.debug(f"Loaded state dict from: {file_path}")

            _load_state_dict(hero_actor_net, "hero_actor_file")
            _load_state_dict(hero_critic_net, "hero_critic_file")
            _load_state_dict(gun_actor_net, "gun_actor_file")
            _load_state_dict(gun_critic_net, "gun_critic_file")

            logger.info("Network state dicts loaded successfully.")

        except (
            ImportError,
            AttributeError,
            KeyError,
            FileNotFoundError,
            TypeError,
            Exception,
        ) as e:
            # Catch TypeError here as well for constructor issues
            logger.error(
                f"Error reconstructing or loading network models: {e}", exc_info=True
            )
            return None

        # --- Reconstruct Optimizers (AFTER moving networks to device) ---
        try:
            learning_rate = state_info.get("learning_rate", DEFAULT_LEARNING_RATE_PPO)
            OptimizerClass = get_class(
                state_info.get("optimizer_class", "torch.optim.AdamW")
            )

            # Move networks to device BEFORE creating optimizers that need their params
            hero_actor_net.to(device)
            hero_critic_net.to(device)
            gun_actor_net.to(device)
            gun_critic_net.to(device)

            hero_actor_optimizer = OptimizerClass(
                hero_actor_net.parameters(), lr=learning_rate
            )
            hero_critic_optimizer = OptimizerClass(
                hero_critic_net.parameters(), lr=learning_rate
            )
            gun_actor_optimizer = OptimizerClass(
                gun_actor_net.parameters(), lr=learning_rate
            )
            gun_critic_optimizer = OptimizerClass(
                gun_critic_net.parameters(), lr=learning_rate
            )

            # Load optimizer states if files exist
            optim_files = [
                "hero_actor_optim",
                "hero_critic_optim",
                "gun_actor_optim",
                "gun_critic_optim",
            ]
            optimizers = [
                hero_actor_optimizer,
                hero_critic_optimizer,
                gun_actor_optimizer,
                gun_critic_optimizer,
            ]
            for name, optim in zip(optim_files, optimizers):
                optim_file_key = f"{name}_file"
                if optim_file_key in state_info:
                    optim_path = os.path.join(load_path_str, state_info[optim_file_key])
                    if os.path.exists(optim_path):
                        try:
                            optim.load_state_dict(
                                torch.load(optim_path, map_location=device)
                            )
                            logger.info(
                                f"{name.replace('_', ' ').title()} state loaded from {optim_path}."
                            )
                        except Exception as load_err:
                            logger.warning(
                                f"Could not load optimizer state for {name} from {optim_path}: {load_err}. Optimizer will be reinitialized."
                            )
                    else:
                        logger.warning(
                            f"{name.replace('_', ' ').title()} state file not found: {optim_path}. Optimizer will be reinitialized."
                        )
                else:
                    logger.warning(
                        f"Optimizer state file key '{optim_file_key}' not found in config. Optimizer will be reinitialized."
                    )

            logger.info("Optimizers reconstructed.")

        except (
            ImportError,
            AttributeError,
            KeyError,
            FileNotFoundError,
            Exception,
        ) as e:
            logger.error(
                f"Error reconstructing or loading optimizers: {e}", exc_info=True
            )
            return None

        # --- Reconstruct Environment ---
        try:
            # Assuming Environment class can be instantiated without arguments
            # If it needs args, they might need to be saved/loaded too
            env = Environment()
            logger.info("Environment instantiated.")
        except Exception as e:
            logger.error(f"Failed to instantiate Environment: {e}", exc_info=True)
            # Decide if this is critical - maybe return None or continue without env?
            return None  # Likely critical for the agent

        # --- Instantiate Agent with loaded state ---
        try:
            agent = cls(
                hero_actor_net=hero_actor_net,
                hero_critic_net=hero_critic_net,
                gun_actor_net=gun_actor_net,
                gun_critic_net=gun_critic_net,
                env=env,  # Pass the newly created env instance
                optimizer_class=OptimizerClass,
                learning_rate=learning_rate,  # LR is set in optimizers already, but good to pass for consistency
                discount_factor=state_info.get(
                    "discount_factor", DEFAULT_DISCOUNT_FACTOR_PPO
                ),
                horizon=state_info.get("horizon", DEFAULT_HORIZON),
                epochs_per_update=state_info.get(
                    "epochs_per_update", DEFAULT_EPOCHS_PER_UPDATE
                ),
                mini_batch_size=state_info.get(
                    "mini_batch_size", DEFAULT_MINI_BATCH_SIZE_PPO
                ),
                clip_epsilon=state_info.get("clip_epsilon", DEFAULT_CLIP_EPSILON),
                gae_lambda=state_info.get("gae_lambda", DEFAULT_GAE_LAMBDA),
                entropy_coeff=state_info.get("entropy_coeff", DEFAULT_ENTROPY_COEFF),
                vf_coeff=state_info.get("vf_coeff", DEFAULT_VF_COEFF),
                log_window_size=state_info.get("log_window_size", LOGGING_WINDOW),
                save_interval=state_info.get("save_interval", SAVE_INTERVAL),
            )

            # --- Restore specific state variables ---
            # Re-assign the loaded optimizer instances to the agent
            agent.hero_actor_optimizer = hero_actor_optimizer
            agent.hero_critic_optimizer = hero_critic_optimizer
            agent.gun_actor_optimizer = gun_actor_optimizer
            agent.gun_critic_optimizer = gun_critic_optimizer

            # Restore training progress metrics
            agent.total_reward_hero = state_info.get("total_reward_hero", 0.0)
            agent.total_reward_gun = state_info.get("total_reward_gun", 0.0)
            # Note: Rolling deques (like episode_rewards_hero_deque) are usually not saved/loaded
            # as they represent recent history, which becomes irrelevant after loading.
            # total_steps is reset after each update, so no need to load.

            # Ensure networks are in train mode after loading (PPO alternates eval/train)
            agent.hero_actor_net.train()
            agent.hero_critic_net.train()
            agent.gun_actor_net.train()
            agent.gun_critic_net.train()

            logger.info(f"PPO Agent loaded successfully from {load_path_str}")
            return agent

        except Exception as e:
            logger.error(
                f"Error instantiating AgentTheseusPPO during final load step: {e}",
                exc_info=True,
            )
            return None
