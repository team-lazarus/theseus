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
# Ensure these imports point to the actual classes/functions in your project
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
    from theseus.utils import State # Assuming State is defined here
    from theseus.utils.network import Environment # Assuming Environment is defined here
    import theseus.constants as c
    # Assuming GNNs are defined and accept hidden_channels
    from theseus.models.GraphDQN.ActionGNN import HeroGNN as HeroBaseGNN
    from theseus.models.GraphDQN.ActionGNN import GunGNN as GunBaseGNN

    # Define specific Actor/Critic classes or reuse if identical structure
    HeroActorGNN = HeroBaseGNN
    HeroCriticGNN = HeroBaseGNN # Adjust if Critic has different output head
    GunActorGNN = GunBaseGNN
    GunCriticGNN = GunBaseGNN # Adjust if Critic has different output head

except ImportError as e:
    # Setup basic logging if imports fail, to see the error
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import necessary libraries: {e}. Please ensure all dependencies are installed and paths are correct.")
    # Re-raise the error to stop execution if critical imports are missing
    raise ImportError(f"Critical import failed: {e}") from e


# Define constants (can be moved to a config file)
HERO_ACTION_SPACE_SIZE: int = 9 # Example
GUN_ACTION_SPACE_SIZE: int = 8 # Example
LOGGING_WINDOW: int = 50
SAVE_INTERVAL: int = 5 # Episodes
DEFAULT_HORIZON: int = 2048 # Steps per rollout collection
DEFAULT_EPOCHS_PER_UPDATE: int = 10
DEFAULT_MINI_BATCH_SIZE_PPO: int = 64
DEFAULT_CLIP_EPSILON: float = 0.2
DEFAULT_GAE_LAMBDA: float = 0.95
DEFAULT_ENTROPY_COEFF: float = 0.01
DEFAULT_VF_COEFF: float = 0.5
DEFAULT_LEARNING_RATE_PPO: float = 3e-4
DEFAULT_DISCOUNT_FACTOR_PPO: float = 0.99
DEFAULT_HERO_HIDDEN_CHANNELS: int = 64 # Example default
DEFAULT_GUN_HIDDEN_CHANNELS: int = 64 # Example default

# Structure to hold trajectory data
TrajectoryStep = namedtuple("TrajectoryStep", [
    'state_graph_hero', 'state_graph_gun', # Preprocessed graphs
    'move_action', 'shoot_action',
    'move_log_prob', 'shoot_log_prob',
    'hero_value', 'gun_value',
    'hero_reward', 'gun_reward',
    'terminated'
])


class AgentTheseusPPO:
    """
    Agent managing simultaneous training of Hero and Gun GNNs using PPO.

    This agent uses separate Actor and Critic GNNs for hero movement and gun
    actions. It collects trajectories on-policy, calculates advantages using GAE,
    and updates networks using the PPO clipped surrogate objective and value loss.
    It also logs episode metrics to a CSV file during checkpointing.

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
        log_window_size: Number of episodes for calculating rolling averages for console logging.
        save_interval: Frequency (in episodes) for saving model checkpoints and metrics CSV.
        hero_hidden_channels: Number of hidden channels for Hero networks (used for saving/loading).
        gun_hidden_channels: Number of hidden channels for Gun networks (used for saving/loading).

    Attributes:
        (Includes standard PPO attributes and metrics tracking)
        trajectory_buffer: Stores the steps collected during a rollout.
        total_steps: Counter for steps within the current rollout horizon.
        training_summary_data: List storing dicts of episode metrics for CSV logging.
        hero_hidden_channels: Stored architecture parameter.
        gun_hidden_channels: Stored architecture parameter.
        ... (other attributes)
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
        hero_hidden_channels: int = DEFAULT_HERO_HIDDEN_CHANNELS,
        gun_hidden_channels: int = DEFAULT_GUN_HIDDEN_CHANNELS,
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
        # Store hidden channels for saving config
        self.hero_hidden_channels = hero_hidden_channels
        self.gun_hidden_channels = gun_hidden_channels

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
        self.hero_actor_optimizer: optim.Optimizer = optimizer_class(
            self.hero_actor_net.parameters(), lr=learning_rate, eps=1e-5
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
        self.critic_loss_fn: nn.Module = nn.MSELoss()

        # --- Data Collection ---
        self.trajectory_buffer: List[TrajectoryStep] = []
        self.total_steps: int = 0 # Steps collected in the current rollout

        # --- Metrics Tracking ---
        self.log_window_size: int = log_window_size
        self.save_interval: int = save_interval
        self.episode_rewards_hero_deque: Deque[float] = deque(maxlen=self.log_window_size)
        self.episode_rewards_gun_deque: Deque[float] = deque(maxlen=self.log_window_size)
        self.episode_time_alive_deque: Deque[int] = deque(maxlen=self.log_window_size)
        self.total_reward_hero: float = 0.0 # Cumulative reward across all training time
        self.total_reward_gun: float = 0.0  # Cumulative reward across all training time
        # PPO specific metrics (rolling averages for console logging)
        self.hero_actor_loss_deque: Deque[float] = deque(maxlen=self.log_window_size * epochs_per_update)
        self.hero_critic_loss_deque: Deque[float] = deque(maxlen=self.log_window_size * epochs_per_update)
        self.gun_actor_loss_deque: Deque[float] = deque(maxlen=self.log_window_size * epochs_per_update)
        self.gun_critic_loss_deque: Deque[float] = deque(maxlen=self.log_window_size * epochs_per_update)
        self.entropy_hero_deque: Deque[float] = deque(maxlen=self.log_window_size * epochs_per_update)
        self.entropy_gun_deque: Deque[float] = deque(maxlen=self.log_window_size * epochs_per_update)

        # List to store episode summary data for CSV logging and final table
        self.training_summary_data: List[Dict[str, Union[int, float]]] = []
        self.console = Console()

        self.current_state: Optional[State] = None # Track current state across steps/episodes

    def _validate_network(self, network: nn.Module, name: str) -> None:
        """Checks if a network has the required 'preprocess_state' method."""
        if not hasattr(network, "preprocess_state") or not callable(
            getattr(network, "preprocess_state", None)
        ):
            raise AttributeError(
                f"{name} network must have a callable 'preprocess_state' method."
            )

    def _update_metrics(self, ep_reward_hero: float, ep_reward_gun: float, time_alive: int) -> None:
        """Updates rolling and cumulative reward metrics after an episode."""
        self.episode_rewards_hero_deque.append(ep_reward_hero)
        self.episode_rewards_gun_deque.append(ep_reward_gun)
        self.episode_time_alive_deque.append(time_alive)
        self.total_reward_hero += ep_reward_hero # Track overall cumulative reward
        self.total_reward_gun += ep_reward_gun

    def _log_episode_metrics(self, episode: int, steps: int) -> None:
        """Logs key performance metrics for the completed episode to the console."""
        avg_rew_hero = np.mean(self.episode_rewards_hero_deque) if self.episode_rewards_hero_deque else 0.0
        avg_rew_gun = np.mean(self.episode_rewards_gun_deque) if self.episode_rewards_gun_deque else 0.0
        avg_time_alive = np.mean(self.episode_time_alive_deque) if self.episode_time_alive_deque else 0.0

        avg_loss_actor_h = np.mean(self.hero_actor_loss_deque) if self.hero_actor_loss_deque else float("nan")
        avg_loss_critic_h = np.mean(self.hero_critic_loss_deque) if self.hero_critic_loss_deque else float("nan")
        avg_loss_actor_g = np.mean(self.gun_actor_loss_deque) if self.gun_actor_loss_deque else float("nan")
        avg_loss_critic_g = np.mean(self.gun_critic_loss_deque) if self.gun_critic_loss_deque else float("nan")
        avg_entropy_h = np.mean(self.entropy_hero_deque) if self.entropy_hero_deque else float("nan")
        avg_entropy_g = np.mean(self.entropy_gun_deque) if self.entropy_gun_deque else float("nan")

        metrics_list = [
            f"TimeAlive={steps}",
            f"AvgTimeAlive={avg_time_alive:.2f}",
            f"AvgR_Hero={avg_rew_hero:.3f}",
            f"AvgR_Gun={avg_rew_gun:.3f}",
            # f"CumR_Hero={self.total_reward_hero:.2f}", # Total cumulative can get very large
            # f"CumR_Gun={self.total_reward_gun:.2f}",
            f"ALoss_H={avg_loss_actor_h:.4f}", f"CLoss_H={avg_loss_critic_h:.4f}",
            f"ALoss_G={avg_loss_actor_g:.4f}", f"CLoss_G={avg_loss_critic_g:.4f}",
            f"Entropy_H={avg_entropy_h:.3f}", f"Entropy_G={avg_entropy_g:.3f}",
            f"RolloutProg={self.total_steps}/{self.horizon}",
        ]
        # Use 1-based episode number for logging display
        log_str = f"Ep {episode + 1} Summary | " + " | ".join(metrics_list)
        self.logger.info(log_str)

    def train(self, num_episodes: Optional[int] = None) -> None:
        """Runs the main PPO training loop."""
        self.logger.info(
            f"Starting PPO training on {self.device} for {num_episodes or 'infinite'} episodes..."
        )
        self.logger.info(f"Collect Horizon: {self.horizon} steps")
        self.logger.info(f"Save interval: Every {self.save_interval} episodes")
        # Reset episode data storage at the start of training
        self.training_summary_data = []

        # Reset environment state at the beginning of training
        self.current_state = self._initialize_episode()
        if self.current_state is None:
             self.logger.critical("Initial environment reset failed. Stopping training.")
             return

        progress_columns = [
            TextColumn("[progress.description]{task.description}"), BarColumn(),
            MofNCompleteColumn(), TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            TimeRemainingColumn(),
        ]
        total_episodes_for_progress = num_episodes
        if num_episodes is None:
            progress_columns = [
                TextColumn("[progress.description]{task.description}"), BarColumn(),
                TextColumn("Episode {task.completed}"), # Display completed count directly
            ]
            total_episodes_for_progress = None # Indicate infinite run to Progress
            self.logger.warning("Training infinitely. Progress bar will not show total or ETA.")

        with Progress(*progress_columns, transient=False) as progress:
            episode_task: TaskID = progress.add_task(
                "[cyan]Training Episodes...", total=total_episodes_for_progress
            )
            # Use 0-based internal episode counter, display 1-based
            episode_iterator = range(num_episodes) if num_episodes is not None else count()
            completed_episodes = 0

            try:
                for episode in episode_iterator:
                    # Run steps until horizon or episode end
                    ep_reward_hero, ep_reward_gun, time_alive, terminated, truncated = self._run_episode_or_rollout(
                        episode, progress, episode_task
                    )

                    # --- Logging and Metrics (only if episode finished) ---
                    if terminated or truncated:
                        self._update_metrics(ep_reward_hero, ep_reward_gun, time_alive)
                        self._log_episode_metrics(episode, time_alive) # Pass 0-based index

                        # Store data for CSV and final summary table
                        if time_alive > 0: # Avoid adding data for zero-step episodes
                            self.training_summary_data.append({
                                "Episode": episode + 1, # Store 1-based index for user display
                                "Time_Alive": time_alive,
                                "Reward_Hero": ep_reward_hero, # Cumulative reward for this episode
                                "Reward_Gun": ep_reward_gun,   # Cumulative reward for this episode
                            })
                        else:
                             self.logger.warning(f"Episode {episode+1} ended with 0 time alive. Not adding to summary data.")

                        # Checkpoint based on completed episode count (1-based)
                        self._save_checkpoint_if_needed(episode + 1)
                        progress.update(episode_task, advance=1)
                        completed_episodes += 1

                    # --- Learning Step Trigger (based on collected steps) ---
                    if self.total_steps >= self.horizon:
                        self.logger.info(f"Horizon {self.horizon} reached. Starting PPO update.")
                        self._update_policy() # Perform PPO update using trajectory buffer
                        # Update progress description after policy update completes
                        progress.update(episode_task, description=f"[cyan]Ep. {episode+1} (Updating Policy...)")


            except RuntimeError as e:
                self.logger.critical(f"Stopping training due to runtime error in episode {episode+1}: {e}", exc_info=True)
            except KeyboardInterrupt:
                 self.logger.warning("Training interrupted by user.")
            except Exception as e:
                self.logger.critical(f"Unexpected error during episode {episode+1}: {e}", exc_info=True)
            finally:
                progress.stop()
                # Final progress bar description
                if num_episodes is not None:
                    final_desc = "[green]Training Finished" if completed_episodes == num_episodes else "[yellow]Training Stopped Early"
                    progress.update(episode_task, description=final_desc, completed=completed_episodes)
                else:
                    final_desc = "[yellow]Training Stopped (Infinite Mode)"
                    progress.update(episode_task, description=final_desc) # Completed count is already tracked

                # Display summary table and save final CSV if any episodes ran
                if completed_episodes > 0:
                    self._display_training_summary(completed_episodes)
                    self.logger.info("Attempting to save final episode metrics CSV...")
                    last_save_dir = self._get_last_save_directory()
                    if last_save_dir:
                         self._save_episode_metrics_csv(last_save_dir)
                         self.logger.info(f"Final metrics CSV saved in: {last_save_dir}")
                    else:
                         # Optionally save to a default location if no checkpoints were made
                         final_save_dir = os.path.join("model_saves_ppo", "final_run_metrics")
                         os.makedirs(final_save_dir, exist_ok=True)
                         self._save_episode_metrics_csv(final_save_dir)
                         self.logger.warning(f"No checkpoint directory found. Final metrics saved to: {final_save_dir}")

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
                   These values represent the outcome of the segment run within this call.
                   If the horizon was reached mid-episode, rewards/steps are partial for that ep.
        """
        segment_reward_hero: float = 0.0
        segment_reward_gun: float = 0.0
        segment_steps: int = 0
        terminated: bool = False
        truncated: bool = False

        # Start from the current state (might be from a previous step/episode)
        if self.current_state is None:
            self.current_state = self._initialize_episode()
            if self.current_state is None:
                raise RuntimeError("Failed to get initial state for episode/rollout.")
            self.logger.debug(f"Episode {episode_num + 1} segment started.")

        # Loop until horizon is met or episode ends
        # Loop condition ensures we don't exceed horizon steps within this call
        while self.total_steps < self.horizon:
            if self.current_state is None: # Should not happen if initialized correctly
                 self.logger.error("Critical: current_state became None during rollout.")
                 terminated = True # Force stop this segment
                 break

            # --- Preprocess State for GNNs ---
            try:
                with torch.no_grad(): # No gradients needed for data collection forward passes
                    graph_hero_actor = self.hero_actor_net.preprocess_state(self.current_state)
                    graph_gun_actor = self.gun_actor_net.preprocess_state(self.current_state)
                    # Assume critics use the same preprocessing
                    graph_hero_critic = self.hero_critic_net.preprocess_state(self.current_state)
                    graph_gun_critic = self.gun_critic_net.preprocess_state(self.current_state)

                if graph_hero_actor is None or graph_gun_actor is None or \
                   graph_hero_critic is None or graph_gun_critic is None:
                    self.logger.warning(f"Preprocessing failed at step {segment_steps} in Ep {episode_num + 1}. Ending segment.")
                    terminated = True # Treat as failure
                    break

                # Move graphs to device once
                graph_hero_actor = graph_hero_actor.to(self.device)
                graph_gun_actor = graph_gun_actor.to(self.device)
                graph_hero_critic = graph_hero_critic.to(self.device)
                graph_gun_critic = graph_gun_critic.to(self.device)

            except Exception as e:
                self.logger.error(f"Error preprocessing state in Ep {episode_num + 1}, Step {segment_steps}: {e}", exc_info=True)
                terminated = True # Critical failure
                break

            # --- Select Actions and Get Values ---
            with torch.no_grad():
                move_action, move_log_prob = self._sample_action(self.hero_actor_net, graph_hero_actor)
                shoot_action, shoot_log_prob = self._sample_action(self.gun_actor_net, graph_gun_actor)
                hero_value = self._get_value(self.hero_critic_net, graph_hero_critic)
                gun_value = self._get_value(self.gun_critic_net, graph_gun_critic)

            if move_action is None or shoot_action is None or hero_value is None or gun_value is None:
                 self.logger.warning(f"Action sampling or value estimation failed in Ep {episode_num + 1}. Ending segment.")
                 terminated = True
                 break

            # --- Step Environment ---
            step_result = self._step_environment(move_action, shoot_action)
            next_state, reward_hero, reward_gun, terminated, truncated = step_result

            # --- Store Transition ---
            # Store graphs on CPU to save GPU memory if buffer gets large
            step_data = TrajectoryStep(
                state_graph_hero=graph_hero_actor.cpu(),
                state_graph_gun=graph_gun_actor.cpu(),
                move_action=move_action,
                shoot_action=shoot_action,
                move_log_prob=move_log_prob.cpu(),
                shoot_log_prob=shoot_log_prob.cpu(),
                hero_value=hero_value.cpu(),
                gun_value=gun_value.cpu(),
                hero_reward=reward_hero,
                gun_reward=reward_gun,
                terminated=terminated # Store terminated flag for GAE calculation
            )
            self.trajectory_buffer.append(step_data)

            # --- Update State and Counters ---
            self.current_state = next_state
            self.total_steps += 1     # Increment global step counter for horizon check
            segment_steps += 1        # Increment step counter for this segment/episode
            segment_reward_hero += reward_hero
            segment_reward_gun += reward_gun

            # Update progress bar description periodically
            if segment_steps % 20 == 0:
                avg_r_hero_disp = np.mean(self.episode_rewards_hero_deque) if self.episode_rewards_hero_deque else 0.0
                avg_r_gun_disp = np.mean(self.episode_rewards_gun_deque) if self.episode_rewards_gun_deque else 0.0
                progress.update(
                    task_id,
                    description=(
                        f"[cyan]Ep. {episode_num + 1}[/cyan] [yellow]Step {segment_steps}[/yellow] "
                        f"| Rollout: [b]{self.total_steps}/{self.horizon}[/b] "
                        f"| AvgR Gun(win): [b]{avg_r_gun_disp:.2f}[/b] " # Rolling avg
                        f"| AvgR Hero(win): [b]{avg_r_hero_disp:.2f}[/b]" # Rolling avg
                    ),
                )

            # --- Check for Episode End (Termination or Truncation) ---
            if terminated or truncated:
                self.logger.debug(f"Episode {episode_num + 1} ended at step {segment_steps} ({'Terminated' if terminated else 'Truncated'}). Rollout steps: {self.total_steps}/{self.horizon}")
                # Reset environment for the *next* episode/segment start
                self.current_state = self._initialize_episode()
                if self.current_state is None:
                    self.logger.error("Failed to reset env after episode end. Training might halt.")
                    # The outer loop's None check will likely catch this
                break # Exit the inner loop (rollout step loop) because episode ended

            # Horizon check moved to the 'while' condition

        # Return rewards/steps accumulated *during this call* and the final status
        return segment_reward_hero, segment_reward_gun, segment_steps, terminated, truncated


    def _initialize_episode(self) -> Optional[State]:
        """Resets the environment and returns the initial state."""
        try:
            initial_state: State = self.env.initialise_environment()
            # Basic check if state looks valid (adapt as needed)
            if not isinstance(initial_state, State): # Use the specific State class from your imports
                raise TypeError(f"Env did not return expected State object, got {type(initial_state)}")
            return initial_state
        except Exception as e:
            self.logger.error(f"Failed to initialize environment: {e}", exc_info=True)
            return None

    def _sample_action(self, actor_net: nn.Module, state_graph: Union[HeteroData, Batch]) -> Tuple[Optional[int], Optional[torch.Tensor]]:
        """Samples an action from the actor network's output distribution."""
        try:
            actor_net.eval() # Set to eval mode for sampling
            logits: torch.Tensor = actor_net(state_graph)
            actor_net.train() # Set back to train mode immediately

            # Ensure logits tensor is valid
            if logits.numel() == 0:
                self.logger.warning(f"{type(actor_net).__name__} produced empty logits.")
                return None, None
            if logits.ndim > 1 and logits.shape[0] == 1: # Remove batch dim if present (expected size 1)
                logits = logits.squeeze(0)
            elif logits.ndim > 1 and logits.shape[0] != 1:
                 self.logger.warning(f"{type(actor_net).__name__} received unexpected batch size > 1 for sampling: {logits.shape}. Using first element.")
                 logits = logits[0]
            elif logits.ndim == 0: # Should be 1D for Categorical
                 self.logger.warning(f"{type(actor_net).__name__} produced scalar output instead of logits vector.")
                 return None, None


            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return action.item(), log_prob

        except Exception as e:
            self.logger.error(f"Error sampling action with {type(actor_net).__name__}: {e}", exc_info=True)
            return None, None

    def _get_value(self, critic_net: nn.Module, state_graph: Union[HeteroData, Batch]) -> Optional[torch.Tensor]:
        """Gets the value estimate from the critic network."""
        try:
            critic_net.eval() # Set to eval mode for value estimation
            value: torch.Tensor = critic_net(state_graph)
            critic_net.train() # Set back to train mode

            # Ensure value is a scalar tensor
            if value.numel() != 1:
                 self.logger.warning(f"{type(critic_net).__name__} produced non-scalar value: shape {value.shape}. Taking first element.")
                 value = value.flatten()[0]

            return value.squeeze() # Ensure it's tensor(value) not tensor([value])

        except Exception as e:
            self.logger.error(f"Error getting value with {type(critic_net).__name__}: {e}", exc_info=True)
            return None


    def _step_environment(
        self, move_action: int, shoot_action: int
    ) -> Tuple[Optional[State], float, float, bool, bool]:
        """Steps the environment with the chosen actions."""
        try:
            # Combine actions into the format expected by env.step
            # Adjust this based on your environment's action spec
            combined_action_list: List[int] = [move_action, shoot_action, 0, 0] # Example format
            step_result: Tuple = self.env.step(combined_action_list)

            # --- Unpack and Validate Step Result ---
            # Adapt this based on what your env.step actually returns
            if not isinstance(step_result, tuple) or len(step_result) < 3:
                 raise TypeError(f"Env step returned unexpected format: {type(step_result)}, value: {step_result}")

            # Assuming format: (next_state, (reward_hero, reward_gun), terminated, [truncated], [info])
            next_s = step_result[0]
            reward_tuple = step_result[1]
            terminated_flag = step_result[2]

            # Handle optional truncated flag (common in newer Gym/Gymnasium)
            truncated_flag = False
            if len(step_result) > 3 and isinstance(step_result[3], bool):
                truncated_flag = step_result[3]
            elif len(step_result) > 3 and not isinstance(step_result[3], bool):
                 # If a 4th element exists but isn't bool, it might be 'info' dict
                 pass # Ignore info for now

            # --- Validate State ---
            next_state: Optional[State] = None
            # Only expect a State object if not terminated/truncated
            if not bool(terminated_flag) and not bool(truncated_flag):
                if isinstance(next_s, State): # Use your actual State class
                    next_state = next_s
                else:
                    # Log error if next state is invalid when episode should continue
                    self.logger.error(f"Env step returned invalid non-terminal state type: {type(next_s)}")
                    # Treat as critical failure -> terminate episode
                    return None, 0.0, 0.0, True, True
            # If terminated/truncated, next_s might be None or a final state, handle as episode end

            # --- Validate Rewards ---
            if not isinstance(reward_tuple, (tuple, list)) or len(reward_tuple) < 2:
                 self.logger.error(f"Env step returned invalid reward format: {reward_tuple}. Using [0, 0].")
                 reward_hero, reward_gun = 0.0, 0.0
            else:
                 # Convert rewards to float, handle potential None values
                 reward_hero = float(0 if reward_tuple[0] is None else reward_tuple[0])
                 reward_gun = float(0 if reward_tuple[1] is None else reward_tuple[1])

            # --- Convert Flags ---
            terminated: bool = bool(terminated_flag)
            truncated: bool = bool(truncated_flag)

            return next_state, reward_hero, reward_gun, terminated, truncated

        except Exception as e:
            self.logger.error(f"Error during environment step: {e}", exc_info=True)
            # Return critical failure state
            return None, 0.0, 0.0, True, True


    def _calculate_gae_and_returns(self, last_state: Optional[State], last_terminated: bool, last_truncated: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates Generalized Advantage Estimation (GAE) and returns (value targets).
        Processes the stored trajectory_buffer.

        Args:
             last_state: The state *after* the last step in the buffer. Needed for bootstrapping if not terminal.
             last_terminated: Whether the episode terminated at the end of the buffer.
             last_truncated: Whether the episode truncated at the end of the buffer (implies not terminated).

        Returns:
            Tuple: (hero_advantages, hero_returns, gun_advantages, gun_returns) all as tensors on self.device.
        """
        hero_advantages_list = []
        gun_advantages_list = []
        hero_last_gae_lam = 0.0
        gun_last_gae_lam = 0.0

        # --- Get value of the last state for bootstrapping ---
        last_hero_value = 0.0
        last_gun_value = 0.0
        # Bootstrap only if the episode didn't end *terminally* (termination OR truncation might end the rollout)
        should_bootstrap = not last_terminated # If terminated, value is 0. If truncated, bootstrap.
        if should_bootstrap and last_state is not None:
            try:
                with torch.no_grad():
                    # Preprocess the very last state observed
                    graph_h_last = self.hero_critic_net.preprocess_state(last_state)
                    graph_g_last = self.gun_critic_net.preprocess_state(last_state)
                    if graph_h_last is not None and graph_g_last is not None:
                        graph_h_last = graph_h_last.to(self.device)
                        graph_g_last = graph_g_last.to(self.device)
                        val_h = self._get_value(self.hero_critic_net, graph_h_last)
                        val_g = self._get_value(self.gun_critic_net, graph_g_last)
                        # Ensure values were successfully retrieved
                        if val_h is not None: last_hero_value = val_h.item()
                        if val_g is not None: last_gun_value = val_g.item()
                        # self.logger.debug(f"GAE Bootstrap values: Hero={last_hero_value:.4f}, Gun={last_gun_value:.4f}")
                    else:
                        self.logger.warning("Preprocessing failed for GAE bootstrap state. Using value 0.")

            except Exception as e:
                self.logger.warning(f"Error getting value for GAE bootstrap state: {e}. Using value 0.")
        # else:
            # self.logger.debug(f"GAE: No bootstrap needed (Terminated={last_terminated}) or last_state is None.")


        # --- Iterate backwards through the trajectory buffer ---
        num_steps = len(self.trajectory_buffer)
        for i in reversed(range(num_steps)):
            step = self.trajectory_buffer[i]
            current_hero_value = step.hero_value.item()
            current_gun_value = step.gun_value.item()
            reward_hero = step.hero_reward
            reward_gun = step.gun_reward
            # Mask is 1.0 if *not* terminated, 0.0 if terminated
            # This prevents bootstrapping rewards beyond a terminal state encountered mid-trajectory
            terminated_mask = 1.0 - float(step.terminated)

            # Calculate delta (TD error) using the *next* state's value (last_hero_value from previous iteration)
            delta_hero = reward_hero + self.discount_factor * last_hero_value * terminated_mask - current_hero_value
            delta_gun = reward_gun + self.discount_factor * last_gun_value * terminated_mask - current_gun_value

            # Calculate GAE advantage for this step
            # Adv_t = delta_t + gamma * lambda * Adv_{t+1} * mask_{t+1}
            adv_hero = delta_hero + self.discount_factor * self.gae_lambda * hero_last_gae_lam * terminated_mask
            adv_gun = delta_gun + self.discount_factor * self.gae_lambda * gun_last_gae_lam * terminated_mask

            hero_advantages_list.append(adv_hero)
            gun_advantages_list.append(adv_gun)

            # Update GAE lambda term and value for the *next* (previous in time) iteration
            hero_last_gae_lam = adv_hero
            gun_last_gae_lam = adv_gun
            last_hero_value = current_hero_value # The current step's value becomes the 'next' value in the next iteration
            last_gun_value = current_gun_value

        # Reverse advantages list to match original trajectory order
        hero_advantages_list.reverse()
        gun_advantages_list.reverse()

        # Convert advantages to tensors
        hero_adv_tensor = torch.tensor(hero_advantages_list, dtype=torch.float32, device=self.device)
        gun_adv_tensor = torch.tensor(gun_advantages_list, dtype=torch.float32, device=self.device)

        # --- Calculate returns (value targets) ---
        # Returns = Advantages + Values (from the trajectory buffer)
        values_hero = torch.stack([step.hero_value for step in self.trajectory_buffer]).to(self.device).squeeze()
        values_gun = torch.stack([step.gun_value for step in self.trajectory_buffer]).to(self.device).squeeze()

        # Ensure dimensions match if squeeze removed a dimension unnecessarily
        if values_hero.ndim == 0 and hero_adv_tensor.ndim == 1:
            values_hero = values_hero.unsqueeze(0)
        if values_gun.ndim == 0 and gun_adv_tensor.ndim == 1:
            values_gun = values_gun.unsqueeze(0)

        hero_returns = hero_adv_tensor + values_hero
        gun_returns = gun_adv_tensor + values_gun

        # --- Normalize Advantages (optional but generally recommended for PPO) ---
        # Normalize across the batch of advantages from this rollout
        hero_adv_tensor = (hero_adv_tensor - hero_adv_tensor.mean()) / (hero_adv_tensor.std() + 1e-8)
        gun_adv_tensor = (gun_adv_tensor - gun_adv_tensor.mean()) / (gun_adv_tensor.std() + 1e-8)

        return hero_adv_tensor, hero_returns, gun_adv_tensor, gun_returns


    def _update_policy(self) -> None:
        """Performs the PPO learning update using the collected trajectory buffer."""
        if not self.trajectory_buffer:
            self.logger.warning("Attempted PPO update with empty trajectory buffer.")
            return

        # Determine the status of the *last* step in the buffer for GAE calculation
        # We need the state *after* this last step for potential bootstrapping.
        last_state = self.current_state # This holds the state after the final step of the rollout
        last_step_info = self.trajectory_buffer[-1]
        # 'terminated' in the last step tells us if *that specific step* ended the episode terminally.
        last_terminated = last_step_info.terminated
        # We infer 'truncated' based on whether the horizon was reached *without* termination.
        # If the buffer is full (total_steps == horizon) and the last step wasn't terminal, it was truncated.
        # This assumes the environment correctly sets 'terminated=True' on the actual final step.
        # last_truncated = (self.total_steps == self.horizon) and (not last_terminated)
        # Simplified: Bootstrap if the last step wasn't terminal. GAE calculation handles the mask.
        last_truncated = not last_terminated # For bootstrapping logic in GAE func

        # --- 1. Calculate Advantages and Returns ---
        try:
            hero_advantages, hero_returns, gun_advantages, gun_returns = self._calculate_gae_and_returns(
                last_state, last_terminated, last_truncated # Pass status of the end of the buffer
            )
            # self.logger.debug(f"GAE/Returns calculated. Shape Adv_H: {hero_advantages.shape}, Ret_H: {hero_returns.shape}")
        except Exception as e:
            self.logger.error(f"Error calculating GAE/Returns: {e}", exc_info=True)
            self.trajectory_buffer.clear() # Clear buffer on critical error
            self.total_steps = 0
            return

        # --- 2. Prepare Data for Optimization Epochs ---
        # Extract data from buffer (move relevant tensors back to device if needed)
        # Graphs are stored on CPU, actions/logprobs need converting
        hero_graphs = [step.state_graph_hero for step in self.trajectory_buffer] # List of HeteroData on CPU
        gun_graphs = [step.state_graph_gun for step in self.trajectory_buffer]   # List of HeteroData on CPU
        move_actions = torch.tensor([step.move_action for step in self.trajectory_buffer], dtype=torch.long, device=self.device)
        shoot_actions = torch.tensor([step.shoot_action for step in self.trajectory_buffer], dtype=torch.long, device=self.device)
        # Old log probs were calculated on device, stored on CPU, move back
        old_move_log_probs = torch.stack([step.move_log_prob for step in self.trajectory_buffer]).to(self.device).squeeze()
        old_shoot_log_probs = torch.stack([step.shoot_log_prob for step in self.trajectory_buffer]).to(self.device).squeeze()

        # Ensure dimensions are consistent (in case buffer had size 1)
        if old_move_log_probs.ndim == 0: old_move_log_probs = old_move_log_probs.unsqueeze(0)
        if old_shoot_log_probs.ndim == 0: old_shoot_log_probs = old_shoot_log_probs.unsqueeze(0)


        data_size = len(self.trajectory_buffer)
        indices = np.arange(data_size)

        # --- 3. Optimization Loop (Multiple Epochs over the Collected Data) ---
        # self.logger.debug(f"Starting PPO optimization ({self.epochs_per_update} epochs, {data_size} samples, {self.mini_batch_size} batch size)")

        # Put networks in training mode for the update phase
        self.hero_actor_net.train()
        self.hero_critic_net.train()
        self.gun_actor_net.train()
        self.gun_critic_net.train()

        for epoch in range(self.epochs_per_update):
            np.random.shuffle(indices) # Shuffle data each epoch
            epoch_actor_loss_h, epoch_critic_loss_h, epoch_entropy_h = 0.0, 0.0, 0.0
            epoch_actor_loss_g, epoch_critic_loss_g, epoch_entropy_g = 0.0, 0.0, 0.0
            num_batches = 0

            for start in range(0, data_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]
                num_batches += 1

                # --- Get Minibatch Data ---
                # Batch graphs for the minibatch
                # Graphs need to be moved to device here
                try:
                    mb_hero_graphs_list = [hero_graphs[i].to(self.device) for i in mb_indices]
                    mb_gun_graphs_list = [gun_graphs[i].to(self.device) for i in mb_indices]
                    mb_hero_graphs_batch = Batch.from_data_list(mb_hero_graphs_list)
                    mb_gun_graphs_batch = Batch.from_data_list(mb_gun_graphs_list)
                except Exception as e:
                     self.logger.error(f"Error creating PyG Batch for minibatch (Epoch {epoch}, Start {start}): {e}", exc_info=True)
                     continue # Skip this minibatch

                # Slice tensors using minibatch indices
                mb_move_actions = move_actions[mb_indices]
                mb_shoot_actions = shoot_actions[mb_indices]
                mb_old_move_log_probs = old_move_log_probs[mb_indices]
                mb_old_shoot_log_probs = old_shoot_log_probs[mb_indices]
                mb_hero_advantages = hero_advantages[mb_indices]
                mb_gun_advantages = gun_advantages[mb_indices]
                mb_hero_returns = hero_returns[mb_indices]
                mb_gun_returns = gun_returns[mb_indices]

                try:
                    # --- Forward pass for current policy/value estimates on Minibatch ---
                    # Hero
                    hero_logits = self.hero_actor_net(mb_hero_graphs_batch)
                    hero_dist = Categorical(logits=hero_logits)
                    new_move_log_probs = hero_dist.log_prob(mb_move_actions)
                    hero_entropy = hero_dist.entropy().mean() # Average entropy over minibatch
                    # Critic value prediction needs to match return shape ([B])
                    hero_values = self.hero_critic_net(mb_hero_graphs_batch).squeeze()

                    # Gun
                    gun_logits = self.gun_actor_net(mb_gun_graphs_batch)
                    gun_dist = Categorical(logits=gun_logits)
                    new_shoot_log_probs = gun_dist.log_prob(mb_shoot_actions)
                    gun_entropy = gun_dist.entropy().mean() # Average entropy over minibatch
                    # Critic value prediction needs to match return shape ([B])
                    gun_values = self.gun_critic_net(mb_gun_graphs_batch).squeeze()


                    # --- Calculate Hero Losses ---
                    # Actor Loss (PPO Clipped Surrogate Objective)
                    # ratio = exp(log_prob_new - log_prob_old)
                    ratio_hero = torch.exp(new_move_log_probs - mb_old_move_log_probs)
                    surr1_hero = ratio_hero * mb_hero_advantages
                    surr2_hero = torch.clamp(ratio_hero, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_hero_advantages
                    actor_loss_hero = -torch.min(surr1_hero, surr2_hero).mean()

                    # Critic Loss (Value Function Loss - typically MSE)
                    # Ensure value predictions have same shape as returns (e.g., [B])
                    if hero_values.shape != mb_hero_returns.shape:
                         hero_values = hero_values.view_as(mb_hero_returns)
                    critic_loss_hero = self.critic_loss_fn(hero_values, mb_hero_returns)

                    # Total Hero Loss = Policy Loss + Value Loss - Entropy Bonus
                    total_loss_hero = actor_loss_hero + self.vf_coeff * critic_loss_hero - self.entropy_coeff * hero_entropy

                    # --- Calculate Gun Losses ---
                    # Actor Loss
                    ratio_gun = torch.exp(new_shoot_log_probs - mb_old_shoot_log_probs)
                    surr1_gun = ratio_gun * mb_gun_advantages
                    surr2_gun = torch.clamp(ratio_gun, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_gun_advantages
                    actor_loss_gun = -torch.min(surr1_gun, surr2_gun).mean()

                    # Critic Loss
                    if gun_values.shape != mb_gun_returns.shape:
                         gun_values = gun_values.view_as(mb_gun_returns)
                    critic_loss_gun = self.critic_loss_fn(gun_values, mb_gun_returns)

                    # Total Gun Loss
                    total_loss_gun = actor_loss_gun + self.vf_coeff * critic_loss_gun - self.entropy_coeff * gun_entropy

                    # --- Optimization Step (Hero) ---
                    self.hero_actor_optimizer.zero_grad()
                    self.hero_critic_optimizer.zero_grad()
                    total_loss_hero.backward()
                    # Optional: Gradient Clipping (applied to parameters of each network)
                    # nn.utils.clip_grad_norm_(self.hero_actor_net.parameters(), max_norm=0.5)
                    # nn.utils.clip_grad_norm_(self.hero_critic_net.parameters(), max_norm=0.5)
                    self.hero_actor_optimizer.step()
                    self.hero_critic_optimizer.step()

                    # --- Optimization Step (Gun) ---
                    self.gun_actor_optimizer.zero_grad()
                    self.gun_critic_optimizer.zero_grad()
                    total_loss_gun.backward()
                    # Optional: Gradient Clipping
                    # nn.utils.clip_grad_norm_(self.gun_actor_net.parameters(), max_norm=0.5)
                    # nn.utils.clip_grad_norm_(self.gun_critic_net.parameters(), max_norm=0.5)
                    self.gun_actor_optimizer.step()
                    self.gun_critic_optimizer.step()

                    # --- Store Loss Metrics (for console logging window) ---
                    epoch_actor_loss_h += actor_loss_hero.item()
                    epoch_critic_loss_h += critic_loss_hero.item()
                    epoch_entropy_h += hero_entropy.item()
                    epoch_actor_loss_g += actor_loss_gun.item()
                    epoch_critic_loss_g += critic_loss_gun.item()
                    epoch_entropy_g += gun_entropy.item()

                except Exception as e:
                    self.logger.error(f"Error during PPO optimization step (Epoch {epoch}, Batch {start // self.mini_batch_size}): {e}", exc_info=True)
                    # Decide whether to continue or break epoch/update
                    # Breaking inner loop for safety on error
                    break # Exit minibatch loop for this epoch

            # End of epoch - log average losses for the epoch if needed, update rolling deques
            if num_batches > 0:
                self.hero_actor_loss_deque.append(epoch_actor_loss_h / num_batches)
                self.hero_critic_loss_deque.append(epoch_critic_loss_h / num_batches)
                self.entropy_hero_deque.append(epoch_entropy_h / num_batches)
                self.gun_actor_loss_deque.append(epoch_actor_loss_g / num_batches)
                self.gun_critic_loss_deque.append(epoch_critic_loss_g / num_batches)
                self.entropy_gun_deque.append(epoch_entropy_g / num_batches)
            # self.logger.debug(f"Epoch {epoch} finished. Avg Actor Loss H: {epoch_actor_loss_h / num_batches:.4f}")

        # --- 4. Clear Trajectory Buffer and Reset Step Counter ---
        self.trajectory_buffer.clear()
        self.total_steps = 0 # Reset step counter for the next rollout collection phase
        self.logger.debug("PPO update finished. Trajectory buffer cleared.")


    def _save_checkpoint_if_needed(self, episode: int) -> None:
        """Saves a checkpoint of the agent's state and episode metrics CSV if the save interval is reached."""
        # episode is 1-based index here
        if self.save_interval > 0 and episode > 0 and episode % self.save_interval == 0:
            self.logger.info(f"Reached save interval at episode {episode}. Saving checkpoint...")
            save_path = self.dump() # dump returns the directory path where files were saved
            if save_path:
                self.logger.info(f"Checkpoint state saved successfully to directory: {save_path}")
                # Now save the metrics CSV to the same directory
                self._save_episode_metrics_csv(save_path)
            else:
                self.logger.error(f"Failed to save checkpoint state for episode {episode}. Metrics CSV will not be saved for this interval.")

    def _save_episode_metrics_csv(self, save_dir: str) -> None:
        """
        Saves the collected episode metrics to a CSV file within the specified directory.

        Args:
            save_dir: The directory path where the checkpoint was saved.
        """
        if not self.training_summary_data:
            self.logger.warning("No episode summary data available to save to CSV.")
            return

        csv_filename = "episode_metrics.csv"
        csv_filepath = os.path.join(save_dir, csv_filename)

        try:
            self.logger.info(f"Saving episode metrics to: {csv_filepath}")
            df = pd.DataFrame(self.training_summary_data)

            # Calculate average rewards per step (handle potential division by zero)
            # Ensure 'Time_Alive' is numeric and replace 0s with NaN before division
            time_alive_numeric = pd.to_numeric(df['Time_Alive'], errors='coerce')
            df['Avg_Reward_Hero'] = (df['Reward_Hero'] / time_alive_numeric.replace(0, np.nan)).fillna(0)
            df['Avg_Reward_Gun'] = (df['Reward_Gun'] / time_alive_numeric.replace(0, np.nan)).fillna(0)

            # Select and rename columns for the CSV output
            csv_df = df[[
                'Episode',
                'Time_Alive',
                'Reward_Hero', # Cumulative reward for the episode
                'Avg_Reward_Hero', # Average reward per step for the episode
                'Reward_Gun', # Cumulative reward for the episode
                'Avg_Reward_Gun' # Average reward per step for the episode
            ]].copy() # Use copy to avoid potential SettingWithCopyWarning

            # Define clearer column names for the CSV file
            csv_df.columns = [
                'Episode',
                'TimeAlive',
                'CumRewardHero',
                'AvgRewardHeroPerStep',
                'CumRewardGun',
                'AvgRewardGunPerStep'
            ]

            # Save to CSV, overwriting if it exists for this checkpoint
            csv_df.to_csv(csv_filepath, index=False, float_format='%.4f') # Format floats
            self.logger.info("Episode metrics CSV saved successfully.")

        except KeyError as e:
             self.logger.error(f"Missing expected column in training_summary_data: {e}. Cannot save CSV.", exc_info=True)
        except Exception as e:
            self.logger.error(f"Failed to save episode metrics CSV to {csv_filepath}: {e}", exc_info=True)

    def _get_last_save_directory(self, base_save_dir: str = "model_saves_ppo") -> Optional[str]:
         """Finds the most recent checkpoint directory based on timestamp naming."""
         if not os.path.isdir(base_save_dir):
             return None
         try:
             # List directories matching the expected pattern
             subdirs = [d for d in os.listdir(base_save_dir)
                        if os.path.isdir(os.path.join(base_save_dir, d)) and d.startswith("theseus_ppo_")]
             if not subdirs:
                 return None
             # Sort by name (which includes timestamp) to find the latest
             latest_subdir_name = max(subdirs)
             return os.path.join(base_save_dir, latest_subdir_name)
         except Exception as e:
             self.logger.error(f"Error finding last save directory in {base_save_dir}: {e}")
             return None


    def _display_training_summary(self, total_episodes_completed: int) -> None:
        """Displays a summary table of training performance based on episode data."""
        if not self.training_summary_data or total_episodes_completed == 0:
            self.logger.info("No training data recorded or no episodes completed, skipping summary table.")
            return

        self.logger.info("Generating Training Summary Table...")
        try:
            df = pd.DataFrame(self.training_summary_data)
            # Verify necessary columns exist
            required_cols = ['Episode', 'Reward_Hero', 'Reward_Gun', 'Time_Alive']
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Training summary data missing columns: {required_cols}. Cannot generate table.")
                return

            # Determine block size for averaging (e.g., 10 blocks or min 1 episode per block)
            block_size = max(1, total_episodes_completed // 10)
            num_blocks = (total_episodes_completed + block_size - 1) // block_size

            summary_rows = []
            for i in range(num_blocks):
                start_episode = i * block_size + 1 # 1-based index
                end_episode = min((i + 1) * block_size, total_episodes_completed)
                # Filter DataFrame based on the 'Episode' column (which is 1-based)
                block_data = df[(df['Episode'] >= start_episode) & (df['Episode'] <= end_episode)]

                if block_data.empty: continue # Skip if no data in this block range

                # Calculate averages for the block
                avg_reward_hero = block_data['Reward_Hero'].mean()
                avg_reward_gun = block_data['Reward_Gun'].mean()
                avg_time_alive = block_data['Time_Alive'].mean()
                summary_rows.append((
                    f"{start_episode}-{end_episode}", # Episode range
                    f"{avg_reward_hero:.3f}",         # Avg cumulative hero reward per ep in block
                    f"{avg_reward_gun:.3f}",          # Avg cumulative gun reward per ep in block
                    f"{avg_time_alive:.2f}"           # Avg time alive per ep in block
                ))

            # Create and print the table using Rich
            table = Table(title=f"Training Summary (Completed {total_episodes_completed} Episodes)")
            table.add_column("Episode Block", justify="center", style="cyan", no_wrap=True)
            table.add_column("Avg Ep Hero Reward", justify="right", style="magenta") # Clarified name
            table.add_column("Avg Ep Gun Reward", justify="right", style="green")   # Clarified name
            table.add_column("Avg Ep Time Alive", justify="right", style="yellow") # Clarified name
            for row in summary_rows: table.add_row(*row)
            self.console.print(table)

        except Exception as e:
             self.logger.error(f"Error generating training summary table: {e}", exc_info=True)


    def dump(self, save_dir: str = "model_saves_ppo") -> Optional[str]:
        """Saves the PPO agent state (networks, optimizers, config) to a timestamped directory."""
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name: str = f"theseus_ppo_{timestamp}"
        dpath: str = os.path.join(save_dir, base_name)

        try:
            os.makedirs(dpath, exist_ok=True)
            self.logger.info(f"Saving PPO agent state to directory: {dpath}")

            # Define filenames within the directory
            filenames: Dict[str, str] = {
                "hero_actor": "hero_actor_state.pth",
                "hero_critic": "hero_critic_state.pth",
                "gun_actor": "gun_actor_state.pth",
                "gun_critic": "gun_critic_state.pth",
                "hero_actor_optim": "hero_actor_optimizer.pth",
                "hero_critic_optim": "hero_critic_optimizer.pth",
                "gun_actor_optim": "gun_actor_optimizer.pth",
                "gun_critic_optim": "gun_critic_optimizer.pth",
                "config": f"{base_name}_config.yaml" # Config specific to this save
            }

            # Save network state dictionaries
            torch.save(self.hero_actor_net.state_dict(), os.path.join(dpath, filenames["hero_actor"]))
            torch.save(self.hero_critic_net.state_dict(), os.path.join(dpath, filenames["hero_critic"]))
            torch.save(self.gun_actor_net.state_dict(), os.path.join(dpath, filenames["gun_actor"]))
            torch.save(self.gun_critic_net.state_dict(), os.path.join(dpath, filenames["gun_critic"]))

            # Save optimizer state dictionaries
            torch.save(self.hero_actor_optimizer.state_dict(), os.path.join(dpath, filenames["hero_actor_optim"]))
            torch.save(self.hero_critic_optimizer.state_dict(), os.path.join(dpath, filenames["hero_critic_optim"]))
            torch.save(self.gun_actor_optimizer.state_dict(), os.path.join(dpath, filenames["gun_actor_optim"]))
            torch.save(self.gun_critic_optimizer.state_dict(), os.path.join(dpath, filenames["gun_critic_optim"]))

            # --- Prepare Configuration Dictionary ---
            state_info: Dict[str, Any] = {
                "agent_type": "PPO",
                # File references
                "hero_actor_file": filenames["hero_actor"],
                "hero_critic_file": filenames["hero_critic"],
                "gun_actor_file": filenames["gun_actor"],
                "gun_critic_file": filenames["gun_critic"],
                "hero_actor_optim_file": filenames["hero_actor_optim"],
                "hero_critic_optim_file": filenames["hero_critic_optim"],
                "gun_actor_optim_file": filenames["gun_actor_optim"],
                "gun_critic_optim_file": filenames["gun_critic_optim"],
                # Class paths for reconstruction
                "hero_actor_class": f"{type(self.hero_actor_net).__module__}.{type(self.hero_actor_net).__name__}",
                "hero_critic_class": f"{type(self.hero_critic_net).__module__}.{type(self.hero_critic_net).__name__}",
                "gun_actor_class": f"{type(self.gun_actor_net).__module__}.{type(self.gun_actor_net).__name__}",
                "gun_critic_class": f"{type(self.gun_critic_net).__module__}.{type(self.gun_critic_net).__name__}",
                # Network Architecture Details (Crucial for loading)
                "hero_hidden_channels": self.hero_hidden_channels,
                "gun_hidden_channels": self.gun_hidden_channels,
                # PPO Hyperparameters
                "learning_rate": self.hero_actor_optimizer.param_groups[0]['lr'], # Get LR from an optimizer
                "discount_factor": self.discount_factor,
                "horizon": self.horizon,
                "epochs_per_update": self.epochs_per_update,
                "mini_batch_size": self.mini_batch_size,
                "clip_epsilon": self.clip_epsilon,
                "gae_lambda": self.gae_lambda,
                "entropy_coeff": self.entropy_coeff,
                "vf_coeff": self.vf_coeff,
                # Other Training State/Config
                "log_window_size": self.log_window_size,
                "save_interval": self.save_interval,
                "total_reward_hero": self.total_reward_hero, # Save total cumulative reward
                "total_reward_gun": self.total_reward_gun,
                "optimizer_class": f"{type(self.hero_actor_optimizer).__module__}.{type(self.hero_actor_optimizer).__name__}",
            }

            # Save configuration to YAML file
            yaml_path: str = os.path.join(dpath, filenames["config"])
            with open(yaml_path, "w") as f:
                yaml.dump(state_info, f, default_flow_style=False, sort_keys=False)

            self.logger.info("PPO Agent state saved successfully.")
            # Return the path to the directory where everything was saved
            return dpath

        except AttributeError as e:
             # Specifically catch if hidden_channels aren't attributes of self
             self.logger.error(f"Failed to dump agent state due to missing attribute (likely 'hero_hidden_channels' or 'gun_hidden_channels' not set in __init__): {e}", exc_info=True)
             return None
        except Exception as e:
            self.logger.error(f"Failed to dump PPO agent state to {dpath}: {e}", exc_info=True)
            return None

    @classmethod
    def load(cls, load_path: Union[str, os.PathLike]) -> Optional[Self]:
        """Loads a PPO agent state from a checkpoint directory."""
        logger = logging.getLogger("agent-theseus-ppo-load") # Use a distinct logger for loading
        logger.info(f"Attempting to load PPO agent state from: {load_path}")

        load_path_str: str = str(load_path)
        if not os.path.isdir(load_path_str):
            logger.error(f"Load path is not a valid directory: {load_path_str}")
            return None

        # Find the configuration file within the directory
        yaml_files = [f for f in os.listdir(load_path_str) if f.endswith('_config.yaml')]
        if not yaml_files:
            logger.error(f"No '_config.yaml' file found in directory: {load_path_str}")
            return None
        # Use the first config file found (assuming only one per checkpoint)
        yaml_path: str = os.path.join(load_path_str, yaml_files[0])
        logger.info(f"Using configuration file: {yaml_path}")

        try:
            with open(yaml_path, "r") as f:
                state_info: Dict[str, Any] = yaml.safe_load(f)
                print(state_info)
        except Exception as e:
            logger.error(f"Error reading YAML configuration file {yaml_path}: {e}", exc_info=True)
            return None

        # --- Verify Agent Type ---
        if state_info.get("agent_type") != "PPO":
            logger.error(f"Config file indicates agent type is '{state_info.get('agent_type')}', expected 'PPO'.")
            return None

        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading models onto device: {device}")

        # --- Helper to dynamically import classes ---
        def get_class(class_path: str) -> Type:
            try:
                module_path, class_name = class_path.rsplit('.', 1)
                logger.info(module_path)
                module = importlib.import_module(module_path)
                return getattr(module, class_name)
            except (ImportError, AttributeError, ValueError) as e:
                logger.error(f"Failed to import class '{class_path}': {e}")
                raise # Re-raise to be caught by the outer try-except

        # --- Reconstruct Networks using saved architecture ---
        try:
            # Get Network Architecture Details from config
            hero_hidden = state_info.get('hero_hidden_channels')
            gun_hidden = state_info.get('gun_hidden_channels')

            # Check if critical architecture info is present
            if hero_hidden is None or gun_hidden is None:
                 logger.error("Missing 'hero_hidden_channels' or 'gun_hidden_channels' in config file. Cannot reconstruct networks.")
                 return None
                 # Or fallback to defaults with a strong warning:
                 # logger.warning("Missing hidden_channels in config, using defaults. Model might not load correctly.")
                 # hero_hidden = DEFAULT_HERO_HIDDEN_CHANNELS
                 # gun_hidden = DEFAULT_GUN_HIDDEN_CHANNELS

            logger.info(f"Using Hero Hidden Channels: {hero_hidden}")
            logger.info(f"Using Gun Hidden Channels: {gun_hidden}")

            # Get Network Classes
            HeroActorClass = get_class(state_info['hero_actor_class'])
            HeroCriticClass = get_class(state_info['hero_critic_class'])
            GunActorClass = get_class(state_info['gun_actor_class'])
            GunCriticClass = get_class(state_info['gun_critic_class'])

            # Instantiate networks using the loaded hidden_channels parameter
            # Adjust constructor call if your GNNs need other args (e.g., out_channels)
            hero_actor_net = HeroActorClass(hidden_channels=hero_hidden)
            # Assume critic output size is 1, pass if needed by constructor
            hero_critic_net = HeroCriticClass(hidden_channels=hero_hidden) # Adjust if needed: out_channels=1
            gun_actor_net = GunActorClass(hidden_channels=gun_hidden)
            gun_critic_net = GunCriticClass(hidden_channels=gun_hidden) # Adjust if needed: out_channels=1

            # Load state dictionaries
            def _load_state_dict(net: nn.Module, file_key: str):
                """Helper to load state dict with error checking."""
                if file_key not in state_info:
                     raise KeyError(f"File key '{file_key}' not found in config.")
                file_path = os.path.join(load_path_str, state_info[file_key])
                if not os.path.exists(file_path):
                     raise FileNotFoundError(f"Network state file not found: {file_path}")
                net.load_state_dict(torch.load(file_path, map_location=device))
                logger.debug(f"Loaded state dict for '{file_key}' from: {file_path}")

            _load_state_dict(hero_actor_net, "hero_actor_file")
            _load_state_dict(hero_critic_net, "hero_critic_file")
            _load_state_dict(gun_actor_net, "gun_actor_file")
            _load_state_dict(gun_critic_net, "gun_critic_file")

            logger.info("Network state dicts loaded successfully.")

        except (ImportError, AttributeError, KeyError, FileNotFoundError, TypeError, Exception) as e:
            # Catch broad exceptions during network reconstruction/loading
            logger.error(f"Fatal error reconstructing or loading network models: {e}", exc_info=True)
            return None

        # --- Reconstruct Optimizers ---
        try:
            learning_rate = state_info.get("learning_rate", DEFAULT_LEARNING_RATE_PPO)
            OptimizerClass = get_class(state_info.get('optimizer_class', 'torch.optim.AdamW')) # Default to AdamW

            # Move networks to the target device *before* creating optimizers
            hero_actor_net.to(device)
            hero_critic_net.to(device)
            gun_actor_net.to(device)
            gun_critic_net.to(device)

            # Instantiate optimizers with network parameters
            hero_actor_optimizer = OptimizerClass(hero_actor_net.parameters(), lr=learning_rate)
            hero_critic_optimizer = OptimizerClass(hero_critic_net.parameters(), lr=learning_rate)
            gun_actor_optimizer = OptimizerClass(gun_actor_net.parameters(), lr=learning_rate)
            gun_critic_optimizer = OptimizerClass(gun_critic_net.parameters(), lr=learning_rate)

            # Load optimizer states if the corresponding files exist
            optim_files_keys = ["hero_actor_optim_file", "hero_critic_optim_file", "gun_actor_optim_file", "gun_critic_optim_file"]
            optimizers_list = [hero_actor_optimizer, hero_critic_optimizer, gun_actor_optimizer, gun_critic_optimizer]
            for key, optim in zip(optim_files_keys, optimizers_list):
                if key in state_info:
                    optim_path = os.path.join(load_path_str, state_info[key])
                    if os.path.exists(optim_path):
                        try:
                            optim.load_state_dict(torch.load(optim_path, map_location=device))
                            logger.info(f"{key.replace('_file', '')} state loaded from {optim_path}.")
                        except Exception as load_err:
                             # Log warning but continue, optimizer starts fresh
                             logger.warning(f"Could not load optimizer state for {key} from {optim_path}: {load_err}. Optimizer will use initial state.")
                    else:
                        logger.warning(f"{key.replace('_file', '')} state file specified in config but not found: {optim_path}. Optimizer will use initial state.")
                else:
                     # If key isn't even in config, optimizer starts fresh
                     logger.warning(f"Optimizer state file key '{key}' not found in config. Optimizer will use initial state.")

            logger.info("Optimizers reconstructed (loaded state where possible).")

        except (ImportError, AttributeError, KeyError, FileNotFoundError, Exception) as e:
            logger.error(f"Fatal error reconstructing or loading optimizers: {e}", exc_info=True)
            return None

        # --- Reconstruct Environment ---
        try:
             # Assuming Environment class can be instantiated without arguments from config
             # If it needs args, they would need to be saved/loaded too.
             env = Environment()
             logger.info("Environment instantiated.")
        except Exception as e:
             logger.error(f"Fatal error instantiating Environment: {e}", exc_info=True)
             return None # Agent likely cannot function without environment

        # --- Instantiate the Agent class with loaded components ---
        try:
            # Use the loaded hyperparameters and components
            agent = cls(
                hero_actor_net=hero_actor_net,
                hero_critic_net=hero_critic_net,
                gun_actor_net=gun_actor_net,
                gun_critic_net=gun_critic_net,
                env=env, # Pass the newly created env instance
                optimizer_class=OptimizerClass,
                learning_rate=learning_rate, # Pass LR used for optimizers
                discount_factor=state_info.get('discount_factor', DEFAULT_DISCOUNT_FACTOR_PPO),
                horizon=state_info.get('horizon', DEFAULT_HORIZON),
                epochs_per_update=state_info.get('epochs_per_update', DEFAULT_EPOCHS_PER_UPDATE),
                mini_batch_size=state_info.get('mini_batch_size', DEFAULT_MINI_BATCH_SIZE_PPO),
                clip_epsilon=state_info.get('clip_epsilon', DEFAULT_CLIP_EPSILON),
                gae_lambda=state_info.get('gae_lambda', DEFAULT_GAE_LAMBDA),
                entropy_coeff=state_info.get('entropy_coeff', DEFAULT_ENTROPY_COEFF),
                vf_coeff=state_info.get('vf_coeff', DEFAULT_VF_COEFF),
                log_window_size=state_info.get('log_window_size', LOGGING_WINDOW),
                save_interval=state_info.get('save_interval', SAVE_INTERVAL),
                # Pass the loaded hidden channel info to the constructor
                hero_hidden_channels=hero_hidden,
                gun_hidden_channels=gun_hidden,
             )

            # --- Restore specific agent state variables ---
            # Assign the optimizers (potentially with loaded state) to the agent instance
            agent.hero_actor_optimizer = hero_actor_optimizer
            agent.hero_critic_optimizer = hero_critic_optimizer
            agent.gun_actor_optimizer = gun_actor_optimizer
            agent.gun_critic_optimizer = gun_critic_optimizer

            # Restore cumulative reward progress (if saved)
            agent.total_reward_hero = state_info.get('total_reward_hero', 0.0)
            agent.total_reward_gun = state_info.get('total_reward_gun', 0.0)
            # Note: Rolling deques (like episode_rewards_*, *_loss_deque) and
            # trajectory_buffer are transient and typically not restored.
            # training_summary_data could optionally be restored if you want to append
            # to the *same* CSV file across multiple loads, but usually logging restarts.

            # Ensure networks are in training mode by default after loading
            agent.hero_actor_net.train()
            agent.hero_critic_net.train()
            agent.gun_actor_net.train()
            agent.gun_critic_net.train()

            logger.info(f"PPO Agent loaded successfully from {load_path_str}")
            return agent

        except Exception as e:
            # Catch errors during the final agent instantiation phase
            logger.error(f"Fatal error during final agent instantiation: {e}", exc_info=True)
            return None