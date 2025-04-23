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
from typing import (
    List, Tuple, Optional, Type, Self, Deque, Dict, Any, Union
)

try:
    from torch import nn, optim
    from torch.distributions import Categorical
    from torch_geometric.data import Batch, HeteroData
    from rich.progress import (
        Progress, BarColumn, TextColumn, TimeRemainingColumn,
        MofNCompleteColumn, TaskID
    )
    from rich.table import Table
    from rich.console import Console
    from theseus.utils import State
    from theseus.utils.network import Environment
    import theseus.constants as c
    from theseus.models.GraphDQN.ActionGNN import HeroGNN as HeroBaseGNN
    from theseus.models.GraphDQN.ActionGNN import GunGNN as GunBaseGNN

    HeroActorGNN = HeroBaseGNN
    HeroCriticGNN = HeroBaseGNN
    GunActorGNN = GunBaseGNN
    GunCriticGNN = GunBaseGNN

except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.error(
        f"Failed to import necessary libraries: {e}. "
        "Please ensure all dependencies are installed and paths are correct."
    )
    raise ImportError(f"Critical import failed: {e}") from e


HERO_ACTION_SPACE_SIZE: int = 9
GUN_ACTION_SPACE_SIZE: int = 8
LOGGING_WINDOW: int = 50
SAVE_INTERVAL: int = 5
DEFAULT_HORIZON: int = 2048
DEFAULT_EPOCHS_PER_UPDATE: int = 10
DEFAULT_MINI_BATCH_SIZE_PPO: int = 64
DEFAULT_CLIP_EPSILON: float = 0.2
DEFAULT_GAE_LAMBDA: float = 0.95
DEFAULT_ENTROPY_COEFF: float = 0.01
DEFAULT_VF_COEFF: float = 0.5
DEFAULT_LEARNING_RATE_PPO: float = 3e-4
DEFAULT_DISCOUNT_FACTOR_PPO: float = 0.99
DEFAULT_HERO_HIDDEN_CHANNELS: int = 64
DEFAULT_GUN_HIDDEN_CHANNELS: int = 64

TrajectoryStep = namedtuple("TrajectoryStep", [
    'state_graph_hero', 'state_graph_gun',
    'move_action', 'shoot_action',
    'move_log_prob', 'shoot_log_prob',
    'hero_value', 'gun_value',
    'hero_reward', 'gun_reward',
    'terminated'
])


class AgentTheseusPPO:
    """
    Agent managing simultaneous training of Hero and Gun GNNs using PPO.
    Uses separate Actor/Critic GNNs, collects on-policy trajectories,
    calculates GAE, and updates networks using PPO objectives.
    Logs metrics to CSV and console.
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
        self.logger: logging.Logger = logging.getLogger("agent-theseus-ppo")
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.env: Environment = env

        self._validate_network(hero_actor_net, "Hero Actor")
        self._validate_network(hero_critic_net, "Hero Critic")
        self._validate_network(gun_actor_net, "Gun Actor")
        self._validate_network(gun_critic_net, "Gun Critic")
        self.hero_actor_net: HeroActorGNN = hero_actor_net.to(self.device)
        self.hero_critic_net: HeroCriticGNN = hero_critic_net.to(self.device)
        self.gun_actor_net: GunActorGNN = gun_actor_net.to(self.device)
        self.gun_critic_net: GunCriticGNN = gun_critic_net.to(self.device)
        self.hero_hidden_channels = hero_hidden_channels
        self.gun_hidden_channels = gun_hidden_channels

        self.discount_factor: float = discount_factor
        self.horizon: int = horizon
        self.epochs_per_update: int = epochs_per_update
        self.mini_batch_size: int = mini_batch_size
        self.clip_epsilon: float = clip_epsilon
        self.gae_lambda: float = gae_lambda
        self.entropy_coeff: float = entropy_coeff
        self.vf_coeff: float = vf_coeff

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

        self.trajectory_buffer: List[TrajectoryStep] = []
        self.total_steps: int = 0

        self.log_window_size: int = log_window_size
        self.save_interval: int = save_interval
        self.episode_rewards_hero_deque: Deque[float] = deque(maxlen=self.log_window_size)
        self.episode_rewards_gun_deque: Deque[float] = deque(maxlen=self.log_window_size)
        self.episode_time_alive_deque: Deque[int] = deque(maxlen=self.log_window_size)
        self.total_reward_hero: float = 0.0
        self.total_reward_gun: float = 0.0
        self.hero_actor_loss_deque: Deque[float] = deque(maxlen=self.log_window_size * epochs_per_update)
        self.hero_critic_loss_deque: Deque[float] = deque(maxlen=self.log_window_size * epochs_per_update)
        self.gun_actor_loss_deque: Deque[float] = deque(maxlen=self.log_window_size * epochs_per_update)
        self.gun_critic_loss_deque: Deque[float] = deque(maxlen=self.log_window_size * epochs_per_update)
        self.entropy_hero_deque: Deque[float] = deque(maxlen=self.log_window_size * epochs_per_update)
        self.entropy_gun_deque: Deque[float] = deque(maxlen=self.log_window_size * epochs_per_update)

        self.training_summary_data: List[Dict[str, Union[int, float]]] = []
        self.console = Console()
        self.current_state: Optional[State] = None

    def _validate_network(self, network: nn.Module, name: str) -> None:
        if not hasattr(network, "preprocess_state") or not callable(
            getattr(network, "preprocess_state", None)
        ):
            raise AttributeError(
                f"{name} network must have a callable 'preprocess_state' method."
            )

    def _update_metrics(
        self, ep_reward_hero: float, ep_reward_gun: float, time_alive: int
    ) -> None:
        self.episode_rewards_hero_deque.append(ep_reward_hero)
        self.episode_rewards_gun_deque.append(ep_reward_gun)
        self.episode_time_alive_deque.append(time_alive)
        self.total_reward_hero += ep_reward_hero
        self.total_reward_gun += ep_reward_gun

    def _log_episode_metrics(self, episode: int, steps: int) -> None:
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
            f"ALoss_H={avg_loss_actor_h:.4f}", f"CLoss_H={avg_loss_critic_h:.4f}",
            f"ALoss_G={avg_loss_actor_g:.4f}", f"CLoss_G={avg_loss_critic_g:.4f}",
            f"Entropy_H={avg_entropy_h:.3f}", f"Entropy_G={avg_entropy_g:.3f}",
            f"RolloutProg={self.total_steps}/{self.horizon}",
        ]
        log_str = f"Ep {episode + 1} Summary | " + " | ".join(metrics_list)
        self.logger.info(log_str)

    def train(self, num_episodes: Optional[int] = None) -> None:
        self.logger.info(
            f"Starting PPO training on {self.device} for "
            f"{num_episodes or 'infinite'} episodes..."
        )
        self.logger.info(f"Collect Horizon: {self.horizon} steps")
        self.logger.info(f"Save interval: Every {self.save_interval} episodes")
        self.training_summary_data = []

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
                TextColumn("Episode {task.completed}"),
            ]
            total_episodes_for_progress = None
            self.logger.warning("Training infinitely. Progress bar will not show total or ETA.")

        with Progress(*progress_columns, transient=False) as progress:
            episode_task: TaskID = progress.add_task(
                "[cyan]Training Episodes...", total=total_episodes_for_progress
            )
            episode_iterator = range(num_episodes) if num_episodes is not None else count()
            completed_episodes = 0

            try:
                for episode in episode_iterator:
                    ep_reward_hero, ep_reward_gun, time_alive, terminated, truncated = self._run_episode_or_rollout(
                        episode, progress, episode_task
                    )

                    if terminated or truncated:
                        self._update_metrics(ep_reward_hero, ep_reward_gun, time_alive)
                        self._log_episode_metrics(episode, time_alive)

                        if time_alive > 0:
                            self.training_summary_data.append({
                                "Episode": episode + 1,
                                "Time_Alive": time_alive,
                                "Reward_Hero": ep_reward_hero,
                                "Reward_Gun": ep_reward_gun,
                            })
                        else:
                             self.logger.warning(f"Episode {episode+1} ended with 0 time alive. Not adding to summary data.")

                        self._save_checkpoint_if_needed(episode + 1)
                        progress.update(episode_task, advance=1)
                        completed_episodes += 1

                    if self.total_steps >= self.horizon:
                        self.logger.info(f"Horizon {self.horizon} reached. Starting PPO update.")
                        self._update_policy()
                        progress.update(episode_task, description=f"[cyan]Ep. {episode+1} (Updating Policy...)")

            except RuntimeError as e:
                self.logger.critical(f"Stopping training due to runtime error in episode {episode+1}: {e}", exc_info=True)
            except KeyboardInterrupt:
                 self.logger.warning("Training interrupted by user.")
            except Exception as e:
                self.logger.critical(f"Unexpected error during episode {episode+1}: {e}", exc_info=True)
            finally:
                progress.stop()
                if num_episodes is not None:
                    final_desc = "[green]Training Finished" if completed_episodes == num_episodes else "[yellow]Training Stopped Early"
                    progress.update(episode_task, description=final_desc, completed=completed_episodes)
                else:
                    final_desc = "[yellow]Training Stopped (Infinite Mode)"
                    progress.update(episode_task, description=final_desc)

                if completed_episodes > 0:
                    self._display_training_summary(completed_episodes)
                    self.logger.info("Attempting to save final episode metrics CSV...")
                    last_save_dir = self._get_last_save_directory()
                    if last_save_dir:
                         self._save_episode_metrics_csv(last_save_dir)
                         self.logger.info(f"Final metrics CSV saved in: {last_save_dir}")
                    else:
                         final_save_dir = os.path.join("model_saves_ppo", "final_run_metrics")
                         os.makedirs(final_save_dir, exist_ok=True)
                         self._save_episode_metrics_csv(final_save_dir)
                         self.logger.warning(f"No checkpoint directory found. Final metrics saved to: {final_save_dir}")

        self.logger.info("Training finished.")

    def _run_episode_or_rollout(
        self, episode_num: int, progress: Progress, task_id: TaskID
    ) -> Tuple[float, float, int, bool, bool]:
        segment_reward_hero: float = 0.0
        segment_reward_gun: float = 0.0
        segment_steps: int = 0
        terminated: bool = False
        truncated: bool = False

        if self.current_state is None:
            self.current_state = self._initialize_episode()
            if self.current_state is None:
                raise RuntimeError("Failed to get initial state for episode/rollout.")
            self.logger.debug(f"Episode {episode_num + 1} segment started.")

        while self.total_steps < self.horizon:
            if self.current_state is None:
                 self.logger.error("Critical: current_state became None during rollout.")
                 terminated = True
                 break

            try:
                with torch.no_grad():
                    graph_hero_actor = self.hero_actor_net.preprocess_state(self.current_state)
                    graph_gun_actor = self.gun_actor_net.preprocess_state(self.current_state)
                    graph_hero_critic = self.hero_critic_net.preprocess_state(self.current_state)
                    graph_gun_critic = self.gun_critic_net.preprocess_state(self.current_state)

                if graph_hero_actor is None or graph_gun_actor is None or \
                   graph_hero_critic is None or graph_gun_critic is None:
                    self.logger.warning(f"Preprocessing failed at step {segment_steps} in Ep {episode_num + 1}. Ending segment.")
                    terminated = True
                    break

                graph_hero_actor = graph_hero_actor.to(self.device)
                graph_gun_actor = graph_gun_actor.to(self.device)
                graph_hero_critic = graph_hero_critic.to(self.device)
                graph_gun_critic = graph_gun_critic.to(self.device)

            except Exception as e:
                self.logger.error(f"Error preprocessing state in Ep {episode_num + 1}, Step {segment_steps}: {e}", exc_info=True)
                terminated = True
                break

            with torch.no_grad():
                move_action, move_log_prob = self._sample_action(self.hero_actor_net, graph_hero_actor)
                shoot_action, shoot_log_prob = self._sample_action(self.gun_actor_net, graph_gun_actor)
                hero_value = self._get_value(self.hero_critic_net, graph_hero_critic)
                gun_value = self._get_value(self.gun_critic_net, graph_gun_critic)

            if move_action is None or shoot_action is None or hero_value is None or gun_value is None:
                 self.logger.warning(f"Action sampling or value estimation failed in Ep {episode_num + 1}. Ending segment.")
                 terminated = True
                 break

            step_result = self._step_environment(move_action, shoot_action)
            next_state, reward_hero, reward_gun, terminated, truncated = step_result

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
                terminated=terminated
            )
            self.trajectory_buffer.append(step_data)

            self.current_state = next_state
            self.total_steps += 1
            segment_steps += 1
            segment_reward_hero += reward_hero
            segment_reward_gun += reward_gun

            if segment_steps % 20 == 0:
                avg_r_hero_disp = np.mean(self.episode_rewards_hero_deque) if self.episode_rewards_hero_deque else 0.0
                avg_r_gun_disp = np.mean(self.episode_rewards_gun_deque) if self.episode_rewards_gun_deque else 0.0
                progress.update(
                    task_id,
                    description=(
                        f"[cyan]Ep. {episode_num + 1}[/cyan] [yellow]Step {segment_steps}[/yellow] "
                        f"| Rollout: [b]{self.total_steps}/{self.horizon}[/b] "
                        f"| AvgR Gun(win): [b]{avg_r_gun_disp:.2f}[/b] "
                        f"| AvgR Hero(win): [b]{avg_r_hero_disp:.2f}[/b]"
                    ),
                )

            if terminated or truncated:
                self.logger.debug(f"Episode {episode_num + 1} ended at step {segment_steps} ({'Terminated' if terminated else 'Truncated'}). Rollout steps: {self.total_steps}/{self.horizon}")
                self.current_state = self._initialize_episode()
                if self.current_state is None:
                    self.logger.error("Failed to reset env after episode end. Training might halt.")
                break

        return segment_reward_hero, segment_reward_gun, segment_steps, terminated, truncated

    def _initialize_episode(self) -> Optional[State]:
        try:
            initial_state: State = self.env.initialise_environment()
            if not isinstance(initial_state, State):
                raise TypeError(f"Env did not return expected State object, got {type(initial_state)}")
            return initial_state
        except Exception as e:
            self.logger.error(f"Failed to initialize environment: {e}", exc_info=True)
            return None

    def _sample_action(
        self, actor_net: nn.Module, state_graph: Union[HeteroData, Batch]
    ) -> Tuple[Optional[int], Optional[torch.Tensor]]:
        try:
            actor_net.eval()
            logits: torch.Tensor = actor_net(state_graph)
            actor_net.train()

            if logits.numel() == 0:
                self.logger.warning(f"{type(actor_net).__name__} produced empty logits.")
                return None, None
            if logits.ndim > 1 and logits.shape[0] == 1:
                logits = logits.squeeze(0)
            elif logits.ndim > 1 and logits.shape[0] != 1:
                 self.logger.warning(f"{type(actor_net).__name__} received unexpected batch size > 1 for sampling: {logits.shape}. Using first element.")
                 logits = logits[0]
            elif logits.ndim == 0:
                 self.logger.warning(f"{type(actor_net).__name__} produced scalar output instead of logits vector.")
                 return None, None

            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return action.item(), log_prob

        except Exception as e:
            self.logger.error(f"Error sampling action with {type(actor_net).__name__}: {e}", exc_info=True)
            return None, None

    def _get_value(
        self, critic_net: nn.Module, state_graph: Union[HeteroData, Batch]
    ) -> Optional[torch.Tensor]:
        try:
            critic_net.eval()
            value: torch.Tensor = critic_net(state_graph)
            critic_net.train()

            if value.numel() != 1:
                 self.logger.warning(f"{type(critic_net).__name__} produced non-scalar value: shape {value.shape}. Taking first element.")
                 value = value.flatten()[0]

            return value.squeeze()

        except Exception as e:
            self.logger.error(f"Error getting value with {type(critic_net).__name__}: {e}", exc_info=True)
            return None

    def _step_environment(
        self, move_action: int, shoot_action: int
    ) -> Tuple[Optional[State], float, float, bool, bool]:
        try:
            combined_action_list: List[int] = [move_action, shoot_action, 0, 0]
            step_result: Tuple = self.env.step(combined_action_list)

            if not isinstance(step_result, tuple) or len(step_result) < 3:
                 raise TypeError(f"Env step returned unexpected format: {type(step_result)}, value: {step_result}")

            next_s = step_result[0]
            reward_tuple = step_result[1]
            terminated_flag = step_result[2]

            truncated_flag = False
            if len(step_result) > 3 and isinstance(step_result[3], bool):
                truncated_flag = step_result[3]

            next_state: Optional[State] = None
            if not bool(terminated_flag) and not bool(truncated_flag):
                if isinstance(next_s, State):
                    next_state = next_s
                else:
                    self.logger.error(f"Env step returned invalid non-terminal state type: {type(next_s)}")
                    return None, 0.0, 0.0, True, True

            if not isinstance(reward_tuple, (tuple, list)) or len(reward_tuple) < 2:
                 self.logger.error(f"Env step returned invalid reward format: {reward_tuple}. Using [0, 0].")
                 reward_hero, reward_gun = 0.0, 0.0
            else:
                 reward_hero = float(0 if reward_tuple[0] is None else reward_tuple[0])
                 reward_gun = float(0 if reward_tuple[1] is None else reward_tuple[1])

            terminated: bool = bool(terminated_flag)
            truncated: bool = bool(truncated_flag)

            return next_state, reward_hero, reward_gun, terminated, truncated

        except Exception as e:
            self.logger.error(f"Error during environment step: {e}", exc_info=True)
            return None, 0.0, 0.0, True, True

    def _calculate_gae_and_returns(
        self, last_state: Optional[State], last_terminated: bool, last_truncated: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hero_advantages_list = []
        gun_advantages_list = []
        hero_last_gae_lam = 0.0
        gun_last_gae_lam = 0.0

        last_hero_value = 0.0
        last_gun_value = 0.0
        should_bootstrap = not last_terminated
        if should_bootstrap and last_state is not None:
            try:
                with torch.no_grad():
                    graph_h_last = self.hero_critic_net.preprocess_state(last_state)
                    graph_g_last = self.gun_critic_net.preprocess_state(last_state)
                    if graph_h_last is not None and graph_g_last is not None:
                        graph_h_last = graph_h_last.to(self.device)
                        graph_g_last = graph_g_last.to(self.device)
                        val_h = self._get_value(self.hero_critic_net, graph_h_last)
                        val_g = self._get_value(self.gun_critic_net, graph_g_last)
                        if val_h is not None: last_hero_value = val_h.item()
                        if val_g is not None: last_gun_value = val_g.item()
                    else:
                        self.logger.warning("Preprocessing failed for GAE bootstrap state. Using value 0.")
            except Exception as e:
                self.logger.warning(f"Error getting value for GAE bootstrap state: {e}. Using value 0.")

        num_steps = len(self.trajectory_buffer)
        for i in reversed(range(num_steps)):
            step = self.trajectory_buffer[i]
            current_hero_value = step.hero_value.item()
            current_gun_value = step.gun_value.item()
            reward_hero = step.hero_reward
            reward_gun = step.gun_reward
            terminated_mask = 1.0 - float(step.terminated)

            delta_hero = reward_hero + self.discount_factor * last_hero_value * terminated_mask - current_hero_value
            delta_gun = reward_gun + self.discount_factor * last_gun_value * terminated_mask - current_gun_value

            adv_hero = delta_hero + self.discount_factor * self.gae_lambda * hero_last_gae_lam * terminated_mask
            adv_gun = delta_gun + self.discount_factor * self.gae_lambda * gun_last_gae_lam * terminated_mask

            hero_advantages_list.append(adv_hero)
            gun_advantages_list.append(adv_gun)

            hero_last_gae_lam = adv_hero
            gun_last_gae_lam = adv_gun
            last_hero_value = current_hero_value
            last_gun_value = current_gun_value

        hero_advantages_list.reverse()
        gun_advantages_list.reverse()

        hero_adv_tensor = torch.tensor(hero_advantages_list, dtype=torch.float32, device=self.device)
        gun_adv_tensor = torch.tensor(gun_advantages_list, dtype=torch.float32, device=self.device)

        values_hero = torch.stack([step.hero_value for step in self.trajectory_buffer]).to(self.device).squeeze()
        values_gun = torch.stack([step.gun_value for step in self.trajectory_buffer]).to(self.device).squeeze()

        if values_hero.ndim == 0 and hero_adv_tensor.ndim == 1:
            values_hero = values_hero.unsqueeze(0)
        if values_gun.ndim == 0 and gun_adv_tensor.ndim == 1:
            values_gun = values_gun.unsqueeze(0)

        hero_returns = hero_adv_tensor + values_hero
        gun_returns = gun_adv_tensor + values_gun

        hero_adv_tensor = (hero_adv_tensor - hero_adv_tensor.mean()) / (hero_adv_tensor.std() + 1e-8)
        gun_adv_tensor = (gun_adv_tensor - gun_adv_tensor.mean()) / (gun_adv_tensor.std() + 1e-8)

        return hero_adv_tensor, hero_returns, gun_adv_tensor, gun_returns

    def _update_policy(self) -> None:
        if not self.trajectory_buffer:
            self.logger.warning("Attempted PPO update with empty trajectory buffer.")
            return

        last_state = self.current_state
        last_step_info = self.trajectory_buffer[-1]
        last_terminated = last_step_info.terminated
        last_truncated = not last_terminated

        try:
            hero_advantages, hero_returns, gun_advantages, gun_returns = self._calculate_gae_and_returns(
                last_state, last_terminated, last_truncated
            )
        except Exception as e:
            self.logger.error(f"Error calculating GAE/Returns: {e}", exc_info=True)
            self.trajectory_buffer.clear()
            self.total_steps = 0
            return

        hero_graphs = [step.state_graph_hero for step in self.trajectory_buffer]
        gun_graphs = [step.state_graph_gun for step in self.trajectory_buffer]
        move_actions = torch.tensor([step.move_action for step in self.trajectory_buffer], dtype=torch.long, device=self.device)
        shoot_actions = torch.tensor([step.shoot_action for step in self.trajectory_buffer], dtype=torch.long, device=self.device)
        old_move_log_probs = torch.stack([step.move_log_prob for step in self.trajectory_buffer]).to(self.device).squeeze()
        old_shoot_log_probs = torch.stack([step.shoot_log_prob for step in self.trajectory_buffer]).to(self.device).squeeze()

        if old_move_log_probs.ndim == 0: old_move_log_probs = old_move_log_probs.unsqueeze(0)
        if old_shoot_log_probs.ndim == 0: old_shoot_log_probs = old_shoot_log_probs.unsqueeze(0)

        data_size = len(self.trajectory_buffer)
        indices = np.arange(data_size)

        self.hero_actor_net.train()
        self.hero_critic_net.train()
        self.gun_actor_net.train()
        self.gun_critic_net.train()

        for epoch in range(self.epochs_per_update):
            np.random.shuffle(indices)
            epoch_actor_loss_h, epoch_critic_loss_h, epoch_entropy_h = 0.0, 0.0, 0.0
            epoch_actor_loss_g, epoch_critic_loss_g, epoch_entropy_g = 0.0, 0.0, 0.0
            num_batches = 0

            for start in range(0, data_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]
                num_batches += 1

                try:
                    mb_hero_graphs_list = [hero_graphs[i].to(self.device) for i in mb_indices]
                    mb_gun_graphs_list = [gun_graphs[i].to(self.device) for i in mb_indices]
                    mb_hero_graphs_batch = Batch.from_data_list(mb_hero_graphs_list)
                    mb_gun_graphs_batch = Batch.from_data_list(mb_gun_graphs_list)
                except Exception as e:
                     self.logger.error(f"Error creating PyG Batch for minibatch (Epoch {epoch}, Start {start}): {e}", exc_info=True)
                     continue

                mb_move_actions = move_actions[mb_indices]
                mb_shoot_actions = shoot_actions[mb_indices]
                mb_old_move_log_probs = old_move_log_probs[mb_indices]
                mb_old_shoot_log_probs = old_shoot_log_probs[mb_indices]
                mb_hero_advantages = hero_advantages[mb_indices]
                mb_gun_advantages = gun_advantages[mb_indices]
                mb_hero_returns = hero_returns[mb_indices]
                mb_gun_returns = gun_returns[mb_indices]

                try:
                    hero_logits = self.hero_actor_net(mb_hero_graphs_batch)
                    hero_dist = Categorical(logits=hero_logits)
                    new_move_log_probs = hero_dist.log_prob(mb_move_actions)
                    hero_entropy = hero_dist.entropy().mean()
                    hero_values = self.hero_critic_net(mb_hero_graphs_batch).squeeze()

                    gun_logits = self.gun_actor_net(mb_gun_graphs_batch)
                    gun_dist = Categorical(logits=gun_logits)
                    new_shoot_log_probs = gun_dist.log_prob(mb_shoot_actions)
                    gun_entropy = gun_dist.entropy().mean()
                    gun_values = self.gun_critic_net(mb_gun_graphs_batch).squeeze()

                    ratio_hero = torch.exp(new_move_log_probs - mb_old_move_log_probs)
                    surr1_hero = ratio_hero * mb_hero_advantages
                    surr2_hero = torch.clamp(ratio_hero, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_hero_advantages
                    actor_loss_hero = -torch.min(surr1_hero, surr2_hero).mean()

                    if hero_values.shape != mb_hero_returns.shape:
                         hero_values = hero_values.view_as(mb_hero_returns)
                    critic_loss_hero = self.critic_loss_fn(hero_values, mb_hero_returns)

                    total_loss_hero = actor_loss_hero + self.vf_coeff * critic_loss_hero - self.entropy_coeff * hero_entropy

                    ratio_gun = torch.exp(new_shoot_log_probs - mb_old_shoot_log_probs)
                    surr1_gun = ratio_gun * mb_gun_advantages
                    surr2_gun = torch.clamp(ratio_gun, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_gun_advantages
                    actor_loss_gun = -torch.min(surr1_gun, surr2_gun).mean()

                    if gun_values.shape != mb_gun_returns.shape:
                         gun_values = gun_values.view_as(mb_gun_returns)
                    critic_loss_gun = self.critic_loss_fn(gun_values, mb_gun_returns)

                    total_loss_gun = actor_loss_gun + self.vf_coeff * critic_loss_gun - self.entropy_coeff * gun_entropy

                    self.hero_actor_optimizer.zero_grad()
                    self.hero_critic_optimizer.zero_grad()
                    total_loss_hero.backward()
                    self.hero_actor_optimizer.step()
                    self.hero_critic_optimizer.step()

                    self.gun_actor_optimizer.zero_grad()
                    self.gun_critic_optimizer.zero_grad()
                    total_loss_gun.backward()
                    self.gun_actor_optimizer.step()
                    self.gun_critic_optimizer.step()

                    epoch_actor_loss_h += actor_loss_hero.item()
                    epoch_critic_loss_h += critic_loss_hero.item()
                    epoch_entropy_h += hero_entropy.item()
                    epoch_actor_loss_g += actor_loss_gun.item()
                    epoch_critic_loss_g += critic_loss_gun.item()
                    epoch_entropy_g += gun_entropy.item()

                except Exception as e:
                    self.logger.error(f"Error during PPO optimization step (Epoch {epoch}, Batch {start // self.mini_batch_size}): {e}", exc_info=True)
                    break

            if num_batches > 0:
                self.hero_actor_loss_deque.append(epoch_actor_loss_h / num_batches)
                self.hero_critic_loss_deque.append(epoch_critic_loss_h / num_batches)
                self.entropy_hero_deque.append(epoch_entropy_h / num_batches)
                self.gun_actor_loss_deque.append(epoch_actor_loss_g / num_batches)
                self.gun_critic_loss_deque.append(epoch_critic_loss_g / num_batches)
                self.entropy_gun_deque.append(epoch_entropy_g / num_batches)

        self.trajectory_buffer.clear()
        self.total_steps = 0
        self.logger.debug("PPO update finished. Trajectory buffer cleared.")

    def _save_checkpoint_if_needed(self, episode: int) -> None:
        if self.save_interval > 0 and episode > 0 and episode % self.save_interval == 0:
            self.logger.info(f"Reached save interval at episode {episode}. Saving checkpoint...")
            save_path = self.dump()
            if save_path:
                self.logger.info(f"Checkpoint state saved successfully to directory: {save_path}")
                self._save_episode_metrics_csv(save_path)
            else:
                self.logger.error(f"Failed to save checkpoint state for episode {episode}. Metrics CSV will not be saved for this interval.")

    def _save_episode_metrics_csv(self, save_dir: str) -> None:
        if not self.training_summary_data:
            self.logger.warning("No episode summary data available to save to CSV.")
            return

        csv_filename = "episode_metrics.csv"
        csv_filepath = os.path.join(save_dir, csv_filename)

        try:
            self.logger.info(f"Saving episode metrics to: {csv_filepath}")
            df = pd.DataFrame(self.training_summary_data)

            time_alive_numeric = pd.to_numeric(df['Time_Alive'], errors='coerce')
            df['Avg_Reward_Hero'] = (df['Reward_Hero'] / time_alive_numeric.replace(0, np.nan)).fillna(0)
            df['Avg_Reward_Gun'] = (df['Reward_Gun'] / time_alive_numeric.replace(0, np.nan)).fillna(0)

            csv_df = df[[
                'Episode',
                'Time_Alive',
                'Reward_Hero',
                'Avg_Reward_Hero',
                'Reward_Gun',
                'Avg_Reward_Gun'
            ]].copy()

            csv_df.columns = [
                'Episode',
                'TimeAlive',
                'CumRewardHero',
                'AvgRewardHeroPerStep',
                'CumRewardGun',
                'AvgRewardGunPerStep'
            ]

            csv_df.to_csv(csv_filepath, index=False, float_format='%.4f')
            self.logger.info("Episode metrics CSV saved successfully.")

        except KeyError as e:
             self.logger.error(f"Missing expected column in training_summary_data: {e}. Cannot save CSV.", exc_info=True)
        except Exception as e:
            self.logger.error(f"Failed to save episode metrics CSV to {csv_filepath}: {e}", exc_info=True)

    def _get_last_save_directory(self, base_save_dir: str = "model_saves_ppo") -> Optional[str]:
         if not os.path.isdir(base_save_dir):
             return None
         try:
             subdirs = [d for d in os.listdir(base_save_dir)
                        if os.path.isdir(os.path.join(base_save_dir, d)) and d.startswith("theseus_ppo_")]
             if not subdirs:
                 return None
             latest_subdir_name = max(subdirs)
             return os.path.join(base_save_dir, latest_subdir_name)
         except Exception as e:
             self.logger.error(f"Error finding last save directory in {base_save_dir}: {e}")
             return None

    def _display_training_summary(self, total_episodes_completed: int) -> None:
        if not self.training_summary_data or total_episodes_completed == 0:
            self.logger.info("No training data recorded or no episodes completed, skipping summary table.")
            return

        self.logger.info("Generating Training Summary Table...")
        try:
            df = pd.DataFrame(self.training_summary_data)
            required_cols = ['Episode', 'Reward_Hero', 'Reward_Gun', 'Time_Alive']
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Training summary data missing columns: {required_cols}. Cannot generate table.")
                return

            block_size = max(1, total_episodes_completed // 10)
            num_blocks = (total_episodes_completed + block_size - 1) // block_size

            summary_rows = []
            for i in range(num_blocks):
                start_episode = i * block_size + 1
                end_episode = min((i + 1) * block_size, total_episodes_completed)
                block_data = df[(df['Episode'] >= start_episode) & (df['Episode'] <= end_episode)]

                if block_data.empty: continue

                avg_reward_hero = block_data['Reward_Hero'].mean()
                avg_reward_gun = block_data['Reward_Gun'].mean()
                avg_time_alive = block_data['Time_Alive'].mean()
                summary_rows.append((
                    f"{start_episode}-{end_episode}",
                    f"{avg_reward_hero:.3f}",
                    f"{avg_reward_gun:.3f}",
                    f"{avg_time_alive:.2f}"
                ))

            table = Table(title=f"Training Summary (Completed {total_episodes_completed} Episodes)")
            table.add_column("Episode Block", justify="center", style="cyan", no_wrap=True)
            table.add_column("Avg Ep Hero Reward", justify="right", style="magenta")
            table.add_column("Avg Ep Gun Reward", justify="right", style="green")
            table.add_column("Avg Ep Time Alive", justify="right", style="yellow")
            for row in summary_rows: table.add_row(*row)
            self.console.print(table)

        except Exception as e:
             self.logger.error(f"Error generating training summary table: {e}", exc_info=True)

    def dump(self, save_dir: str = "model_saves_ppo") -> Optional[str]:
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
                "config": f"{base_name}_config.yaml"
            }

            torch.save(self.hero_actor_net.state_dict(), os.path.join(dpath, filenames["hero_actor"]))
            torch.save(self.hero_critic_net.state_dict(), os.path.join(dpath, filenames["hero_critic"]))
            torch.save(self.gun_actor_net.state_dict(), os.path.join(dpath, filenames["gun_actor"]))
            torch.save(self.gun_critic_net.state_dict(), os.path.join(dpath, filenames["gun_critic"]))

            torch.save(self.hero_actor_optimizer.state_dict(), os.path.join(dpath, filenames["hero_actor_optim"]))
            torch.save(self.hero_critic_optimizer.state_dict(), os.path.join(dpath, filenames["hero_critic_optim"]))
            torch.save(self.gun_actor_optimizer.state_dict(), os.path.join(dpath, filenames["gun_actor_optim"]))
            torch.save(self.gun_critic_optimizer.state_dict(), os.path.join(dpath, filenames["gun_critic_optim"]))

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
                "hero_actor_class": f"{type(self.hero_actor_net).__module__}.{type(self.hero_actor_net).__name__}",
                "hero_critic_class": f"{type(self.hero_critic_net).__module__}.{type(self.hero_critic_net).__name__}",
                "gun_actor_class": f"{type(self.gun_actor_net).__module__}.{type(self.gun_actor_net).__name__}",
                "gun_critic_class": f"{type(self.gun_critic_net).__module__}.{type(self.gun_critic_net).__name__}",
                "hero_hidden_channels": self.hero_hidden_channels,
                "gun_hidden_channels": self.gun_hidden_channels,
                "learning_rate": self.hero_actor_optimizer.param_groups[0]['lr'],
                "discount_factor": self.discount_factor,
                "horizon": self.horizon,
                "epochs_per_update": self.epochs_per_update,
                "mini_batch_size": self.mini_batch_size,
                "clip_epsilon": self.clip_epsilon,
                "gae_lambda": self.gae_lambda,
                "entropy_coeff": self.entropy_coeff,
                "vf_coeff": self.vf_coeff,
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
             self.logger.error(f"Failed to dump agent state due to missing attribute: {e}", exc_info=True)
             return None
        except Exception as e:
            self.logger.error(f"Failed to dump PPO agent state to {dpath}: {e}", exc_info=True)
            return None

    @classmethod
    def load(cls, load_path: Union[str, os.PathLike]) -> Optional[Self]:
        logger = logging.getLogger("agent-theseus-ppo-load")
        logger.info(f"Attempting to load PPO agent state from: {load_path}")

        load_path_str: str = str(load_path)
        if not os.path.isdir(load_path_str):
            logger.error(f"Load path is not a valid directory: {load_path_str}")
            return None

        yaml_files = [f for f in os.listdir(load_path_str) if f.endswith('_config.yaml')]
        if not yaml_files:
            logger.error(f"No '_config.yaml' file found in directory: {load_path_str}")
            return None
        yaml_path: str = os.path.join(load_path_str, yaml_files[0])
        logger.info(f"Using configuration file: {yaml_path}")

        try:
            with open(yaml_path, "r") as f:
                state_info: Dict[str, Any] = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error reading YAML configuration file {yaml_path}: {e}", exc_info=True)
            return None

        if state_info.get("agent_type") != "PPO":
            logger.error(f"Config file indicates agent type is '{state_info.get('agent_type')}', expected 'PPO'.")
            return None

        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading models onto device: {device}")

        def get_class(class_path: str) -> Type:
            try:
                module_path, class_name = class_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                return getattr(module, class_name)
            except (ImportError, AttributeError, ValueError) as e:
                logger.error(f"Failed to import class '{class_path}': {e}")
                raise

        try:
            hero_hidden = state_info.get('hero_hidden_channels')
            gun_hidden = state_info.get('gun_hidden_channels')

            if hero_hidden is None or gun_hidden is None:
                 logger.error("Missing 'hero_hidden_channels' or 'gun_hidden_channels' in config file. Cannot reconstruct networks.")
                 return None

            logger.info(f"Using Hero Hidden Channels: {hero_hidden}")
            logger.info(f"Using Gun Hidden Channels: {gun_hidden}")

            HeroActorClass = get_class(state_info['hero_actor_class'])
            HeroCriticClass = get_class(state_info['hero_critic_class'])
            GunActorClass = get_class(state_info['gun_actor_class'])
            GunCriticClass = get_class(state_info['gun_critic_class'])

            hero_actor_net = HeroActorClass(hidden_channels=hero_hidden)
            hero_critic_net = HeroCriticClass(hidden_channels=hero_hidden, out_channels=1)
            gun_actor_net = GunActorClass(hidden_channels=gun_hidden)
            gun_critic_net = GunCriticClass(hidden_channels=gun_hidden, out_channels=1)

            def _load_state_dict(net: nn.Module, file_key: str):
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
            logger.error(f"Fatal error reconstructing or loading network models: {e}", exc_info=True)
            return None

        try:
            learning_rate = state_info.get("learning_rate", DEFAULT_LEARNING_RATE_PPO)
            OptimizerClass = get_class(state_info.get('optimizer_class', 'torch.optim.AdamW'))

            hero_actor_net.to(device)
            hero_critic_net.to(device)
            gun_actor_net.to(device)
            gun_critic_net.to(device)

            hero_actor_optimizer = OptimizerClass(hero_actor_net.parameters(), lr=learning_rate)
            hero_critic_optimizer = OptimizerClass(hero_critic_net.parameters(), lr=learning_rate)
            gun_actor_optimizer = OptimizerClass(gun_actor_net.parameters(), lr=learning_rate)
            gun_critic_optimizer = OptimizerClass(gun_critic_net.parameters(), lr=learning_rate)

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
                             logger.warning(f"Could not load optimizer state for {key} from {optim_path}: {load_err}. Optimizer will use initial state.")
                    else:
                        logger.warning(f"{key.replace('_file', '')} state file specified in config but not found: {optim_path}. Optimizer will use initial state.")
                else:
                     logger.warning(f"Optimizer state file key '{key}' not found in config. Optimizer will use initial state.")

            logger.info("Optimizers reconstructed (loaded state where possible).")

        except (ImportError, AttributeError, KeyError, FileNotFoundError, Exception) as e:
            logger.error(f"Fatal error reconstructing or loading optimizers: {e}", exc_info=True)
            return None

        try:
             env = Environment()
             logger.info("Environment instantiated.")
        except Exception as e:
             logger.error(f"Fatal error instantiating Environment: {e}", exc_info=True)
             return None

        try:
            agent = cls(
                hero_actor_net=hero_actor_net,
                hero_critic_net=hero_critic_net,
                gun_actor_net=gun_actor_net,
                gun_critic_net=gun_critic_net,
                env=env,
                optimizer_class=OptimizerClass,
                learning_rate=learning_rate,
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
                hero_hidden_channels=hero_hidden,
                gun_hidden_channels=gun_hidden,
             )

            agent.hero_actor_optimizer = hero_actor_optimizer
            agent.hero_critic_optimizer = hero_critic_optimizer
            agent.gun_actor_optimizer = gun_actor_optimizer
            agent.gun_critic_optimizer = gun_critic_optimizer

            agent.total_reward_hero = state_info.get('total_reward_hero', 0.0)
            agent.total_reward_gun = state_info.get('total_reward_gun', 0.0)

            agent.hero_actor_net.train()
            agent.hero_critic_net.train()
            agent.gun_actor_net.train()
            agent.gun_critic_net.train()

            logger.info(f"PPO Agent loaded successfully from {load_path_str}")
            return agent

        except Exception as e:
            logger.error(f"Fatal error during final agent instantiation: {e}", exc_info=True)
            return None
