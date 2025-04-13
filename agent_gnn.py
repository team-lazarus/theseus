import torch
import random
import logging
import os
import yaml
import numpy as np
from collections import deque
from datetime import datetime
from itertools import count
from typing import List, Tuple, Optional, Type, Self

from torch import nn, optim
from torch_geometric.data import Batch, HeteroData

from theseus.utils import State, ExperienceReplayMemory
from theseus.utils.network import Environment
import theseus.constants as c
from theseus.models.GraphDQN.ActionGNN import HeroGNN, GunGNN

# Action space sizes
HERO_ACTION_SPACE_SIZE = 9
GUN_ACTION_SPACE_SIZE = 8

# --- Default Configuration for Logging/Saving ---
LOGGING_WINDOW = 100  # Calculate average rewards over this many episodes
SAVE_INTERVAL = 500  # Save a checkpoint every this many episodes (adjust as needed)


class AgentTheseusGNN(object):
    """Agent managing simultaneous training of Hero (movement) and Gun (shooting) GNNs."""

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
        epsilon_init: float = 0.9,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.05,
        mini_batch_size: int = 64,
        target_sync_rate: int = 500,
        replay_memory_size: int = c.REPLAY_MEMORY_SIZE,
        log_window_size: int = LOGGING_WINDOW,
        save_interval: int = SAVE_INTERVAL,
    ) -> None:
        """Initializes the dual-GNN agent."""
        self.logger = logging.getLogger("agent-theseus-gnn")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.env = env

        # Network Setup
        self._validate_network(hero_policy_net, "Hero Policy")
        self._validate_network(hero_target_net, "Hero Target")
        self._validate_network(gun_policy_net, "Gun Policy")
        self._validate_network(gun_target_net, "Gun Target")
        self.hero_policy_net = hero_policy_net.to(self.device)
        self.hero_target_net = hero_target_net.to(self.device)
        self.gun_policy_net = gun_policy_net.to(self.device)
        self.gun_target_net = gun_target_net.to(self.device)

        # Training Parameters
        self.discount_factor = discount_factor
        self.mini_batch_size = mini_batch_size
        self.target_sync_rate = target_sync_rate
        self.sync_steps_taken = 0
        self.log_window_size = log_window_size
        self.save_interval = save_interval

        # Epsilon
        self.epsilon_init = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_init

        # Optimization Setup
        self.loss_fn = loss_fn_class()
        self.hero_optimizer = optimizer_class(
            self.hero_policy_net.parameters(), lr=learning_rate, amsgrad=True
        )
        self.gun_optimizer = optimizer_class(
            self.gun_policy_net.parameters(), lr=learning_rate, amsgrad=True
        )

        # Memory
        self.memory = ExperienceReplayMemory(replay_memory_size)

        # Metric Tracking
        self.episode_rewards_hero_deque = deque(maxlen=self.log_window_size)
        self.episode_rewards_gun_deque = deque(maxlen=self.log_window_size)
        self.total_reward_hero = 0.0
        self.total_reward_gun = 0.0
        self.hero_loss_deque = deque(
            maxlen=self.log_window_size * 10
        )  # Track more samples for avg loss
        self.gun_loss_deque = deque(maxlen=self.log_window_size * 10)

        # Initialization
        self.hero_target_net.load_state_dict(self.hero_policy_net.state_dict())
        self.gun_target_net.load_state_dict(self.gun_policy_net.state_dict())
        self.hero_target_net.eval()
        self.gun_target_net.eval()

    def _validate_network(self, network: nn.Module, name: str) -> None:
        """Checks if a network has the required preprocess_state method."""
        if not hasattr(network, "preprocess_state") or not callable(
            network.preprocess_state
        ):
            raise AttributeError(
                f"{name} network must have a 'preprocess_state' method."
            )

    def _update_metrics(self, ep_reward_hero: float, ep_reward_gun: float) -> None:
        """Updates rolling and cumulative metrics after an episode."""
        self.episode_rewards_hero_deque.append(ep_reward_hero)
        self.episode_rewards_gun_deque.append(ep_reward_gun)
        self.total_reward_hero += ep_reward_hero
        self.total_reward_gun += ep_reward_gun
        # Note: Loss metrics are updated in _calculate_and_apply_loss

    def _log_episode_metrics(self, episode: int, steps: int) -> None:
        """Logs key metrics for the completed episode."""
        # Calculate averages safely, handling empty deques
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
        avg_loss_hero = (
            np.mean(self.hero_loss_deque) if self.hero_loss_deque else float("nan")
        )
        avg_loss_gun = (
            np.mean(self.gun_loss_deque) if self.gun_loss_deque else float("nan")
        )

        # Prepare metrics string - easy to add more key-value pairs
        metrics_list = [
            f"Steps={steps}",
            f"Epsilon={self.epsilon:.4f}",
            f"AvgR_Hero={avg_rew_hero:.3f}",  # Average Hero Reward (rolling window)
            f"AvgR_Gun={avg_rew_gun:.3f}",  # Average Gun Reward (rolling window)
            f"CumR_Hero={self.total_reward_hero:.2f}",  # Cumulative Hero Reward (total)
            f"CumR_Gun={self.total_reward_gun:.2f}",  # Cumulative Gun Reward (total)
            f"AvgL_Hero={avg_loss_hero:.4f}",  # Average Hero Loss (rolling window)
            f"AvgL_Gun={avg_loss_gun:.4f}",  # Average Gun Loss (rolling window)
            f"Memory={len(self.memory)}",
            # Add more metrics here if tracked, e.g.:
            # f"AvgQ_Hero={avg_q_hero:.3f}",
        ]
        # Join metrics with a clear separator for readability
        log_str = f"Ep {episode} Summary | " + " | ".join(metrics_list)
        self.logger.info(log_str)

    def train(self, num_episodes: Optional[int] = None) -> None:
        """Runs the main training loop."""
        self.logger.info(
            f"Starting training on {self.device} for {num_episodes or 'infinite'} episodes..."
        )
        episode_iterator = range(num_episodes) if num_episodes is not None else count()

        for episode in episode_iterator:
            self.logger.info(
                f"[green]Starting Episode: {episode} (epsilon: {self.epsilon:.4f})[/]",
                extra={"markup": True},
            )

            try:
                ep_rewards_hero, ep_rewards_gun, steps = self._run_episode()
                self._update_metrics(ep_rewards_hero, ep_rewards_gun)
                self._log_episode_metrics(episode, steps)
                self._learn()
                self._decay_epsilon()
                self._save_checkpoint_if_needed(episode)

            except (
                RuntimeError
            ) as e:  # Catch specific runtime errors like env init failure
                self.logger.critical(
                    f"Stopping training due to runtime error in episode {episode}: {e}",
                    exc_info=True,
                )
                break
            except Exception as e:
                self.logger.critical(
                    f"Unexpected error during episode {episode}: {e}", exc_info=True
                )
                break  # Stop on other critical errors too

        self.logger.info("Training finished.")

    def _run_episode(self) -> Tuple[float, float, int]:
        """Runs a single episode, returns rewards and step count."""
        state = self._initialize_episode()  # Can raise RuntimeError
        terminated = False
        truncated = False
        episode_reward_hero = 0.0
        episode_reward_gun = 0.0
        episode_steps = 0

        while not terminated and not truncated:
            episode_steps += 1
            move_action, shoot_action = self._select_actions(state)
            next_state, reward_hero, reward_gun, terminated, truncated = (
                self._step_environment(move_action, shoot_action)
            )

            if next_state is not None:
                self.memory.append(
                    (
                        state,
                        move_action,
                        shoot_action,
                        next_state,
                        reward_hero,
                        reward_gun,
                        terminated,
                    )
                )
                self.sync_steps_taken += 1
                state = next_state
            else:
                self.logger.warning("Environment step failed, terminating episode.")
                terminated = True  # Force end episode if step fails

            episode_reward_hero += reward_hero
            episode_reward_gun += reward_gun

        return episode_reward_hero, episode_reward_gun, episode_steps

    def _initialize_episode(self) -> State:
        """Resets the environment and returns the initial state."""
        try:
            initial_state = self.env.initialise_environment()
            if not isinstance(initial_state, State):
                raise TypeError(
                    f"Environment did not return State object, got {type(initial_state)}"
                )
            self.logger.debug("Environment initialized for new episode.")
            return initial_state
        except Exception as e:
            self.logger.error(f"Failed to initialize environment: {e}", exc_info=True)
            raise RuntimeError("Environment initialization failed.") from e

    def _select_actions(self, state: State) -> Tuple[int, int]:
        """Selects movement and shooting actions using epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            move_action = random.randrange(HERO_ACTION_SPACE_SIZE)
            shoot_action = random.randrange(GUN_ACTION_SPACE_SIZE)
            return move_action, shoot_action
        else:
            move_action = self._predict_action(self.hero_policy_net, state)
            shoot_action = self._predict_action(self.gun_policy_net, state)
            # Fallback to random if prediction fails
            move_action = (
                move_action
                if move_action is not None
                else random.randrange(HERO_ACTION_SPACE_SIZE)
            )
            shoot_action = (
                shoot_action
                if shoot_action is not None
                else random.randrange(GUN_ACTION_SPACE_SIZE)
            )
            return move_action, shoot_action

    def _predict_action(self, policy_net: nn.Module, state: State) -> Optional[int]:
        """Predicts the best action using a given policy network."""
        try:
            graph_data = policy_net.preprocess_state(state)
            if graph_data is None:
                self.logger.warning(
                    f"Preprocessing failed for {type(policy_net).__name__}."
                )
                return None

            graph_data = graph_data.to(self.device)
            policy_net.eval()
            with torch.no_grad():
                q_values = policy_net(graph_data)
            policy_net.train()

            print(f"{type(policy_net).__name__}: {q_values}")
            if q_values.numel() == 0:
                self.logger.warning(
                    f"{type(policy_net).__name__} produced empty Q-values."
                )
                return None

            if q_values.ndim > 1:
                q_values = q_values.squeeze(0)
            if q_values.numel() == 0:
                return None  # Check again after squeeze

            action = q_values.argmax().item()
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
        """Combines actions, steps the environment, and unpacks results."""
        try:
            combined_action_list = [move_action, shoot_action, 0, 0]
            step_result = self.env.step(combined_action_list)
            next_state, reward_tuple, terminated = step_result

            if not isinstance(next_state, State):
                raise TypeError(
                    f"Environment step returned invalid state type: {type(next_state)}"
                )

            reward_hero = float(reward_tuple[0])
            reward_gun = float(reward_tuple[1])
            truncated = False  # Assume no truncation from env structure

            return next_state, reward_hero, reward_gun, terminated, truncated

        except Exception as e:
            self.logger.error(f"Error during environment step: {e}", exc_info=True)
            return None, 0.0, 0.0, True, True  # Indicate failure, force termination

    def _learn(self) -> None:
        """Performs optimization if enough samples are available and syncs target networks."""
        if len(self.memory) < self.mini_batch_size:
            return  # Wait for more samples

        mini_batch = self.memory.sample(self.mini_batch_size)
        self._optimize_step(mini_batch)
        # Syncing happens based on steps taken, checked in optimize step helper
        self._sync_target_networks_if_needed()

    def _optimize_step(self, mini_batch: List[Tuple]) -> None:
        """Performs a single optimization step for both Hero and Gun networks."""
        (
            states,
            move_actions,
            shoot_actions,
            next_states,
            hero_rewards,
            gun_rewards,
            terminations,
        ) = zip(*mini_batch)

        try:
            # Preprocess states for policy networks
            h_graph_s, g_graph_s = self._preprocess_batch(
                states, self.hero_policy_net, self.gun_policy_net
            )
            # Preprocess next_states for target networks
            # Assuming target preprocessing is same as policy for now
            h_graph_ns, g_graph_ns = self._preprocess_batch(
                next_states, self.hero_target_net, self.gun_target_net
            )
        except (
            ValueError
        ):  # Raised by _preprocess_batch if filtering results in empty batch
            self.logger.warning(
                "Skipping optimization step due to preprocessing failure."
            )
            return

        # Convert components to tensors
        move_actions_t = torch.tensor(
            move_actions, dtype=torch.long, device=self.device
        )
        shoot_actions_t = torch.tensor(
            shoot_actions, dtype=torch.long, device=self.device
        )
        hero_rewards_t = torch.tensor(
            hero_rewards, dtype=torch.float, device=self.device
        )
        gun_rewards_t = torch.tensor(gun_rewards, dtype=torch.float, device=self.device)
        terminations_t = torch.tensor(
            terminations, dtype=torch.float, device=self.device
        )

        # --- Optimize Hero Network ---
        self._calculate_and_apply_loss(
            self.hero_policy_net,
            self.hero_target_net,
            self.hero_optimizer,
            h_graph_s,
            move_actions_t,
            h_graph_ns,
            hero_rewards_t,
            terminations_t,
        )

        # --- Optimize Gun Network ---
        self._calculate_and_apply_loss(
            self.gun_policy_net,
            self.gun_target_net,
            self.gun_optimizer,
            g_graph_s,
            shoot_actions_t,
            g_graph_ns,
            gun_rewards_t,
            terminations_t,
        )

    def _preprocess_batch(
        self, states: Tuple[State], net1: nn.Module, net2: nn.Module
    ) -> Tuple[Batch, Batch]:
        """Preprocesses a batch of states for two networks."""
        graphs1_list = []
        graphs2_list = []
        for state in states:
            # Individual preprocessing can fail, handle gracefully
            try:
                graph1 = net1.preprocess_state(state)
                graph2 = net2.preprocess_state(state)
                if graph1 is not None and graph2 is not None:
                    graphs1_list.append(graph1)
                    graphs2_list.append(graph2)
                # else: log warning? - Handled if list becomes empty
            except Exception as e:
                self.logger.warning(
                    f"Error preprocessing state in batch: {e}", exc_info=False
                )  # Avoid flooding logs

        if not graphs1_list:  # Check if any valid graphs were produced
            raise ValueError("Empty batch after preprocessing")  # Signal failure

        batch1 = Batch.from_data_list(graphs1_list).to(self.device)
        batch2 = Batch.from_data_list(graphs2_list).to(self.device)
        return batch1, batch2

    def _calculate_and_apply_loss(
        self,
        policy_net: nn.Module,
        target_net: nn.Module,
        optimizer: optim.Optimizer,
        batch_states: Batch,
        actions_t: torch.Tensor,
        batch_next_states: Batch,
        rewards_t: torch.Tensor,
        terminations_t: torch.Tensor,
    ) -> None:
        """Calculates loss, performs optimization for one network, stores loss."""
        # --- Target Q Calculation ---
        # Robust Target Calculation: Filter batch_next_states for non-terminal indices BEFORE passing to target_net
        # This requires map from original batch index to graph index in batch_next_states if filtering happens in _preprocess_batch
        # --- Current Simplified approach (assumes target_net handles empty graphs / batching robustly) ---
        non_terminal_mask = terminations_t == 0
        # If all are terminal, next_q is zeros
        if not non_terminal_mask.any():
            max_next_q = torch.zeros_like(rewards_t)
        else:
            # Need a way to select only non_terminal graphs for target_net input
            # For simplicity now, pass all and zero out terminal state values later
            target_net.eval()
            with torch.no_grad():
                # Pass all next states; assumes target_net handles potential empty graphs if preprocessing failed for some
                next_q_values_all = target_net(batch_next_states)
                # Select max Q value
                max_next_q_all = next_q_values_all.max(dim=1)[0]
                # Zero out Q-values for terminal states
                max_next_q = torch.zeros_like(rewards_t)
                # This assumes indices align directly; Needs care if filtering happened!
                max_next_q[non_terminal_mask] = max_next_q_all[non_terminal_mask]

        target_q = (
            rewards_t + self.discount_factor * max_next_q
        )  # (1 - terminations_t) implicitly handled by zeroing out terminal Qs

        # --- Current Q Calculation ---
        policy_net.train()
        current_q_all = policy_net(batch_states)
        current_q = current_q_all.gather(
            dim=1, index=actions_t.unsqueeze(dim=1)
        ).squeeze(dim=1)

        # --- Optimization ---
        loss = self.loss_fn(current_q, target_q)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), clip_value=1.0)
        optimizer.step()

        # Store loss
        loss_value = loss.item()
        if policy_net is self.hero_policy_net:
            self.hero_loss_deque.append(loss_value)
        elif policy_net is self.gun_policy_net:
            self.gun_loss_deque.append(loss_value)

    def _sync_target_networks_if_needed(self) -> None:
        """Copies weights from policy networks to target networks periodically."""
        if self.sync_steps_taken >= self.target_sync_rate:
            self.logger.info(
                f"Syncing target networks (steps: {self.sync_steps_taken})"
            )
            self.hero_target_net.load_state_dict(self.hero_policy_net.state_dict())
            self.gun_target_net.load_state_dict(self.gun_policy_net.state_dict())
            self.hero_target_net.eval()
            self.gun_target_net.eval()
            self.sync_steps_taken = 0  # Reset counter

    def _decay_epsilon(self) -> None:
        """Decays the exploration rate."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def _save_checkpoint_if_needed(self, episode: int) -> None:
        """Saves model checkpoints periodically."""
        # Use self.save_interval configured during init
        if self.save_interval > 0 and episode > 0 and episode % self.save_interval == 0:
            self.logger.info(f"Saving checkpoint at episode {episode}")
            save_path = self.dump()  # Call internal dump method
            if save_path:
                self.logger.info(f"Checkpoint saved to: {save_path}")
            else:
                self.logger.error(f"Failed to save checkpoint for episode {episode}.")

    def dump(self, save_dir: str = "model_saves") -> Optional[str]:
        """Saves agent state (networks, optimizers) to a timestamped directory."""
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        base_name = f"theseus_gnn_{timestamp}"
        dpath = os.path.join(save_dir, base_name)

        try:
            os.makedirs(dpath, exist_ok=True)
            self.logger.info(f"Saving agent state to: {dpath}")

            hero_policy_file = "hero_policy.pth"
            hero_target_file = "hero_target.pth"
            gun_policy_file = "gun_policy.pth"
            gun_target_file = "gun_target.pth"
            hero_optim_file = "hero_optimizer.pth"
            gun_optim_file = "gun_optimizer.pth"

            # Save networks
            torch.save(self.hero_policy_net, os.path.join(dpath, hero_policy_file))
            torch.save(self.hero_target_net, os.path.join(dpath, hero_target_file))
            torch.save(self.gun_policy_net, os.path.join(dpath, gun_policy_file))
            torch.save(self.gun_target_net, os.path.join(dpath, gun_target_file))

            # Save optimizers
            torch.save(
                self.hero_optimizer.state_dict(), os.path.join(dpath, hero_optim_file)
            )
            torch.save(
                self.gun_optimizer.state_dict(), os.path.join(dpath, gun_optim_file)
            )

            # Save hyperparameters and state
            state_info = {
                "hero_policy_file": hero_policy_file,
                "hero_target_file": hero_target_file,
                "gun_policy_file": gun_policy_file,
                "gun_target_file": gun_target_file,
                "hero_optim_file": hero_optim_file,
                "gun_optim_file": gun_optim_file,
                "hero_policy_class": type(self.hero_policy_net).__name__,
                "gun_policy_class": type(self.gun_policy_net).__name__,
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
                # Consider saving total rewards if needed for tracking across loads
                "total_reward_hero": self.total_reward_hero,
                "total_reward_gun": self.total_reward_gun,
            }
            yaml_path = os.path.join(dpath, f"{base_name}.yaml")
            with open(yaml_path, "w") as f:
                yaml.dump(state_info, f, default_flow_style=False)

            self.logger.info("Agent state saved successfully.")
            return dpath

        except Exception as e:
            self.logger.error(f"Failed to dump agent state: {e}", exc_info=True)
            return None

    @classmethod
    def load(cls, load_path: str | os.PathLike) -> Optional[Self]:
        """Loads agent state from a specified directory."""
        logger = logging.getLogger("agent-theseus-gnn-load")
        logger.info(f"Attempting to load agent state from: {load_path}")

        if not os.path.isdir(load_path):
            logger.error(f"Load path is not a valid directory: {load_path}")
            return None

        base_name = os.path.basename(load_path)
        yaml_path = os.path.join(load_path, f"{base_name}.yaml")
        if not os.path.exists(yaml_path):
            logger.error(f"YAML configuration file not found: {yaml_path}")
            return None
        try:
            with open(yaml_path, "r") as f:
                state_info = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error reading YAML {yaml_path}: {e}", exc_info=True)
            return None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading models onto device: {device}")

        try:
            hp_path = os.path.join(load_path, state_info["hero_policy_file"])
            ht_path = os.path.join(load_path, state_info["hero_target_file"])
            gp_path = os.path.join(load_path, state_info["gun_policy_file"])
            gt_path = os.path.join(load_path, state_info["gun_target_file"])

            hero_policy_net = torch.load(hp_path, map_location=device)
            hero_target_net = torch.load(ht_path, map_location=device)
            gun_policy_net = torch.load(gp_path, map_location=device)
            gun_target_net = torch.load(gt_path, map_location=device)

            if not all(
                isinstance(net, nn.Module)
                for net in [
                    hero_policy_net,
                    hero_target_net,
                    gun_policy_net,
                    gun_target_net,
                ]
            ):
                raise TypeError("Loaded network file is not a valid nn.Module.")
        except Exception as e:
            logger.error(f"Error loading network models: {e}", exc_info=True)
            return None

        learning_rate = state_info.get("learning_rate", 1e-4)
        optimizer_class: Type[optim.Optimizer] = optim.AdamW
        try:
            hero_optimizer = optimizer_class(
                hero_policy_net.parameters(), lr=learning_rate
            )
            gun_optimizer = optimizer_class(
                gun_policy_net.parameters(), lr=learning_rate
            )
        except Exception as e:
            logger.error(f"Failed to create optimizers: {e}", exc_info=True)
            return None

        try:
            ho_path = os.path.join(load_path, state_info["hero_optim_file"])
            go_path = os.path.join(load_path, state_info["gun_optim_file"])
            if os.path.exists(ho_path):
                hero_optimizer.load_state_dict(torch.load(ho_path, map_location=device))
            else:
                logger.warning(f"Hero optimizer state file not found: {ho_path}")
            if os.path.exists(go_path):
                gun_optimizer.load_state_dict(torch.load(go_path, map_location=device))
            else:
                logger.warning(f"Gun optimizer state file not found: {go_path}")
        except Exception as e:
            logger.error(f"Error loading optimizer states: {e}", exc_info=True)
            # Continue with fresh optimizers

        env = Environment()
        loss_fn_class: Type[nn.Module] = nn.MSELoss

        try:
            agent = cls(
                hero_policy_net=hero_policy_net,
                hero_target_net=hero_target_net,
                gun_policy_net=gun_policy_net,
                gun_target_net=gun_target_net,
                env=env,
                loss_fn_class=loss_fn_class,
                optimizer_class=optimizer_class,
                learning_rate=learning_rate,
                discount_factor=state_info.get("discount_factor", 0.99),
                epsilon_init=state_info.get("epsilon", 0.05),
                epsilon_decay=state_info.get("epsilon_decay", 0.9995),
                epsilon_min=state_info.get("epsilon_min", 0.05),
                mini_batch_size=state_info.get("mini_batch_size", 64),
                target_sync_rate=state_info.get("target_sync_rate", 500),
                log_window_size=state_info.get("log_window_size", LOGGING_WINDOW),
                save_interval=state_info.get("save_interval", SAVE_INTERVAL),
            )
        except Exception as e:
            logger.error(
                f"Error instantiating AgentTheseusGNN during load: {e}", exc_info=True
            )
            return None

        agent.hero_optimizer = hero_optimizer
        agent.gun_optimizer = gun_optimizer
        agent.epsilon = state_info.get("epsilon", agent.epsilon_min)
        agent.sync_steps_taken = state_info.get("sync_steps_taken", 0)
        # Restore cumulative rewards if needed for consistent logging across runs
        agent.total_reward_hero = state_info.get("total_reward_hero", 0.0)
        agent.total_reward_gun = state_info.get("total_reward_gun", 0.0)

        agent.hero_target_net.eval()
        agent.gun_target_net.eval()

        logger.info(f"Agent loaded successfully from {load_path}")
        return agent
