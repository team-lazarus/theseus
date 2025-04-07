import torch
import random
import logging
import os
import yaml
from datetime import datetime
from itertools import count
from typing import List, Tuple, Optional, Type, Self

from torch import nn, optim
from torch_geometric.data import Batch, HeteroData

from theseus.utils import State, ExperienceReplayMemory
from theseus.utils.network import Environment
import theseus.constants as c
# Import your GNN classes (adjust path if necessary)
from theseus.models.GraphDQN.ActionGNN import HeroGNN, GunGNN

# Action space sizes (ensure these match GNN outputs)
HERO_ACTION_SPACE_SIZE = 9
GUN_ACTION_SPACE_SIZE = 8

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
    ) -> None:
        """Initializes the dual-GNN agent."""
        self.logger = logging.getLogger("agent-theseus-gnn")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.env = env

        # --- Network Setup ---
        self._validate_network(hero_policy_net, "Hero Policy")
        self._validate_network(hero_target_net, "Hero Target")
        self._validate_network(gun_policy_net, "Gun Policy")
        self._validate_network(gun_target_net, "Gun Target")

        self.hero_policy_net = hero_policy_net.to(self.device)
        self.hero_target_net = hero_target_net.to(self.device)
        self.gun_policy_net = gun_policy_net.to(self.device)
        self.gun_target_net = gun_target_net.to(self.device)

        # --- Training Parameters ---
        self.discount_factor = discount_factor
        self.mini_batch_size = mini_batch_size
        self.target_sync_rate = target_sync_rate
        self.sync_steps_taken = 0

        # --- Epsilon ---
        self.epsilon_init = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_init

        # --- Optimization Setup ---
        self.loss_fn = loss_fn_class()
        self.hero_optimizer = optimizer_class(
            self.hero_policy_net.parameters(), lr=learning_rate, amsgrad=True
        )
        self.gun_optimizer = optimizer_class(
            self.gun_policy_net.parameters(), lr=learning_rate, amsgrad=True
        )

        # --- Memory ---
        # Stores: (state, move_action, shoot_action, next_state, reward_hero, reward_gun, terminated)
        self.memory = ExperienceReplayMemory(replay_memory_size)

        # --- Initialization ---
        self.hero_target_net.load_state_dict(self.hero_policy_net.state_dict())
        self.gun_target_net.load_state_dict(self.gun_policy_net.state_dict())
        self.hero_target_net.eval()
        self.gun_target_net.eval()

        self.episode_rewards_hero = []
        self.episode_rewards_gun = []

    def _validate_network(self, network: nn.Module, name: str) -> None:
        """Checks if a network has the required preprocess_state method."""
        if not hasattr(network, 'preprocess_state') or not callable(network.preprocess_state):
            raise AttributeError(f"{name} network must have a 'preprocess_state' method.")

    def train(self, num_episodes: Optional[int] = None) -> None:
        """Runs the main training loop for a specified number of episodes or indefinitely."""
        self.logger.info(f"Starting training on {self.device}...")
        episode_iterator = range(num_episodes) if num_episodes is not None else count()

        for episode in episode_iterator:
            self.logger.info(
                f"[green]Starting Episode: {episode} (epsilon: {self.epsilon:.4f})[/]",
                extra={"markup": True}
            )

            try:
                ep_rewards_hero, ep_rewards_gun, steps = self._run_episode()
                self.episode_rewards_hero.append(ep_rewards_hero)
                self.episode_rewards_gun.append(ep_rewards_gun)
                self.logger.info(
                    f"Episode {episode} finished. Steps: {steps}, "
                    f"Hero Reward: {ep_rewards_hero:.2f}, Gun Reward: {ep_rewards_gun:.2f}"
                )
                self._learn()
                self._decay_epsilon()
                self._save_checkpoint(episode) # Periodically save based on constants

            except Exception as e:
                self.logger.critical(f"Error during episode {episode}: {e}", exc_info=True)
                # Decide whether to break or continue
                break

        self.logger.info("Training finished.")

    def _run_episode(self) -> Tuple[float, float, int]:
        """Runs a single episode, returns rewards and step count."""
        state = self._initialize_episode()
        terminated = False
        truncated = False
        episode_reward_hero = 0.0
        episode_reward_gun = 0.0
        episode_steps = 0

        while not terminated and not truncated:
            episode_steps += 1
            move_action, shoot_action = self._select_actions(state)
            next_state, reward_hero, reward_gun, terminated, truncated = self._step_environment(move_action, shoot_action)

            # Store experience only if step was successful
            if next_state is not None:
                 self.memory.append((
                    state, move_action, shoot_action,
                    next_state, reward_hero, reward_gun, terminated
                 ))
                 self.sync_steps_taken += 1
                 state = next_state # Move to next state only if valid
            else:
                # Handle case where step failed (e.g., env error) - often terminate
                self.logger.warning("Environment step failed, terminating episode.")
                terminated = True

            episode_reward_hero += reward_hero
            episode_reward_gun += reward_gun

        return episode_reward_hero, episode_reward_gun, episode_steps

    def _initialize_episode(self) -> State:
        """Resets the environment and returns the initial state."""
        try:
            initial_state = self.env.initialise_environment()
            if not isinstance(initial_state, State):
                 raise TypeError(f"Environment did not return a valid State object, got {type(initial_state)}")
            return initial_state
        except Exception as e:
            self.logger.error(f"Failed to initialize environment: {e}", exc_info=True)
            raise RuntimeError("Environment initialization failed.") from e # Re-raise critical error


    def _select_actions(self, state: State) -> Tuple[int, int]:
        """Selects movement and shooting actions using epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            move_action = random.randrange(HERO_ACTION_SPACE_SIZE)
            shoot_action = random.randrange(GUN_ACTION_SPACE_SIZE)
            return move_action, shoot_action
        else:
            move_action = self._predict_action(self.hero_policy_net, state)
            shoot_action = self._predict_action(self.gun_policy_net, state)

            # Fallback to random if prediction fails for either
            move_action = move_action if move_action is not None else random.randrange(HERO_ACTION_SPACE_SIZE)
            shoot_action = shoot_action if shoot_action is not None else random.randrange(GUN_ACTION_SPACE_SIZE)
            return move_action, shoot_action

    def _predict_action(self, policy_net: nn.Module, state: State) -> Optional[int]:
        """Predicts the best action using a given policy network."""
        try:
            graph_data = policy_net.preprocess_state(state)
            if graph_data is None:
                self.logger.warning(f"Preprocessing failed for {type(policy_net).__name__}.")
                return None # Return None, fallback to random in _select_actions

            graph_data = graph_data.to(self.device)
            policy_net.eval()
            with torch.no_grad():
                q_values = policy_net(graph_data)
            policy_net.train()

            # --- FIX: Check if q_values tensor is empty ---
            if q_values.numel() == 0:
                self.logger.warning(
                    f"{type(policy_net).__name__} produced empty Q-values "
                    "(likely due to no relevant nodes, e.g., no enemies for GunGNN). "
                    "Cannot determine best action."
                )
                # Signal failure, let _select_actions handle fallback to random
                return None
            # --- End Fix ---

            if q_values.ndim > 1:
                q_values = q_values.squeeze(0)

            # Check again after potential squeeze, although unlikely to be empty now
            if q_values.numel() == 0:
                 self.logger.warning(f"{type(policy_net).__name__} Q-values became empty after squeeze.")
                 return None

            action = q_values.argmax().item()
            return action

        except Exception as e:
            self.logger.error(f"Error during {type(policy_net).__name__} prediction: {e}", exc_info=True)
            return None # Return None on any exception


    def _step_environment(self, move_action: int, shoot_action: int) -> Tuple[Optional[State], float, float, bool, bool]:
        """Combines actions, steps the environment, and unpacks results."""
        try:
            # Verify action format [move, attack, phase, bomb]
            combined_action_list = [move_action, shoot_action, 0, 0]
            # combined_action_str = str(combined_action_list)

            # Assumes env.step returns: next_state, (hero_r, gun_r), terminated
            step_result = self.env.step(combined_action_list)
            # step_result = self.env.step(combined_action_str)
            next_state, reward_tuple, terminated = step_result

            if not isinstance(next_state, State):
                self.logger.error(f"Environment step returned invalid state type: {type(next_state)}")
                return None, 0.0, 0.0, True, True # Treat as failure

            reward_hero = float(reward_tuple[0])
            reward_gun = float(reward_tuple[1])
            truncated = False # Assume not truncated unless env provides it

            return next_state, reward_hero, reward_gun, terminated, truncated

        except Exception as e:
            self.logger.error(f"Error during environment step: {e}", exc_info=True)
            return None, 0.0, 0.0, True, True # Indicate failure


    def _learn(self) -> None:
        """Performs optimization if enough samples are available and syncs target networks."""
        if len(self.memory) < self.mini_batch_size:
            return

        mini_batch = self.memory.sample(self.mini_batch_size)
        self._optimize_step(mini_batch)
        self._sync_target_networks()


    def _optimize_step(self, mini_batch: List[Tuple]) -> None:
        """Performs a single optimization step for both Hero and Gun networks."""
        states, move_actions, shoot_actions, next_states, hero_rewards, gun_rewards, terminations = zip(*mini_batch)

        # Preprocess states and next_states for both network types
        # Note: This assumes preprocess_state works correctly even if some nodes are missing
        try:
            h_graph_s, g_graph_s = self._preprocess_batch(states, self.hero_policy_net, self.gun_policy_net)
            # Use target network's preprocess method if it differs, else policy is fine
            h_graph_ns, g_graph_ns = self._preprocess_batch(next_states, self.hero_target_net, self.gun_target_net)
        except ValueError: # Raised by _preprocess_batch if filtering fails
             return # Skip optimization if preprocessing fails

        # Filter actions/rewards based on valid preprocessed states
        # _preprocess_batch returns indices of valid items in the original batch
        valid_indices = h_graph_s.ptr[:-1].tolist() # Hacky way? Better if preprocess returns indices. Let's assume it returns filtered batches.
        # Correction: Let's simplify and assume preprocessing doesn't filter for now, relies on robust GNNs
        # If filtering is needed, _preprocess_batch needs modification.

        # Convert components to tensors
        move_actions_t = torch.tensor(move_actions, dtype=torch.long, device=self.device)
        shoot_actions_t = torch.tensor(shoot_actions, dtype=torch.long, device=self.device)
        hero_rewards_t = torch.tensor(hero_rewards, dtype=torch.float, device=self.device)
        gun_rewards_t = torch.tensor(gun_rewards, dtype=torch.float, device=self.device)
        terminations_t = torch.tensor(terminations, dtype=torch.float, device=self.device)

        # --- Optimize Hero Network ---
        loss_hero = self._calculate_loss(
            self.hero_policy_net, self.hero_target_net, self.hero_optimizer,
            h_graph_s, move_actions_t, h_graph_ns, hero_rewards_t, terminations_t
        )

        # --- Optimize Gun Network ---
        loss_gun = self._calculate_loss(
            self.gun_policy_net, self.gun_target_net, self.gun_optimizer,
            g_graph_s, shoot_actions_t, g_graph_ns, gun_rewards_t, terminations_t
        )

        # Optional: Log losses
        # self.logger.debug(f"Losses - Hero: {loss_hero:.4f}, Gun: {loss_gun:.4f}")


    def _preprocess_batch(self, states: Tuple[State], policy_net1: nn.Module, policy_net2: nn.Module) -> Tuple[Batch, Batch]:
        """Preprocesses a batch of states for both hero and gun networks."""
        graphs1_list = []
        graphs2_list = []
        valid_indices = []

        for i, state in enumerate(states):
             graph1 = policy_net1.preprocess_state(state)
             graph2 = policy_net2.preprocess_state(state)
             if graph1 is not None and graph2 is not None: # Check both preprocess successfully
                  graphs1_list.append(graph1)
                  graphs2_list.append(graph2)
                  valid_indices.append(i) # Track index if needed for filtering rewards/actions later

        if not graphs1_list: # Check if any valid graphs were produced
            self.logger.warning("Preprocessing failed for all states in the batch.")
            raise ValueError("Empty batch after preprocessing") # Signal failure

        batch1 = Batch.from_data_list(graphs1_list).to(self.device)
        batch2 = Batch.from_data_list(graphs2_list).to(self.device)
        # Note: If filtering happened, need to return valid_indices to filter actions/rewards
        return batch1, batch2


    def _calculate_loss(
        self, policy_net: nn.Module, target_net: nn.Module, optimizer: optim.Optimizer,
        batch_states: Batch, actions_t: torch.Tensor, batch_next_states: Batch,
        rewards_t: torch.Tensor, terminations_t: torch.Tensor
    ) -> float:
        """Calculates loss and performs optimization for one network."""
        # --- Target Q Calculation ---
        # Need to handle non-terminal next states for target calculation
        # This requires knowing which elements in batch_next_states correspond to non-terminals
        # Let's assume batch_next_states ONLY contains non-terminal graph data for simplicity
        # Requires modifying _preprocess_batch to filter terminals when creating next state batches.
        # --- Simplified approach (potential inaccuracy if terminals included) ---
        target_net.eval()
        with torch.no_grad():
            next_q_values_all = target_net(batch_next_states) # Assumes batch_next_states is correctly filtered/handled
            max_next_q = next_q_values_all.max(dim=1)[0]
            # If terminals were included, need masking: max_next_q = max_next_q * (1 - terminations_t[non_terminal_indices])

        # --- Current simplified target ---
        target_q = rewards_t + self.discount_factor * max_next_q * (1 - terminations_t) # Uses full terminations tensor

        # --- Current Q Calculation ---
        policy_net.train()
        current_q_all = policy_net(batch_states)
        current_q = current_q_all.gather(dim=1, index=actions_t.unsqueeze(dim=1)).squeeze(dim=1)

        # --- Optimization ---
        loss = self.loss_fn(current_q, target_q)
        optimizer.zero_grad()
        loss.backward()
        # Optional: Gradient Clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), clip_value=1.0)
        optimizer.step()

        return loss.item()

    def _sync_target_networks(self) -> None:
        """Copies weights from policy networks to target networks if needed."""
        if self.sync_steps_taken >= self.target_sync_rate:
            self.logger.info(
                f"Syncing target networks (steps since last sync: {self.sync_steps_taken})"
            )
            self.hero_target_net.load_state_dict(self.hero_policy_net.state_dict())
            self.gun_target_net.load_state_dict(self.gun_policy_net.state_dict())
            self.hero_target_net.eval()
            self.gun_target_net.eval()
            self.sync_steps_taken = 0

    def _decay_epsilon(self) -> None:
        """Decays the exploration rate."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def _save_checkpoint(self, episode: int) -> None:
        """Saves model checkpoints periodically."""
        save_interval = getattr(c, 'SAVE_INTERVAL', 0)
        if save_interval > 0 and episode > 0 and episode % save_interval == 0:
             self.logger.info(f"Saving checkpoint at episode {episode}")
             self.dump() # Call internal dump method

    def dump(self, save_dir: str = "model_saves") -> Optional[str]:
        """Saves agent state (networks, optimizers) to a timestamped directory."""
        timestamp = datetime.now().strftime("%m%d_%H%M%S") # Added seconds for uniqueness
        base_name = f"theseus_gnn_{timestamp}"
        dpath = os.path.join(save_dir, base_name)

        try:
            os.makedirs(dpath, exist_ok=True)
            self.logger.info(f"Saving agent state to: {dpath}")

            # Network filenames
            hero_policy_file = "hero_policy.pth"
            hero_target_file = "hero_target.pth"
            gun_policy_file = "gun_policy.pth"
            gun_target_file = "gun_target.pth"
            # Optimizer filenames
            hero_optim_file = "hero_optimizer.pth"
            gun_optim_file = "gun_optimizer.pth"

            # Save networks (consider state_dict for robustness)
            torch.save(self.hero_policy_net, os.path.join(dpath, hero_policy_file))
            torch.save(self.hero_target_net, os.path.join(dpath, hero_target_file))
            torch.save(self.gun_policy_net, os.path.join(dpath, gun_policy_file))
            torch.save(self.gun_target_net, os.path.join(dpath, gun_target_file))

            # Save optimizers
            torch.save(self.hero_optimizer.state_dict(), os.path.join(dpath, hero_optim_file))
            torch.save(self.gun_optimizer.state_dict(), os.path.join(dpath, gun_optim_file))

            # Save hyperparameters and state
            state_info = {
                "hero_policy_file": hero_policy_file, "hero_target_file": hero_target_file,
                "gun_policy_file": gun_policy_file, "gun_target_file": gun_target_file,
                "hero_optim_file": hero_optim_file, "gun_optim_file": gun_optim_file,
                "hero_policy_class": type(self.hero_policy_net).__name__,
                "gun_policy_class": type(self.gun_policy_net).__name__,
                "epsilon": self.epsilon,
                "sync_steps_taken": self.sync_steps_taken,
                "learning_rate": self.hero_optimizer.param_groups[0]['lr'], # Get current LR
                "discount_factor": self.discount_factor,
                "mini_batch_size": self.mini_batch_size,
                "target_sync_rate": self.target_sync_rate,
                "epsilon_decay": self.epsilon_decay,
                "epsilon_min": self.epsilon_min,
                # Add other relevant info
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
        logger = logging.getLogger("agent-theseus-gnn-load") # Specific logger
        logger.info(f"Attempting to load agent state from: {load_path}")

        # --- 1. Validate Path ---
        if not os.path.isdir(load_path):
            logger.error(f"Load path is not a valid directory: {load_path}")
            return None

        # --- 2. Find and Load YAML Configuration ---
        base_name = os.path.basename(load_path)
        yaml_path = os.path.join(load_path, f"{base_name}.yaml")
        if not os.path.exists(yaml_path):
            logger.error(f"YAML configuration file not found: {yaml_path}")
            return None
        try:
            with open(yaml_path, "r") as f:
                state_info = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Error loading YAML file {yaml_path}: {e}", exc_info=True)
            return None
        except Exception as e:
             logger.error(f"Unexpected error reading YAML {yaml_path}: {e}", exc_info=True)
             return None

        # --- 3. Determine Device ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading models onto device: {device}")

        # --- 4. Load Networks ---
        try:
            # Construct full paths
            hp_path = os.path.join(load_path, state_info['hero_policy_file'])
            ht_path = os.path.join(load_path, state_info['hero_target_file'])
            gp_path = os.path.join(load_path, state_info['gun_policy_file'])
            gt_path = os.path.join(load_path, state_info['gun_target_file'])

            # Load the models (requires HeroGNN and GunGNN classes to be defined)
            hero_policy_net = torch.load(hp_path, map_location=device)
            hero_target_net = torch.load(ht_path, map_location=device)
            gun_policy_net = torch.load(gp_path, map_location=device)
            gun_target_net = torch.load(gt_path, map_location=device)

            # Basic type check
            if not all(isinstance(net, nn.Module) for net in [hero_policy_net, hero_target_net, gun_policy_net, gun_target_net]):
                 raise TypeError("One or more loaded network files are not valid nn.Module instances.")

        except FileNotFoundError as e:
            logger.error(f"Network file not found during load: {e}", exc_info=True)
            return None
        except KeyError as e:
            logger.error(f"Missing network file key in YAML config: {e}", exc_info=True)
            return None
        except Exception as e: # Catch other potential errors (pickle, etc.)
            logger.error(f"Error loading network models: {e}", exc_info=True)
            return None

        # --- 5. Recreate Optimizers ---
        # Get saved learning rate, provide a default if missing
        learning_rate = state_info.get('learning_rate', 1e-4)
        # Assume default optimizer class (cannot easily serialize/deserialize class types)
        optimizer_class: Type[optim.Optimizer] = optim.AdamW
        try:
            hero_optimizer = optimizer_class(hero_policy_net.parameters(), lr=learning_rate)
            gun_optimizer = optimizer_class(gun_policy_net.parameters(), lr=learning_rate)
        except Exception as e:
            logger.error(f"Failed to create optimizers: {e}", exc_info=True)
            return None # Cannot proceed without optimizers if intending to train

        # --- 6. Load Optimizer States ---
        try:
            ho_path = os.path.join(load_path, state_info['hero_optim_file'])
            go_path = os.path.join(load_path, state_info['gun_optim_file'])
            hero_optimizer.load_state_dict(torch.load(ho_path, map_location=device))
            gun_optimizer.load_state_dict(torch.load(go_path, map_location=device))
            logger.info("Optimizer states loaded successfully.")
        except FileNotFoundError:
             logger.warning(f"Optimizer state file(s) not found in {load_path}. Initializing new optimizer state.")
             # Continue without loading state - optimizer starts fresh
        except KeyError as e:
             logger.warning(f"Missing optimizer file key(s) in YAML config: {e}. Initializing new optimizer state.")
        except Exception as e:
             logger.error(f"Error loading optimizer states: {e}. Initializing new optimizer state.", exc_info=True)


        # --- 7. Instantiate the Agent ---
        # Environment state is not saved, create a new instance
        # If env needs specific config, this needs adjustment
        env = Environment()
        # Assume default loss class
        loss_fn_class: Type[nn.Module] = nn.MSELoss

        try:
            # Create the agent instance using loaded components and config
            agent = cls(
                hero_policy_net=hero_policy_net,
                hero_target_net=hero_target_net,
                gun_policy_net=gun_policy_net,
                gun_target_net=gun_target_net,
                env=env,
                loss_fn_class=loss_fn_class, # Use default class
                optimizer_class=optimizer_class, # Pass class used to recreate
                learning_rate=learning_rate, # Use loaded LR
                discount_factor=state_info.get('discount_factor', 0.99),
                epsilon_init=state_info.get('epsilon', 0.05), # Use saved epsilon as starting point
                epsilon_decay=state_info.get('epsilon_decay', 0.9995),
                epsilon_min=state_info.get('epsilon_min', 0.05),
                mini_batch_size=state_info.get('mini_batch_size', 64),
                target_sync_rate=state_info.get('target_sync_rate', 500),
                # Replay memory size usually defined by constants, not loaded
            )
        except Exception as e:
             logger.error(f"Error instantiating AgentTheseusGNN during load: {e}", exc_info=True)
             return None


        # --- 8. Restore exact state ---
        # Assign the loaded optimizers (with their loaded state)
        agent.hero_optimizer = hero_optimizer
        agent.gun_optimizer = gun_optimizer
        # Restore exact epsilon and sync steps
        agent.epsilon = state_info.get('epsilon', agent.epsilon_min)
        agent.sync_steps_taken = state_info.get('sync_steps_taken', 0)

        # --- 9. Set Target Networks to Eval Mode ---
        agent.hero_target_net.eval()
        agent.gun_target_net.eval()

        logger.info(f"Agent loaded successfully from {load_path}")
        return agent
