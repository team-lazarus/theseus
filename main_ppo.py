import logging
import torch
import os
from rich.logging import RichHandler

# --- PPO Agent Import ---
from theseus.agent_ppo import AgentTheseusPPO # Assuming agent saved as agent_ppo.py

# --- Network Imports (Assuming GNNs can act as Actor/Critic) ---
# If you have separate Critic classes, import them here.
from theseus.models.GraphDQN.ActionGNN import HeroGNN
from theseus.models.GraphDQN.ActionGNN import GunGNN

# --- Utility Imports ---
from theseus.utils.network import Environment
# import theseus.constants as c # Constants might be defined within Agent or here

# --- Configuration ---
HIDDEN_CHANNELS = 16 # Example hidden size
HERO_ACTION_SPACE_SIZE = 9
GUN_ACTION_SPACE_SIZE = 8

# --- PPO Hyperparameters ---
PPO_LEARNING_RATE = 3e-4       # Typical PPO learning rate
PPO_DISCOUNT_FACTOR = 0.99     # Gamma
PPO_HORIZON = 2048             # Steps collected per rollout (N)
PPO_EPOCHS_PER_UPDATE = 10     # Optimization epochs per rollout (K)
PPO_MINI_BATCH_SIZE = 64       # Minibatch size for optimization
PPO_CLIP_EPSILON = 0.2         # PPO clipping parameter (epsilon)
PPO_GAE_LAMBDA = 0.95          # GAE parameter (lambda)
PPO_ENTROPY_COEFF = 0.01       # Entropy bonus coefficient
PPO_VF_COEFF = 0.5             # Value function loss coefficient
PPO_LOG_WINDOW = 50            # Episodes for rolling average metrics
PPO_SAVE_INTERVAL = 200        # Save checkpoint every N episodes
NUM_TRAINING_EPISODES = 50000  # Total episodes to train for

SAVED_MODEL="model_saves_ppo/theseus_ppo_20250414_231934"


def train_ppo():
    """Initializes and trains the combined AgentTheseusPPO."""
    train_logger = logging.getLogger("train_loop_ppo")
    train_logger.info("Initializing environment and PPO GNN agent...")

    env = Environment()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_logger.info(f"Using device: {device}")

    # --- Instantiate Actor and Critic Networks ---
    # CRITICAL ASSUMPTION: HeroGNN/GunGNN can output a single value when out_channels=1
    # If not, you need dedicated Critic GNN classes.
    agent = AgentTheseusPPO.load(SAVED_MODEL)
    if agent == None:
        try:
            hero_actor = HeroGNN(
                hidden_channels=HIDDEN_CHANNELS, out_channels=HERO_ACTION_SPACE_SIZE
            )
            hero_critic = HeroGNN( # Using HeroGNN as critic
                hidden_channels=HIDDEN_CHANNELS, out_channels=1 # Output a single value
            )
            gun_actor = GunGNN(
                hidden_channels=HIDDEN_CHANNELS, out_channels=GUN_ACTION_SPACE_SIZE
            )
            gun_critic = GunGNN( # Using GunGNN as critic
                hidden_channels=HIDDEN_CHANNELS, out_channels=1 # Output a single value
            )
            train_logger.info("Actor and Critic GNNs instantiated.")
        except Exception as e:
            train_logger.critical(f"Failed to instantiate GNNs: {e}", exc_info=True)
            return

        # --- Instantiate PPO Agent ---
        try:
            agent = AgentTheseusPPO(
                hero_actor_net=hero_actor,
                hero_critic_net=hero_critic,
                gun_actor_net=gun_actor,
                gun_critic_net=gun_critic,
                env=env,
                learning_rate=PPO_LEARNING_RATE,
                discount_factor=PPO_DISCOUNT_FACTOR,
                horizon=PPO_HORIZON,
                epochs_per_update=PPO_EPOCHS_PER_UPDATE,
                mini_batch_size=PPO_MINI_BATCH_SIZE,
                clip_epsilon=PPO_CLIP_EPSILON,
                gae_lambda=PPO_GAE_LAMBDA,
                entropy_coeff=PPO_ENTROPY_COEFF,
                vf_coeff=PPO_VF_COEFF,
                log_window_size=PPO_LOG_WINDOW,
                save_interval=PPO_SAVE_INTERVAL,
                # Add other necessary args like optimizer_class if not default
            )
            train_logger.info("AgentTheseusPPO initialized successfully.")
        except Exception as e:
            train_logger.critical(
                f"Failed to initialize AgentTheseusPPO: {e}", exc_info=True
            )
            return

    # --- Start Training ---
    train_logger.info(f"Starting PPO training for {NUM_TRAINING_EPISODES} episodes...")
    agent.train(num_episodes=NUM_TRAINING_EPISODES)


# --- Main Execution ---
if __name__ == "__main__":
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO",
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)], # Enable markup
    )
    # Silence verbose loggers if necessary
    logging.getLogger("torch_geometric").setLevel(logging.WARNING)
    # Add others here if needed, e.g., environment logs if too noisy

    main_logger = logging.getLogger("main_ppo")
    main_logger.info(
        "[bold green]>>> Starting Theseus PPO GNN Training <<<[/]", extra={"markup": True}
    )

    try:
        train_ppo() # Call the PPO training function
    except KeyboardInterrupt:
        main_logger.warning("[bold yellow]>>> Training interrupted by user <<<[/]", extra={"markup": True})
    except Exception as main_e:
        main_logger.critical(
            f"Unhandled exception in main PPO training loop: {main_e}", exc_info=True
        )
    finally:
        main_logger.info(
            "[bold red]>>> PPO Training finished or stopped <<<[/]", extra={"markup": True}
        )
        # Add cleanup if needed (e.g., close environment)
