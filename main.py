import logging
import torch
import os
from rich.logging import RichHandler

# Adjust imports
from theseus.agent_gnn import AgentTheseusGNN # Import the new agent
from theseus.models.GraphDQN.ActionGNN import HeroGNN, GunGNN
from theseus.utils.network import Environment
import theseus.constants as c

# --- Configuration (same as before) ---
HIDDEN_CHANNELS = 16
HERO_ACTION_SPACE_SIZE = 9
GUN_ACTION_SPACE_SIZE = 8
# ... other hyperparameters ...
LEARNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.99
# ... etc ...

# --- Main Training Function ---
def train():
    """Initializes and trains the combined AgentTheseusGNN."""
    train_logger = logging.getLogger("train_loop")
    train_logger.info("Initializing environment and combined GNN agent...")

    env = Environment()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_logger.info(f"Using device: {device}")

    # --- Initialize Networks ---
    # Ensure input channels match graph_rep.py if GunGNN needs it specified
    hero_policy = HeroGNN(hidden_channels=HIDDEN_CHANNELS, out_channels=HERO_ACTION_SPACE_SIZE)
    hero_target = HeroGNN(hidden_channels=HIDDEN_CHANNELS, out_channels=HERO_ACTION_SPACE_SIZE)
    gun_policy = GunGNN(hidden_channels=HIDDEN_CHANNELS, out_channels=GUN_ACTION_SPACE_SIZE) # Add in_channels if needed
    gun_target = GunGNN(hidden_channels=HIDDEN_CHANNELS, out_channels=GUN_ACTION_SPACE_SIZE) # Add in_channels if needed

    # --- Initialize Combined Agent ---
    try:
        agent = AgentTheseusGNN(
            hero_policy_net=hero_policy,
            hero_target_net=hero_target,
            gun_policy_net=gun_policy,
            gun_target_net=gun_target,
            env=env, # Pass the environment instance
            learning_rate=LEARNING_RATE,
            discount_factor=DISCOUNT_FACTOR,
            # ... pass other hyperparameters ...
        )
        train_logger.info("AgentTheseusGNN initialized.")
    except Exception as e:
        train_logger.critical(f"Failed to initialize AgentTheseusGNN: {e}", exc_info=True)
        return

    # --- Start Training ---
    # The train method now contains the loop
    agent.train() # You can optionally pass num_episodes

# --- Main Execution ---
if __name__ == "__main__":
    # --- Logging Setup (same as before) ---
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO",
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
    )
    logging.getLogger("torch_geometric").setLevel(logging.WARNING)

    main_logger = logging.getLogger("main")
    main_logger.info("[bold green] Starting Theseus GNN Training [/]", extra={"markup": True})

    try:
        train()
    except KeyboardInterrupt:
        main_logger.warning("Training interrupted by user.")
    except Exception as main_e:
        main_logger.critical(f"Unhandled exception in main training loop: {main_e}", exc_info=True)
    finally:
        main_logger.info("[bold red] Training finished or interrupted. [/]", extra={"markup": True})
        # Add cleanup if needed
