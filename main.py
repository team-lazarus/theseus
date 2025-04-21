import logging
import torch
import os
from rich.logging import RichHandler
from argparse import ArgumentParser

# Adjust imports
from theseus.agent_gnn import AgentTheseusGNN  # Import the new agent
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


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="AgentTheseus", description="")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--train", action="store_true")
    group.add_argument("-p", "--play", action="store_true")
    parser.add_argument("-e", "--episodes", type=int)
    parser.add_argument("--path")

    return parser


def train(path: str = "", episodes: int = 9999) -> None:
    """Initializes and trains the combined AgentTheseusGNN."""
    train_logger = logging.getLogger("train_loop")
    train_logger.info("Initializing environment and combined GNN agent...")

    if not path == "":
        agent = AgentTheseusGNN.load(path)
        if not agent == None:
            agent.train(episodes)

        return

    env = Environment()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_logger.info(f"Using device: {device}")

    hero_policy = HeroGNN(
        hidden_channels=HIDDEN_CHANNELS, out_channels=HERO_ACTION_SPACE_SIZE
    )
    hero_target = HeroGNN(
        hidden_channels=HIDDEN_CHANNELS, out_channels=HERO_ACTION_SPACE_SIZE
    )
    gun_policy = GunGNN(
        hidden_channels=HIDDEN_CHANNELS, out_channels=GUN_ACTION_SPACE_SIZE
    )  # Add in_channels if needed
    gun_target = GunGNN(
        hidden_channels=HIDDEN_CHANNELS, out_channels=GUN_ACTION_SPACE_SIZE
    )  # Add in_channels if needed

    try:
        agent = AgentTheseusGNN(
            hero_policy_net=hero_policy,
            hero_target_net=hero_target,
            gun_policy_net=gun_policy,
            gun_target_net=gun_target,
            env=env,  # Pass the environment instance
            learning_rate=LEARNING_RATE,
            discount_factor=DISCOUNT_FACTOR,
        )
        train_logger.info("AgentTheseusGNN initialized.")
    except Exception as e:
        train_logger.critical(
            f"Failed to initialize AgentTheseusGNN: {e}", exc_info=True
        )
        return

    agent.train(episodes)


def play(path: str):
    pass


# --- Main Execution ---
if __name__ == "__main__":
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="INFO",
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    logging.getLogger("torch_geometric").setLevel(logging.WARNING)

    main_logger = logging.getLogger("main")
    main_logger.info(
        "[bold green] Starting Theseus GNN Training [/]", extra={"markup": True}
    )

    args = get_parser().parse_args()

    try:
        if args.train:
            train(args.path, args.episodes)
        elif args.play:
            play(args.path)
    except KeyboardInterrupt:
        main_logger.warning("Training interrupted by user.")
    except Exception as main_e:
        main_logger.critical(
            f"Unhandled exception in main training loop: {main_e}", exc_info=True
        )
    finally:
        main_logger.info(
            "[bold red] Training finished or interrupted. [/]", extra={"markup": True}
        )
