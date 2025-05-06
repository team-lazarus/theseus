import logging
import torch
import os
from rich.logging import RichHandler
from argparse import ArgumentParser

from theseus.agent_ppo import AgentTheseusPPO
from theseus.agent_gnn import AgentTheseusGNN
from theseus.models.GraphDQN.ActionGNN import HeroGNN, GunGNN
from theseus.utils.network import Environment
import theseus.constants as c


HIDDEN_CHANNELS = 16
HERO_ACTION_SPACE_SIZE = 9
GUN_ACTION_SPACE_SIZE = 8

LEARNING_RATE = 1e-4
DISCOUNT_FACTOR = 0.99


# --- PPO Hyperparameters ---
PPO_LEARNING_RATE = 3e-4  # Typical PPO learning rate
PPO_DISCOUNT_FACTOR = 0.99  # Gamma
PPO_HORIZON = 2048  # Steps collected per rollout (N)
PPO_EPOCHS_PER_UPDATE = 10  # Optimization epochs per rollout (K)
PPO_MINI_BATCH_SIZE = 64  # Minibatch size for optimization
PPO_CLIP_EPSILON = 0.2  # PPO clipping parameter (epsilon)
PPO_GAE_LAMBDA = 0.95  # GAE parameter (lambda)
PPO_ENTROPY_COEFF = 0.01  # Entropy bonus coefficient
PPO_VF_COEFF = 0.5  # Value function loss coefficient
PPO_LOG_WINDOW = 50  # Episodes for rolling average metrics
PPO_SAVE_INTERVAL = 200  # Save checkpoint every N episodes
NUM_TRAINING_EPISODES = 50000  # Total episodes to train for


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="AgentTheseus", description="")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--train", action="store_true")
    group.add_argument("-p", "--play", action="store_true")
    parser.add_argument("--model", default="PPO", choices=["DQN", "PPO"])
    parser.add_argument("-e", "--episodes", default=9999, type=int)
    parser.add_argument("--path")

    return parser


def train(path: str = "", episodes: int = 9999) -> None:
    """Initializes and trains the combined AgentTheseusGNN."""
    train_logger = logging.getLogger("train_loop")
    train_logger.info("Initializing environment and combined GNN agent...")

    if path != "" and path != None:
    # try:
        agent = AgentTheseusGNN.load(path)
        if not agent == None:
            agent.train(episodes)

        return
    # except Exception as e:
    #     train_logger.critical("Failed to load model, proceeding with new run.")

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
    )
    gun_target = GunGNN(
        hidden_channels=HIDDEN_CHANNELS, out_channels=GUN_ACTION_SPACE_SIZE
    )

    try:
        agent = AgentTheseusGNN(
            hero_policy_net=hero_policy,
            hero_target_net=hero_target,
            gun_policy_net=gun_policy,
            gun_target_net=gun_target,
            env=env,
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


def train_ppo(path: str = "", episodes: int = 9999):
    """Initializes and trains the combined AgentTheseusPPO."""
    train_logger = logging.getLogger("train_loop_ppo")
    train_logger.info("Initializing environment and PPO GNN agent...")

    if path != "" and path != None:
    # try:
        agent = AgentTheseusPPO.load(path)
        if not agent == None:
            agent.train(episodes)

        return
    # except Exception as e:
    #     train_logger.critical("Failed to load model, proceeding with new run.")

    env = Environment()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_logger.info(f"Using device: {device}")

    # --- Instantiate Actor and Critic Networks ---
    # CRITICAL ASSUMPTION: HeroGNN/GunGNN can output a single value when out_channels=1
    # If not, you need dedicated Critic GNN classes.
    # agent = AgentTheseusPPO.load(SAVED_MODEL)

    hero_actor = HeroGNN(
        hidden_channels=HIDDEN_CHANNELS, out_channels=HERO_ACTION_SPACE_SIZE
    )
    hero_critic = HeroGNN(
        hidden_channels=HIDDEN_CHANNELS, out_channels=1
    )
    gun_actor = GunGNN(
        hidden_channels=HIDDEN_CHANNELS, out_channels=GUN_ACTION_SPACE_SIZE
    )
    gun_critic = GunGNN(
        hidden_channels=HIDDEN_CHANNELS, out_channels=1
    )
    train_logger.info("Actor and Critic GNNs instantiated.")

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


def play(
    model_dir_path: str, # Path to the directory containing model files
) -> None:
    """
    Runs continuous play using directly loaded policy networks. MINIMAL VERSION.

    Args:
        model_dir_path: Path to the directory containing 'hero_policy_state.pth'
                        and 'gun_policy_state.pth'.
    """
    print(f"--- Starting Minimal Play (Models from: {model_dir_path}) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load Networks Directly (Minimal Error Checks) ---
    # Assumes standard filenames saved by AgentTheseusGNN dump (state_dict versions)
    hero_policy_path = os.path.join(model_dir_path, "hero_actor_state.pth")
    gun_policy_path = os.path.join(model_dir_path, "gun_actor_state.pth")

    # CRITICAL: You need to know the hidden_channels and out_channels for your models
    # to instantiate them before loading state_dict.
    # These should ideally be saved in a config file within model_dir_path.
    # Using placeholders here - YOU MUST REPLACE THESE WITH ACTUAL VALUES.
    HERO_HIDDEN = 16 # Example - MUST MATCH SAVED MODEL
    GUN_HIDDEN = 16  # Example - MUST MATCH SAVED MODEL
    HERO_OUT = 9
    GUN_OUT = 8

    hero_net = HeroGNN(hidden_channels=HERO_HIDDEN, out_channels=HERO_OUT)
    gun_net = GunGNN(hidden_channels=GUN_HIDDEN, out_channels=GUN_OUT)

    hero_net.load_state_dict(torch.load(hero_policy_path, map_location=device))
    gun_net.load_state_dict(torch.load(gun_policy_path, map_location=device))

    hero_net.to(device).eval()
    gun_net.to(device).eval()
    print("Policy networks loaded.")

    # --- Initialize Environment ---
    env = Environment()
    state = env.initialise_environment()
    print("Environment initialized. Starting play loop (Ctrl+C to stop)...")

    # --- Continuous Play Loop ---
    while True:
        try:
            move_action, shoot_action = 0, 0

            with torch.no_grad():
                h_graph = hero_net.preprocess_state(state)
                if h_graph: # Simplified check
                    h_q = hero_net(h_graph.to(device))
                    if h_q.numel() > 0: move_action = h_q.squeeze().argmax().item()

                g_graph = gun_net.preprocess_state(state)
                if g_graph: # Simplified check
                    g_q = gun_net(g_graph.to(device))
                    if g_q.numel() > 0: shoot_action = g_q.squeeze().argmax().item()

            action_list = [move_action, shoot_action, 0, 0]
            next_s, _, terminated_flag = env.step(action_list)[:3]

            if bool(terminated_flag):
                print("Episode terminated. Resetting...")
                state = env.initialise_environment()
            else:
                state = next_s

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt. Exiting play.")
            break
        except Exception as e:
            print(f"\nERROR in play loop: {e}. Attempting to reset...")
            try:
                state = env.initialise_environment()
            except Exception as reset_e:
                print(f"CRITICAL: Reset failed after error: {reset_e}. Exiting.")
                break

    print("--- Minimal play finished. ---")


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
        "[bold green] Starting Theseus [/]", extra={"markup": True}
    )

    args = get_parser().parse_args()

    print(args)
    try:
        if args.train:
            match args.model:
                case "DQN":
                    train(args.path, args.episodes)
                case "PPO":
                    train_ppo(args.path, args.episodes)
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
