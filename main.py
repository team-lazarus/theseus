import logging
from rich.logging import RichHandler

from theseus import AgentTheseus
from theseus.models import PolicyDQN

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("main")
logger.info("[bold green] Starting Thesus [/]", extra={"markup": True})

if __name__ == "__main__":
    agent = AgentTheseus(PolicyDQN(32,81), PolicyDQN(32,81))
    agent.train()
