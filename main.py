import logging
from rich.logging import RichHandler

from thesus import AgentThesus

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("main")
logger.info("[bold green] Starting Thesus [/]", extra={"markup": True})

if __name__ == "__main__":
    agent = AgentThesus(None)
    agent.train()
