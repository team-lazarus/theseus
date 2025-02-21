import logging
from torch import nn
from itertools import count
from thesus.utils.network import Environment, ActionSpace

class AgentThesus(object):
    def __init__(self, policy_network: nn.Module):
        self.environment = Environment()
        self.environment.initialise_environment()

        self.policy_network = policy_network
        self.logger = logging.getLogger("agent-thesus")

    def train(self):
        for episode in count():
            self.logger.info(f"[blue]starting episode: {episode}[/]")
            self.train_episode()
    
    def train_episode(self):
        terminated = False
        while not terminated:
            action = self.environment.action_space.sample()
            next_state, reward, terminated = self.environment.step(action)

