import random
import logging
from torch import nn
from itertools import count

import theseus.constants as c
from theseus.utils import ExperienceReplayMemory
from theseus.utils.network import Environment, ActionSpace

class AgentThesus(object):
    def __init__(self, policy_network: nn.Module, train: bool = True, epsilon_init: float = 1.0, epsilon_decay: float = 0.9995, epsilon_min: float = 0.05):
        self.environment = Environment()

        self.policy_network = policy_network
        self.policy_network.to(self.device)

        self.epsilon_init = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train = train
        self.logger = logging.getLogger("agent-thesus")

        self.reward_history = []

        if train:
            self.memory = ExperienceReplayMemory(c.REPLAY_MEMORY_SIZE)

    def train(self):
        epsilon = self.epsilon_init
        for episode in count():
            self.logger.info(f"[blue]starting episode: {episode}[/]", {"markup" : True})
            self.train_episode()

    def train_episode(self):
        terminated = False
        episode_reward = 0.0
        state = self.environment.initialise_environment()

        while not terminated:
            if self.train and random.random() < self.epsilon:
                action = self.environment.action_space.sample()
            else:
                with torch.no_grad():
                    x = self.policy_network.preprocess_state(state)
                    action = self.policy_network.forward(x.unsqueeze(0)).squeeze().argmax()

            next_state, reward, terminated = self.environment.step(action.item())

            # accumulate reward
            episode_reward += reward

            if train:
                self.memory.append((state, action, next_state, reward, terminated))
            
            state = next_state
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        self.reward_history.append(episode_reward)