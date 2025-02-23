import torch
import random
import logging
from torch import nn
from typing import List
import torch.optim as optim
from itertools import count

import theseus.constants as c
from theseus.utils import ExperienceReplayMemory
from theseus.utils.network import Environment, ActionSpace


class AgentTheseus(object):
    def __init__(
        self,
        policy_network: nn.Module,
        target_network: nn.Module,
        *,
        loss_fn=nn.MSELoss(),
        optimizer=optim.AdamW,
        learning_rate=1e-3,
        discount_factor=0.95,
        training: bool = True,
        epsilon_init: float = 1.0,
        epsilon_decay: float = 0.9995,
        epsilon_min: float = 0.05,
        mini_batch_size: int = 32,
        target_sync_rate: int = 16,
    ):
        self.training = training
        self.environment = Environment()
        self.logger = logging.getLogger("agent-theseus")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.training and type(policy_network) != type(target_network):
            raise ValueError(
                "The type of policy network & target network should be the same"
            )

        self.policy_network = policy_network
        self.policy_network.to(self.device)
        self.target_network = target_network

        self.epsilon_init = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon_init
        self.discount_factor = discount_factor

        self.mini_batch_size = mini_batch_size
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.optimizer = optimizer(
            policy_network.parameters(), lr=self.learning_rate, amsgrad=True
        )

        self.reward_per_episode = []
        self.target_sync_rate = target_sync_rate
        self.sync_steps_taken = 0

        if self.training:
            self.memory = ExperienceReplayMemory(c.REPLAY_MEMORY_SIZE)

            self.target_network.load_state_dict(self.policy_network.state_dict())

    def train(self):
        epsilon = self.epsilon_init
        for episode in count():
            self.logger.info(f"[blue]starting episode: {episode}[/]", extra={"markup": True})
            self.train_episode()
            if self.training:
                self.learn()

    def train_episode(self):
        terminated = False
        episode_reward = 0.0
        state = self.environment.initialise_environment()

        while not terminated:
            if self.training and random.random() < self.epsilon:
                action = self.environment.action_space.sample()
            else:
                with torch.no_grad():
                    x = self.policy_network.preprocess_state(state)
                    action = (
                        self.policy_network.forward(x.unsqueeze(0)).squeeze().argmax()
                    ).item()

            next_state, reward, terminated = self.environment.step(action)

            # accumulate reward
            episode_reward += reward

            if self.training:
                self.memory.append((state, action, next_state, reward, terminated))
                self.target_step_counter += 1

            state = next_state
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        self.reward_per_episode.append(episode_reward)

    def learn(self):
        if len(self.memory) > self.mini_batch_size:
            mini_batch = self.memory.sample(self.mini_batch_size)
            self.optimize(mini_batch)

        if self.sync_steps_taken > self.network_sync_rate:
            self.target_network.load_state_dict(self.policy_network.state_dict())
            self.sync_steps_taken = 0

    def optimize(self, mini_batch: List):
        states, actions, next_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            x = self.target_network.preprocess_state(next_state)
            target_q = (
                reward
                + (1 - terminations)
                * self.discount_factor
                * target_network(x).max(dim=1)[0]
            )

        x = self.policy_network.preprocess_state(state)
        current_q = (
            self.policy_network(x)
            .gather(dim=1, index=actions.unsqueeze(dim=1))
            .squeeze()
        )

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
