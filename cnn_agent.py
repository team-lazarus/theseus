import random
import torch
from theseus.agent import AgentTheseus
from theseus.models.CNNDQN.models import create_dual_head_policies


class DualActionAgent:
    def __init__(
        self,
        movement_actions=9,
        attack_actions=8,
        input_channels=3,
        non_spatial_dims=4,
        grid_size=32,
        *,
        loss_fn=torch.nn.MSELoss(),
        optimizer=torch.optim.AdamW,
        learning_rate=1e-3,
        discount_factor=0.95,
        epsilon_init=0.1,
        epsilon_decay=0.9995,
        epsilon_min=0.05,
        mini_batch_size=32,
        target_sync_rate=16,
    ):
        # Create policies with shared feature extractor
        movement_policy, attack_policy = create_dual_head_policies(
            movement_actions=movement_actions,
            attack_actions=attack_actions,
            input_channels=input_channels,
            non_spatial_dims=non_spatial_dims,
            grid_size=grid_size,
            shared_extractor=True,
        )

        # Create target networks
        movement_target = type(movement_policy)(n_actions=movement_actions)
        movement_target.feature_extractor = movement_policy.feature_extractor

        attack_target = type(attack_policy)(n_actions=attack_actions)
        attack_target.feature_extractor = attack_policy.feature_extractor

        # Create individual agents
        self.movement_agent = AgentTheseus(
            policy_network=movement_policy,
            target_network=movement_target,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            training=True,
            epsilon_init=epsilon_init,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            mini_batch_size=mini_batch_size,
            target_sync_rate=target_sync_rate,
        )

        self.attack_agent = AgentTheseus(
            policy_network=attack_policy,
            target_network=attack_target,
            loss_fn=loss_fn,
            optimizer=optimizer,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            training=True,
            epsilon_init=epsilon_init,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            mini_batch_size=mini_batch_size,
            target_sync_rate=target_sync_rate,
        )

    def select_actions(self, state):
        # Select both movement and attack actions based on current state
        # Select movement action
        if (
            self.movement_agent.training
            and random.random() < self.movement_agent.epsilon
        ):
            movement_action = random.randint(
                0, self.movement_agent.policy_network.layer3.out_features - 1
            )
        else:
            with torch.no_grad():
                x = self.movement_agent.policy_network.preprocess_state(state)
                movement_action = (
                    self.movement_agent.policy_network(x).squeeze().argmax()
                ).item()

        # Select attack action
        if self.attack_agent.training and random.random() < self.attack_agent.epsilon:
            attack_action = random.randint(
                0, self.attack_agent.policy_network.layer3.out_features - 1
            )
        else:
            with torch.no_grad():
                x = self.attack_agent.policy_network.preprocess_state(state)
                attack_action = (
                    self.attack_agent.policy_network(x).squeeze().argmax()
                ).item()

        return movement_action, attack_action

    def train_episode(self, environment):
        # Train the agent for one episode in the given environment
        terminated = False
        episode_reward = 0.0
        state = environment.initialise_environment()

        while not terminated:
            # Select actions
            movement_action, attack_action = self.select_actions(state)

            # Take actions in environment - implement this function in your environment
            next_state, reward, terminated = environment.step_dual_action(
                movement_action, attack_action
            )

            # Store experiences in respective memories
            self.movement_agent.memory.append(
                (state, movement_action, next_state, reward, terminated)
            )
            self.attack_agent.memory.append(
                (state, attack_action, next_state, reward, terminated)
            )

            # Update step counters
            self.movement_agent.sync_steps_taken += 1
            self.attack_agent.sync_steps_taken += 1

            # Move to next state
            state = next_state
            episode_reward += reward

            # Update exploration rates
            self.movement_agent.epsilon = max(
                self.movement_agent.epsilon * self.movement_agent.epsilon_decay,
                self.movement_agent.epsilon_min,
            )
            self.attack_agent.epsilon = max(
                self.attack_agent.epsilon * self.attack_agent.epsilon_decay,
                self.attack_agent.epsilon_min,
            )

        # Return total reward for this episode
        return episode_reward

    def learn(self):
        # Update both policy networks from experiences
        # Learn from movement experiences
        if len(self.movement_agent.memory) > self.movement_agent.mini_batch_size:
            mini_batch = self.movement_agent.memory.sample(
                self.movement_agent.mini_batch_size
            )
            self.movement_agent.optimize(mini_batch)

        # Learn from attack experiences
        if len(self.attack_agent.memory) > self.attack_agent.mini_batch_size:
            mini_batch = self.attack_agent.memory.sample(
                self.attack_agent.mini_batch_size
            )
            self.attack_agent.optimize(mini_batch)

        # Update target networks if needed
        if self.movement_agent.sync_steps_taken > self.movement_agent.target_sync_rate:
            self.movement_agent.target_network.load_state_dict(
                self.movement_agent.policy_network.state_dict()
            )
            self.movement_agent.sync_steps_taken = 0

        if self.attack_agent.sync_steps_taken > self.attack_agent.target_sync_rate:
            self.attack_agent.target_network.load_state_dict(
                self.attack_agent.policy_network.state_dict()
            )
            self.attack_agent.sync_steps_taken = 0

    def train(self, environment, num_episodes=1000):
        # Train the agent for multiple episodes
        episode_rewards = []

        for episode in range(num_episodes):
            # Train for one episode
            episode_reward = self.train_episode(environment)
            episode_rewards.append(episode_reward)

            # Learn from experiences
            self.learn()

            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = sum(episode_rewards[-10:]) / 10
                print(
                    f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, "
                    f"Epsilon Move: {self.movement_agent.epsilon:.2f}, "
                    f"Epsilon Attack: {self.attack_agent.epsilon:.2f}"
                )

        return episode_rewards

    def save_models(self, movement_path, attack_path):
        # Save both policy networks
        torch.save(self.movement_agent.policy_network.state_dict(), movement_path)
        torch.save(self.attack_agent.policy_network.state_dict(), attack_path)

    def load_models(self, movement_path, attack_path):
        # Load both policy networks
        self.movement_agent.policy_network.load_state_dict(torch.load(movement_path))
        self.movement_agent.target_network.load_state_dict(
            self.movement_agent.policy_network.state_dict()
        )

        self.attack_agent.policy_network.load_state_dict(torch.load(attack_path))
        self.attack_agent.target_network.load_state_dict(
            self.attack_agent.policy_network.state_dict()
        )
