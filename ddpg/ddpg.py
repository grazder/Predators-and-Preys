import random
from collections import namedtuple, deque
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from ddpg.buffer import Buffer


GAMMA = 0.99
TAU = 0.002
CRITIC_LR = 5e-4
ACTOR_LR = 2e-4
BATCH_SIZE = 128
TRANSITIONS = 1000000
DEVICE = "cpu"


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.model(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1))


class RandomNoise:
    def __init__(self, mu=0.0, theta=0.1, sigma=0.1):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.action_dim = 1
        self.low = -1.0
        self.high = 1.0
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state


class DDPG:
    def __init__(self, state_dim, action_dim, low, high, eps, mem_sz):
        self.low = low
        self.high = high
        self.eps = eps

        self.memory = Buffer(state_dim, action_dim, 100_000)
        self.noise = RandomNoise()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)

        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optim = Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.critic_criterion = nn.MSELoss()

    def update(self):
        if len(self.memory) < BATCH_SIZE * 16:
            return

        state, action, next_state, reward, done = self.memory.sample(BATCH_SIZE)
        state = torch.tensor(np.array(state), device=DEVICE, dtype=torch.float)
        action = torch.tensor(np.array(action), device=DEVICE, dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), device=DEVICE, dtype=torch.float)
        reward = torch.tensor(np.array(reward), device=DEVICE, dtype=torch.float)
        done = torch.tensor(np.array(done), device=DEVICE, dtype=torch.float)

        with torch.no_grad():
            na = self.actor(next_state)

        temp_1 = self.critic(state, action)
        temp_2 = self.critic_target(next_state, na)
        temp_3 = reward.unsqueeze(1) + GAMMA * temp_2.detach()

        critic_loss = self.critic_criterion(temp_1, temp_3)
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()

        self.critic_optim.zero_grad()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.7)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.7)

        self.actor_optim.step()
        self.critic_optim.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * TAU + target_param.data * (1.0 - TAU))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * TAU + target_param.data * (1.0 - TAU))

    def act(self, state):
        with torch.no_grad():
            action = np.clip(self.actor(state).view(-1)
                             + self.eps * self.noise.noise(), self.low, self.high)
        return action.numpy()

    def save(self):
        torch.save(self.actor.state_dict(), "agent.pkl")