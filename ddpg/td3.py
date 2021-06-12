from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import random
import copy

from ddpg.buffer import Buffer

GAMMA = 0.99
TAU = 0.002
CRITIC_LR = 5e-4
ACTOR_LR = 2e-4
DEVICE = "cpu"
BATCH_SIZE = 128
EPS = 0.2


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
        return self.model(torch.cat([state, action], dim=-1)).view(-1)


def soft_update(target, source):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)


class TD3:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic_1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_2 = Critic(state_dim, action_dim).to(DEVICE)

        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=ACTOR_LR)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=ACTOR_LR)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        self.replay_buffer = Buffer(state_dim, action_dim, 100_000)
        self.criterion = nn.MSELoss()

    def update(self, transition):
        self.replay_buffer.push(*transition)
        if len(self.replay_buffer) > BATCH_SIZE * 16:
            # Sample batch
            transitions = self.replay_buffer.sample(BATCH_SIZE)
            state, action, next_state, reward, done = transitions
            state = torch.tensor(np.array(state), device=DEVICE, dtype=torch.float)
            action = torch.tensor(np.array(action), device=DEVICE, dtype=torch.float)
            next_state = torch.tensor(np.array(next_state), device=DEVICE, dtype=torch.float)
            reward = torch.tensor(np.array(reward), device=DEVICE, dtype=torch.float)
            done = torch.tensor(np.array(done), device=DEVICE, dtype=torch.float)

            # Update critic
            with torch.no_grad():
                noise = (torch.randn_like(action) * EPS).clamp(-1, 1)
                next_action = (self.target_actor(next_state) + noise).clamp(-1, 1)

                next_q1_value = self.target_critic_1(next_state, next_action)
                next_q2_value = self.target_critic_2(next_state, next_action)
                next_q_value = torch.min(next_q1_value, next_q2_value)

                target_q_value = reward + GAMMA * (1 - done) * next_q_value

            q1_value = self.critic_1(state, action)
            q2_value = self.critic_2(state, action)

            q1_loss = self.criterion(q1_value, target_q_value)
            q2_loss = self.criterion(q2_value, target_q_value)

            loss = q1_loss + q2_loss

            self.critic_1_optim.zero_grad()
            self.critic_2_optim.zero_grad()

            loss.backward()

            self.critic_1_optim.step()
            self.critic_2_optim.step()

            # Update actor
            policy_loss = -self.critic_1(state, self.actor(state)).mean()

            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()

            soft_update(self.target_critic_1, self.critic_1)
            soft_update(self.target_critic_2, self.critic_2)
            soft_update(self.target_actor, self.actor)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float, device=DEVICE)
            return self.actor(state).cpu().numpy()[0]

    def save(self):
        torch.save(self.actor.state_dict(), "agent.pkl")