from collections import deque
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor


class Buffer(object):
    def __init__(self, state_dim: int, action_dim: int, max_size: int):
        self.max_size = max_size

        self.state_buff = torch.zeros(max_size, state_dim).float()
        self.action_buff = torch.zeros(max_size, action_dim).float()
        self.reward_buff = torch.zeros(max_size).float()
        self.next_state_buff = torch.zeros(max_size, state_dim).float()
        self.done_buff = torch.zeros(max_size).float()

        self.filled_i = 0
        self.curr_size = 0

    def push(self, state: np.ndarray, action: np.ndarray,
             next_state: np.ndarray, reward: float, done: bool):
        self.state_buff[self.filled_i] = torch.Tensor(state)
        self.action_buff[self.filled_i] = torch.Tensor(action)
        self.reward_buff[self.filled_i] = torch.Tensor([reward])
        self.next_state_buff[self.filled_i] = torch.Tensor(next_state)
        self.done_buff[self.filled_i] = torch.Tensor([done])

        self.curr_size = min(self.max_size, self.curr_size + 1)
        self.filled_i = (self.filled_i + 1) % self.max_size

    def sample(self, batch_size: int, norm_rew: bool = True) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        indices = np.random.choice(self.curr_size, batch_size, replace=False)
        indices = torch.Tensor(indices).long()

        if norm_rew:
            mean = torch.mean(self.reward_buff[:self.curr_size])
            std = torch.std(self.reward_buff[:self.curr_size])
            rew = (self.reward_buff[indices] - mean) / std
        else:
            rew = self.reward_buff[indices]

        return self.state_buff[indices], self.action_buff[indices], self.next_state_buff[indices], \
               self.reward_buff[indices],  self.done_buff[indices]

    def __len__(self):
        return self.curr_size