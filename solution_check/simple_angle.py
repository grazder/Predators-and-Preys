import numpy as np
import numpy as np
import torch
from torch import nn, Tensor
import math
import os

DEVICE = 'cpu'


def generate_features(state_dict):
    features = []

    for predator in state_dict['predators']:
        x_pred, y_pred, r_pred, speed_pred = predator['x_pos'], predator['y_pos'], predator['radius'], predator['speed']

        features += [x_pred, y_pred]

        prey_list = []

        for prey in state_dict['preys']:
            x_prey, y_prey, r_prey, speed_prey, alive = prey['x_pos'], prey['y_pos'], \
                                                        prey['radius'], prey['speed'], prey['is_alive']
            angle = np.arctan2(y_prey - y_pred, x_prey - x_pred) / np.pi
            distance = np.sqrt((y_prey - y_pred) ** 2 + (x_prey - x_pred) ** 2)

            prey_list += [[angle, distance, int(alive), r_prey]]

        prey_list = sorted(prey_list, key=lambda x: x[1])
        prey_list = [item for sublist in prey_list for item in sublist]
        features += prey_list

        obs_list = []

        for obs in state_dict['obstacles']:
            x_obs, y_obs, r_obs = obs['x_pos'], obs['y_pos'], obs['radius']
            angle = np.arctan2(y_obs - y_pred, x_obs - x_pred) / np.pi
            distance = np.sqrt((y_obs - y_pred) ** 2 + (x_obs - x_pred) ** 2)

            obs_list += [[angle, distance, r_obs]]

        obs_list = sorted(obs_list, key=lambda x: x[1])
        obs_list = [item for sublist in obs_list for item in sublist]
        features += obs_list

    return np.array(features, dtype=np.float32)


def calc_distance(first, second):
    return ((first["x_pos"] - second["x_pos"]) ** 2 + (first["y_pos"] - second["y_pos"]) ** 2) ** 0.5


# class Actor(nn.Module):
#     def __init__(self, state_dim: int, action_dim: int,
#                  hidden_dim: int = 64, norm_in: bool = True):
#         super().__init__()
#         action_dim = action_dim * 2
#         if norm_in:
#             self.in_fn = nn.BatchNorm1d(state_dim)
#             self.in_fn.weight.data.fill_(1)
#             self.in_fn.bias.data.fill_(0)
#         else:
#             self.in_fn = lambda x: x
#
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, action_dim)
#         self.fc3.weight.data.uniform_(-3e-3, 3e-3)
#         self.nonlin = nn.ReLU()
#         self.out_fn = nn.Tanh()
#
#     def forward(self, states: Tensor) -> Tensor:
#         batch_size, _ = states.shape
#         h1 = self.nonlin(self.fc1(states))
#         h2 = self.nonlin(self.fc2(h1))
#         out = self.out_fn(self.fc3(h2))
#
#         norms = np.zeros((out.shape[0], out.shape[1] // 2))
#         for i in range(0, out.shape[1], 2):
#             angle = torch.atan2(*out[:, i:i + 2].T)
#             normalized = (angle / math.pi).view(batch_size, -1).cpu().detach().numpy().reshape(1, -1)
#             norms[:, i // 2] = normalized
#
#         norms = torch.Tensor(norms)
#         return norms

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


class PredatorAgent:
    def __init__(self):
        self.model = Actor(104, 2).to(torch.device(DEVICE))
        #state_dict = torch.load('solution_check/agent_simple_angle_2.pkl')
        state_dict = torch.load('agent.pkl')
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def act(self, state):
        with torch.no_grad():
            features = generate_features(state)
            features = torch.tensor(np.array([features]), dtype=torch.float, device=DEVICE)
            action = self.model(features).cpu().numpy()[0]

            return action


class PreyAgent:
    def act(self, state_dict):
        action = []
        for prey in state_dict["preys"]:
            closest_predator = None
            for predator in state_dict["predators"]:
                if closest_predator is None:
                    closest_predator = predator
                else:
                    if calc_distance(closest_predator, prey) > calc_distance(prey, predator):
                        closest_predator = predator
            if closest_predator is None:
                action.append(0.)
            else:
                action.append(1 + np.arctan2(closest_predator["y_pos"] - prey["y_pos"],
                                             closest_predator["x_pos"] - prey["x_pos"]) / np.pi)
        return action
