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


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64, temperature=30):
        super().__init__()

        self.temp = temperature

        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )
        # self.model[-1].weight.data.uniform_(-3e-3, 3e-3) # ?

    def forward(self, state):
        out = self.model(state)

        return torch.tanh(out / self.temp)


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
