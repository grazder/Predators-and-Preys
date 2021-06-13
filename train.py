from predators_and_preys_env.env import PredatorsAndPreysEnv, DEFAULT_CONFIG
from ddpg.agents import PredatorAgent, PreyAgent
from ddpg.rewards import pred_reward
from ddpg.features import generate_features

from ddpg.td3 import TD3
from ddpg.ddpg import DDPG
import torch

from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error

TRANSITIONS = 1000000
EPS = 0.2

if __name__ == "__main__":

    env = PredatorsAndPreysEnv(render=False)
    predator_agent = PredatorAgent()
    prey_agent = PreyAgent()

    episodes_sampled = 0
    steps_sampled = 0

    state_dict = env.reset()
    rewards = []
    next_features = generate_features(state_dict)
    next_features = torch.FloatTensor(next_features)
    print(next_features.shape[0], DEFAULT_CONFIG['game']['num_preds'])

    # td3 = TD3(state_dim=next_features.shape[0], action_dim=DEFAULT_CONFIG['game']['num_preds'])
    ddpg = DDPG(state_dim=next_features.shape[0], action_dim=DEFAULT_CONFIG['game']['num_preds'],
               low=-1, high=1, eps=1.0, mem_sz=60000)

    for i in tqdm(range(TRANSITIONS)):
        steps = 0

        train_features = next_features

        # Epsilon-greedy policy
        predator_action = ddpg.act(train_features)
        #predator_action = np.clip(predator_action + EPS * np.random.randn(*predator_action.shape), -1, +1)

        next_state_dict, _, done = env.step(predator_action, prey_agent.act(state_dict))
        # reward = pred_reward(state_dict)

        # mse
        default_action = predator_agent.act(state_dict)
        reward = -mean_squared_error(default_action, predator_action)

        next_features = generate_features(next_state_dict)
        next_features = torch.FloatTensor(next_features)

        ddpg.memory.push(train_features, predator_action, next_features, reward, done)
        ddpg.update()

        #td3.update((train_features, predator_action, next_features, reward, done))

        rewards.append(reward)
        state_dict = next_state_dict if not done else env.reset()

        if (i + 1) % 5_000 == 0:
            print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            # td3.save()
            ddpg.save()
            rewards = []
