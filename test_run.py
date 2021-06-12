from predators_and_preys_env.env import PredatorsAndPreysEnv
import numpy as np
from examples.simple_chasing_agents.agents import ChasingPredatorAgent
from examples.simple_chasing_agents.agents import FleeingPreyAgent
import time

env = PredatorsAndPreysEnv(render=False)
env.seed(42)
predator_agent = ChasingPredatorAgent()
prey_agent = FleeingPreyAgent()

num_of_games = 25

done = True
step_count = 0
state_dict = None
print(f"Test run for {num_of_games} episodes")
while num_of_games > 0:
    if done:
        state_dict = env.reset()
        if num_of_games < 25:
            print(f"Game {25 - num_of_games} -- {step_count} steps")
        num_of_games -= 1
        step_count = 0

    state_dict, reward, done = env.step(predator_agent.act(state_dict), prey_agent.act(state_dict))
    step_count += 1
