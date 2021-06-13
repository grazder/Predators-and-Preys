from predators_and_preys_env.env import PredatorsAndPreysEnv
from solution_check.simple_angle import PredatorAgent, PreyAgent
#from solution_check.simpl_dimpl import PredatorAgent, PreyAgent

env = PredatorsAndPreysEnv(render=True)
predator_agent = PredatorAgent()
prey_agent = PreyAgent()

a = None
done = True
state_dict = None

import numpy as np

#12
for i in range(1):
    state_dict = env.reset()
for i in range(100000):
    if done:
        state_dict = env.reset()

    state_dict, _, done = env.step(predator_agent.act(state_dict), prey_agent.act(state_dict))
