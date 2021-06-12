import math

def distance(agent1, agent2) -> float:
    return math.sqrt((agent1["x_pos"] - agent2["x_pos"]) ** 2 + (agent1["y_pos"] - agent2["y_pos"]) ** 2)


def is_collision(agent1, agent2) -> bool:
    return distance(agent1, agent2) + 1e-3 < agent1["radius"] + agent2["radius"]


def pred_reward(state_dict) -> float:
    rew = 0.

    for predator in state_dict["predators"]:
        for prey in state_dict["preys"]:
            if is_collision(predator, prey):
                rew += 10.

            rew -= 0.1 * distance(predator, prey)

    return rew
