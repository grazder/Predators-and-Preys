import numpy as np
#from predators_and_preys_env.agent import PredatorAgent, PreyAgent

def distance(first, second):
    return ((first["x_pos"] - second["x_pos"]) ** 2 + (first["y_pos"] - second["y_pos"]) ** 2) ** 0.5

def is_collision(agent1, agent2) -> bool:
    return distance(agent1, agent2) + 1e-3 < agent1["radius"] + agent2["radius"]

EPS = 0.01
DELTA = 10
BACK_DELTA = 10

class PredatorAgent:
    def __init__(self):
        self.priv_state = None
        self.random_delay = [-1] * 3
        self.random_angle = [[-1] * BACK_DELTA + [-1] * DELTA] * 3

    def act(self, state_dict):
        action = []

        for j, predator in enumerate(state_dict["predators"]):
            closest_prey = None
            angle_value = 0.

            if self.random_delay[j] < 0:
                for i, prey in enumerate(state_dict["preys"]):
                    if not prey["is_alive"]:
                        continue
                    if closest_prey is None:
                        closest_prey = prey
                    else:
                        if distance(closest_prey, predator) > distance(prey, predator):
                            closest_prey = prey

                if closest_prey is not None:
                    angle_value = np.arctan2(closest_prey["y_pos"] - predator["y_pos"],
                                             closest_prey["x_pos"] - predator["x_pos"]) / np.pi

                if self.priv_state is not None:
                    priv_predator = self.priv_state["predators"][j]

                    if abs(priv_predator["y_pos"] - predator["y_pos"]) + abs(
                            priv_predator["x_pos"] - predator["x_pos"]) < EPS:
                        self.random_delay[j] = DELTA + BACK_DELTA
                        self.random_angle[j] = [1 + angle_value] * BACK_DELTA + [np.random.uniform(-1, 1)] * DELTA
            else:
                self.random_delay[j] -= 1
                angle_value = self.random_angle[j][self.random_delay[j]]

            action.append(angle_value)

        self.priv_state = state_dict
        return action


class PreyAgent:
    def __init__(self):
        self.priv_state = None
        self.good_spots = []

    def act(self, state_dict):
        action = []

        obstacles = state_dict['obstacles']
        N = len(obstacles)
        for i in range(N):
            for j in range(i + 1, N):
                if 0.8 <= distance(obstacles[i], obstacles[j]) - obstacles[i]['radius'] - obstacles[j]['radius'] < 1:
                    self.good_spots.append(((obstacles[i]['x_pos'] + obstacles[j]['x_pos']) / 2,
                                                (obstacles[i]['y_pos'] + obstacles[j]['y_pos']) / 2))

        for prey in state_dict["preys"]:
            closest_predator = None
            for predator in state_dict["predators"]:
                if closest_predator is None:
                    closest_predator = predator
                else:
                    if distance(closest_predator, prey) > distance(prey, predator):
                        closest_predator = predator

            closest_spot = None
            spot_dist = None
            if len(self.good_spots) > 0:
                closest_spot = self.good_spots[0]
                for x, y in self.good_spots:
                    spot_dist = np.sqrt((x - prey['x_pos']) ** 2 + (y - prey['y_pos']) ** 2)

                    if spot_dist < np.sqrt((closest_spot[0] - prey['x_pos']) ** 2 + (closest_spot[1] - prey['y_pos']) ** 2):
                        closest_spot = (x, y)

                spot_dist = np.sqrt((closest_spot[0] - prey['x_pos']) ** 2 + (closest_spot[1] - prey['y_pos']) ** 2)
            predator_dist = distance(closest_predator, prey)

            if closest_predator is None:
                action.append(0.)
            else:
                if closest_spot is not None and predator_dist > spot_dist:
                    action.append(np.arctan2(closest_spot[1] - prey["y_pos"],
                                             closest_spot[0] - prey["x_pos"]) / np.pi)
                else:
                    action.append(1 + np.arctan2(closest_predator["y_pos"] - prey["y_pos"],
                                                 closest_predator["x_pos"] - prey["x_pos"]) / np.pi)

        self.priv_state = state_dict
        return action
