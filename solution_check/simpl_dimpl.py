import numpy as np
from solution_check.ClosestPath import ClosestPath

def distance(first, second):
    return ((first["x_pos"] - second["x_pos"]) ** 2 + (first["y_pos"] - second["y_pos"]) ** 2) ** 0.5


def usual_distance(x_1, y_1, x_2, y_2):
    return np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)


EPS = 0.02
BACK_DELTA = 10
RANDOM_DELTA = 20
DELTA = 30
SPLIT_NUMBER = 10


class PredatorAgent:
    def __init__(self):
        self.priv_state = None
        self.random_delay = [-1] * 2
        self.pather = None
        self.paths = [None] * 2
        self.random_angle = [[-1] * BACK_DELTA + [-1] * RANDOM_DELTA] * 2

    def find_angle(self, x, y, j):
        while len(self.paths[j]) > 0 and usual_distance(self.paths[j][0][0], self.paths[j][0][1], x, y) < 1:
            self.paths[j].pop(0)

        if len(self.paths[j]) == 0:
            print('zeroing')
            return 0

        angle_value = np.arctan2(self.paths[j][0][1] - y, self.paths[j][0][0] - x) / np.pi
        return angle_value

    def act(self, state_dict):
        action = []

        if self.pather is None:
            self.pather = ClosestPath(state_dict, number=SPLIT_NUMBER)

        for j, predator in enumerate(state_dict["predators"]):
            closest_prey = None
            angle_value = 0.

            if self.random_delay[j] < 0:
                self.paths[j] = None

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

                        self.paths[j] = self.pather.find_predator_paths(state_dict, predator)
                        if self.paths[j] is not None:
                            print('yeah found')
                            self.random_delay[j] = DELTA
                            angle_value = self.find_angle(predator['x_pos'], predator['y_pos'], j)
                            print(j, angle_value)
                        else:
                            print('fck go back')
                            self.random_delay[j] = RANDOM_DELTA + BACK_DELTA
                            self.random_angle[j] = [1 + angle_value] * BACK_DELTA + [np.random.uniform(-1, 1)] * RANDOM_DELTA

            else:
                self.random_delay[j] -= 1
                if self.paths[j] is not None:
                    angle_value = self.find_angle(predator['x_pos'], predator['y_pos'], j)
                    print(j, angle_value)
                else:
                    angle_value = self.random_angle[j][self.random_delay[j]]

            action.append(angle_value)

        self.priv_state = state_dict
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
                    if distance(closest_predator, prey) > distance(prey, predator):
                        closest_predator = predator
            if closest_predator is None:
                action.append(0.)
            else:
                action.append(1 + np.arctan2(closest_predator["y_pos"] - prey["y_pos"],
                                             closest_predator["x_pos"] - prey["x_pos"]) / np.pi)
        return action