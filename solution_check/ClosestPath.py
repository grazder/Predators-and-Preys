from predators_and_preys_env.env import PredatorsAndPreysEnv, DEFAULT_CONFIG
import numpy as np


class ClosestPath:
    def __init__(self, state, number=100):
        x_limit = DEFAULT_CONFIG['game']['x_limit']
        y_limit = DEFAULT_CONFIG['game']['y_limit']

        predator_radius = 1

        x_step = 2 * x_limit / number
        y_step = 2 * y_limit / number

        x_coords = np.arange(-x_limit + x_step, x_limit, x_step)
        y_coords = np.arange(-y_limit + y_step, y_limit, y_step)

        self.field_grid = np.array(np.meshgrid(x_coords, y_coords)).T
        self.field_obsticle = np.zeros(self.field_grid.shape[:2])

        ### Bounding Check

        for i in range(self.field_grid.shape[0]):
            for j in range(self.field_grid.shape[1]):
                x, y = self.field_grid[i][j]

                if x + predator_radius > x_limit or x - predator_radius < -x_limit or \
                        y + predator_radius > y_limit or y - predator_radius < -y_limit:
                    self.field_obsticle[i][j] = 1

                for obs in state['obstacles']:
                    x_obs, y_obs, rad_obs = obs['x_pos'], obs['y_pos'], obs['radius']

                    if np.sqrt((x - x_obs) ** 2 + (y - y_obs) ** 2) < predator_radius + rad_obs:
                        self.field_obsticle[i][j] = 1

        ### Adjacency Matrix

        self.height = self.field_grid.shape[0]
        self.width = self.field_grid.shape[1]
        self.N = self.height * self.width

        self.adj_matrix = []

        def get_distance(i, j, vert):
            if i >= 0 and i < self.height and \
                    j >= 0 and j < self.width:

                if 1 - self.field_obsticle[i][j]:
                    self.adj_matrix[vert] += [i * self.width + j]

        for vert in range(self.N):
            self.adj_matrix.append([])
            i, j = vert // self.height, vert % self.height

            if self.field_obsticle[i][j] == 0:
                get_distance(i + 1, j, vert)
                get_distance(i - 1, j, vert)
                get_distance(i, j + 1, vert)
                get_distance(i, j - 1, vert)

    def find_path(self, start, end, path=[]):
        path = path + [start]

        if start == end:
            return path

        if len(path) > 10:
            return None

        for node in self.adj_matrix[start]:
            if node not in path:
                newpath = self.find_path(node, end, path)

                if newpath:
                    return newpath

        return None

    def find_closest_vertex(self, x, y):
        min_dist = 10000
        min_dot = None

        for i in range(self.field_grid.shape[0]):
            for j in range(self.field_grid.shape[1]):
                x_grid, y_grid = self.field_grid[i][j]
                distance = np.sqrt((x_grid - x) ** 2 + (y_grid - y) ** 2)

                if distance < min_dist and (1 - self.field_obsticle[i][j]):
                    min_dist = distance
                    min_dot = (i, j)

        return min_dot[0] * self.width + min_dot[1]

    def find_predator_paths(self, state, predator):
        # predator_paths = []
        # for predator in state['predators']:
        x, y = predator['x_pos'], predator['y_pos']

        pred_vert = self.find_closest_vertex(x, y)

        min_distance = 1000000
        min_path = None
        for prey in state['preys']:
            x, y = prey['x_pos'], prey['y_pos']

            if prey['is_alive']:
                prey_vert = self.find_closest_vertex(x, y)
                path = self.find_path(pred_vert, prey_vert)

                if path is not None:
                    path = [self.field_grid[x // self.width][x % self.width] for x in path]
                    if len(path) < min_distance:
                        min_distance = len(path)
                        min_path = path

        return min_path
        # predator_paths.append(min_path)

        # return predator_paths
