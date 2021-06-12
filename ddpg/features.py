import numpy as np

def generate_features(state_dict):
    features = []

    for predator in state_dict['predators']:
        x_pred, y_pred, r_pred, speed_pred = predator['x_pos'], predator['y_pos'], predator['radius'], predator['speed']

        features += [x_pred, y_pred]

        angle_min = 1000
        distance_min = 1000
        for prey in state_dict['preys']:
            x_prey, y_prey, r_prey, speed_prey, alive = prey['x_pos'], prey['y_pos'], \
                                                        prey['radius'], prey['speed'], prey['is_alive']
            angle = np.arctan2(y_prey - y_pred, x_prey - x_pred) / np.pi
            distance = np.sqrt((y_prey - y_pred) ** 2 + (x_prey - x_pred) ** 2)

            features += [angle, distance, int(alive), r_prey]

            if distance < distance_min:
                distance_min = distance
                angle_min = angle

        features += [angle_min, distance_min]

        for obs in state_dict['obstacles']:
            x_obs, y_obs, r_obs = obs['x_pos'], obs['y_pos'], obs['radius']
            angle = np.arctan2(y_obs - y_pred, x_obs - x_pred) / np.pi
            distance = np.sqrt((y_obs - y_pred) ** 2 + (x_obs - x_pred) ** 2)

            features += [angle, distance, r_obs]

    return np.array(features, dtype=np.float32)
