import numpy as np
from stable_baselines3 import SAC
from global_value import Glob


class Rover:
    def __init__(self):
        self.glob = Glob()
        self.model_path = "sac.zip"
        self.model = SAC.load(self.model_path)

    def get_obs(self, pos, direction, goal, obstacles):
        pos = np.array(pos)
        goal = np.array(goal)
        H = goal - pos
        obstacles = np.array(obstacles)

        # 1. rotate goal
        rotation_matrix = np.array([[np.cos(direction), -np.sin(direction)],
                                    [np.sin(direction), np.cos(direction)]])
        goal_vector = np.dot(rotation_matrix.T, H)

        # 2. goal angle
        angle_to_goal = np.arctan2(H[1], H[0]) - direction

        # 3. rotate obstacles
        visible_obstacles = []
        for obstacle in obstacles:
            relative_position = obstacle - pos
            rotated_obstacle = np.dot(rotation_matrix.T, relative_position)
            angle_to_obstacle = np.arctan2(rotated_obstacle[1], rotated_obstacle[0])

            # fov check
            if -self.glob.fov / 2 <= angle_to_obstacle <= self.glob.fov / 2:
                visible_obstacles.append((rotated_obstacle, np.linalg.norm(relative_position)))

        # 4. two closest obstacles
        visible_obstacles.sort(key=lambda x: x[1])
        closest_obstacles = [obs[0] for obs in visible_obstacles[:2]]

        # 5. if none obstacle
        while len(closest_obstacles) < 2:
            closest_obstacles.append(np.array([self.glob.none_obstacle_distance, self.glob.none_obstacle_distance]))

        state = np.concatenate([goal_vector, closest_obstacles[0], closest_obstacles[1], [angle_to_goal]]).astype(
            np.float32)

        return state

    def path_planning(self, pos, direction, goal, obstacles):
        obs = self.get_obs(pos, direction, goal, obstacles)
        action, _states = self.model.predict(obs, deterministic=True)

        v = action[0]
        w = action[1]

        # v, w usage
        # direction = direction + w
        # movement = np.array([v * np.cos(direction), v * np.sin(direction)])
        # pos = pos + movement

        return v, w


rover = Rover()
v, w = rover.path_planning(pos=[1, 3], direction=np.pi, goal=[10, 6], obstacles=[[4, 50], [6, 40]])
print(v, w)
