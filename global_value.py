import numpy as np


class Glob:
    def __init__(self):
        # env setting
        self.max_step = 500
        self.map_size = 150
        self.goal_threshold = 10
        self.obstacle_radius = 10
        self.num_obstacles = 20
        self.obstacle_max_velocity = 1
        self.sudden_obstacle_distance = 20
        self.sudden_obstacle_appear_rate = 15
        self.generate_start_goal_threshold = 50
        self.observe_obstacle_num = 3

        # car setting
        self.max_steering = np.pi / 6
        self.fov = np.pi

        # reward setting
        self.reward1 = -500
        self.reward2 = -0.5
        self.reward3 = 200

        # else
        self.none_obstacle_distance = 1000