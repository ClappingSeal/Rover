import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from DRL_env import DRLEnv
from global_value import Glob


class Rover:
    def __init__(self):
        self.glob = Glob()
        self.model_path = "sac.zip"
        self.model = SAC.load(self.model_path)
        self.env = DRLEnv()
        self.obs, _ = self.env.reset()
        self.env.goal_position = [30, 20]
        self.env.obstacles = [[10, 3], [20, 6]]

    def path_planning(self):
        # Get the current observation from the environment
        obs = self.obs

        # Predict the next action using the trained model
        action, _states = self.model.predict(obs, deterministic=True)
        self.obs, reward, terminated, truncated, info = self.env.step(action)

        # Retrieve current position, direction, goal, and obstacles
        pos = self.env.car_position
        direction = self.env.direction
        goal = self.env.goal_position
        obstacles = self.env.obstacles

        return pos, direction, goal, obstacles, terminated, truncated


# Initialize the rover
rover = Rover()

# Lists to store the robot's path
path_x = []
path_y = []

# Simulation parameters
max_steps = 500  # Maximum number of steps

for _ in range(max_steps):
    pos, direction, goal, obstacles, terminated, truncated = rover.path_planning()
    direction = direction % np.pi
    print(direction)
    # print(pos, direction, goal, obstacles)

    # Save the current position
    path_x.append(pos[0])
    path_y.append(pos[1])

    # Check for termination conditions
    if terminated or truncated:
        if terminated:
            print("Reached the goal!")
        else:
            print("Episode terminated.")
        break

# Plotting
plt.figure(figsize=(8, 8))
plt.plot(path_x, path_y, label='Robot Path', color='blue')
plt.scatter(goal[0], goal[1], marker='*', color='green', s=200, label='Goal')

# Plot obstacles
for obs in obstacles:
    circle = plt.Circle((obs[0], obs[1]), rover.env.obstacle_radius, color='red', alpha=0.5)
    plt.gca().add_patch(circle)
plt.scatter([obs[0] for obs in obstacles], [obs[1] for obs in obstacles], color='red', label='Obstacles')

# Set titles and labels in English
plt.title('Robot Path Planning')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
