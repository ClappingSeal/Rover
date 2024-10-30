import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Wedge
from stable_baselines3 import SAC
from DRL_env import DRLEnv
from global_value import Glob

# 환경 초기화 및 모델 로드
glob = Glob()
env = DRLEnv()
model_path = "logs_sac/best_model.zip"  # SAC 모델 경로
model = SAC.load(model_path)

# 시각화 설정
fig, ax = plt.subplots()
ax.set_xlim(-env.map_size, env.map_size)
ax.set_ylim(-env.map_size, env.map_size)
ax.set_aspect('equal')

# 시각화
car_plot, = ax.plot([], [], 'ro', markersize=6, label="Car")  # 차량의 위치
goal_plot, = ax.plot([],  [], 'go', markersize=8, label="Goal")  # 목표 위치
obstacle_plots = []  # 장애물 위치 plot들을 담을 리스트

# 자동차 시야각
fov_radius = env.map_size / 2  # 부채꼴 반경을 맵 크기의 절반으로 설정
fov_wedge = Wedge((0, 0), fov_radius, 0, 0, color='#9370DB', alpha=0.3)  # 부채꼴 (중심점, 반경, 시작 각도, 끝 각도)
ax.add_patch(fov_wedge)


# 초기화 함수
def init():
    car_plot.set_data([], [])
    goal_plot.set_data([], [])
    fov_wedge.set_center((0, 0))
    return [car_plot, goal_plot, fov_wedge] + obstacle_plots


# 업데이트 함수
def update(frame):
    global obs
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # 차량 위치 업데이트
    car_plot.set_data(env.car_position[0], env.car_position[1])

    # 목표 위치 업데이트
    goal_plot.set_data(env.goal_position[0], env.goal_position[1])

    # 장애물 위치 업데이트
    while len(obstacle_plots) < len(env.obstacles):
        new_obstacle_plot, = ax.plot([], [], 'bo', markersize=glob.obstacle_radius * 2 - 4)
        obstacle_plots.append(new_obstacle_plot)

    for i, obs_pos in enumerate(env.obstacles):
        obstacle_plots[i].set_data(obs_pos[0], obs_pos[1])

    # 시야각 (FOV) 부채꼴 업데이트
    fov_wedge.set_center((env.car_position[0], env.car_position[1]))
    fov_wedge.set_theta1(np.degrees(env.direction - glob.fov / 2))
    fov_wedge.set_theta2(np.degrees(env.direction + glob.fov / 2))

    if terminated or truncated:
        obs, _ = env.reset()

    return [car_plot, goal_plot, fov_wedge] + obstacle_plots


obs, _ = env.reset()
ani = FuncAnimation(fig, update, frames=200, init_func=init, blit=True, interval=100)
ani.save('sac_car_fov.gif', writer='imagemagick', fps=10)
plt.legend()
plt.show()
