import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env
from global_value import Glob


class DRLEnv(gym.Env):
    def __init__(self):
        super(DRLEnv, self).__init__()
        self.glob = Glob()
        self.max_step = self.glob.max_step
        self.current_step = 0
        self.map_size = self.glob.map_size  # [cm]

        self.car_position = None
        self.goal_position = None
        self.goal_threshold = self.glob.goal_threshold  # [cm]
        self.max_steering = self.glob.max_steering
        self.direction = None

        # 장애물 정보
        self.obstacles = []
        self.obstacle_radius = self.glob.obstacle_radius  # [cm]
        self.sudden_obstacle = self.glob.sudden_obstacle_distance  # [cm]
        self.sudden_obstacle_appear_rate = self.glob.sudden_obstacle_appear_rate
        self.obstacle_directions = []  # 장애물 방향 저장
        self.obstacle_speeds = []  # 장애물 속도 저장
        self.observe_obstacle_num = self.glob.observe_obstacle_num  # 장애물 관측 갯수

        # D435i FOV : 85.2 degree
        self.fov = self.glob.fov  # 90

        # Action space (speed, steering angle)
        self.action_space = spaces.Box(
            low=np.array([2, -self.max_steering]),
            high=np.array([5, self.max_steering]),
            dtype=np.float32
        )

        # Observation space (Goal vector + 2 obstacle vectors + angle to goal)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3 + 2 * self.observe_obstacle_num,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, **kwargs):
        while True:
            self.car_position = np.random.uniform(-self.map_size, self.map_size, size=2)
            self.goal_position = np.random.uniform(-self.map_size, self.map_size, size=2)
            distance = np.linalg.norm(self.goal_position - self.car_position)
            if distance >= self.glob.generate_start_goal_threshold:
                break

        self.current_step = 0
        self.direction = np.random.uniform(-np.pi, np.pi)

        # 장애물 초기화
        self.obstacles = []
        self.obstacle_directions = []
        self.obstacle_speeds = []

        # 장애물 생성
        for _ in range(self.glob.num_obstacles):
            self._generate_obstacle()

        remaining_distance = self.goal_position - self.car_position
        state = self._get_observation(remaining_distance)

        return state, {}

    def step(self, action):
        # 장애물 이동
        self._move_obstacles()

        # 장애물 생성
        if self.current_step % self.sudden_obstacle_appear_rate == 0:
            self._generate_sudden_obstacle()

        # action 처리
        action = np.array(action)
        v = action[0]
        w = action[1]
        self.direction = self.direction + w
        movement = np.array([v * np.cos(self.direction), v * np.sin(self.direction)])

        # 상태 업데이트
        self.car_position = self.car_position + movement
        self.current_step = self.current_step + 1
        remaining_distance = self.goal_position - self.car_position
        state = self._get_observation(remaining_distance)

        # 충돌 체크
        for obstacle in self.obstacles:
            if np.linalg.norm(self.car_position - obstacle) <= self.obstacle_radius:
                reward = self.glob.reward1
                print('collide')
                return self._get_observation(remaining_distance), reward, True, False, {}

        # 보상 계산
        reward = 0

        goal_remain = np.linalg.norm(remaining_distance)
        reward_scale = self.glob.reward2
        reward += reward_scale + (0.03 * reward_scale * goal_remain)

        # 조향각 안정화 벌점
        reward -= (10 * abs(w))

        # 목표 도달 여부 체크
        if goal_remain < self.goal_threshold:
            reward += self.glob.reward3
            print('goal')
            terminated = True
        else:
            terminated = False

        truncated = self.current_step >= self.max_step
        return state, reward, terminated, truncated, {}

    def _get_observation(self, remaining_distance):
        # 목표 회전
        rotation_matrix = np.array([[np.cos(self.direction), -np.sin(self.direction)],
                                    [np.sin(self.direction), np.cos(self.direction)]])
        goal_vector = np.dot(rotation_matrix.T, remaining_distance)

        # 목표 각도 계산
        angle_to_goal = np.arctan2(remaining_distance[1], remaining_distance[0]) - self.direction

        # 장애물 회전
        visible_obstacles = []
        for obstacle in self.obstacles:
            relative_position = obstacle - self.car_position
            rotated_obstacle = np.dot(rotation_matrix.T, relative_position)
            angle_to_obstacle = np.arctan2(rotated_obstacle[1], rotated_obstacle[0])

            # 시야각 내 장애물만 추가
            if -self.fov / 2 <= angle_to_obstacle <= self.fov / 2:
                visible_obstacles.append((rotated_obstacle, np.linalg.norm(relative_position)))

        # 가장 가까운 두 개의 장애물
        visible_obstacles.sort(key=lambda x: x[1])
        closest_obstacles = [obs[0] for obs in visible_obstacles[:self.observe_obstacle_num]]

        # 장애물이 없는 경우 기본값 추가
        while len(closest_obstacles) < self.observe_obstacle_num:
            closest_obstacles.append(np.array([self.glob.none_obstacle_distance, self.glob.none_obstacle_distance]))

        state = np.concatenate(
            [np.array(goal_vector)] + [np.array(obs) for obs in closest_obstacles[:self.observe_obstacle_num]] + [
                np.array([angle_to_goal])]).astype(np.float32)

        return state

    def _generate_obstacle(self):
        while True:
            # 장애물 위치 및 속도, 방향 설정
            obstacle = np.random.uniform(-self.map_size, self.map_size, size=2)
            min_distance_to_obstacles = np.min([np.linalg.norm(obstacle - o) for o in self.obstacles], initial=np.inf)
            distance_to_car = np.linalg.norm(obstacle - self.car_position)
            distance_to_goal = np.linalg.norm(obstacle - self.goal_position)

            # 장애물이 겹치지 않도록 거리 확인
            if (min_distance_to_obstacles >= self.obstacle_radius * 2 and
                    distance_to_car >= self.obstacle_radius * 2 and
                    distance_to_goal >= self.obstacle_radius * 2):
                self.obstacles.append(obstacle)
                self.obstacle_directions.append(np.random.uniform(0, 2 * np.pi))
                self.obstacle_speeds.append(np.random.uniform(0, self.glob.obstacle_max_velocity))
                break

    def _generate_sudden_obstacle(self):
        while True:
            # 장애물 생성 거리 유지
            angle_to_obstacle = np.random.uniform(self.direction - self.fov / 2 / 3, self.direction + self.fov / 2 / 3)
            offset = np.array([self.glob.sudden_obstacle_distance * np.cos(angle_to_obstacle),
                               self.glob.sudden_obstacle_distance * np.sin(angle_to_obstacle)])
            obstacle = self.car_position + offset

            self.obstacles.append(obstacle)
            self.obstacle_directions.append(np.random.uniform(0, 2 * np.pi))
            self.obstacle_speeds.append(np.random.uniform(0, self.glob.obstacle_max_velocity))
            break

    def _move_obstacles(self):
        for i in range(len(self.obstacles)):
            # 장애물 이동
            movement = np.array([self.obstacle_speeds[i] * np.cos(self.obstacle_directions[i]),
                                 self.obstacle_speeds[i] * np.sin(self.obstacle_directions[i])])
            self.obstacles[i] = self.obstacles[i] + movement

            # 장애물 튕기기
            if np.any(np.abs(self.obstacles[i]) > self.map_size):
                self.obstacle_directions[i] += np.pi

    def render(self):
        pass

    def close(self):
        pass


# 환경 검사
env = DRLEnv()
check_env(env)
