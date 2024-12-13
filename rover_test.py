from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import time
import logging
import math
from datetime import datetime
import numpy as np
from stable_baselines3 import SAC
from global_value import Glob
import pyrealsense2 as rs

logging.getLogger('dronekit').setLevel(logging.CRITICAL)


class RoverDRL:
    def __init__(self, model_path="logs_sac/best_model.zip"):
        self.glob = Glob()
        self.model_path = model_path
        self.model = SAC.load(self.model_path)

    def get_obs(self, pos, direction, goal, obstacles):
        pos = np.array(pos)
        goal = np.array(goal)
        H = goal - pos
        obstacles = np.array(obstacles)

        # Rotate goal relative to robot's direction
        rotation_matrix = np.array([[np.cos(direction), -np.sin(direction)],
                                    [np.sin(direction), np.cos(direction)]])
        goal_vector = np.dot(rotation_matrix.T, H)

        # Goal angle
        angle_to_goal = -np.arctan2(H[1], H[0]) + direction
        angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi

        # Rotate obstacles and filter visible ones
        visible_obstacles = []
        for obstacle in obstacles:
            relative_position = obstacle - pos
            rotated_obstacle = np.dot(rotation_matrix.T, relative_position)
            # angle_to_obstacle = np.arctan2(rotated_obstacle[1], rotated_obstacle[0])

            visible_obstacles.append((rotated_obstacle, np.linalg.norm(relative_position)))

        # Select three closest obstacles
        visible_obstacles.sort(key=lambda x: x[1])
        closest_obstacles = [obs[0] for obs in visible_obstacles[:3]]

        # Pad with default values if fewer than three obstacles
        while len(closest_obstacles) < 3:
            closest_obstacles.append(np.array([self.glob.none_obstacle_distance, self.glob.none_obstacle_distance]))

        # Create state vector
        state = np.concatenate(
            [goal_vector] + [np.array(obs) for obs in closest_obstacles] + [np.array([angle_to_goal])]).astype(
            np.float32)
        return state

    def path_planning(self, pos, direction, goal, obstacles):
        obs = self.get_obs(pos, direction, goal, obstacles)
        action, _states = self.model.predict(obs, deterministic=True)

        v = action[0]  # Linear velocity
        w = action[1]  # Angular velocity
        return v, w


class Rover:
    def __init__(self, connection_string='COM3', baudrate=57600):
        print('rover connecting...')
        # self.vehicle = connect(connection_string, wait_ready=False, baud=baudrate, timeout=100)
        self.vehicle = connect('tcp:127.0.0.1:5762', wait_ready=False, timeout=100)
        self.init_lat = 35.2265396
        self.init_lon = 126.8404010
        self.max_steering_angle = 30
        self.min_steering_angle = -30

    def arm(self):
        self.vehicle._master.mav.command_long_send(
            self.vehicle._master.target_system,
            self.vehicle._master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        while not self.vehicle.armed:
            print("Waiting for vehicle to arm...")
            time.sleep(1)
        print("Vehicle armed!")
        self.vehicle.mode = VehicleMode("GUIDED")
        time.sleep(0.1)

    def get_pos(self):
        LATITUDE_CONVERSION = 111000  # meters per degree latitude
        LONGITUDE_CONVERSION = 88.649 * 1000  # meters per degree longitude

        delta_lat = self.vehicle.location.global_relative_frame.lat - self.init_lat
        delta_lon = self.vehicle.location.global_relative_frame.lon - self.init_lon

        y = delta_lat * LATITUDE_CONVERSION  # North-South (Y-axis)
        x = delta_lon * LONGITUDE_CONVERSION  # East-West (X-axis)
        return x, y

    def get_direction(self):
        heading = self.vehicle.heading
        # Convert heading from North=0 to Y-axis=0, X-axis positive counterclockwise
        converted_heading = (-heading + 90) % 360
        return math.radians(converted_heading)

    def set_speed_and_steering(self, speed, steering_angle):
        if self.vehicle.mode != VehicleMode("GUIDED"):
            self.vehicle.mode = VehicleMode("GUIDED")
            print("Changing vehicle mode to GUIDED...")
            time.sleep(1)

        # Get the current heading in radians
        current_heading = math.radians(self.vehicle.heading)

        # Create rotation matrix
        R = [
            [math.cos(current_heading), -math.sin(current_heading)],
            [math.sin(current_heading), math.cos(current_heading)]
        ]

        # Compute waypoint based on steering angle
        radius = 10  # Radius for computing waypoint
        steering_radians = math.radians(steering_angle)
        waypoint_local = [
            radius * math.sin(steering_radians),
            radius * math.cos(steering_radians)
        ]

        # Apply rotation matrix to compute global waypoint
        R_T = [
            [R[0][0], R[1][0]],
            [R[0][1], R[1][1]]
        ]
        waypoint_global = [
            R_T[0][0] * waypoint_local[0] + R_T[0][1] * waypoint_local[1],
            R_T[1][0] * waypoint_local[0] + R_T[1][1] * waypoint_local[1]
        ]

        # Get current position
        current_lat = self.vehicle.location.global_relative_frame.lat
        current_lon = self.vehicle.location.global_relative_frame.lon

        # Convert waypoint to global coordinates
        LATITUDE_CONVERSION = 111000  # Meters per degree latitude
        LONGITUDE_CONVERSION = 88.649 * 1000  # Meters per degree longitude

        target_lat = current_lat + waypoint_global[1] / LATITUDE_CONVERSION
        target_lon = current_lon + waypoint_global[0] / LONGITUDE_CONVERSION

        # Set the target location for the waypoint
        target_location = LocationGlobalRelative(
            target_lat, target_lon, self.vehicle.location.global_relative_frame.alt
        )

        # Set vehicle speed
        self.vehicle.groundspeed = speed

        # Navigate to the target location
        self.vehicle.simple_goto(target_location)

class DepthGridManager:
    def __init__(self, grid_size=(8, 3)):
        self.grid_size = grid_size
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(self.config)

    def calculate_grid_depth_average(self, depth_frame):
        depth_array = np.asanyarray(depth_frame.get_data())
        height, width = depth_array.shape
        cell_height = height // self.grid_size[1]
        cell_width = width // self.grid_size[0]
        grid_averages = np.zeros((self.grid_size[1], self.grid_size[0]))

        for row in range(self.grid_size[1]):
            for col in range(self.grid_size[0]):
                start_y = row * cell_height
                end_y = min((row + 1) * cell_height, height)
                start_x = col * cell_width
                end_x = min((col + 1) * cell_width, width)

                if row >= self.grid_size[1] // 2:
                    grid_averages[row, col] = 10000
                else:
                    cell = depth_array[start_y:end_y, start_x:end_x]
                    valid_depths = cell[cell > 0]
                    if valid_depths.size > 0:
                        grid_averages[row, col] = np.mean(valid_depths)
                    else:
                        grid_averages[row, col] = 10000
        return grid_averages

    def retain_top_n_smallest_in_top_half(self, grid_averages, n=3):
        top_half = grid_averages[:self.grid_size[1] // 2, :]
        flat_top_half = top_half.flatten()
        smallest_indices = np.argsort(flat_top_half)[:n]
        mask = np.full(flat_top_half.shape, 10000)
        mask[smallest_indices] = flat_top_half[smallest_indices]
        top_half_masked = mask.reshape(top_half.shape)
        grid_averages[:self.grid_size[1] // 2, :] = top_half_masked
        return grid_averages

    def get_relative_positions(self, grid_averages, depth_frame):
        positions = []
        height, width = np.asanyarray(depth_frame.get_data()).shape
        cell_height = height // self.grid_size[1]
        cell_width = width // self.grid_size[0]

        for row in range(self.grid_size[1]):
            for col in range(self.grid_size[0]):
                if grid_averages[row, col] < 10000:
                    x = (col + 0.5) * cell_width
                    y = (row + 0.5) * cell_height
                    depth = grid_averages[row, col] / 1000.0
                    x_m = (x - width / 2) / (width / 2) * depth
                    y_m = depth
                    positions.append([x_m, y_m])
        return positions

    def process(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return [[10, 10]]

        grid_averages = self.calculate_grid_depth_average(depth_frame)
        grid_averages = self.retain_top_n_smallest_in_top_half(grid_averages, n=3)
        positions = self.get_relative_positions(grid_averages, depth_frame)

        return positions


def multiply(arr, x):
    return np.array(arr) * x


def compute_absolute_obstacles(pos, direction, obstacles):
    # Create rotation matrix R based on the rover's current direction
    R = np.array([
        [np.cos(direction), -np.sin(direction)],
        [np.sin(direction), np.cos(direction)]
    ])

    # Compute absolute positions of obstacles
    absolute_obstacles = []
    for obstacle in obstacles:
        relative_position = np.array(obstacle)
        absolute_position = pos + np.dot(R, relative_position)  # Apply rotation and add to current position
        absolute_obstacles.append(absolute_position.tolist())

    return absolute_obstacles


if __name__ == "__main__":
    vision_close_parameter = 0.5  # 작을 수록 장애물이 더 가깝다고 생각
    goal_close_parameter = 1  # 작을 수록 골이 더 가깝다고 생각
    obs_close_threshold = 2  # 장애물 인지 반경, 이 밖으로는 인지 안함

    rover = Rover()
    rover_drl = RoverDRL()
    rover.arm()
    depth = DepthGridManager()

    pos = rover.get_pos()
    direction = rover.get_direction()
    goal = [-30, 30]
    obstacles = []

    while np.linalg.norm(np.array(goal) - np.array(pos)) > 5:
        relative_obstacles = depth.process()
        filtered_relative_obstacles = [obstacle for obstacle in relative_obstacles if
                                       np.linalg.norm(obstacle) <= obs_close_threshold]
        filtered_relative_obstacles = multiply(filtered_relative_obstacles, vision_close_parameter)
        obstacles = compute_absolute_obstacles(pos, direction, filtered_relative_obstacles)

        v, w = rover_drl.path_planning(multiply(pos, goal_close_parameter), direction,
                                       multiply(goal, goal_close_parameter), obstacles=obstacles)
        w = -w / np.pi * 180  # 각도 변환: 반시계 + → 시계 +

        # 로버 움직임
        rover.set_speed_and_steering(v, w)

        # 로버 상태 업데이트
        pos = rover.get_pos()
        direction = rover.get_direction()
        print(f"Updated Position: {pos}, Updated Direction: {direction}")
