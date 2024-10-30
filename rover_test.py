from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import time
import logging
import math
from datetime import datetime
import numpy as np
from stable_baselines3 import SAC
from global_value import Glob

logging.getLogger('dronekit').setLevel(logging.CRITICAL)


class RoverDRL:
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


class Rover:
    def __init__(self, connection_string='COM17', baudrate=57600):
        print('rover connecting...')

        # Connecting values
        self.connection_string = connection_string
        self.baudrate = baudrate
        # self.vehicle = connect(self.connection_string, wait_ready=False, baud=self.baudrate, timeout=100)
        self.vehicle = connect('tcp:127.0.0.1:5762', wait_ready=False, timeout=100)

        # Position value
        self.init_lat = self.vehicle.location.global_relative_frame.lat
        self.init_lon = self.vehicle.location.global_relative_frame.lon

        self.max_steering_angle = 45
        self.min_steering_angle = -45

    def arm(self):
        # Send ARM command using MAVLink
        self.vehicle._master.mav.command_long_send(
            self.vehicle._master.target_system,
            self.vehicle._master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0)  # 1 to arm

        # Wait until the vehicle is armed
        while not self.vehicle.armed:
            print("Waiting for vehicle to arm...")
            time.sleep(1)
        print("Vehicle armed!")

        self.vehicle.mode = VehicleMode("GUIDED")
        time.sleep(0.1)

    def disarm(self):
        # Send DISARM command using MAVLink
        self.vehicle._master.mav.command_long_send(
            self.vehicle._master.target_system,
            self.vehicle._master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0)  # 0 to disarm

        # Wait until the vehicle is disarmed
        while self.vehicle.armed:
            print("Waiting for vehicle to disarm...")
            time.sleep(1)
        print("Vehicle disarmed")

    # move non-blocking
    def goto_location(self, x, y, speed=10):
        LATITUDE_CONVERSION = 111000
        LONGITUDE_CONVERSION = 88.649 * 1000

        target_lat = self.init_lat + (x / LATITUDE_CONVERSION)
        target_lon = self.init_lon - (y / LONGITUDE_CONVERSION)
        target_location = LocationGlobalRelative(target_lat, target_lon,
                                                 self.vehicle.location.global_relative_frame.alt)

        self.vehicle.groundspeed = speed
        self.vehicle.simple_goto(target_location)

        print(f"Moving to: Lat: {target_lat}, Lon: {target_lon} at {speed} m/s")

    # move blocking
    def goto_location_block(self, x, y, speed=10):
        LATITUDE_CONVERSION = 111000
        LONGITUDE_CONVERSION = 88.649 * 1000

        target_lat = self.init_lat + (x / LATITUDE_CONVERSION)
        target_lon = self.init_lon - (y / LONGITUDE_CONVERSION)

        def get_distance(lat1, lon1, lat2, lon2):
            R = 6371000  # 지구 반지름 (미터)

            d_lat = math.radians(lat2 - lat1)
            d_lon = math.radians(lon2 - lon1)

            a = (math.sin(d_lat / 2) * math.sin(d_lat / 2) +
                 math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
                 math.sin(d_lon / 2) * math.sin(d_lon / 2))
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            return R * c

        if self.vehicle.mode != VehicleMode("GUIDED"):
            self.vehicle.mode = VehicleMode("GUIDED")
            time.sleep(0.1)

        target_location = LocationGlobalRelative(target_lat, target_lon,
                                                 self.vehicle.location.global_relative_frame.alt)
        self.vehicle.groundspeed = speed
        self.vehicle.simple_goto(target_location)

        print(f"Moving to: Lat: {target_lat}, Lon: {target_lon}")

        while True:
            current_location = self.vehicle.location.global_relative_frame
            distance_to_target = get_distance(current_location.lat, current_location.lon, target_lat, target_lon)

            print("Current position: ", self.get_pos())

            # 도착 확인
            if distance_to_target < 3:  # 3미터 내에 도달하면 도착으로 간주
                print("Arrived at target location!")
                break

            time.sleep(0.5)

    def set_speed_and_steering(self, speed, steering_angle):
        steering_angle = max(min(steering_angle, self.max_steering_angle), self.min_steering_angle)
        current_heading = self.vehicle.heading
        target_heading = (current_heading + steering_angle) % 360
        target_heading_rad = math.radians(target_heading)

        # Calculate velocity components based on target heading
        vx = speed * math.cos(target_heading_rad)  # North-South velocity (X axis)
        vy = speed * math.sin(target_heading_rad)  # East-West velocity (Y axis)

        # Check if the vehicle is in GUIDED mode
        if self.vehicle.mode != VehicleMode("GUIDED"):
            self.vehicle.mode = VehicleMode("GUIDED")
            print("Changing vehicle mode to GUIDED...")
            time.sleep(1)

        # Send velocity command using MAVLink (velocity in X (North) and Y (East) directions)
        self.vehicle._master.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms (not used)
            self.vehicle._master.target_system,  # target_system
            self.vehicle._master.target_component,  # target_component
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # MAV_FRAME
            0b0000111111000111,  # type_mask (only control velocities)
            0, 0, 0,  # x, y, z positions (not used)
            vx, vy, 0,  # velocity in m/s (north, east, down), using X (north) and Y (east)
            0, 0, 0,  # yaw rate not used
            0, 0  # yaw and yaw_rate not used
        )

    # get position (m)
    def get_pos(self):
        LATITUDE_CONVERSION = 111000
        LONGITUDE_CONVERSION = 88.649 * 1000

        delta_lat = self.vehicle.location.global_relative_frame.lat - self.init_lat
        delta_lon = self.vehicle.location.global_relative_frame.lon - self.init_lon

        x = delta_lat * LATITUDE_CONVERSION
        y = -delta_lon * LONGITUDE_CONVERSION

        return x, y

    def get_direction(self):
        return self.vehicle.heading

    # get position (degree)
    def get_time_pos(self):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 현재 시간
        latitude = self.vehicle.location.global_relative_frame.lat  # 위도
        longitude = self.vehicle.location.global_relative_frame.lon  # 경도

        return current_time, latitude, longitude

    def emergency_stop(self):
        """로버를 즉시 멈추는 함수"""
        print("Emergency stop initiated!")

        # Check if the vehicle is in GUIDED mode
        if self.vehicle.mode != VehicleMode("GUIDED"):
            self.vehicle.mode = VehicleMode("GUIDED")
            print("Changing vehicle mode to GUIDED for emergency stop...")
            time.sleep(1)

        # Send velocity command with zero velocity (to stop the rover)
        self.vehicle._master.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms (not used)
            self.vehicle._master.target_system,  # target_system
            self.vehicle._master.target_component,  # target_component
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # MAV_FRAME
            0b0000111111000111,  # type_mask (only control velocities)
            0, 0, 0,  # x, y, z positions (not used)
            0, 0, 0,  # velocity in m/s (north, east, down), setting velocities to 0
            0, 0, 0,  # yaw rate not used
            0, 0  # yaw and yaw_rate not used
        )

        print("Rover has been stopped.")

    # end
    def close_connection(self):
        self.vehicle.close()


def convert_angle(angle):
    """
    Convert an angle from the y-axis based, clockwise system to an x-axis based, counter-clockwise system,
    and ensure the result is within the range of -pi to pi.

    Args:
        angle (float): Angle in degrees, where 0 degrees is along the y-axis (positive) and 359 degrees is clockwise.

    Returns:
        float: Converted angle in radians, where 0 degrees is along the x-axis (positive) and positive values are counter-clockwise,
               bounded between -pi and pi.
    """
    # 입력 받은 각도를 라디안으로 변환
    angle_rad = np.radians(angle)

    # 새로운 좌표계로 변환 (pi/2에서 빼면 x축 기준 반시계방향 좌표계로 변환)
    converted_angle_rad = np.pi / 2 - angle_rad

    # 변환된 값을 -pi에서 pi 범위 내로 조정
    converted_angle_rad = np.arctan2(np.sin(converted_angle_rad), np.cos(converted_angle_rad))

    return converted_angle_rad


def multiply(arr, x=10):
    return np.array(arr) * x


if __name__ == "__main__":
    rover = Rover()
    rover_drl = RoverDRL()
    rover.arm()

    pos = multiply(np.array(rover.get_pos()))
    goal = multiply(np.array([30, 10]))

    try:
        while np.linalg.norm(pos - goal) > 20:
            pos = multiply(np.array(rover.get_pos()))
            direction = convert_angle(rover.get_direction())
            goal = multiply([30, 10])
            obstacles = multiply([[10, 3], [20, 6]])

            v, w = rover_drl.path_planning(pos=pos, direction=direction, goal=goal, obstacles=obstacles)
            w *= 180 / np.pi
            rover.set_speed_and_steering(v, w)
            print(rover.get_pos())
            time.sleep(0.1)

    except KeyboardInterrupt:
        rover.emergency_stop()
        rover.disarm()
        rover.close_connection()
