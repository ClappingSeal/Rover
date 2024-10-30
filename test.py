import numpy as np
import random
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from PIL import Image, ImageDraw
import io


def dynamic_rrt_star_path_planning(start, goal, obstacles, obstacle_speeds, bounds, max_iter, extend_length,
                                   search_radius, goal_threshold):
    def rrt_star_path_planning(start, goal, obstacles, bounds, max_iter, extend_length, search_radius, goal_threshold):
        def collision_check(new_node, obstacles):
            point = Point(new_node[0], new_node[1])
            for obstacle in obstacles:
                if obstacle.contains(point):
                    return False
            return True

        def distance(point1, point2):
            return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

        def steer(from_node, to_node, extend_length):
            dist = distance(from_node, to_node)
            if dist <= extend_length:
                return to_node

            ratio = extend_length / dist
            new_node = (from_node[0] + ratio * (to_node[0] - from_node[0]),
                        from_node[1] + ratio * (to_node[1] - from_node[1]))
            return new_node

        def find_nearest_node(nodes, random_node):
            min_dist = float("inf")
            nearest_node = None
            for node in nodes:
                dist = distance(node, random_node)
                if dist < min_dist:
                    min_dist = dist
                    nearest_node = node
            return nearest_node

        def find_near_nodes(nodes, new_node, search_radius):
            near_nodes = []
            for node in nodes:
                if distance(node, new_node) < search_radius:
                    near_nodes.append(node)
            return near_nodes

        nodes = [start]
        connections = {}
        connections[start] = (None, 0)
        goal_reached = False

        for _ in range(max_iter):
            random_node = (random.uniform(bounds[0], bounds[1]), random.uniform(bounds[2], bounds[3]))
            nearest_node = find_nearest_node(nodes, random_node)
            new_node = steer(nearest_node, random_node, extend_length)

            if collision_check(new_node, obstacles):
                min_cost = connections[nearest_node][1] + distance(nearest_node, new_node)
                best_node = nearest_node

                near_nodes = find_near_nodes(nodes, new_node, search_radius)
                for near_node in near_nodes:
                    if collision_check(near_node, obstacles):
                        cost = connections[near_node][1] + distance(near_node, new_node)
                        if cost < min_cost:
                            min_cost = cost
                            best_node = near_node

                connections[new_node] = (best_node, min_cost)
                nodes.append(new_node)

                for near_node in near_nodes:
                    if collision_check(near_node, obstacles):
                        new_cost = connections[new_node][1] + distance(new_node, near_node)
                        if new_cost < connections[near_node][1]:
                            connections[near_node] = (new_node, new_cost)

                if distance(new_node, goal) <= goal_threshold and not goal_reached:
                    if collision_check(goal, obstacles):
                        nodes.append(goal)
                        connections[goal] = (new_node, connections[new_node][1] + distance(new_node, goal))
                        goal_reached = True

        path = []
        if goal_reached:
            current_node = goal
            while current_node is not None and current_node != start:
                path.insert(0, current_node)
                current_node = connections[current_node][0]
            path.insert(0, start)

        # Plotting
        plt.figure()
        for obstacle in obstacles:
            x, y = obstacle.exterior.xy
            plt.fill(x, y, "gray", alpha=0.5)

        for node, connection in connections.items():
            if connection[0] is not None:
                plt.plot([connection[0][0], node[0]], [connection[0][1], node[1]], 'r-', linewidth=0.5)

        if path:
            path_x = [point[0] for point in path]
            path_y = [point[1] for point in path]
            plt.plot(path_x, path_y, 'b-', linewidth=4)

        plt.plot(start[0], start[1], 'go', markersize=10)
        plt.plot(goal[0], goal[1], 'bo', markersize=10)
        plt.xlim(bounds[0], bounds[1])
        plt.ylim(bounds[2], bounds[3])
        plt.grid(True)
        plt.show()

        return path

    def update_obstacles(obstacles, obstacle_speeds):
        updated_obstacles = []
        for i, obstacle in enumerate(obstacles):
            speed = obstacle_speeds[i]
            obstacle = Polygon([(x + speed[0], y + speed[1]) for x, y in obstacle.exterior.coords[:-1]])
            updated_obstacles.append(obstacle)
        return updated_obstacles

    path = []

    for _ in range(5):  # Update the environment 5 times
        path = rrt_star_path_planning(start, goal, obstacles, bounds, max_iter, extend_length, search_radius,
                                      goal_threshold)

        if not path:
            print("Path not found. Updating obstacles.")
            obstacles = update_obstacles(obstacles, obstacle_speeds)
        else:
            print("Path found.")
            break

    return path


# Parameters
start = (1, 1)
goal = (9, 9)
obstacles = [
    Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
    Polygon([(6, 6), (8, 6), (8, 8), (6, 8)]),
]
obstacle_speeds = [
    (0.1, 0.05),
    (-0.1, -0.1),
]
bounds = (0, 10, 0, 10)
max_iter = 2000
extend_length = 0.7
search_radius = 1
goal_threshold = 0.5

# Run Dynamic RRT* path planning
path = dynamic_rrt_star_path_planning(start, goal, obstacles, obstacle_speeds, bounds, max_iter, extend_length,
                                      search_radius, goal_threshold)
print("Path:", path)


def create_gif(obstacles, obstacle_speeds, path, bounds, gif_filename, n_frames=50):
    def update_obstacles(obstacles, obstacle_speeds):
        updated_obstacles = []
        for i, obstacle in enumerate(obstacles):
            speed = obstacle_speeds[i]
            obstacle = Polygon([(x + speed[0], y + speed[1]) for x, y in obstacle.exterior.coords[:-1]])
            updated_obstacles.append(obstacle)
        return updated_obstacles

    def draw_frame(obstacles, path, current_path_index, bounds):
        plt.figure()
        for obstacle in obstacles:
            x, y = obstacle.exterior.xy
            plt.fill(x, y, "gray", alpha=0.5)

        plt.plot(path[:current_path_index + 1, 0], path[:current_path_index + 1, 1], 'b-', linewidth=4)
        plt.plot(path[current_path_index, 0], path[current_path_index, 1], 'go', markersize=10)
        plt.xlim(bounds[0], bounds[1])
        plt.ylim(bounds[2], bounds[3])
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return Image.open(buf)

    images = []
    path = np.array(path)
    path_index = 0

    for frame in range(n_frames):
        if frame % (n_frames // len(path)) == 0 and path_index < len(path) - 1:
            path_index += 1
        obstacles = update_obstacles(obstacles, obstacle_speeds)
        frame_image = draw_frame(obstacles, path, path_index, bounds)
        images.append(frame_image)

    images[0].save(gif_filename, save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)


# 사용 예시
create_gif(obstacles, obstacle_speeds, path, bounds, 'dynamic_rrt_star.gif')
