# The copy of Astar_25D_GPU_2.1 + and get_action function + State convert fucntion

import torch
import numpy as np
import time
from sortedcontainers import SortedList
import matplotlib.pyplot as plt

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import matplotlib

# matplotlib.use('TkAgg')


def draw_object_for_network(img_size, mario_s, mushroom_s, original_map_s,
                            mario_position, mushroom_position, window_extense_size=10):
    """
    Draw state images for network input (simplified version without sub_state_2)

    Parameters:
        img_size: Image dimensions (height, width)
        mario_s: Mario small template image
        mushroom_s: Mushroom small template image
        original_map_s: Original map height data
        mario_position: Current Mario position [x, y]
        mushroom_position: Mushroom position [x, y]
        window_extense_size: Window extension size

    Returns:
        state_show: Combined state image [img_size, img_size, 3]
        dynamic_ROI: Dynamic window coordinates [x_start, y_start, x_end, y_end]
    """
    # Initialize state image
    state_show = np.zeros([img_size[0], img_size[1], 3])

    # Helper function to draw objects
    def draw_object(channel, template, position):
        h, w = template.shape
        x, y = position

        # Calculate bounds
        x_start = max(x - h // 2, 0)
        x_end = min(x + h // 2, img_size[0])
        y_start = max(y - w // 2, 0)
        y_end = min(y + w // 2, img_size[1])

        # Calculate template region
        temp_x_start = max(h // 2 - (x - x_start), 0)
        temp_x_end = h // 2 + (x_end - x)
        temp_y_start = max(w // 2 - (y - y_start), 0)
        temp_y_end = w // 2 + (y_end - y)

        # Draw object
        channel[x_start:x_end, y_start:y_end] = template[temp_x_start:temp_x_end, temp_y_start:temp_y_end]

    # Draw Mario (channel 0)
    draw_object(state_show[:, :, 0], mario_s, mario_position)

    # Draw Mushroom (channel 1)
    draw_object(state_show[:, :, 1], mushroom_s, mushroom_position)

    # Map height (channel 2)
    state_show[:, :, 2] = original_map_s

    # Calculate dynamic window by extending window_extense_size pixels in all directions
    min_x = min(mario_position[0], mushroom_position[0])
    max_x = max(mario_position[0], mushroom_position[0])
    min_y = min(mario_position[1], mushroom_position[1])
    max_y = max(mario_position[1], mushroom_position[1])

    # Extend the window in all directions by window_extense_size pixels
    window_x_start = max(min_x - window_extense_size, 0)
    window_y_start = max(min_y - window_extense_size, 0)
    window_x_end = min(max_x + window_extense_size, img_size[0])
    window_y_end = min(max_y + window_extense_size, img_size[1])


    # # Calculate required size with protection against too small values
    # raw_size = max(max_x - min_x, max_y - min_y)
    # required_size = min(
    #     max(raw_size + 2 * window_extense_size, 10),  # Minimum size of 10
    #     min(img_size[0], img_size[1])  # Cannot exceed image dimensions
    # )
    #
    # # Recalculate start positions if window exceeds bounds
    # if window_x_end - window_x_start < required_size:
    #     window_x_start = max(0, window_x_end - required_size)
    # if window_y_end - window_y_start < required_size:
    #     window_y_start = max(0, window_y_end - required_size)

    # Get the actual cropped region (may be smaller than required_size)
    cropped = state_show[window_x_start:window_x_end, window_y_start:window_y_end]
    actual_height, actual_width = cropped.shape[:2]

    # Create centered dynamic state
    dynamic_state = np.zeros_like(state_show)

    # Calculate center position
    center_pos_x = (img_size[0] - actual_height) // 2
    center_pos_y = (img_size[1] - actual_width) // 2

    # Place the cropped region in the center
    dynamic_state[
    center_pos_x:center_pos_x + actual_height,
    center_pos_y:center_pos_y + actual_width
    ] = cropped

    return state_show, dynamic_state

class ExactGPUAStar:
    def __init__(self, device='cuda'):
        # Set highest precision to match CPU calculations
        torch.set_float32_matmul_precision('highest')
        self.device = device

        # Define movement directions (8-connected grid)
        self.directions = torch.tensor([
            [-1, 0], [0, 1], [1, 0], [0, -1],  # 4-directional
            [-1, 1], [1, 1], [1, -1], [-1, -1]  # 8-directional
        ], dtype=torch.int16, device=device)

        # Exact movement costs matching CPU version
        self.move_costs = torch.tensor([
            1.0, 1.0, 1.0, 1.0,  # Straight moves
            1.0, 1.0,  # Diagonal moves (exact √2)
            1.0, 1.0
        ], device=device)

    def astar(self, np_map, start, end, heuristic_factor=1.0):
        """A* implementation that exactly matches CPU version"""

        start = tuple(start)
        end = tuple(end)

        # Convert map to GPU tensor
        map_tensor = torch.from_numpy(np_map).float().to(self.device)
        height, width = map_tensor.shape

        # Initialize open set with sorted list for deterministic ordering
        open_set = SortedList(key=lambda x: (x[0], x[1], x[2]))  # f, x, y
        open_set.add((0.0, start[0], start[1]))

        # Tracking dictionaries
        g_scores = {tuple(start): 0.0}  # Cost from start to node
        came_from = {}  # Path reconstruction
        closed_set = set()  # Visited nodes

        while open_set:
            # Get node with lowest f score
            current_f, cx, cy = open_set.pop(0)
            current = (cx, cy)

            # Check if we've reached the goal
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(current)
                return path[::-1], len(closed_set)

            # Skip if already processed
            if current in closed_set:
                continue
            closed_set.add(current)

            # Generate valid neighbors
            neighbors = self.get_neighbors(cx, cy, height, width)
            if neighbors.nelement() == 0:
                continue

            # Compute all costs in parallel on GPU
            costs = self.compute_costs(
                current, neighbors, end,
                map_tensor, g_scores[current]
            )

            # Process each neighbor
            for idx, (nx, ny) in enumerate(neighbors.cpu().numpy()):
                neighbor = (nx, ny)
                if neighbor in closed_set:
                    continue

                # Calculate tentative g score (exact match to CPU)
                tentative_g = costs[idx, 0].item()

                # Update if we found a better path
                if tentative_g < g_scores.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + heuristic_factor * costs[idx, 1].item()

                    # Update priority queue (remove old entry if exists)
                    open_set.discard((g_scores.get(neighbor, float('inf')), nx, ny))
                    open_set.add((f_score, nx, ny))

        return None, len(closed_set)

    def get_neighbors(self, x, y, h, w):
        """Generate valid neighboring cells"""
        neighbors = torch.tensor([x, y], device=self.device) + self.directions
        valid_mask = (neighbors[:, 0] >= 0) & (neighbors[:, 0] < h) & \
                     (neighbors[:, 1] >= 0) & (neighbors[:, 1] < w)
        return neighbors[valid_mask]

    def compute_costs(self, current, neighbors, end, map_tensor, current_g):
        """Calculate costs exactly matching CPU version"""
        current_z = map_tensor[current[0], current[1]]
        end_z = map_tensor[end[0], end[1]]
        neighbor_zs = map_tensor[neighbors[:, 0], neighbors[:, 1]]

        # Determine move type (0-3: straight, 4-7: diagonal)
        move_type = ((neighbors[:, 0] != current[0]) &
                     (neighbors[:, 1] != current[1])).long() * 4

        # Calculate g cost (exact match to CPU version)
        dz = torch.abs(neighbor_zs - current_z)
        g_cost = current_g + self.move_costs[move_type] * (1 + dz)

        # Calculate h cost (3D Euclidean distance)
        dx = neighbors[:, 0].float() - end[0]
        dy = neighbors[:, 1].float() - end[1]
        dz_h = neighbor_zs - end_z
        h_cost = torch.sqrt(dx ** 2 + dy ** 2 + dz_h ** 2)

        # Add small tie-breaker to match CPU behavior
        # h_cost += 0.00001 * (torch.abs(dx) + torch.abs(dy))
        return torch.stack([g_cost, h_cost], dim=1)


def path_length(path, map_data):
    """Calculate path length matching CPU version"""
    length = 0.0
    for i in range(len(path) - 1):
        current = [path[i][0], path[i][1], map_data[path[i][0], path[i][1]]]
        next_p = [path[i + 1][0], path[i + 1][1], map_data[path[i + 1][0], path[i + 1][1]]]
        length += np.linalg.norm(np.array(current) - np.array(next_p))
    return length


def visualize_results(map_data, path, title):
    """Visualize the path on the map"""
    vis_map = np.copy(map_data)
    for i, j in path:
        vis_map[i][j] = np.max(map_data) * 1.2  # Make path stand out

    plt.figure(figsize=(10, 10))
    plt.imshow(vis_map.T, cmap='terrain')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def get_action(CurrentPosition, NextPosition):

    [x, y] = NextPosition
    [x0, y0] = CurrentPosition
    # 判断[x,y] 和 start 之间的关系来决定应该推荐的动作是什么
    if x == x0 - 1 and y == y0:
        a = 0
    elif x == x0 - 1 and y == y0 + 1:
        a = 1
    elif x == x0 and y == y0 + 1:
        a = 2
    elif x == x0 + 1 and y == y0 + 1:
        a = 3
    elif x == x0 + 1 and y == y0:
        a = 4
    elif x == x0 + 1 and y == y0 - 1:
        a = 5
    elif x == x0 and y == y0 - 1:
        a = 6
    else:  # x == x0 - 1 and y == y0 - 1:
        a = 7

    a = np.int64(a)
    return a

# def load_templates(basepath, img_size=(100, 100)):
#     """Load all required templates"""
#     return (
#         np.load(basepath + "Mario_s.npy"),  # mario_s
#         np.load(basepath + "Mushroom_s.npy"),  # mushroom_s
#         np.load(basepath + 'Map_s5.npy')  # original_map_s
#     )



if __name__ == '__main__':
    # Load your map data
    basepath = 'C:/Users/Guoming/Desktop/AOA_clean_version/map_s/Map_s6.npy'
    original_map = np.load(basepath) * 100  # Match your CPU version's scaling

    # Define start and end points (match your test case)
    start_pos = [15, 15]
    end_pos = [85, 85]

    # Create GPU A* planner
    planner = ExactGPUAStar()

    # Run pathfinding
    start_time = time.time()
    path, search_area = planner.astar(original_map, start_pos, end_pos)
    gpu_time = time.time() - start_time

    if path:
        # Calculate path length
        length = path_length(path, original_map)

        # Print results
        print(f"Path length: {length:.2f}")  # Should match 166.24
        print(f"Search area: {search_area}")  # Should match ~4862
        print(f"Computation time: {gpu_time:.4f} seconds")

        # Visualize results
        # title = f"GPU A* (Exact CPU Match)\nLength: {length:.2f}, Area: {search_area}"
        # visualize_results(original_map, path, title)
        for i, j in path:
            original_map[i][j] = 1  # 注意划线的颜色
        plt.figure('3D-A GPU V2 result')
        plt.clf()
        plt.imshow(original_map)
        plt.title('3D-A-GPU V2, Path length: ' + str(np.round(length, 2)) + ", Area:" + str(search_area))
        plt.axis(False)
        plt.show()
    else:
        print("No path found!")