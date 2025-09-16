# GPU加速版本的A*路径规划，结合神经网络启发式（预计算优化版）
## center the ROI
import torch
import numpy as np
import time
from sortedcontainers import SortedList
import matplotlib.pyplot as plt
import matplotlib
import torch.nn as nn
import torch.nn.functional as Func

matplotlib.use('TkAgg')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*72*72, 128)    #16 47 47  for 200  # 16*11*11*4 for 100  #16 72 72 for 300
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, input):
        x1 = self.pool(Func.relu(self.conv1(input)))
        x2 = self.pool(Func.relu(self.conv2(x1)))
        # x2 = Func.relu(self.conv2(x1))
        x3 = x2.view(-1, 16*72*72)
        x4 = Func.relu(self.fc1(x3))
        x5 = Func.relu(self.fc2(x4))
        x6 = self.fc3(x5)
        return x6


class GPU_AStar_Net:
    def __init__(self, net, device='cuda'):
        torch.set_float32_matmul_precision('highest')
        self.device = device
        self.net = net
        self.heuristic_cache = None  # 预计算启发式值矩阵
        self.cache_endpoint = None  # 缓存的目标点

        # Define movement directions (8-connected grid)
        self.directions = torch.tensor([
            [0, -1], [0, 1], [-1, 0], [1, 0],  # 4-directional
            [-1, -1], [-1, 1], [1, -1], [1, 1]  # 8-directional
        ], dtype=torch.int16, device=device)

        # Movement costs matching CPU version
        self.move_costs = torch.tensor([
            1.0, 1.0, 1.0, 1.0,  # Straight moves
            1., 1., 1., 1.  # Diagonal moves
        ], device=device)

    def precompute_heuristics(self, np_map, mario_s, mushroom_s, end, window_extense_size):
        """预计算所有点到终点的启发式值"""
        print("Precomputing heuristic values...")
        t_start = time.time()
        height, width = np_map.shape
        self.heuristic_cache = torch.zeros((height, width), device=self.device)
        self.cache_endpoint = end

        # 批量处理所有点
        for x in range(height):
            for y in range(width):
                # 绘制动态窗口
                dynamic_window = self.draw_object_for_network(
                    img_size=(300, 300),
                    mario_s=mario_s,
                    mushroom_s=mushroom_s,
                    original_map_s=np_map/100.0,
                    mario_position=[x, y],
                    mushroom_position=[end[0], end[1]],
                    window_extense_size=window_extense_size
                )

                # plt.figure(figsize=(5, 5))
                # plt.imshow(dynamic_window)
                # plt.title("dynamic_window")
                # plt.axis('off')
                # plt.pause(1)

                # 准备网络输入
                State = np.transpose(dynamic_window, [2, 0, 1])
                State = np.expand_dims(State, 0)
                inputs = torch.from_numpy(State).float().to(device)

                # 计算网络启发值
                with torch.no_grad():
                    Net_h = self.net(inputs).item()

                linear_h_2D = np.linalg.norm((np.array([x, y]) - np.array(end)), ord=2) - 3
                if linear_h_2D < 10:
                    h_val = 0.0
                else:
                    h_val = Net_h * 1 + 10

                self.heuristic_cache[x, y] = h_val

        print(f"Heuristic precomputation completed in {time.time() - t_start:.2f} seconds")

    def astar(self, np_map, mario_s, mushroom_s, start, end, map_plot, heuristic_factor=1.0, window_extense_size=10):
        """GPU加速的A*算法，使用预计算启发式值"""
        # 检查是否需要重新预计算
        if self.heuristic_cache is None or self.cache_endpoint != end:
            self.precompute_heuristics(np_map, mario_s, mushroom_s, end, window_extense_size)

        # 转换地图到GPU张量
        map_tensor = torch.from_numpy(np_map).float().to(self.device)
        height, width = map_tensor.shape

        # 初始化开放集合
        open_set = SortedList(key=lambda x: (x[0], x[1], x[2]))  # f, x, y
        open_set.add((0.0, start[0], start[1]))

        # 跟踪字典
        g_scores = {start: 0.0}  # 从起点到节点的成本
        came_from = {}  # 路径重建
        closed_set = set()  # 已访问节点

        while open_set:
            # 获取f值最低的节点
            current_f, cx, cy = open_set.pop(0)
            current = (cx, cy)

            # 检查是否到达目标
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(current)
                return path[::-1], len(closed_set), maze_plot

            # 跳过已处理的节点
            if current in closed_set:
                continue
            closed_set.add(current)

            # 生成有效邻居
            neighbors = self.get_neighbors(cx, cy, height, width)
            if neighbors.nelement() == 0:
                continue

            # 计算所有成本（使用预计算启发式值）
            costs = self.compute_costs_with_cache(
                current, neighbors, end,
                map_tensor, g_scores[current],
                heuristic_factor
            )

            # 处理每个邻居
            for idx, (nx, ny) in enumerate(neighbors.cpu().numpy()):
                neighbor = (nx, ny)
                maze_plot[nx, ny] = 50
                if neighbor in closed_set:
                    continue

                # 计算暂定g分数
                tentative_g = costs[idx, 0].item()

                # 如果找到更好的路径则更新
                if tentative_g < g_scores.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + heuristic_factor * costs[idx, 1].item()

                    # 更新优先级队列
                    open_set.discard((g_scores.get(neighbor, float('inf')), nx, ny))
                    open_set.add((f_score, nx, ny))

        return None, len(closed_set), maze_plot

    def get_neighbors(self, x, y, h, w):
        """生成有效相邻单元格"""
        neighbors = torch.tensor([x, y], device=self.device) + self.directions
        valid_mask = (neighbors[:, 0] >= 0) & (neighbors[:, 0] < h) & \
                     (neighbors[:, 1] >= 0) & (neighbors[:, 1] < w)
        return neighbors[valid_mask]

    def compute_costs_with_cache(self, current, neighbors, end, map_tensor, current_g, heuristic_factor):
        """使用预计算启发式值的代价计算"""
        current_z = map_tensor[current[0], current[1]]
        neighbor_zs = map_tensor[neighbors[:, 0], neighbors[:, 1]]

        # 确定移动类型
        move_type = ((neighbors[:, 0] != current[0]) &
                     (neighbors[:, 1] != current[1])).long() * 4

        # 计算g成本
        dz = torch.abs(neighbor_zs - current_z)
        g_cost = current_g + self.move_costs[move_type] * (1 + dz)

        # 从预计算矩阵中获取h成本
        h_cost = self.heuristic_cache[neighbors[:, 0], neighbors[:, 1]]

        return torch.stack([g_cost, h_cost], dim=1)

    def draw_object_for_network(self, img_size, mario_s, mushroom_s, original_map_s,
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

        return dynamic_state


def path_length(path, map_data):
    """计算路径长度"""
    length = 0.0
    for i in range(len(path) - 1):
        current = [path[i][0], path[i][1], map_data[path[i][0], path[i][1]]]
        next_p = [path[i + 1][0], path[i + 1][1], map_data[path[i + 1][0], path[i + 1][1]]]
        length += np.linalg.norm(np.array(current) - np.array(next_p))
    return length


def load_templates(basepath, img_size=(100, 100)):
    """加载所有需要的模板"""
    return (
        np.load(basepath + "Mario_s.npy"),  # mario_s
        np.load(basepath + "Mushroom_s.npy"),  # mushroom_s
        np.load(basepath + 'Map_s5_resized_300.npy')  # original_map_s
    )


if __name__ == '__main__':
    # 主程序
    # 加载神经网络模型
    win = 5
    model_name = 'Dynamic_State_300.pth'

    net = Net()
    params = torch.load(model_name)
    net.load_state_dict(params)
    net.to(device)
    net.eval()  # 设置为评估模式

    # 加载地图和模板
    basepath = './map_s/'
    mario_s, mushroom_s, original_map = load_templates(basepath)
    original_map *= 100     #
    maze_plot = np.copy(original_map)*0
    maze_path = np.copy(original_map)*0

    # 定义起点和终点
    start = (20, 25)
    end = (85, 70)

    # 创建GPU加速的A*规划器
    planner = GPU_AStar_Net(net)

    t_start = time.time()
    # 运行路径规划
    path, search_area, maze_plot = planner.astar(original_map, mario_s, mushroom_s,
                                                 start, end, maze_plot, heuristic_factor=1,
                                                 window_extense_size=win)

    # 输出结果
    print("Total time:", time.time() - t_start)
    if path:
        length = path_length(path, original_map)
        print('The length is:', length)
        print('The Search_area is:', search_area)

        # 可视化结果

        plt.figure(f'3D-A-Net search area, window_extense: {win}')
        plt.clf()
        plt.imshow(maze_plot)
        plt.title(f'win size: {win}, length: {np.round(length, 2)}, Area: {search_area}')
        plt.axis(False)
        plt.show()

        for i, j in path:
            maze_path[i][j] = 100
        plt.figure(f'3D-A-Net path, window_extense: {win}')
        plt.clf()
        plt.imshow(maze_path)  #, cmap='cool'
        plt.title(f'win size: {win}, length: {np.round(length, 2)}, Area: {search_area}')
        plt.axis(False)
        plt.show()
    else:
        print("No path found!")