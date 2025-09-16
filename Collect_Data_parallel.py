from Astar_25D_GPU_Collect import *
import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time


def single_rollout(args):
    """单个rollout进程的任务函数"""
    original_map, mario, mushroom, start, end, fixedMap = args
    Dynamic_State = []
    Static_State = []
    H_value = []
    Action = []
    linear_3D = []
    linear_2D = []

    Original_map = np.copy(original_map) * 100
    maze_plot = np.copy(original_map)

    planner = ExactGPUAStar()
    path, search_area = planner.astar(Original_map, start, end)

    h_ = path_length(path, Original_map)
    action = get_action(path[0], path[1])

    static_state, dynamic_state = draw_object_for_network(
        img_size=(100, 100),
        mario_s=mario,
        mushroom_s=mushroom,
        original_map_s=Original_map / 100,
        mario_position=start,
        mushroom_position=end
    )

    dynamic_state = np.transpose(dynamic_state, [2, 0, 1])
    static_state = np.transpose(static_state, [2, 0, 1])

    Dynamic_State.append(np.array(dynamic_state, dtype=np.float16))
    Static_State.append(np.array(static_state, dtype=np.float16))
    Action.append(action)
    H_value.append(float(h_))
    start_3D_position = [start[0], start[1], Original_map[start[0], start[1]]]
    end_3D_position = [end[0], end[1], Original_map[end[0], end[1]]]
    linear_2D.append(np.linalg.norm((np.array(start) - np.array(end)), ord=2))
    linear_3D.append(np.linalg.norm((np.array(start_3D_position) - np.array(end_3D_position)), ord=2))

    for i in range(len(path) - 1):
        h_ = path_length(path[i + 1:], Original_map)
        action = get_action(path[i], path[i + 1])

        static_state, dynamic_state = draw_object_for_network(
            img_size=(100, 100),
            mario_s=mario,
            mushroom_s=mushroom,
            original_map_s=Original_map / 100,
            mario_position=[path[i + 1][0], path[i + 1][1]],
            mushroom_position=end
        )

        dynamic_state = np.transpose(dynamic_state, [2, 0, 1])
        static_state = np.transpose(static_state, [2, 0, 1])

        Dynamic_State.append(dynamic_state)
        Static_State.append(static_state)
        Action.append(action)
        H_value.append(float(h_))
        start_3D_position = [path[i + 1][0], path[i + 1][1], Original_map[path[i + 1][0], path[i + 1][1]]]
        end_3D_position = [end[0], end[1], Original_map[end[0], end[1]]]
        linear_3D.append(np.linalg.norm((np.array(start_3D_position) - np.array(end_3D_position)), ord=2))
        linear_2D.append(np.linalg.norm((np.array(path[i + 1]) - np.array(end)), ord=2))

    return Dynamic_State, Static_State, Action, H_value, linear_2D, linear_3D


def generate_start_end_pairs(Episodes):
    """生成起始点和终点对"""
    start_end_pairs = []
    for ep in range(Episodes):
        while True:
            a = np.random.rand(1)
            if a >= 0.5:
                start = [np.random.randint(10, 30), np.random.randint(10, 30)]
                end = [np.random.randint(60, 90), np.random.randint(60, 90)]
            else:
                start = [np.random.randint(60, 90), np.random.randint(10, 30)]
                end = [np.random.randint(10, 30), np.random.randint(60, 90)]
            dis = np.linalg.norm((np.array(start) - np.array(end)), ord=2)
            if dis > 70:
                start_end_pairs.append((start, end))
                break
    return start_end_pairs


def parallel_rollout(original_map, mario, mushroom, Episodes, fixedMap=False, num_workers=None):
    """并行执行rollout"""
    if num_workers is None:
        num_workers = mp.cpu_count()

    print(f"Using {num_workers} workers for parallel processing")

    # 生成所有的起始点和终点对
    start_end_pairs = generate_start_end_pairs(Episodes)

    # 准备参数
    args_list = [(original_map, mario, mushroom, start, end, fixedMap)
                 for start, end in start_end_pairs]

    # 使用进程池并行执行
    all_results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(single_rollout, args) for args in args_list]

        for i, future in enumerate(futures):
            try:
                result = future.result()
                all_results.append(result)
                print(f"Completed episode {i + 1}/{Episodes}")
            except Exception as e:
                print(f"Error in episode {i + 1}: {e}")

    # 合并所有结果
    Dynamic_State_all = []
    Static_State_all = []
    Action_all = []
    H_value_all = []
    linear_2D_all = []
    linear_3D_all = []

    for result in all_results:
        Dynamic_State, Static_State, Action, H_value, linear_2D, linear_3D = result
        # Dynamic_State_all.extend(Dynamic_State)
        # Static_State_all.extend(Static_State)
        # Action_all.extend(Action)
        # H_value_all.extend(H_value)
        # linear_2D_all.extend(linear_2D)
        # linear_3D_all.extend(linear_3D)

        # 确保所有数据都是16位浮点
        Dynamic_State_all.extend([np.array(x, dtype=np.float16) for x in Dynamic_State])
        Static_State_all.extend([np.array(x, dtype=np.float16) for x in Static_State])
        Action_all.extend([np.float16(x) for x in Action])
        H_value_all.extend([np.float16(x) for x in H_value])
        linear_2D_all.extend([np.float16(x) for x in linear_2D])
        linear_3D_all.extend([np.float16(x) for x in linear_3D])

    return (Dynamic_State_all, Static_State_all, Action_all,
            H_value_all, linear_2D_all, linear_3D_all)

def load_templates(basepath, img_size=(100, 100)):
    """Load all required templates"""
    return (
        np.load(basepath + "Mario_s.npy"),  # mario_s
        np.load(basepath + "Mushroom_s.npy"),  # mushroom_s
        np.load(basepath + 'Map_s5.npy')  # original_map_s
    )


if __name__ == '__main__':

    basepath = 'C:/Users/Guoming/Desktop/AOA_clean_version/map_s/'
    mario, mushroom, original_map = load_templates(basepath)
    original_map *= 1

    fixedMap = False
    Episodes = 1000

    debug = 0
    if debug == 1:
        start = [10, 50]
        end = [55, 75]
        Dynamic_State = []
        Static_State = []
        H_value = []
        Action = []
        linear_3D = []
        linear_2D = []

        # 使用单进程版本进行调试
        args = (original_map, mario, mushroom, start, end, fixedMap)
        Dynamic_State, Static_State, _, H_value, linear_2D, _ = single_rollout(args)

        print('H true:', H_value)
        print('linear_2D:', linear_2D)

        X_H = np.linspace(0, len(H_value), len(H_value))
        X_2D = np.linspace(0, len(linear_2D), len(linear_2D))

        for i in H_value:
            if i < 0:
                print("H", i)

        plt.scatter(X_H, H_value)
        plt.scatter(X_2D, linear_2D)
        plt.legend(["H true", "2D"])
        plt.show()

        # Visualization
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        Static_State_0 = np.transpose(Static_State[0], (1, 2, 0))
        plt.imshow(Static_State_0)
        plt.title("Global State")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        Dynamic_State_0 = np.transpose(Dynamic_State[0], (1, 2, 0))
        plt.imshow(Dynamic_State_0)
        plt.title("Dynamic Window")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    else:
        print("Starting parallel rollout...")
        start_time = time.time()

        # 使用并行版本
        Dynamic_State, Static_State, Action, H_value, linear_2D, linear_3D = parallel_rollout(
            original_map, mario, mushroom, Episodes, fixedMap, num_workers=10
        )

        end_time = time.time()
        print(f"Parallel rollout completed in {end_time - start_time:.2f} seconds")

        assert len(Dynamic_State) == len(Action) == len(H_value)
        print("Size of the data:", len(Dynamic_State))

        current_time = datetime.datetime.now()
        np.save(
            'Dynamic_State_' + str(Episodes) + '_' + str(current_time.day) + '_' + str(current_time.hour) + '_' + str(
                current_time.minute) + '.npy', Dynamic_State)
        np.save(
            'Static_State_' + str(Episodes) + '_' + str(current_time.day) + '_' + str(current_time.hour) + '_' + str(
                current_time.minute) + '.npy', Static_State)
        np.save('Action_' + str(Episodes) + '_' + str(current_time.day) + '_' + str(current_time.hour) + '_' + str(
            current_time.minute) + '.npy', Action)
        np.save('H_value_' + str(Episodes) + '_' + str(current_time.day) + '_' + str(current_time.hour) + '_' + str(
            current_time.minute) + '.npy', H_value)
        np.save('linear_2D' + str(Episodes) + '_' + str(current_time.day) + '_' + str(current_time.hour) + '_' + str(
            current_time.minute) + '.npy', linear_2D)
        np.save('linear_3D' + str(Episodes) + '_' + str(current_time.day) + '_' + str(current_time.hour) + '_' + str(
            current_time.minute) + '.npy', linear_3D)

        print("Data saved, Rollout done!")