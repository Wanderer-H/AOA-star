# In this code, we modify the Astar method to run on the path planning Environment.
# load the 2.5D map that includes the altitude info. (x, y） -> (x, y, z)
# Author: Martin Huang
# 10/18/2022
# import gym_3d_path_planning_v18
import matplotlib.pyplot as plt
import matplotlib
# import gym
import numpy as np
import time
import datetime

matplotlib.use('TkAgg')  # 显示结果 注释掉就会不显示（如果不想显示还可以先import lib,在plt那一句之前加上matplotlib.use('agg')）


class Node:
    def __init__(self, parent=None, position=None):    #这个类需要传入的属性
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):                           #利用__eq__方法来进行判断，这个方法默认有两个参数，一个是self,另一个是other.也就是用自身的属性和other对象的属性分别进行比较，如果比对成功则返回True，失败则返回False。你也可以自定义想要比较的属性有哪些，也不一定是全部的属性都一样才相等。
        return self.position == other.position


def a_star_CPU(maze, start, end, maze_plot, heuristic_factor=1.0):

    debug = False
    start = (start[0], start[1])
    end = (end[0], end[1])
    show_flag = False
    # show_flag = True if maze_plot is not None else False  # 控制是否要画图

    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    start_node.x = start[0]
    start_node.y = start[1]
    start_node.z = maze[start[0], start[1]]

    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    open_list = []
    closed_list = []
    open_list.append(start_node)
    # print(open_list)
    t = time.time()
    count = 0
    while len(open_list) > 0:
        count += 1
        t1 = time.time()
        current_node = open_list[0]
        current_index = 0

        for index, item in enumerate(open_list):               #enumerate()可以返回排序号以及数据本体
            if item.f < current_node.f:  # find the node index of minimum f
                current_node = item
                current_index = index
        t2 = time.time()

        open_list.pop(current_index)  # pop the minimum and append into the closed_list
        closed_list.append(current_node)
        # print("Path length:", len(closed_list))
        # print('current_node == end_node:', current_node.position, end_node.position)

        if np.array_equal(np.array(current_node.position), np.array(end)):
            path = []
            current_p = current_node
            if show_flag: show_state(maze_plot)
            while current_p is not None:
                path.append(current_p.position)
                current_p = current_p.parent
            Search_area = len(open_list) + len(closed_list)
            # print("Searched area is:", Search_area)
            return path[::-1], maze_plot, Search_area

        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:

            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
            w = 1   # 搜索边界控制
            if node_position[0] > len(maze)-w or node_position[0] < w or node_position[1] > len(maze[0])-w or \
                    node_position[1] < w:
                continue
            new_node = Node(current_node, node_position)   # current_node is the parent of node_position

            children.append(new_node)
            # print("children中的元素个数", len(children))
        if children is None:
            return None, None, None
        t3 = time.time()

        for child in children:
            t41 = time.time()
            # if show_flag:
            maze_plot[child.position[0], child.position[1]] = 50
            # print('len(closed_list):', len(closed_list))  # longer and longer
            # if child in closed_list:
            #     continue
            t42 = time.time()
            # child.g = current_node.g + 1
            child_3D_position = [child.position[0], child.position[1], maze[child.position[0], child.position[1]]]
            # print("child_3D:", child_3D_position)
            current_3D_position = [current_node.position[0], current_node.position[1],
                                   maze[current_node.position[0], current_node.position[1]]]
            child.g = current_node.g + np.linalg.norm((np.array(child_3D_position) - np.array(current_3D_position)), ord=2)
            # child.h = ((child.position[0] - end_node.position[0]) ** 2) + (
            #             (child.position[1] - end_node.position[1]) ** 2)
            end_3D_position = [end_node.position[0], end_node.position[1],
                               maze[end_node.position[0], end_node.position[1]]]
            child.h = np.linalg.norm((np.array(child_3D_position) - np.array(end_3D_position)), ord=2)

            # child.h = np.linalg.norm((np.array(child.position) - np.array(end)), ord=2) - 3
            # print("child_3D:", child_3D_position, current_3D_position, end_3D_position)
            child.f = child.g + heuristic_factor * child.h
            # print(child.position)
            # print("child:", child.f, child.g, child.h)
            t43 = time.time()

            if child in closed_list:
                for index, close_node in enumerate(closed_list):
                    if child.position == close_node.position:
                        if child.g < close_node.g:  # sorry, you are not the best
                            open_list.append(child)  # put into the open list again
                            closed_list.pop(index)  # pop out from the close list
                        continue

            if child not in open_list and child not in closed_list:
                open_list.append(child)
                if count % 100 == 0 and show_flag:
                    show_state(maze_plot)
                #     # print("len(open_list):", len(open_list))
                #     # print("time:", time.time() - t)
                #     t = time.time()

            if child in open_list:
                for open_node in open_list:
                    if child.position == open_node.position:
                        if child.g < open_node.g:
                            open_node.g = child.g
                            open_node.f = child.f
                            open_node.h = child.h
                            open_node.parent = child.parent
                        continue


            # print("open_list:", len(open_list))

            # open_list.append(child)
            # print("open_list:", len(open_list))
            t44 = time.time()
            # maze_plot[current_node.position[0], current_node.position[1]] = 1

        t4 = time.time()
        if debug:
            print("portion 1:", t2 - t1)
            print("portion 2:", t3 - t2)
            print("portion 3:", t4 - t3)
            print("portion 4_1:", t42 - t41)
            print("portion 4_2:", t43 - t42)
            print("portion 4_3:", t44 - t43)
            print("Total time:", t4 - t1)
            print("\n")
    # 如果循环结束仍未找到路径
    return None, None, None  # No path found

def show_state(maze_plot):
    plt.figure('A* search area')  #这里显示的是访问过的节点  openlist是待探索节点，closelist是已探索节点
    # plt.title('A* search area')
    plt.clf()
    # maze_plot = maze_plot.T
    # maze_plot = np.rot90(maze_plot)
    plt.imshow(maze_plot.T)#, cmap='terrain')
    # plt.imshow(maze_plot, extent=[0, maze_plot.shape[1], 0, maze_plot.shape[0]])
    plt.axis(False)
    plt.tight_layout()
    plt.draw()      #用draw可以hold，show不可以
    plt.pause(0.001)


def Astar_action(map, start, end, maze_plot, heuristic_factor=1.0):

    path, maze_plot, Search_area = a_star_CPU(map, start, end, maze_plot, heuristic_factor)

    [x, y] = path[1]
    [x0, y0] = start
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
    return a, path, maze_plot, Search_area

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

def Plot_3D_A_modified(start, end, Env):
    t_start = time.time()
    Original_map = np.copy(Env.original_map_s)
    maze_plot = np.copy(Env.original_map_s)

    action, path, _, Search_area = Astar_action(Original_map, start, end, maze_plot, heuristic_factor=1.0)

    length = path_length(path, Original_map)

    return Original_map, length, Search_area, path


def path_length(path, map):
    length = 0
    # print("len(path):", len(path))
    for i in range(len(path)-1):
        current_position = [path[i][0], path[i][1], map[path[i][0], path[i][1]]]
        next_position = [path[i+1][0], path[i+1][1], map[path[i+1][0], path[i+1][1]]]
        length += np.linalg.norm((np.array(current_position) - np.array(next_position)), ord=2)
    return length


if __name__ == '__main__':
    # main()
    t_start = time.time()
    basepath = './map_s/Map_s6.npy'

    Original_map = np.load(basepath)*100    #注意地图要不要乘系数
    maze_plot = np.copy(Original_map) * 0
    maze_path = np.copy(Original_map)

    start = (20, 25)
    end = (85, 70)

    path, maze_plot, Search_area = a_star_CPU(Original_map, start, end, maze_plot, heuristic_factor=1)
    # path, _, Search_area = a_star_CPU(Original_map, start, end, None, heuristic_factor=1)
    # print("The reference action is:", action)
    print("Total time:", time.time() - t_start)
    length = path_length(path, Original_map)
    print('The length is:', length)
    print('The Search_area is:', Search_area)
    # for i, j in path:
    #     maze_plot[i][j] = 1        #注意划线的颜色
    plt.figure('3D-A search area')
    plt.clf()
    plt.imshow(maze_plot)
    plt.title('3D-A-CPU, Path length: ' + str(np.round(length, 2)) + ", Area:" + str(Search_area))
    plt.axis(False)
    plt.show()

    for i, j in path:
        maze_path[i][j] = 1        #注意划线的颜色
    plt.figure('3D-A result')
    plt.clf()
    plt.imshow(maze_path, cmap='cool')
    plt.title('3D-A-CPU, Path length: ' + str(np.round(length, 2)) + ", Area:" + str(Search_area))
    plt.axis(False)
    plt.show()

    # current_time = datetime.datetime.now()
    # name = str(current_time.month) + '_' + str(current_time.day) + '_' + str(current_time.hour) + '_' + str(current_time.minute)
    # print(name)
