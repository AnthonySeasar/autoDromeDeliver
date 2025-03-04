import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.image import imread
import matplotlib.style as mplstyle
from tqdm import tqdm
import itertools


class DroneEnv(gym.Env):

    metadata = {"render_modes": ['human']}
    reward_range = (-float("inf"), float("inf"))

    def __init__(self, action_type='continuous', mission_type='fly',control_type='velocity', max_building_num=100):
        '''
        :param action_type: 
            'continuous': 连续动作空间
            'discrete': 离散动作空间
        :param control_type: 
            'velocity': 控制速度
            'acceleration': 控制加速度
        :param mission_type: 
            'fly': 要求无人机飞到目标的, 无需投下包裹, 无需返回
            'drop': 无人机飞到目标的, 无人机需要投下包裹, 无需返回
            'back': 无人机飞到目标的, 无人机需要返回, 会自动投下包裹
            'all': 要求无人机飞到目标的, 并投下包裹, 然后返回
        '''

        super(DroneEnv, self).__init__()
        # 假设无人机的参数
        drone_horizontal_accele = 5     # 无人机水平加速度, m/s^2
        drone_climb_accele = 2          # 无人机爬升加速度, m/s^2
        drone_descent_accele = 10       # 无人机下降加速度, m/s^2
        self.max_acceleration = np.array([drone_horizontal_accele, drone_climb_accele, drone_descent_accele])  # 无人机最大加速度, m/s^2
        self.max_velocity = np.array([600, 600, 600])  # 无人机最大速度, m/s, 600m/s=30km/h
        self.max_resultant_velocity = np.linalg.norm(self.max_velocity)  # 无人机最大合速度, m/s
        self.drone_size = np.array([0.2, 0.2, 0])  # 无人机的尺寸, WDH, 高度忽略不计, m
        self.package_max_size = np.array([0.5, 0.5, 0.3])  # 包裹的最大尺寸, m
        # 障碍物参数
        self.max_building_num = max_building_num  # 最大建筑物数量
        self.min_building_w, self.min_building_h, self.min_building_d = 1, 1, 1  # 建筑物的最小尺寸, m
        self.max_building_w, self.max_building_h, self.max_building_d = 5, 20, 5  # 建筑物的最大尺寸, m
        # 地图参数
        self.map_max_h = 500  # 地图最大高度, m
        self.start_inner_radius = 500  # 起始内部半径, m
        self.start_outer_radius = 1000  # 起始外部半径, m
        self.aim_xyz = np.array([0, 0, 20])  # 目标点坐标
        self.red_area_size = np.array([0.5, 0.5, 0])   # 红区区域尺寸, m
        self.yellow_area_size = np.array([2, 2, 0])  # 黄区区域尺寸, m
        self.map_margin = self.max_building_w + 1  # 地图边缘留白, m
        self.map_r = self.start_outer_radius + self.map_margin   # 地图半径, m
        self.return_tolerence = 20  # 无人机返回容差, m
        self.map_max = np.array([self.map_r, self.map_r, self.map_max_h]) # 地图最大坐标
        # 定义观测空间. 实际坐标, 实际速度 = obs[:6] * (map_max, max_velocity)
        # self.observation_space = spaces.Box(
        #     low=np.array([-1, -1, 0, -1, -1, -1]),
        #     high=np.array([1, 1, 1, 1, 1, 1]),
        #     dtype=np.float32
        # )   # x, y, z, xV, yV, zV (无人机坐标, xyz轴的速度)
        self.observation_space = spaces.Box(
            low=np.array([-1, -1, 0, -1, -1, -1, -1, -1, 0, -1, -1]),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            dtype=np.float32
        )  #  x, y, z, xV, yV, zV, aimX, aimY, aimZ, xySin, xyCos (无人机坐标, xyz轴的速度, 目标点坐标, 目标点与无人机的xy轴夹角)
        # 定义动作空间
        self.mission_type = mission_type # 任务类型
        self.action_type = action_type  # 动作类型
        self.control_type = control_type  # 控制类型
        drop_action_space = spaces.Discrete(2)  # 0: 不投下包裹, 1: 投下包裹
        if action_type == 'discrete':
            fly_action_space = spaces.Discrete(6)  # 0: 前进, 1: 左转, 2: 右转, 3: 后退, 4: 左前, 5: 右前
        else:
            fly_action_space = spaces.Box(
                low=np.array([-1, -1, -1]),     # 三轴方向速度 or 三轴方向加速度
                high=np.array([1, 1, 1]),
                dtype=np.float32
            )  # xA, yA, zA 三轴方向加速度
        # 定义标准化观测空间系数
        if self.control_type == 'acceleration':
            self.normalize_array = np.concatenate((self.map_max, self.max_acceleration, self.map_max, np.ones(2)))
        else:
            self.normalize_array = np.concatenate((self.map_max, self.max_velocity, self.map_max, np.ones(2)))
        if mission_type == 'fly' or mission_type == 'back':
            self.action_space = fly_action_space
        else: 
            self.action_space = spaces.Tuple((fly_action_space, drop_action_space))
        # 定义环境参数
        self.time_step = 0.1        # 时间步长, s
        self.time_out = 10          # 单程最大时间, s
        # 初始化render参数
        self.fig = None
        # 定义顶点转化矩阵, 将3维的size转为顶点坐标
        self.vertices_to_map = np.array([
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], 
            [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],  # 底面四个顶点, 中心在原点
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], 
            [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]  # 顶面四个顶点
        ])
        self.vertices_to_faces = np.array([     # (8, 3) -> (6, 4, 3) 的转换矩阵, 将立方体8个顶点转为6个面片, 每个面片由4个顶点组成
            [0, 1, 2, 3],   # 0, 1, 2, 3 表示四个顶点的索引
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7]
        ])

    def reset(self, seed=None, return_info=False, option={}):
        '''
        :param seed: 随机种子
        :param return_info: 是否返回额外信息
        :param option: 额外参数, dict, 包含以下参数:
            reset_buildings: bool: 是否重新生成障碍物
            buildings: list: 障碍物坐标
            pos: list: 无人机初始坐标
            random_aim: bool: 是否随机设置目标点坐标 (default: False)
        :return: 环境状态
        '''
        super(DroneEnv, self).reset(seed=seed)

        # 环境状态初始化
        self.buildings_center = None  # 障碍物中心坐标
        self.buidlings_size = None  # 障碍物尺寸
        self.collision_size = None  # 无人机碰撞尺寸
        self.carried_package = True  # 无人机是否携带包裹
        self.package_size = None  # 包裹尺寸
        self.state = None  # 无人机状态, x, y, z, xV, yV, zV
        self.done = False  # 环境是否结束
        self.step_count = 0  # 交互次数
        self.journey_distance = 0  # 飞行距离
        self.journey_trajectory = []  # 飞行轨迹
        self.reward = 0         # step奖励
        self.rewards = [0]       # 各步奖励
        self.max_step = int(self.time_out / self.time_step)  # 最大步数

        # 初始化障碍物
        if option.get('reset_buildings', False):    # reset_buildings为True, 则重新生成障碍物
            if option.get('buildings', None) is not None:
                self.buildings_center = option['buildings']
        if self.buildings_center is None:
            self.buildings_center, self.buidlings_size = self.generate_buildings(
                np.random.randint(1, self.max_building_num))

        # 初始化目标点
        if option.get('random_aim', False):
            r = np.random.uniform(0, self.start_inner_radius)
            theta = np.random.uniform(0, 2 * np.pi)     # 随机极角
            self.aim_xyz = np.array([r * np.cos(theta), r * np.sin(theta), 20])  # 转为笛卡尔坐标
        else:
            self.aim_xyz = np.array([0, 0, 20])  # 目标点坐标

        # 初始化无人机状态, 在内部半径和外部半径之间随机选择出发位置
        if option.get('pos', None) is not None:
            x, y, z = option['pos']
        else:
            r = np.random.uniform(self.start_inner_radius,
                                  self.start_outer_radius)     # 随机极半径
            theta = np.random.uniform(0, 2 * np.pi)     # 随机极角
            x, y, z = r * np.cos(theta), r * np.sin(theta), self.package_max_size[2]  # 转为笛卡尔坐标

        # self.state = np.array([x, y, z, 0, 0, 0], dtype=np.float32)    # 无人机的状态, x, y, z, xV, yV, zV
        self.state = np.concatenate((np.array([x, y, z]), np.zeros(3), self.aim_xyz, np.zeros(2)))
        self.reletive_pos = self.state[:3] - self.aim_xyz  # 无人机相对位置 = 绝对位置 - 目标位置
        target_r = np.linalg.norm(self.reletive_pos[:2])
        self.state[-2:] = self.reletive_pos[:2] / target_r
        self.start_pos = np.array([x, y, z])  # 起始位置
        self.journey_trajectory.append(self.start_pos)

        # 理想单程总路程
        self.total_distance = np.linalg.norm(self.reletive_pos)   # 出发点到目标点的欧式距离
        self.previous_distance = self.total_distance  # 上次剩余距离
        self.min_distance = self.total_distance  # 最小剩余距离

        # 初始化包裹
        self.package_size = np.random.uniform(0, self.package_max_size)

        # 计算无人机的碰撞尺寸, max(包裹尺寸, 无人机尺寸)
        self.collision_size = np.maximum(self.package_size, self.drone_size)

        state = np.copy(self.state)
        state[:3] = state[:3] - state[6:9]  # 无人机相对位置 = 绝对位置 - 目标位置
        state /= self.normalize_array       # 归一化
        return state

    def generate_buildings(self, num):
        '''生成num个随机的建筑物的xyz中心坐标和尺寸'''
        buildings_center = np.zeros((num, 3))  # 建筑物的xyz中心坐标
        buildings_size = np.zeros((num, 3))  # 建筑物的尺寸, width, depth, height
        buildings_size[:, 0] = np.random.uniform(
            self.min_building_w, self.max_building_w, num)  # 宽度随机
        buildings_size[:, 1] = np.random.uniform(
            self.min_building_d, self.max_building_d, num)  # 长度随机
        buildings_size[:, 2] = np.random.uniform(
            self.min_building_h, self.max_building_h, num)  # 高度随机
        buildings_center[:, :2] = np.random.uniform(
            -self.start_outer_radius, self.start_outer_radius, (num, 2))  # 随机生成xy位置
        buildings_center[:, 2] = buildings_size[:, 2] / 2  # 高度置中
        return buildings_center, buildings_size

    def _center_size_to_vertices(self, center, size):
        '''将建筑物中心坐标和尺寸转换为顶点坐标'''
        vertices = np.tile(size[..., None, :], (1, 8, 1)
                           )     # (num, 3) -> (num, 8, 3)
        vertices = vertices * self.vertices_to_map + \
            center[..., None, :]  # 中心处于原点的8个顶点+位置偏移
        return vertices
        
    def _step_info(self):
        return {
            'step': self.step_count,
            'aim': self.aim_xyz,
            'journey_distance': self.journey_distance,
            'journey_trajectory': self.journey_trajectory,
            'rewards': self.rewards,
        }

    def _action_transfer(self, discrete_action):
        '''将离散fly动作转为实际的加速度or速度'''
        action = np.zeros(3)
        action[discrete_action // 2] = discrete_action % 2 * 2 - 1  # -1: 反向加速, 1: 前向加速
        return action   # shape (x, y, z) in value (-1, 1) 

    def step(self, action):
        '''
        :param action: 
            fly: fly_action_space
                continuous and velocity: xV, yV, zV
                continuous and acceleration: xA, yA, zA
                discrete: 0: 前进, 1: 左转, 2: 右转, 3: 后退, 4: 左前, 5: 右前
            drop or all: drop_action_space
                (fly_action_space, drop_action_space)
        :return: 环境状态, 奖励, 是否终止, 额外信息
        '''

        if self.done:
            return self.state, 0, True, self._step_info()
            
        self.reward = 0
        self.step_count += 1
        fly_action = action

        self.reletive_pos = self.state[:3] - self.aim_xyz  # 无人机相对位置 = 绝对位置 - 目标位置
        xy_distance = np.abs(self.reletive_pos)[:2]    # 无人机距离目标点的xy距离
        if np.all(xy_distance <= (self.return_tolerence - self.collision_size[:2])):
            self.reward += 200
            if self.mission_type == 'fly' or self.carried_package == False:
                self.done = True
            elif self.mission_type == 'back':
                self.carried_package = False
                self.aim_xyz = self.start_pos.copy()
                self.state[6:9] = self.aim_xyz
                self.max_step = 2 * int(self.time_out / self.time_step)  # 最大步数
        if self.mission_type == 'drop' or self.mission_type == 'all':
            fly_action, drop_action = action[0], action[1]      # drop_action: 0: 不投下包裹, 1: 投下包裹
            # 计算投放包裹的奖励
            if self.carried_package and drop_action :  # 如果投下包裹
                self.carried_package = False
                self.collision_size = self.drone_size
                # 部分在黄区内
                if np.all(xy_distance < self.yellow_area_size[:2] + self.collision_size[:2]):
                    self.reward += 25
                # 全部在黄区内
                if np.all(xy_distance <= (self.yellow_area_size[:2] - self.collision_size[:2])):
                    self.reward += 25
                # 部分在红区内
                if np.all(xy_distance < (self.red_area_size[:2] + self.collision_size[:2])):
                    self.reward += 25
                # 全部在红区内
                if np.all(xy_distance <= self.red_area_size[:2] - self.collision_size[:2]):
                    self.reward += 25
                if self.mission_type == 'drop':
                    self.done = True
                else:
                    self.aim_xyz = self.start_pos.copy()
                    self.state[6:9] = self.aim_xyz
                    self.max_step = 2 * int(self.time_out / self.time_step)  # 最大步数

        # 计算无人机当前状态
        if self.action_type == 'discrete':
            fly_action = self._action_transfer(fly_action)
        if self.control_type == 'acceleration':
            acceleration = fly_action * self.max_acceleration
        else:   # fly_action 是速度
            acceleration = (fly_action * self.max_velocity - self.state[3:6]) / self.time_step
        
        # 更新无人机速度, v_t = v_t-1 + a_t*t
        self.state[3:6] += acceleration * self.time_step
        self.state[3:6] = np.clip(self.state[3:6], -self.max_velocity, self.max_velocity)
        # 更新无人机位置, x_t = x_t-1 + v_t*t
        self.state[0:3] += self.state[3:6] * self.time_step 
        target_r = np.linalg.norm(self.reletive_pos[:2])
        self.state[-2:] = self.reletive_pos[:2] / target_r

        # 计算距离奖励, 理想最大奖励是欧氏直线距离
        remaining_distance = np.linalg.norm(self.reletive_pos)
        # 奖励 = -距离, 距离越远, 奖励越低
        # self.reward -= remaining_distance / self.total_distance * 5       # 剩余距离接近0时, 奖励约为0
        # self.reward += (10 - remaining_distance) / self.total_distance * 5       # 剩余距离接近0时, 奖励约为0
        self.reward -= np.log(1 + (remaining_distance / self.total_distance * 70))
        # self.reward -= (remaining_distance / self.total_distance * 2 - 1 ) * 5
        # self.reward -= (remaining_distance / self.total_distance)**2 * 5    # 用距离的平方, 拉大远距离惩罚
        # self.reward -= (((remaining_distance / self.total_distance) * 2 - 1) ** 3 ) * 5    
        # self.reward -= remaining_distance     # 用距离的平方, 拉大远距离惩罚
        # 奖励 = max( 历史最短剩余距离 - 当前剩余距离 ) / 总路程
        # if remaining_distance < self.min_distance:
        #     reward += (self.min_distance - remaining_distance) / self.total_distance * 10
        #     self.min_distance = remaining_distance
        # 奖励 = (上次剩余距离 - 当前剩余距离) / 总路程, 此奖励不合适, 远离目标的惩罚过高
        # reward += (self.previous_distance - remaining_distance) / \
        #     self.total_distance * 50
        # self.previous_distance = remaining_distance
        # 奖励 = (总路程 - 剩余距离) / 总路程, 此奖励不合适, 重复走过路径时可以获得重复奖励
        # reward += (self.total_distance - np.linalg.norm(self.state[:3] - self.aim_xyz)) / self.total_distance

        # 计算耗时奖励. 若无耗时奖励, 无人机可能会0速度停在目标附近. 奖励不应该是t的函数, 违反了平稳性假设
        # self.reward -= self.step_count * 0.1

        # 飞行路程 = 合速度 * 时间
        resultant_velocity = float(np.linalg.norm(self.state[3:6]))
        self.journey_distance += resultant_velocity * self.time_step
        self.journey_trajectory.append(self.state[:3].copy())      

        # 计算速度奖励, 速度越快, 奖励越高, 鼓励无人机快速到达目标. 控制速度模式时, 会导致无人机走折线刷速度奖励
        # self.reward += resultant_velocity / self.max_resultant_velocity * 3

        # 计算路程奖励, 路程越远, 奖励越高, 鼓励无人机找到最短路径
        # self.reward -= self.journey_distance / self.total_distance

        # 判断碰撞
        for building_center, building_size in zip(self.buildings_center, self.buidlings_size):
            if np.all(np.abs(self.state[:3] - building_center) < self.collision_size + building_size):
                self.done = True
                self.reward -= 100

       # 判断出界
        if np.any(np.abs(self.state[:2]) > self.map_r) or self.state[2] > self.map_max_h or self.state[2] < 0:
            self.done = True
            self.reward -= 100

        # 判断是否回归出发点
        if self.journey_distance >= self.total_distance and not self.carried_package \
                and np.linalg.norm(self.state[:3] - self.start_pos) < self.return_tolerence:
            self.done = True
            self.reward += 100

        # 判断是否超时
        if self.step_count > self.max_step:
            self.done = True
            # self.reward -= 100
        self.rewards.append(self.reward)

        state = np.copy(self.state)
        state[:3] = state[:3] - state[6:9]  # 无人机相对位置 = 绝对位置 - 目标位置
        state /= self.normalize_array       # 归一化
        return state, self.reward / 100, self.done, self._step_info()
        # return np.copy(self.state) / self.normalize_array, self.reward / 100, self.done, self._step_info()

    def render(self, mode='human', speed=1):
        '''
        :Aparam mode: 渲染模式, human或rgb_array
        :Aparam speed: 速度, 1为正常速度, 2为2倍速度
        :return: 无返回值
        '''
        def render_circle(R, Z, color):
            mplstyle.use('fast')        # 简化绘图, 加速绘制速度
            plt.ion()       # 开启交互模式
            theta = np.linspace(0, 2*np.pi, 50)
            x = R * np.cos(theta)
            y = R * np.sin(theta)
            z = np.zeros_like(x)
            z.fill(Z)
            self.ax.plot(x, y, z, color=color, linestyle='--', linewidth=1)

        if self.fig is None:    # 第一次调用render, 创建画布
            self.fig = plt.figure(figsize=(8, 8))  # 设置画布大小
            self.fig.canvas.manager.set_window_title('Drone Environment')
            
            self.ax = self.fig.add_subplot(111, projection='3d')  # 创建三维坐标系
            self.ax.set_xlim(-self.map_r, self.map_r)
            self.ax.set_ylim(-self.map_r, self.map_r)
            self.ax.set_zlim(0, self.map_max_h)
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
            self.ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
            self.ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.5))
            # 绘制无人机出发半径
            render_circle(self.start_inner_radius, 0, 'red')
            render_circle(self.start_outer_radius, 0, 'red')
            # 绘制投放区
            yellow_vertices = np.tile(
                self.yellow_area_size[None, :], (8, 1)) * self.vertices_to_map
            self.ax.add_collection3d(Poly3DCollection(
                yellow_vertices[self.vertices_to_faces], facecolors='yellow', edgecolors='yellow', alpha=1, linewidths=0.5))
            red_vertices = np.tile(
                self.red_area_size[None, :], (8, 1)) * self.vertices_to_map
            self.ax.add_collection3d(Poly3DCollection(
                red_vertices[self.vertices_to_faces], facecolors='red', edgecolors='red', alpha=1, linewidths=0.5))

            self.render_aim = self.ax.scatter(self.aim_xyz[0], self.aim_xyz[1], 0, color='red', marker='o', s=10)

            # 绘制障碍物
            buildings_vertices = self._center_size_to_vertices(
                self.buildings_center, self.buidlings_size)
            for building_vertices in buildings_vertices:
                self.ax.add_collection3d(Poly3DCollection(
                    building_vertices[self.vertices_to_faces], facecolors='lightblue', edgecolors='darkblue', alpha=1, linewidths=0.5))
            # 计算无人机和包裹在原点的顶点坐标

            self.drone_vertices = self._center_size_to_vertices(
                np.array([0, 0, 0]), self.drone_size)[0]  # 会返回(1, 8, 3), 去掉0维度
            self.package_vertices = self._center_size_to_vertices(
                np.array([0, 0, 0]), self.package_size)[0]
            # 初始化无人机和包裹的collection
            self.drone_collection = None
            self.package_collection = None
            self.trajectory_plot = None
        
        self.ax.set_title(f'Reward: {self.reward:.1f} \n Aim: {self.aim_xyz} \nXYZ: {self.state[:3]} \n XYZV: {self.state[3:6]}')

        self.render_aim.remove()
        self.render_aim = self.ax.scatter(self.aim_xyz[0], self.aim_xyz[1], 0, color='red', marker='o', s=10)

        # 移除上次的collection和轨迹
        if self.drone_collection is not None:
            self.drone_collection.remove()
        if self.package_collection is not None:
            self.package_collection.remove()
        if self.trajectory_plot is not None:
            self.trajectory_plot[0].remove()

        # 绘制无人机
        drone_vertices = self.drone_vertices + self.state[:3]  # 当前无人机顶点坐标
        self.drone_collection = self.ax.add_collection3d(Poly3DCollection(
            drone_vertices[self.vertices_to_faces], facecolors='lime', edgecolors='darkblue', alpha=1, linewidths=0.5))
        # 绘制包裹
        package_vertices = self.package_vertices + self.state[:3]  # 当前包裹顶点坐标
        self.package_collection = self.ax.add_collection3d(Poly3DCollection(
            package_vertices[self.vertices_to_faces], facecolors='goldenrod', edgecolors='darkgoldenrod', alpha=1, linewidths=0.5))
        # 绘制飞行路线
        x_traj, y_traj, z_traj = zip(*self.journey_trajectory)
        self.trajectory_plot = self.ax.plot(
            x_traj, y_traj, z_traj, color='darkgray', alpha=0.5, linewidth=0.5)
        # 绘制当前state
        time_step = 1 / speed * self.time_step
        plt.pause(time_step)  # 显示并延迟
    
    def save_trajectory(self, file_path):
        with open(file_path, 'w') as f:
            for i, pos in enumerate(self.journey_trajectory):
                line = f"pos: {pos[0]:3.1f}, {pos[1]:3.1f}, {pos[2]:3.1f}, \
                         reward: {self.rewards[i]:.1f}, \
                         distance: {np.linalg.norm(pos - self.aim_xyz):.1f}\n"
                f.write(line)

def test_drone_env(building_num):
    # 障碍物参数
    env = DroneEnv('continuous', 'back', building_num)
    curr_state = env.reset()
    init_state = curr_state.copy()

    np.set_printoptions(formatter={'float': lambda x: format(x, '.2f')})
    loop = tqdm(itertools.count(), desc="DroneEnv")
    carried_package = True
    total_reward = 0

    for i in loop:
        if curr_state[2] < 20:
            zv = 50
        elif curr_state[2] > 21:
            zv = -50
        if carried_package:
            aim = np.array([0, 0, 0])
        else:
            aim = init_state[:3]
        # 计算动作
        xyv = (((curr_state[:2] - aim[:2]) < 0).astype(int) * 2 - 1) * 50
        xyzv = np.array([xyv[0], xyv[1], zv])
        action = xyzv
        state, reward, done, info = env.step(action)
        total_reward += reward
        loop.set_postfix_str(
            f"reward: {reward:.1f}, xyz: {state[:3]}, xyzV: {state[3:6]}")
        env.render()
        if done:
            print("Episode finished after {} timesteps, total reward: {}".format(i+1, total_reward))
            break
    
def manully_test_drone_env(building_num):
    import keyboard
    exite_flag = False
    a = np.array([0, 0, 0], dtype=np.float32)

    env = DroneEnv('continuousV', 'back', building_num)
    np.set_printoptions(formatter={'float': lambda x: format(x, '.2f')})
    loop = tqdm(itertools.count(), desc="DroneEnv")
    carried_package = True
    total_reward = 0
    v = 0.1

    def key_press(event):
        if event.name == '8' and event.event_type == 'down':
            a[0] += v
        elif event.name == '2' and event.event_type == 'down':
            a[0] -= v
        elif event.name == '4' and event.event_type == 'down':
            a[1] -= v
        elif event.name == '6' and event.event_type == 'down':
            a[1] += v
        elif event.name == '7' and event.event_type == 'down':
            a[2] += v
        elif event.name == '9' and event.event_type == 'down':
            a[2] -= v
        elif event.name == 'e' and event.event_type == 'down':
            exite_flag = True
        
    keyboard.hook(key_press)
    while not exite_flag:
        # curr_state = env.reset(option={'reset_buildings': False, 'pos':(600, -600, 100)})
        curr_state = env.reset(option={'random_aim': True})
        done = False
        while True:
            if np.sum(a) != 0:
                state, reward, done, info = env.step(a)
                a = np.zeros(3)
                total_reward += reward
                loop.set_postfix_str(
                    f"reward: {reward:.1f}, xyz: {state[:3]}, xyzV: {state[3:6]}, sincos: {state[9:11]}")
                curr_state = state.copy()
            env.render()
            if done:
                print("total reward: {}".format(total_reward))
                break

if __name__ == '__main__':
    manully_test_drone_env(100)