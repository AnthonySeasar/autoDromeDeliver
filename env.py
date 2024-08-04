import numpy as np
import gym
from gym import spaces


class DroneDeliveryEnv(gym.Env):
    def __init__(self, obstacles, target_area, boundary):
        super(DroneDeliveryEnv, self).__init__()

        self.obstacles = obstacles
        self.target_area = target_area
        self.boundary = boundary

        self.action_space = spaces.Discrete(6)  # 6 actions: up, down, left, right, forward, backward
        self.observation_space = spaces.Box(low=np.array([0, 0, 0], dtype=np.float32),
                                            high=np.array(boundary, dtype=np.float32),
                                            dtype=np.float32)

        self.reset()

    def reset(self):
        self.drone_position = np.random.uniform(0, self.boundary, size=(3,))
        self.failed_attempts = 0  # 记录连续失败次数
        return self.drone_position

    def step(self, action):
        if action == 0 and self.drone_position[0] < self.boundary[0] - 1:
            self.drone_position[0] += 1  # up
        elif action == 1 and self.drone_position[0] > 0:
            self.drone_position[0] -= 1  # down
        elif action == 2 and self.drone_position[1] < self.boundary[1] - 1:
            self.drone_position[1] += 1  # left
        elif action == 3 and self.drone_position[1] > 0:
            self.drone_position[1] -= 1  # right
        elif action == 4 and self.drone_position[2] < self.boundary[2] - 1:
            self.drone_position[2] += 1  # forward
        elif action == 5 and self.drone_position[2] > 0:
            self.drone_position[2] -= 1  # backward

        distance = np.linalg.norm(self.drone_position - self.target_area)
        done = False

        reward = -1
        if distance < 0.5:  # 红色区域内
            reward = 100
            self.failed_attempts = 0  # 重置连续失败次数
            done = True
        elif distance < 2.0:  # 黄色区域内
            reward = 50
            self.failed_attempts = 0
            done = True
        else:
            reward = -1  # 包裹掉到黄区外
            self.failed_attempts += 1

        if self.failed_attempts >= 10:  # 连续10次失败
            done = True
            reward = -100  # 算法失效，负分大惩罚

        # 添加调试输出
        print(f"Action: {action}, Position: {self.drone_position}, Reward: {reward}, Done: {done}")

        return self.drone_position, reward, done, {}

    def render(self):
        pass

    def close(self):
        pass
