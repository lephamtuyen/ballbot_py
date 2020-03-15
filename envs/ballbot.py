import gym
from gym.spaces import Box
import numpy as np
from gym.utils import seeding

class Ballbot2D(gym.Env):

    def __init__(self):
        self.controlUpdate_dt_ = 0.05
        self.timeLimit_ = 10.0
        self.maxTorque_ = 5.0
        self.maxBallSpeed_ = 8.0
        self.maxAngleSpeed_ = 8.0
        self.high = np.array([2.0, 10000, 12 * np.pi, 10000])
        self.action_space = Box(low=-self.maxTorque_, high=self.maxTorque_, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=-self.high, high=self.high, dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        u = np.clip(u, -self.maxTorque_, self.maxTorque_)[0]

        self.x_[2] = -6.7 * self.theta_[0] - 2.1 * u
        self.theta_[2] = 3.79 * u + 25.52 * self.theta_[0]

        self.x_[0] = self.x_[0] + self.x_[1] * self.controlUpdate_dt_
        self.x_[1] = self.x_[1] + self.x_[2] * self.controlUpdate_dt_
        self.x_[1] = np.clip(self.x_[1], -self.maxBallSpeed_, self.maxBallSpeed_)

        self.theta_[0] = self.theta_[0] + self.theta_[1] * self.controlUpdate_dt_
        self.theta_[1] = self.theta_[1] + self.theta_[2] * self.controlUpdate_dt_
        self.theta_[1] = np.clip(self.theta_[1], -self.maxAngleSpeed_, self.maxAngleSpeed_)

        cost1 = self.normAngle(self.theta_[0]) * self.normAngle(self.theta_[0])
        cost2 = .0005 * (u * u)
        costs = cost1 + cost2

        term = self.isViolatingBoxConstraint()

        return self._get_obs(), -costs, term, {}

    def reset(self):
        self.x_ = np.array([0.0, self.np_random.uniform(), self.np_random.uniform()])
        self.theta_ = np.array([self.np_random.uniform() * 12 * np.pi / 180, self.np_random.uniform(), self.np_random.uniform()])

        return self._get_obs()

    def _get_obs(self):
        return np.array([self.x_[0], self.x_[1], self.theta_[0], self.theta_[1]])

    def render(self, mode='human'):
        return None

    def isViolatingBoxConstraint(self):
        state = self._get_obs()
        offsetUp = self.high - state
        offsetLower = state + self.high

        if np.any(offsetUp<0) or np.any(offsetLower<0):
            return True

        return False

    def normAngle(self, x):
        return (x + np.pi) % (2 * np.pi) - np.pi