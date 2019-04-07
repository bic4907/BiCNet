'''
Implemented by ghliu
https://github.com/ghliu/pytorch-ddpg/blob/master/normalized_env.py
'''

import gym
import numpy as np

# https://github.com/openai/gym/blob/master/gym/core.py
class ActionNormalizedEnv(gym.ActionWrapper):
    """ Wrap action """
    def __init__(self, env):
        super(ActionNormalizedEnv, self).__init__(env=env)
        self.action_high = 1.
        self.action_low = -1.

    def action(self, action):
        act_k = (self.action_high - self.action_low)/ 2.
        act_b = (self.action_high + self.action_low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_high - self.action_low)
        act_b = (self.action_high + self.action_low)/ 2.
        return act_k_inv * (action - act_b)

class ObsNormalizedEnv(gym.ObservationWrapper):
    """ Wrap action """
    def __init__(self, env):
        super(ObsNormalizedEnv, self).__init__(env=env)
        self.action_high = 1.
        self.action_low = -1.

    def observation(self, observation):
        obs = np.array([[observation[0][2] - observation[0][0], observation[0][3] - observation[0][1]]])
        return obs