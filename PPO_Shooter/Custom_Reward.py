import gym
from gym import Wrapper

class CustomRewardWrapper(Wrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)
        self.previous_score_diff = 0

    def reset(self, **kwargs):
        self.previous_score_diff = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Compute reward like in football_env_core.py
        score = info.get('score', [0, 0])
        score_diff = score[0] - score[1]
        custom_reward = score_diff - self.previous_score_diff
        self.previous_score_diff = score_diff

        # Optional: add any debug info
        info['custom_reward'] = custom_reward

        return obs, custom_reward, done, info
