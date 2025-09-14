import numpy as np
from gym import Wrapper

class KeeperWaitWrapper(Wrapper):
    def __init__(self, env):
        super(KeeperWaitWrapper, self).__init__(env)
        self.ball_kicked = False
        self.prev_ball_x = None
        self.prev_ball_y = None

    def reset(self):
        self.ball_kicked = False
        self.prev_ball_x = None
        self.prev_ball_y = None
        obs = self.env.reset()
        return obs

    def step(self, actions):
        if not self.ball_kicked:
            actions[1] = 1  # freeze keeper

        obs, reward, done, info = self.env.step(actions)

        # Use only shooter reward
        if isinstance(reward, (list, np.ndarray)):
            reward = float(reward[0])
        else:
            reward = float(reward)

        shooter_obs = obs[0]
        ball_pos = shooter_obs[88:91]
        ball_x, ball_y = ball_pos[0], ball_pos[1]

        if not self.ball_kicked:
            if self.prev_ball_x is not None and self.prev_ball_y is not None:
                if abs(ball_x - self.prev_ball_x) > 0.01 or abs(ball_y - self.prev_ball_y) > 0.01:
                    self.ball_kicked = True
            self.prev_ball_x = ball_x
            self.prev_ball_y = ball_y

        return obs, reward, done, info
