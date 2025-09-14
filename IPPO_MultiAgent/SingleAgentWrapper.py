import gym
from gym.spaces import Box, Discrete
import numpy as np

class SingleAgentEnvWrapper(gym.Env):
    def __init__(self, base_env, agent_id, other_agent_model=None):
        self.base_env = base_env
        self.agent_id = agent_id
        self.other_agent_model = other_agent_model
        self.num_agents = 2
        self.metadata = getattr(self.base_env, 'metadata', {'render.modes': ['human']})

        obs_shape = base_env.observation_space.shape
        if len(obs_shape) != 2 or obs_shape[0] != self.num_agents:
            raise ValueError(f"Expected observation shape ({self.num_agents}, obs_dim), got {obs_shape}")
        obs_dim = obs_shape[1]

        low = base_env.observation_space.low[self.agent_id]
        high = base_env.observation_space.high[self.agent_id]
        self.observation_space = Box(low=low, high=high, shape=(obs_dim,), dtype=base_env.observation_space.dtype)
        self.action_space = Discrete(19)

        self.last_obs = None
        self.last_other_action = 0  # Safe default

    def set_other_agent_model(self, model):
        """Update the other agent's model."""
        self.other_agent_model = model

    def reset(self):
        obs = self.base_env.reset()
        self.last_obs = obs
        self.last_other_action = 0
        return obs[self.agent_id]

    def step(self, action):
        if self.other_agent_model is not None:
            if self.last_obs is None:
                raise RuntimeError("Environment must be reset before stepping.")
            other_obs = self.last_obs[1 - self.agent_id]
            action_other, _ = self.other_agent_model.predict(other_obs[None], deterministic=True)
            self.last_other_action = int(action_other[0])

        actions = [None, None]
        actions[self.agent_id] = int(action)
        actions[1 - self.agent_id] = self.last_other_action

        obs, rewards, dones, infos = self.base_env.step(actions)
        self.last_obs = obs

        done = dones[self.agent_id] if isinstance(dones, (list, tuple, dict)) else dones

        if isinstance(infos, (list, tuple)):
            info = infos[self.agent_id]
        elif isinstance(infos, dict) and self.agent_id in infos:
            info = infos[self.agent_id]
        else:
            info = infos

        return obs[self.agent_id], rewards[self.agent_id], done, info

    def render(self, mode="human"):
        return self.base_env.render(mode=mode)

    def close(self):
        self.base_env.close()
