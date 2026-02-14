
import numpy as np
from typing import Dict, Any, Tuple, Optional
try:
    from gymnasium import spaces
except ImportError:

    from gym import spaces


class CentralizedObservationWrapper:

    def __init__(self, env):
        self.env = env
        self.observation_space = None
        self.action_space = None
        self.agents = None


        self._init_observation_space()

    def _init_observation_space(self):

        if hasattr(self.env, 'agents'):
            self.agents = list(self.env.agents)
        elif hasattr(self.env, 'get_agents'):
            self.agents = list(self.env.get_agents())
        else:
            raise ValueError("无法获取agent列表")


        if len(self.agents) > 0:
            first_agent = self.agents[0]
            local_obs_space = self.env.observation_space(first_agent)


            if isinstance(local_obs_space, spaces.Box):
                local_dim = local_obs_space.shape[0] if len(local_obs_space.shape) > 0 else local_obs_space.shape

                global_dim = local_dim * len(self.agents)


                self.observation_space = spaces.Box(
                    low=local_obs_space.low[0] if len(local_obs_space.shape) > 0 else local_obs_space.low,
                    high=local_obs_space.high[0] if len(local_obs_space.shape) > 0 else local_obs_space.high,
                    shape=(global_dim,),
                    dtype=local_obs_space.dtype
                )
            else:

                self.observation_space = local_obs_space

    def observation_space(self, agent: str):
        if self.observation_space is None:
            self._init_observation_space()
        return self.observation_space

    def action_space(self, agent: str):

        if hasattr(self.env, 'action_space'):
            if callable(self.env.action_space):
                return self.env.action_space(agent)
            else:

                return getattr(self.env, 'action_space', None)
        return None

    def get_agents(self):
        return self.agents

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:

        obs, info = self.env.reset(seed=seed, options=options)


        global_obs = self._merge_observations(obs)


        global_obs_dict = {agent: global_obs for agent in self.agents}

        return global_obs_dict, info

    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]]:

        obs, rewards, terminations, truncations, infos = self.env.step(actions)


        global_obs = self._merge_observations(obs)


        global_obs_dict = {agent: global_obs for agent in self.agents}

        return global_obs_dict, rewards, terminations, truncations, infos

    def _merge_observations(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:

        obs_list = []
        for agent in self.agents:
            if agent in obs_dict:
                agent_obs = np.array(obs_dict[agent], dtype=np.float32)
                obs_list.append(agent_obs)
            else:

                if len(obs_list) > 0:
                    zero_obs = np.zeros_like(obs_list[0])
                else:


                    zero_obs = np.zeros(200, dtype=np.float32)
                obs_list.append(zero_obs)


        global_obs = np.concatenate(obs_list)

        return global_obs

    def __getattr__(self, name):
        return getattr(self.env, name)
