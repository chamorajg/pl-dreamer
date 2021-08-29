import gym
import random
import numpy as np
from typing import Any, List, Tuple, Optional

class DMControlSuiteEnv:

    def __init__(self, 
                name: str, 
                max_episode_length: int = 1000,
                action_repeat:int = 2,
                size: Tuple[int] = (64, 64),
                camera: Optional[Any] = None,
                ):
        domain, task = name.split('_', 1)
        if domain == 'cup':
            domain = 'ball_in_cup'
        if isinstance(domain, str):
            from dm_control import suite
            self._env = suite.load(domain, task)
        else:
            assert task is None
            self._env = domain()
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera
        self._step = 0
        self._max_episode_length = 1000
        self._action_repeat = action_repeat
    
    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            spaces[key] = gym.spaces.Box(
                -np.inf, np.inf, value.shape, dtype=np.float32)
            spaces['image'] = gym.spaces.Box(
                0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        reward = 0
        obs = None
        for k in range(self._action_repeat):
            time_step = self._env.step(action)
            self._step += 1
            obs = dict(time_step.observation)
            obs['image'] = self.render()
            reward += time_step.reward or 0
            done = time_step.last() or self._step == self._max_episode_length
            if done:
                break
        info = {'discount': np.array(time_step.discount, np.float32)}
        return obs["image"], reward, done, info

    def reset(self):
        time_step = self._env.reset()
        self._step = 0
        obs = dict(time_step.observation)
        obs['image'] = self.render()
        return obs["image"]

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)

if __name__ == '__main__':
    import cv2
    env = DMControlSuiteEnv("acrobot_swingup")
    obs = env.reset()
    action_space = env.action_space
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
    # cv2.imshow("Obs", cv2.resize(obs, (640, 640)))
    # cv2.waitKey(0)
    env.step(0.2)