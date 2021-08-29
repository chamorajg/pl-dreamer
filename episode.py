import torch
import numpy as np
from typing import Tuple, Any, Union, Optional, Dict
TensorType = Any

class Episode(object):
    """ Episode Class which contains the related
        attributes of an environment episode in the
        the format similar to queue"""
    
    def __init__(self,
                obs:TensorType,
                action_space:int = 1) -> None:
        """Initializes a list of all episode attributes"""
        self.action_space = action_space
        obs = torch.FloatTensor(np.ascontiguousarray(obs.transpose
                                                        ((2, 0, 1))))
        self.t = 1
        self.obs = [obs]
        self.action = [torch.FloatTensor(torch.zeros(1, self.action_space))]
        self.reward = [0]
        self.done = [False]
    
    def append(self, 
                episode_attrs: Tuple[TensorType]) -> None:
        """ Appends episode attribute to the list."""
        obs, action, reward, done = episode_attrs
        obs = torch.FloatTensor(np.ascontiguousarray(obs.transpose
                                                        ((2, 0, 1))))
        self.t += 1
        self.obs.append(obs)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)
    
    def reset(self, 
            obs:TensorType) -> None:
        """ Resets Episode list of attributes."""
        obs = torch.FloatTensor(np.ascontiguousarray(obs.transpose
                                                        ((2, 0, 1))))
        self.t = 1
        self.obs = [obs]
        self.action = [torch.FloatTensor(torch.zeros(1, self.action_space))]
        self.reward = [0]
        self.done = [False]
    
    def todict(self,) -> Dict:
        episode_dict = dict({'count': self.t,
                                'obs': torch.stack(self.obs),
                                'action': torch.cat(self.action),
                                'reward': torch.FloatTensor(self.reward),
                                'done': torch.BoolTensor(self.done)})
        return episode_dict