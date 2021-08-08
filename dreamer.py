import copy
import time
import random
import numpy as np
from tqdm import tqdm
from typing import Callable, Iterator, Tuple

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.optim import Adam
from torch.utils.data import DataLoader, IterableDataset
from torch.distributions import Categorical, Normal

from env import DMControlSuiteEnv
from planet import PLANet, FreezeParameters


class ExperienceSourceDataset(IterableDataset):
    """
    Implementation from PyTorch Lightning Bolts:
    https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/datamodules/experience_source.py
    Basic experience source dataset. Takes a generate_batch function that returns an iterator.
    The logic for the experience source and how the batch is generated is defined the Lightning model itself
    """

    def __init__(self, generate_batch: Callable):
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterator:
        iterator = self.generate_batch
        return iterator

class DreamerTrainer(pl.LightningModule):

    agent_name = "dreamer"
    def __init__(self, 
                    config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.env = DMControlSuiteEnv('hopper_hop')
        self.model = PLANet(self.config["dreamer"]["dreamer_model"]['obs_space'], 
                            np.array(self.config["dreamer"]["dreamer_model"]['action_space']), 
                            self.config["dreamer"]["dreamer_model"]['num_outputs'], 
                            self.config["dreamer"]["dreamer_model"],
                            self.config['name'])
        # self.model = self.agent
        self.episodes = []
        self.max_length = self.config["dreamer"]['max_length']
        self.length = self.config["dreamer"]['length']
        self.timesteps = 0
        self.explore = self.config["dreamer"]['explore_noise']
        self.fill_batches = []
        self.batch_size = self.config["dreamer"]["batch_size"]
        prefill_episodes = self._prefill_train_batch()
        self._add(prefill_episodes)     

    def compute_dreamer_loss(self,
                         obs,
                         action,
                         reward,
                         imagine_horizon,
                         discount=0.99,
                         lambda_=0.95,
                         kl_coeff=1.0,
                         free_nats=3.0,
                         log=False):
        """Constructs loss for the Dreamer objective
            Args:
                obs (TensorType): Observations (o_t)
                action (TensorType): Actions (a_(t-1))
                reward (TensorType): Rewards (r_(t-1))
                model (TorchModelV2): DreamerModel, encompassing all other models
                imagine_horizon (int): Imagine horizon for actor and critic loss
                discount (float): Discount
                lambda_ (float): Lambda, like in GAE
                kl_coeff (float): KL Coefficient for Divergence loss in model loss
                free_nats (float): Threshold for minimum divergence in model loss
                log (bool): If log, generate gifs
            """
        encoder_weights = list(self.model.encoder.parameters())
        decoder_weights = list(self.model.decoder.parameters())
        reward_weights = list(self.model.reward.parameters())
        dynamics_weights = list(self.model.dynamics.parameters())
        critic_weights = list(self.model.value.parameters())
        model_weights = list(encoder_weights + decoder_weights + reward_weights +
                            dynamics_weights)

        device = self.device
        # PlaNET Model Loss
        latent = self.model.encoder(obs)
        istate = self.model.dynamics.get_initial_state(obs.shape[0], self.device)
        post, prior = self.model.dynamics.observe(latent, action, istate)
        features = self.model.dynamics.get_feature(post)
        image_pred = self.model.decoder(features)
        reward_pred = self.model.reward(features)
        image_loss = -torch.mean(image_pred.log_prob(obs))
        reward_loss = -torch.mean(reward_pred.log_prob(reward))
        prior_dist = self.model.dynamics.get_dist(prior[0], prior[1])
        post_dist = self.model.dynamics.get_dist(post[0], post[1])
        div = torch.mean(
            torch.distributions.kl_divergence(post_dist, prior_dist).sum(dim=2))
        div = torch.clamp(div, min=free_nats)
        model_loss = kl_coeff * div + reward_loss + image_loss

        # Actor Loss
        # [imagine_horizon, batch_length*batch_size, feature_size]
        with torch.no_grad():
            actor_states = [v.detach() for v in post]
        with FreezeParameters(model_weights):
            imag_feat = self.model.imagine_ahead(actor_states, imagine_horizon)
        with FreezeParameters(model_weights + critic_weights):
            reward = self.model.reward(imag_feat).mean
            value = self.model.value(imag_feat).mean
        pcont = discount * torch.ones_like(reward)
        returns = self._lambda_return(reward[:-1], value[:-1], pcont[:-1], value[-1],
                                lambda_)
        discount_shape = pcont[:1].size()
        discount = torch.cumprod(
            torch.cat([torch.ones(*discount_shape).to(device), pcont[:-2]], dim=0),
            dim=0)
        actor_loss = -torch.mean(discount * returns)

        # Critic Loss
        with torch.no_grad():
            val_feat = imag_feat.detach()[:-1]
            target = returns.detach()
            val_discount = discount.detach()
        val_pred = self.model.value(val_feat)
        critic_loss = -torch.mean(val_discount * val_pred.log_prob(target))

        # Logging purposes
        prior_ent = torch.mean(prior_dist.entropy())
        post_ent = torch.mean(post_dist.entropy())

        log_gif = None
        if log:
            log_gif = self._log_summary(obs, action, latent, image_pred)

        return_dict = {
            "model_loss": model_loss,
            "reward_loss": reward_loss,
            "image_loss": image_loss,
            "divergence": div,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "prior_ent": prior_ent,
            "post_ent": post_ent,
        }

        if log_gif is not None:
            return_dict["log_gif"] = log_gif
        return return_dict

    def dreamer_loss(self, train_batch):
        log_gif = False
        if "log_gif" in train_batch:
            log_gif = True

        self.stats_dict = self.compute_dreamer_loss(
            train_batch["obs"],
            train_batch["actions"],
            train_batch["rewards"],
            self.config["dreamer"]["imagine_horizon"],
            self.config["dreamer"]["discount"],
            self.config["dreamer"]["lambda"],
            self.config["dreamer"]["kl_coeff"],
            self.config["dreamer"]["free_nats"],
            log_gif,
        )

        loss_dict = self.stats_dict

        return (loss_dict["model_loss"], loss_dict["actor_loss"],
                loss_dict["critic_loss"])
    
    def _prefill_train_batch(self, ):
        self.timesteps = 0
        dict_keys = ['count', 'obs', 'action', 'reward', 'done']
        obs = self.env.reset()
        episode_obs = [torch.FloatTensor(np.ascontiguousarray(obs.transpose((2, 0, 1))))]
        episode_action = [torch.FloatTensor([[0, 0, 0, 0]])]
        episode_reward = [0]
        episode_done = [False]
        episode_count = 1
        episode_state = [self.model.get_initial_state(self.device)]
        episode_dict = {}
        episodes = []
        
        def initialize(obs):
            episode_obs = [obs]
            episode_action = [0]
            episode_reward = [0]
            episode_state = [self.model.get_initial_state(self.device)]
            episode_done = [False]
            episode_count = 1
            episode_dict = {}
        
        while self.timesteps <= int(self.config["dreamer"]["prefill_timesteps"] / self.config["dreamer"]["env_config"]["action_repeat"]):    
            action, logp, state = self.action_sampler_fn(obs, None, False, self.timesteps)
            obs, reward, done, _ = self.env.step(action)
            episode_count += 1
            obs = obs.transpose((2, 0, 1))
            obs_tensor = torch.FloatTensor(np.ascontiguousarray(obs))
            episode_obs.append(obs_tensor)
            episode_action.append(action)
            episode_done.append(done)
            episode_state.append(state)
            episode_reward.append(reward)
            if done:
                episode_dict.update({'count': episode_count,
                                'obs': torch.stack(episode_obs),
                                'state': self._collate_state(episode_state),
                                'action': torch.cat(episode_action, axis=0),
                                'reward': torch.FloatTensor(episode_reward),
                                'done': torch.BoolTensor(episode_done)})
                obs = self.env.reset()
                initialize(obs)
                episodes.append(episode_dict)
            self.timesteps += 1
        return episodes
    
    def _collate_state(self, state):
        return_state = [[], [], [], []]
        for s in state:
            for i in range(4):
                return_state[i].extend(s[i])
        for i in range(4):
            return_state[i] = torch.stack(return_state[i])
        return return_state
        
    
    def _data_collect(self):
        self.data_obs = self.env.reset()
        self.data_state = self.model.get_initial_state(self.device)
        episode_obs = [torch.FloatTensor(np.ascontiguousarray(self.data_obs.transpose((2, 0, 1))))]
        episode_action = [torch.FloatTensor([[0, 0, 0, 0]])]
        episode_reward = [0]
        episode_done = [False]
        episode_count = 1
        episode_state = [self.data_state]
        episode_dict = {}
        episodes = []
        
        def initialize(obs):
            episode_obs = [obs]
            episode_action = [0]
            episode_reward = [0]
            episode_state = [self.model.get_initial_state(self.device)]
            episode_done = [False]
            episode_count = 1
            episode_dict = {}
        
        for i in tqdm(range(self.config["dreamer"]["max_episode_length"] / self.config["dreamer"]["env_config"]["action_repeat"])):
            action, logp, state = self.action_sampler_fn(obs, state, self.config["explore"], self.timesteps)
            obs, reward, done, _ = self.env.step(action)
            obs = obs.transpose((2, 0, 1))
            obs_tensor = torch.FloatTensor(np.ascontiguousarray(obs))
            episode_count += 1
            episode_obs.append(obs_tensor)
            episode_action.append(action)
            episode_done.append(torch.from_numpy(done))
            episode_reward.append(torch.from_numpy(reward))
            episode_state.append(state)
            self.timesteps += 1
            if done:
                episode_dict.update({'count': episode_count,
                                'state': self._collate_state(episode_state),
                                'obs': torch.stack(episode_obs),
                                'action': torch.stack(episode_action),
                                'reward': torch.FloatTensor(episode_reward),
                                'done': torch.BoolTensor(episode_done),
                                    })
                episodes.append(episode_dict)
                break
            
        return episodes

    def _add(self, batch):
        for b in batch:
            self.timesteps += b["count"]
        self.episodes.extend(batch)
        if len(self.episodes) > self.max_length:
            remove_episode_index = len(self.episodes) - self.max_length
            self.episodes = self.episodes[remove_episode_index:]
    
    def _sample(self, batch_size):
        episodes_buffer = [] #{"count": 0, "obs": [], "reward": [], "action": [], "done": []}
        while len(episodes_buffer) < batch_size:
            rand_index = random.randint(0, len(self.episodes) - 1)
            episode = self.episodes[rand_index]
            if episode["count"] < self.length:
                continue
            available = episode["count"] - self.length
            index = int(random.randint(0, available))
            episodes_buffer.append({"count": self.length,
                                    "obs": episode["obs"][index : index + self.length],
                                    "state": [episode["state"][i][index : index + self.length] for i in range(4)],
                                    "action": episode["action"][index: index + self.length],
                                    "reward": episode["reward"][index: index + self.length],
                                    "done": episode["done"][index: index + self.length],
                                    })
        # return episodes_buffer
        total_batch = {}
        for k in episodes_buffer[0].keys():
            if k == "count":
                total_batch[k] = torch.LongTensor([e[k] for e in episodes_buffer])
            elif k == "state":
                state_batch = []
                for i in range(4):
                    state_batch.append(torch.stack([e[k][i] for e in episodes_buffer], axis=0))
                total_batch[k] = state_batch
            else:
                total_batch[k] = torch.stack([e[k] for e in episodes_buffer], axis=0)
        return total_batch
    
    def _train_batch(self, batch_size):
        for _ in range(self.config["dreamer"]["collect_interval"]):
            total_batch = self._sample(batch_size)
            def return_batch(i):
                return total_batch["count"][i], total_batch["obs"][i], [total_batch["state"][b][i] for b in range(4)],\
                    total_batch["action"][i], total_batch["reward"][i], total_batch["done"][i]
            for i in range(batch_size):
                yield return_batch(i)
    
    def _data_batch(self, 
        batch_size) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor]:
        for _ in range(self.config["dreamer"]["collect_interval"]):
            total_batch = self._sample(batch_size)
            def return_batch(i):
                return total_batch["count"][i], total_batch["obs"][i], [total_batch["state"][b][i] for b in range(4)],\
                    total_batch["action"][i], total_batch["reward"][i], total_batch["done"][i]
            for i in range(batch_size):
                yield return_batch(i)
    
    def action_sampler_fn(self, obs, state, explore, timestep):
        """Action sampler function has two phases. During the prefill phase,
        actions are sampled uniformly [-1, 1]. During training phase, actions
        are evaluated through DreamerPolicy and an additive gaussian is added
        to incentivize exploration.
        """
        # Custom Exploration
        if timestep <= int(self.config["dreamer"]["prefill_timesteps"] / self.config["dreamer"]["env_config"]["action_repeat"]):
            logp = [0.0]
            # Random action in space [-1.0, 1.0]
            action = 2.0 * torch.rand(1, self.model.action_size) - 1.0
            state = self.model.get_initial_state(self.device)
        else:
            # Weird RLLib Handling, this happens when env rests
            # if len(state[0].size()) == 3:
            #     # Very hacky, but works on all envs
            #     state = self.model.get_initial_state(self.device)
            action, logp, state = self.model.policy(obs, state, explore)
            action = Normal(action, self.config["dreamer"]["explore_noise"]).sample()
            action = torch.clamp(action, min=-1.0, max=1.0)

        return action, logp, state
    
    def training_step(self, batch, batch_idx):
        count, obs, state, action, reward, _ = batch
        state = self.model.dynamics.get_initial_state(obs.shape[0], self.device)
        state = state + [action]
        action, _, state = self.action_sampler_fn(obs, state, self.explore, self.timesteps)
        loss = self.dreamer_loss({"obs":obs, "actions":action, "rewards":reward})
        return sum(list(loss))
    
    def training_epoch_end(self, outputs):
        with torch.no_grad():
            data_collection_episodes = self._data_collect()
            self._add(data_collection_episodes)

        if self.current_epoch % self.config["trainer_params"]["val_check_interval"] == 0:
            self.model.eval()
            test_data = self._data_collect()
        
    def validation_step(self, batch, batch_idx):
        return batch
    
    def _collate_fn(self, batch):
        return_batch = {}
        for k in batch[0].keys():
            if k == 'count':
                return_batch[k] = torch.LongTensor([data[k] for data in batch])
            return_batch[k] = torch.stack([data[k] for data in batch])
        return return_batch

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        dataset = ExperienceSourceDataset(self._train_batch(self.batch_size))
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size)
        return dataloader
    
    def val_dataloader(self) -> DataLoader:
        """Get val loader"""
        dataset = ExperienceSourceDataset(self._data_batch(1))
        dataloader = DataLoader(dataset=dataset, batch_size=1)
        return dataloader 
    
    def configure_optimizers(self,):
        encoder_weights = list(self.model.encoder.parameters())
        decoder_weights = list(self.model.decoder.parameters())
        reward_weights = list(self.model.reward.parameters())
        dynamics_weights = list(self.model.dynamics.parameters())
        actor_weights = list(self.model.actor.parameters())
        critic_weights = list(self.model.value.parameters())
        model_opt = Adam(
            [{'params': encoder_weights + decoder_weights + reward_weights + dynamics_weights,
            'lr':self.config["dreamer"]["td_model_lr"]},
            {'params':actor_weights, 'lr':self.config["dreamer"]["actor_lr"]},
            {'params':critic_weights, 'lr':self.config["dreamer"]["critic_lr"]}],
            lr=self.config["dreamer"]["default_lr"],
            weight_decay=self.config["dreamer"]["weight_decay"])
        return model_opt
    
    def _postprocess_gif(self, gif: np.ndarray):
        gif = np.clip(255 * gif, 0, 255).astype(np.uint8)
        B, T, C, H, W = gif.shape
        frames = gif.transpose((1, 2, 3, 0, 4)).reshape((1, T, C, H, B * W))
        return frames
    
    def _log_summary(self, obs, action, embed, image_pred):
        truth = obs[:6] + 0.5
        recon = image_pred.mean[:6]
        init, _ = self.model.dynamics.observe(embed[:6, :5], action[:6, :5])
        init = [itm[:, -1] for itm in init]
        prior = self.model.dynamics.imagine(action[:6, 5:], init)
        openl = self.model.decoder(self.model.dynamics.get_feature(prior)).mean

        mod = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (mod - truth + 1.0) / 2.0
        return torch.cat([truth, mod, error], 3)
    
    def _lambda_return(self, reward, value, pcont, bootstrap, lambda_):
        def agg_fn(x, y):
            return y[0] + y[1] * lambda_ * x

        next_values = torch.cat([value[1:], bootstrap[None]], dim=0)
        inputs = reward + pcont * next_values * (1 - lambda_)

        last = bootstrap
        returns = []
        for i in reversed(range(len(inputs))):
            last = agg_fn(last, [inputs[i], pcont[i]])
            returns.append(last)

        returns = list(reversed(returns))
        returns = torch.stack(returns, dim=0)
        return returns