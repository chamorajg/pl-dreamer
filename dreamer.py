import os
import copy
import time
import random
import numpy as np
from PIL import Image
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
from episode import Episode
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
        self.env = DMControlSuiteEnv(name=self.config["env"], 
                                max_episode_length=self.config["dreamer"]["max_episode_length"],
                                action_repeat=self.config["dreamer"]["env_config"]["action_repeat"])
        self.model = PLANet(self.config["dreamer"]["dreamer_model"]['obs_space'], 
                            np.array(self.config["dreamer"]["dreamer_model"]['action_space']), 
                            self.config["dreamer"]["dreamer_model"]['num_outputs'], 
                            self.config["dreamer"]["dreamer_model"],
                            self.config['name'])
        self.episodes = []
        self.length = self.config["dreamer"]['length']
        self.timesteps = 0
        self._max_experience_size = self.config["dreamer"]['max_experience_size']
        self._action_repeat = self.config["dreamer"]["env_config"]["action_repeat"]
        self._prefill_timesteps = self.config["dreamer"]["prefill_timesteps"]
        self._max_episode_length = self.config["dreamer"]["max_episode_length"]

        self.explore = self.config["dreamer"]['explore_noise']
        self.batch_size = self.config["dreamer"]["batch_size"]
        self.action_space = np.array(self.config["dreamer"]["dreamer_model"]["action_space"]).shape[0]
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
            return_dict["log_gif"] = self._postprocess_gif(log_gif)
        return return_dict

    def dreamer_loss(self, train_batch):
        """ calculates dreamer loss."""

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
        return loss_dict
    
    def _prefill_train_batch(self, ):
        """ Prefill episodes before the training begins."""
        
        self.timesteps = 2
        obs = self.env.reset()
        episode = Episode(obs, self.action_space)
        episodes = []
        
        while self.timesteps < self._prefill_timesteps: 
            action, logp, state = self.prefill_action_sampler_fn(None, 
                                                            self.timesteps)
            obs, reward, done, _ = self.env.step(action.numpy())
            episode.append((obs, action, reward, done))
            self.timesteps += self._action_repeat       
            if done or self.timesteps == self._prefill_timesteps - 1:
                episodes.append(episode.todict())
                obs = self.env.reset() 
                if done:
                    episode.reset(obs)
        del episode
        return episodes        
    
    def _data_collect(self):
        """ Collect data from the policy after every epoch. """
        
        obs = self.env.reset()
        state = self.model.get_initial_state(self.device)
        episode = Episode(obs, self.action_space)
        episodes = []
        
        max_len = self._max_episode_length // self._action_repeat
        for i in range(max_len):
            action, logp, state = self.action_sampler_fn(
                    ((episode.obs[-1] / 255.0) - 0.5).unsqueeze(0).to(
                    self.device), state, self.explore, False)
            obs, reward, done, _ = self.env.step(action.detach().cpu().numpy())
            episode.append((obs, action.detach().cpu(), reward, done))
            if done or i == max_len - 1:
                episodes.append(episode.todict())
                break
        del episode
        return episodes
    
    def _test(self):
        """ Test the model after every few intervals."""
        
        obs = self.env.reset()
        state = self.model.get_initial_state(self.device)
        obs = torch.FloatTensor(np.ascontiguousarray(obs.transpose((2, 0, 1))))
        
        tot_reward = 0
        done = False
        while not done:
            action, logp, state = self.action_sampler_fn(
                        ((obs / 255.0) - 0.5).unsqueeze(0).to(self.device), state, self.explore, True)
            obs, reward, done, _ = self.env.step(action.detach().cpu().numpy())
            obs = obs.transpose((2, 0, 1))
            obs = torch.FloatTensor(np.ascontiguousarray(obs))
            tot_reward += reward
        return tot_reward

    def _add(self, batch):
        """ Adds the collected episode samples as well as the prefilled
            episode samples into the episode memory."""
        
        self.episodes.extend(batch)
        
        if len(self.episodes) > self._max_experience_size:
            remove_episode_index = len(self.episodes) -\
                                        self._max_experience_size
            self.episodes = self.episodes[remove_episode_index:]
        
        if self.config["dreamer"]["save_episodes"] and\
            self.trainer is not None and self.trainer.log_dir is not None:
            save_episodes = np.array(self.episodes)
            if not os.path.exists(f'{self.trainer.log_dir}/episodes'):
                os.makedirs(f'{self.trainer.log_dir}/episodes', exist_ok=True)
            np.savez(f'{self.trainer.log_dir}/episodes/episodes.npz', save_episodes)

    def _sample(self, batch_size):
        """ Samples a batch of episode of length T from the config."""
        
        episodes_buffer = []
        while len(episodes_buffer) < batch_size:
            rand_index = random.randint(0, len(self.episodes) - 1)
            episode = self.episodes[rand_index]
            if episode["count"] < self.length:
                continue
            available = episode["count"] - self.length
            index = int(random.randint(0, available))
            episodes_buffer.append({"count": self.length,
                                    "obs": episode["obs"][index : index + self.length],
                                    "action": episode["action"][index: index + self.length],
                                    "reward": episode["reward"][index: index + self.length],
                                    "done": episode["done"][index: index + self.length],
                                    })
        total_batch = {}
        for k in episodes_buffer[0].keys():
            if k == "count" or k == "state":
                continue
            else:
                total_batch[k] = torch.stack([e[k] for e in episodes_buffer], axis=0)
        return total_batch
    
    def _train_batch(self, batch_size):
        for _ in range(self.config["dreamer"]["collect_interval"]):
            total_batch = self._sample(batch_size)
            def return_batch(i):
                return (total_batch["obs"][i] / 255.0 - 0.5),\
                    total_batch["action"][i], total_batch["reward"][i], total_batch["done"][i]
            for i in range(batch_size):
                yield return_batch(i)
    
    def prefill_action_sampler_fn(self, state, timestep):
        """Action sampler function during prefill phase where
        actions are sampled uniformly [-1, 1].
        """
        # Custom Exploration
        logp = [0.0]
        # Random action in space [-1.0, 1.0]
        action = torch.FloatTensor(1, self.model.action_size).uniform_(-1.0, 
                                                1.0)
        state = self.model.get_initial_state(self.device)
        return action, logp, state
    
    def action_sampler_fn(self, obs, state, explore, test=False):
        """Action sampler during training phase, actions
        are evaluated through DreamerPolicy and 
        an additive gaussian is added
        to incentivize exploration."""
        
        action, logp, state_new = self.model.policy(obs, state, 
                                    explore=not(test))
        if not test:
            action = Normal(action, explore).sample()
        action = torch.clamp(action, min=-1.0, max=1.0)
        return action, logp, state_new
    
    def training_step(self, batch, batch_idx):
        """ Trains the model on the samples collected."""
        
        obs, action, reward, __ = batch
        loss = self.dreamer_loss({"obs":obs, 
                        "actions":action, "rewards":reward, 
                        "log_gif": True})
        outputs = []
        for k, v in loss.items():
            if "loss" in k:
                self.log(k, v)
            if k in ["model_loss", "critic_loss", "actor_loss"]:
                outputs.append(v)
        return sum(outputs)
    
    def training_epoch_end(self, outputs):
        """ Collects data samples after every epoch end and tests the
            model on the environment of maximum length from the config every
            few intervals."""
        
        total_loss = 0
        for out in outputs:
            total_loss += out['loss'].item()
        if len(outputs) != 0:
            total_loss /= len(outputs)     
        self.log('loss', total_loss)

        with torch.no_grad():
            data_collection_episodes = self._data_collect()
            self._add(data_collection_episodes)
            data_dict = data_collection_episodes[0]
            self.log('avg_reward_collection', torch.mean(data_dict['reward']))

        if self.current_epoch > 0 and \
                self.current_epoch % self.config["trainer_params"]["val_check_interval"] == 0:
            self.model.eval()
            episode_reward = self._test()
            self.log('avg_reward_test', episode_reward)
            self.model.train()
    
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
        dataloader = DataLoader(dataset=dataset, 
                                    batch_size=self.batch_size,        
                                    pin_memory=True, 
                                    num_workers=1)
        return dataloader
    
    def configure_optimizers(self,):
        """ Configure optmizers."""

        encoder_weights = list(self.model.encoder.parameters())
        decoder_weights = list(self.model.decoder.parameters())
        reward_weights = list(self.model.reward.parameters())
        dynamics_weights = list(self.model.dynamics.parameters())
        actor_weights = list(self.model.actor.parameters())
        critic_weights = list(self.model.value.parameters())
        model_opt = Adam(
            [
            {'params': encoder_weights + decoder_weights + reward_weights + dynamics_weights,
            'lr':self.config["dreamer"]["td_model_lr"]},
            {'params':actor_weights, 'lr':self.config["dreamer"]["actor_lr"]},
            {'params':critic_weights, 'lr':self.config["dreamer"]["critic_lr"]}],
            lr=self.config["dreamer"]["default_lr"],
            weight_decay=self.config["dreamer"]["weight_decay"])
        return model_opt
    
    def _postprocess_gif(self, gif: np.ndarray):
        gif = gif.detach().cpu().numpy()
        gif = np.clip(255*gif, 0, 255).astype(np.uint8)
        B, T, C, H, W = gif.shape
        frames = gif.transpose((1, 2, 3, 0, 4)).reshape((1, T, C, H, B * W))
        frames = frames.squeeze(0)
        
        def display_image(frame):
            frame = frame.transpose((1, 2, 0))
            return Image.fromarray(frame)
        
        img, *imgs = [display_image(frame) for frame in list(frames)]
        img.save(f'{self.trainer.log_dir}/movies/movie_{self.current_epoch}.gif', format='GIF', append_images=imgs,
         save_all=True, loop=0)
        return frames
    
    def _log_summary(self, obs, action, embed, image_pred):
        truth = obs[:6] + 0.5
        recon = image_pred.mean[:6]
        istate = self.model.dynamics.get_initial_state(6, self.device)
        init, _ = self.model.dynamics.observe(embed[:6, :5], 
                                            action[:6, :5], istate)
        init = [itm[:6, -1] for itm in init]
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