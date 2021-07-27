import copy
import time
import random
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import multiprocessing
from torch.optim import Adam
from torch.multiprocessing import Queue
from torch.distributions import Categorical, normal

from planet import PLANet


def move_gradients_one_model_to_another(from_model, to_model, set_from_gradients_to_zero=False):
        """Copies gradients from from_model to to_model"""
        for from_model, to_model in zip(from_model.parameters(), to_model.parameters()):
            to_model._grad = from_model.grad.clone()
            if set_from_gradients_to_zero: from_model._grad = None


def copy_model_over(from_model, to_model):
    """Copies model parameters from from_model to to_model"""
    for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
        to_model.data.copy_(from_model.data.clone())

def create_actor_distribution(action_types, actor_output, action_size):
    """Creates a distribution that the actor can then use to randomly draw actions"""
    if action_types == "DISCRETE":
        assert actor_output.size()[1] == action_size, "Actor output the wrong size"
        action_distribution = Categorical(actor_output)  # this creates a distribution to sample from
    else:
        assert actor_output.size()[1] == action_size * 2, "Actor output the wrong size"
        means = actor_output[:, :action_size].squeeze(0)
        stds = actor_output[:,  action_size:].squeeze(0)
        if len(means.shape) == 2: means = means.squeeze(-1)
        if len(stds.shape) == 2: stds = stds.squeeze(-1)
        if len(stds.shape) > 1 or len(means.shape) > 1:
            raise ValueError("Wrong mean and std shapes - {} -- {}".format(stds.shape, means.shape))
        action_distribution = normal.Normal(means.squeeze(0), torch.abs(stds))
    return action_distribution

class Dreamer(object):

    agent_name = "dreamer"
    def __init__(self, config):
        super(Dreamer, self).__init__(config)
        self.num_processes = multiprocessing.cpu_count()
        self.worker_processes = min( max(1, self.num_processes - 2),
                                        config['num_workers'])
        self.dreamer = PLANet(input_dim, output_dim)
        self.dreamer_optimizer = Adam(self.dreamer.parameters(), lr=self.config.learning_rate, eps=1e-4)
    
    def run_n_episodes(self):
        """ Runs game to completion n times."""
        start = time.time()
        results_queue = Queue()
        gradient_updates_queue = Queue()
        episode_numer = multiprocessing.Value('i', 0)
        self.optimizer_lock = multiprocessing.Lock()
        episodes_per_process = int(self.config.num_episodes_to_run / self.worker_processes) + 1
        processes = []
        self.dreamer.share_memory()
        self.dreamer_optimizer.share_memory()

        optimizer_worker = multiprocessing.Process(target=self.update_shared_model, args=(gradient_updates_queue,))
        optimizer_worker.start()

        for process_num in range(self.worker_processes):
            worker = DreamerWorker(process_num, copy.deepcopy(self.environment), self.actor_critic, episode_number, self.optimizer_lock,
                                    self.actor_critic_optimizer, self.config, episodes_per_process,
                                    self.hyperparameters["epsilon_decay_rate_denominator"],
                                    self.action_size, self.action_types,
                                    results_queue, copy.deepcopy(self.actor_critic), gradient_updates_queue)
            worker.start()
            processes.append(worker)
        self.print_results(episode_number, results_queue)
        for worker in processes:
            worker.join()
        optimizer_worker.kill()

        time_taken = time.time() - start
        return self.game_full_episode_scores, self.rolling_results, time_taken

    def print_results(self, episode_number, results_queue):
        """Worker that prints out results as they get put into a queue"""
        while True:
            with episode_number.get_lock():
                carry_on = episode_number.value < self.config.num_episodes_to_run
            if carry_on:
                if not results_queue.empty():
                    self.total_episode_score_so_far = results_queue.get()
                    self.save_and_print_result()
            else: break

    def update_shared_model(self, gradient_updates_queue):
        """Worker that updates the shared model with gradients as they get put into the queue"""
        while True:
            gradients = gradient_updates_queue.get()
            with self.optimizer_lock:
                self.actor_critic_optimizer.zero_grad()
                for grads, params in zip(gradients, self.actor_critic.parameters()):
                    params._grad = grads  # maybe need to do grads.clone()
                self.actor_critic_optimizer.step()


class DreamerWorker(torch.multiprocessing.Process):
    """Dreamer worker that will play the game for the designated number of episodes """
    def __init__(self, worker_num, environment, shared_model, counter, optimizer_lock, shared_optimizer,
                 config, episodes_to_run, epsilon_decay_denominator, action_size, action_types, results_queue,
                 local_model, gradient_updates_queue):
        super(DreamerWorker, self).__init__()
        self.environment = environment
        self.config = config
        self.worker_num = worker_num

        self.gradient_clipping_norm = self.config.hyperparameters["gradient_clipping_norm"]
        self.discount_rate = self.config.hyperparameters["discount_rate"]
        self.normalise_rewards = self.config.hyperparameters["normalise_rewards"]

        self.action_size = action_size
        self.set_seeds(self.worker_num)
        self.shared_model = shared_model
        self.local_model = local_model
        self.local_optimizer = Adam(self.local_model.parameters(), lr=0.0, eps=1e-4)
        self.counter = counter
        self.optimizer_lock = optimizer_lock
        self.shared_optimizer = shared_optimizer
        self.episodes_to_run = episodes_to_run
        self.epsilon_decay_denominator = epsilon_decay_denominator
        self.exploration_worker_difference = self.config.exploration_worker_difference
        self.action_types = action_types
        self.results_queue = results_queue
        self.episode_number = 0

        self.gradient_updates_queue = gradient_updates_queue

    def set_seeds(self, worker_num):
        """Sets random seeds for this worker"""
        torch.manual_seed(self.config.seed + worker_num)
        self.environment.seed(self.config.seed + worker_num)

    def run(self):
        """Starts the worker"""
        torch.set_num_threads(1)
        for ep_ix in range(self.episodes_to_run):
            with self.optimizer_lock:
                copy_model_over(self.shared_model, self.local_model)
            epsilon_exploration = self.calculate_new_exploration()
            state = self.reset_game_for_worker()
            done = False
            self.episode_states = []
            self.episode_actions = []
            self.episode_rewards = []
            self.episode_log_action_probabilities = []
            self.critic_outputs = []

            while not done:
                action, action_log_prob, critic_outputs = self.pick_action_and_get_critic_values(self.local_model, state, epsilon_exploration)
                next_state, reward, done, _ =  self.environment.step(action)
                self.episode_states.append(state)
                self.episode_actions.append(action)
                self.episode_rewards.append(reward)
                self.episode_log_action_probabilities.append(action_log_prob)
                self.critic_outputs.append(critic_outputs)
                state = next_state

            total_loss = self.calculate_total_loss()
            self.put_gradients_in_queue(total_loss)
            self.episode_number += 1
            with self.counter.get_lock():
                self.counter.value += 1
                self.results_queue.put(np.sum(self.episode_rewards))

    def calculate_new_exploration(self):
        """Calculates the new exploration parameter epsilon. It picks a random point within 3X above and below the
        current epsilon"""
        with self.counter.get_lock():
            epsilon = 1.0 / (1.0 + (self.counter.value / self.epsilon_decay_denominator))
        epsilon = max(0.0, random.uniform(epsilon / self.exploration_worker_difference, epsilon * self.exploration_worker_difference))
        return epsilon

    def reset_game_for_worker(self):
        """Resets the game environment so it is ready to play a new episode"""
        state = self.environment.reset()
        if self.action_types == "CONTINUOUS": self.noise.reset()
        return state

    def pick_action_and_get_critic_values(self, policy, state, epsilon_exploration=None):
        """Picks an action using the policy"""
        state = torch.from_numpy(state).float().unsqueeze(0)
        model_output = policy.forward(state)
        actor_output = model_output[:, list(range(self.action_size))] #we only use first set of columns to decide action, last column is state-value
        critic_output = model_output[:, -1]
        action_distribution = create_actor_distribution(self.action_types, actor_output, self.action_size)
        action = action_distribution.sample().cpu().numpy()
        if self.action_types == "CONTINUOUS": action += self.noise.sample()
        if self.action_types == "DISCRETE":
            if random.random() <= epsilon_exploration:
                action = random.randint(0, self.action_size - 1)
            else:
                action = action[0]
        action_log_prob = self.calculate_log_action_probability(action, action_distribution)
        return action, action_log_prob, critic_output

    def calculate_log_action_probability(self, actions, action_distribution):
        """Calculates the log probability of the chosen action"""
        policy_distribution_log_prob = action_distribution.log_prob(torch.Tensor([actions]))
        return policy_distribution_log_prob

    def calculate_total_loss(self, 
                                obs, 
                                action, 
                                reward, 
                                model,
                                imagine_horizon,
                                discount=0.99,
                                lambda=0.95,
                                kl_coeff=1.0,
                                free_nats=3.0,
                                log=False):
        """
        Constructs loss for the Dreamer model.
        Args:
            obs (TensorType): Observations (o_t)
            action (TensorType): Actions (a_(t-1))
            reward (TensorType): Rewards (r_(t-1))
            model (nn.Module): DreamerModel having all the other model components encompassing all other models
            imagine_horizon (int): Imagine horizon for actor and critic loss
            discount (float): Discount
            lambda_ (float): Lambda, like in GAE
            kl_coeff (float): KL Coefficient for Divergence loss in model loss
            free_nats (float): Threshold for minimum divergence in model loss
            log (bool): If log, generate gifs
        """
        discounted_returns = self.calculate_discounted_returns()
        if self.normalise_rewards:
            discounted_returns = self.normalise_discounted_returns(discounted_returns)
        critic_loss, advantages = self.calculate_critic_loss_and_advantages(discounted_returns)
        actor_loss = self.calculate_actor_loss(advantages)
        total_loss = actor_loss + critic_loss
        return total_loss

    def calculate_discounted_returns(self):
        """Calculates the cumulative discounted return for an episode which we will then use in a learning iteration"""
        discounted_returns = [0]
        for ix in range(len(self.episode_states)):
            return_value = self.episode_rewards[-(ix + 1)] + self.discount_rate*discounted_returns[-1]
            discounted_returns.append(return_value)
        discounted_returns = discounted_returns[1:]
        discounted_returns = discounted_returns[::-1]
        return discounted_returns

    def normalise_discounted_returns(self, discounted_returns):
        """Normalises the discounted returns by dividing by mean and std of returns that episode"""
        mean = np.mean(discounted_returns)
        std = np.std(discounted_returns)
        discounted_returns -= mean
        discounted_returns /= (std + 1e-5)
        return discounted_returns

    def calculate_critic_loss_and_advantages(self, all_discounted_returns):
        """Calculates the critic's loss and the advantages"""
        critic_values = torch.cat(self.critic_outputs)
        advantages = torch.Tensor(all_discounted_returns) - critic_values
        advantages = advantages.detach()
        critic_loss =  (torch.Tensor(all_discounted_returns) - critic_values)**2
        critic_loss = critic_loss.mean()
        return critic_loss, advantages

    def calculate_actor_loss(self, advantages):
        """Calculates the loss for the actor"""
        action_log_probabilities_for_all_episodes = torch.cat(self.episode_log_action_probabilities)
        actor_loss = -1.0 * action_log_probabilities_for_all_episodes * advantages
        actor_loss = actor_loss.mean()
        return actor_loss

    def put_gradients_in_queue(self, total_loss):
        """Puts gradients in a queue for the optimisation process to use to update the shared model"""
        self.local_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), self.gradient_clipping_norm)
        gradients = [param.grad.clone() for param in self.local_model.parameters()]
        self.gradient_updates_queue.put(gradients)