# -*- coding: utf-8 -*-
"""
Created on Thu May 16 06:58:40 2019

@author: Pichau
"""

import numpy as np


class KArmedBandit:

    def __init__(self, k_arms=10, alpha="1/n", epsilon=0, prior_estimation=0,
                 true_reward=0,  init_deterministic=False, increment_mu=0,
                 increment_sigma=0, use_gradient=False,
                 use_baseline_gradient=True,
                 use_UBC=False,  UBC_param=None):

        """
        :param k_arms: `int`
            number of arms
        :param alpha: `float` or "1/n"
            step size
        :param epsilon: `float` [0,1]
            epsilon-greedy algorithm parameter. (probability of exploration)
        :param prior_estimation: `float` or `list`
            prior estimation for each action
        :param true_reward: `float` or `list`
            True reward for each arm.
        :param init_deterministic: bool
            if True, arm values with be initialised deterministically
        :param increment_mu: `float`
            mean of normally distributed increment to value functions on each 
            time step for non-stationary problems
        :param increment_sigma: `float`
            standard deviation of normally distributed increment to value functions 
            on each time step for non-stationary problems
        :param use_gradient: `bool`
            Use gradient based bandit algorithm
        :param use_baseline_gradient: `bool`
            Use baseline in gradient based bandit algorithm
        :param use_UBC: `bool`
            Use Upper-Confidence-Bound Action Selection
        :param UBC_param: `float`
            controls the degree of exploration in UCB action selection.

        """

        self.k = k_arms
        self.arms = np.arange(self.k)
        self.time = 0
        self.average_reward = 0
        self.true_reward = true_reward
        self.alpha = alpha
        self.epsilon = epsilon
        self.prior_estimation = prior_estimation
        self.init_deterministic = init_deterministic
        self.increment_mu = increment_mu
        self.increment_sigma = increment_sigma
        self.use_gradient = use_gradient
        self.use_baseline_gradient = use_baseline_gradient
        self.use_UBC = use_UBC
        self.UBC_param = UBC_param
        if use_UBC:
            assert UBC_param is not None, 'If `use_UBC=True`, UBC_param need to be set.'


        self.q_true = None  # true value for each arm
        self.q_estimate = None  # the current estimate for the reward of each arm
        self.action_count = None # how many times each arm was used.
        self.best_action = None  # the best action amoung all the arms.
        self.gradient_action_prob = None #gradient action probability


    def initialize(self):
        """
        Initialize the process

        :return: None
        """
        
        if self.init_deterministic:
            if np.isscalar(self.true_reward):
                self.q_true = np.ones(self.k) * self.true_reward
            else:
                self.q_true = self.true_reward
        else:
            self.q_true = np.random.randn(self.k) + self.true_reward
        self.q_estimate = np.zeros(self.k) + self.prior_estimation
        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)


    def simulate(self, epochs=1000):
        """
        Simulate `epochs` times the K bandit arm.
        
        param: epochs
            number of epochs.
        
        return: np.array shape (epochs,2)
            array with the rewards received and if this action was the optimal action.
        """
        self.initialize()
        action_rewards_best_action = np.zeros((epochs,2))
        for t in range(epochs):
            self.time += 1 #added here to avoid problems if using UBC
            action = self.get_action()
            reward = self.step(action)
            action_rewards_best_action[t,0] = reward
            if action == self.best_action:
                action_rewards_best_action[t,1] = 1
            self.increment_q()
        return action_rewards_best_action
    
    def get_action(self):
        """
        Return an action
        
        :return: 
            action
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.arms)

        if self.use_gradient:
            _grad_exp = np.exp(self.q_estimate)
            self.gradient_action_prob = _grad_exp / np.sum(_grad_exp)
            return np.random.choice(self.arms, p=self.gradient_action_prob)
        if self.use_UBC:
            ubc_estimate = self.q_estimate + \
                             self.UBC_param * np.sqrt(
                np.log(self.time) / (self.action_count + np.finfo(float).eps))
            q_best = np.max(ubc_estimate)
            return np.random.choice(
                [action for action, q in enumerate(ubc_estimate) if
                 q == q_best])

        q_best = np.max(self.q_estimate)
        return np.random.choice([action
                                 for action, q in enumerate(self.q_estimate)
                                 if q == q_best])

    def get_reward(self, action, mu=0, sigma=1):

        """

        Get reward from an action

        :param action:
            arm to get the true reward
        :param mu:
            bias to add to the reward. (it might not make sense)
        :param sigma:
            variance of the noise to add to the reward.
        :return: reward
        """
        return sigma * np.random.randn() + mu + self.q_true[action]


    def step(self, action):
        """
            take an action, update estimation for this action

        :param action:
            arm to get the true reward
        :return:
            reward
        """
        # generate the reward
        reward = self.get_reward(action)
        # update time

        #compute the average reward
        self.average_reward += (reward - self.average_reward) / self.time
        #update the action count.
        self.action_count[action] += 1

        if self.use_gradient:
            aux_vec = np.zeros(self.k) #helper vector to avoid loop over all arms.
            aux_vec[action] = 1

            if self.use_baseline_gradient:
                baseline = self.average_reward
            else:
                baseline = 0

            self.q_estimate = self.q_estimate \
                              + self.alpha*(reward - baseline)*(aux_vec-self.gradient_action_prob)

        # update the estimate for the given action.
        elif self.alpha == "1/n":
            self.q_estimate[action] += 1.0 / self.action_count[action] * (reward - self.q_estimate[action])
        else:
            self.q_estimate[action] += self.alpha * (reward - self.q_estimate[action])

        return reward
    
    def increment_q(self):
        """
        Modify value functions by adding guassian noise
        """
        self.q_true += (self.increment_sigma * np.random.randn(self.k) + self.increment_mu)
        self.best_action = np.argmax(self.q_true)