from collections import defaultdict

import numpy as np

from agents.agent import Agent


class QLearningAgent(Agent):
    """
    An implementation of the Q Learning agent.

    """

    def __init__(self, number_of_action, discount_factor=1.0, alpha=0.5, epsilon=0.1):
        self.number_of_action = number_of_action
        self.q = defaultdict(lambda: np.zeros(number_of_action))
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.epsilon = epsilon
        self.policy = self._make_epsilon_greedy_policy()

    def _make_epsilon_greedy_policy(self):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon.

        """

        def policy_fn(state):
            A = np.ones(self.number_of_action, dtype=float) * self.epsilon / self.number_of_action
            best_action = np.argmax(self.q[state])
            A[best_action] += (1.0 - self.epsilon)
            return A

        return policy_fn

    def step(self, state):
        # state = str(state)
        action_probs = self.policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def update(self, state, action, reward, next_state):
        # TD Update
        best_next_action = np.argmax(self.q[next_state])
        td_target = reward + self.discount_factor * self.q[next_state][best_next_action]
        td_delta = td_target - self.q[state][action]
        self.q[state][action] += self.alpha * td_delta
