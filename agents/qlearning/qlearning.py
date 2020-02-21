import itertools
import sys
from collections import defaultdict
from ast import literal_eval

import numpy as np

from agents.agent import Agent


class QLearningAgent(Agent):
    """
    An implementation of the Q Learning agent.

    """

    def __init__(self, env, gamma=1.0, alpha=0.5, epsilon=0.1):
        self.environment = env
        self.number_of_action = env.action_space.n
        self.q = defaultdict(lambda: np.zeros(self.number_of_action))
        self.gamma = gamma
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

    def act(self, state):
        action_probs = self.policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def update(self, state, action, reward, next_state):
        # TD Update
        best_next_action = np.argmax(self.q[next_state])
        td_target = reward + self.gamma * self.q[next_state][best_next_action]
        td_delta = td_target - self.q[state][action]
        self.q[state][action] += self.alpha * td_delta

    def train(self, num_episodes=500, verbose=False):
        stats = {
            'episode_lengths': np.zeros(num_episodes),
            'episode_rewards': np.zeros(num_episodes)}

        for i_episode in range(num_episodes):

            # Print out which episode we're on.
            if (i_episode + 1) % 100 == 0:
                print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
                sys.stdout.flush()

            state = self.environment.reset()

            for t in itertools.count():

                action = self.act(state)
                next_state, reward, done, _ = self.environment.step(action)

                # Update statistics
                stats['episode_rewards'][i_episode] += reward
                stats['episode_lengths'][i_episode] = t

                self.update(state, action, reward, next_state)

                if done:
                    break

                state = next_state
        return stats


class QLearningWithOptionsAgent(QLearningAgent):

    def __init__(self, env, options, gamma=1.0, alpha=0.5, epsilon=0.1, intra_options=False):
        super().__init__(env, gamma, alpha, epsilon)

        self.options = options
        self.number_of_action = env.action_space.n + len(options)
        self.q = defaultdict(lambda: np.zeros(self.number_of_action))

        self.intra_options = intra_options
        if intra_options:
            self.option_q_hat = defaultdict(lambda: np.zeros(self.number_of_action))

        self.policy = self._make_epsilon_greedy_policy()

    def train(self, num_episodes=500, verbose=False):
        stats = {
            'episode_lengths': np.zeros(num_episodes),
            'episode_rewards': np.zeros(num_episodes)}

        for i_episode in range(num_episodes):

            # Print out which episode we're on.
            if (i_episode + 1) % 100 == 0:
                print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
                sys.stdout.flush()

            state = self.environment.reset()

            for t in itertools.count():

                action = self.act(state)
                if action >= self.environment.action_space.n:
                    next_state, reward, done, _, total_steps = self._execute_option(state, action)
                    # Update statistics
                    stats['episode_rewards'][i_episode] += reward
                    stats['episode_lengths'][i_episode] = total_steps
                else:
                    next_state, reward, done, _ = self.environment.step(action)
                    # Update statistics
                    stats['episode_rewards'][i_episode] += reward
                    stats['episode_lengths'][i_episode] = t

                    if self.intra_options:

                        update_options = []
                        for i, o in enumerate(self.options):
                            o_a, _ = o.step(state)
                            if o_a == action:
                                update_options.append(i + self.environment.action_space.n)

                        # Intra option Q learning update
                        self.q[state][action] = self.q[state][action] + self.alpha * (
                                (reward * self.gamma * (self.option_q_hat[next_state][action])) -
                                self.q[state][action])

                        max_o = np.max(self.q[next_state])
                        if done:
                            beta_s = 1
                        else:
                            beta_s = 0
                        self.option_q_hat[next_state][action] = (1 - beta_s) * self.q[next_state][
                            action] + beta_s * max_o

                        for o in update_options:
                            self.q[state][o] += self.alpha * (reward + self.gamma * (
                                    self.option_q_hat[next_state][o] - self.q[state][o]))
                            max_o = np.max(self.q[next_state])
                            if self.options[o - self.environment.action_space.n].termination_condition(
                                    state=next_state):
                                beta_s = 1
                            else:
                                beta_s = 0
                            self.option_q_hat[next_state][o] = (1 - beta_s) * self.q[next_state][
                                o] + beta_s * max_o

                self.update(state, action, reward, next_state)

                if done:
                    break

                state = next_state
        return stats

    def _execute_option(self, state, action):
        done = False
        total_reward = 0
        total_steps = 0

        option = self.options[action - self.environment.action_space.n]

        terminated = False

        while not terminated:
            a, terminated = option.step(state)
            next_state, reward, done, _ = self.environment.step(a)  # taking action

            total_reward += reward * (self.gamma ** total_steps)
            total_steps += 1

            if done:
                break

            state = next_state

        return state, total_reward, done, None, total_steps
