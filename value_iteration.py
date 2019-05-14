import numpy as np
import gym
import envs.gridworld
from collections import defaultdict

from agents.qlearning.qlearning_agent import QLearningAgent


def value(policy, states, transition_probabilities, reward, discount, threshold=1e-2):
    value_state = defaultdict(lambda: 0.0)
    for state in states:
        value_state[state] = 0.0

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for state in states:
            state = str(state)
            vs = value_state[state]
            action = policy[state]
            for new_state in states:
                new_state = str(new_state)
                value_state[state] += transition_probabilities[state][action][new_state] * (
                        reward[new_state] + discount * value_state[new_state])
            diff = max(diff, abs(vs - value_state[state]))

    return value_state


def q_to_policy(q, offset=0):
    optimal_policy = {}
    for state in q:
        optimal_policy[state] = np.argmax(q[state]) + offset
    return optimal_policy


if __name__ == '__main__':
    gamma = 0.9
    env = gym.make('FourRooms-v1')
    agent = QLearningAgent(env, gamma=gamma, alpha=0.12, epsilon=0.1)
    agent.train(1000)
    policy = q_to_policy(agent.q)
    rewards = defaultdict(lambda: 0.0)
    for s in env.possibleStates:
        rewards[str(s)] = env._get_reward(s)
    v = value(policy,
              env.possibleStates,
              env.transition_probability,
              rewards, gamma)
