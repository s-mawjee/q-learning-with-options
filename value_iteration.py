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


def optimal_value(states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-2):
    value_state = defaultdict(lambda: 0.0)
    for state in states:
        value_state[state] = 0.0

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in states:
            s = str(s)
            max_v = float("-inf")
            for a in range(n_actions):
                tp = transition_probabilities[s][a]
                comp_v = float("inf")
                for
                    max_v = max(max_v, np.dot(tp, reward + discount * v))

            new_diff = abs(v[s] - max_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = max_v

    return v


def find_policy(n_states, n_actions, transition_probabilities, reward, discount,
                threshold=1e-2, v=None, stochastic=True):
    """
    Find the optimal policy.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    v: Value function (if known). Default None.
    stochastic: Whether the policy should be stochastic. Default True.
    -> Action probabilities for each state or action int for each state
        (depending on stochasticity).
    """

    if v is None:
        v = optimal_value(n_states, n_actions, transition_probabilities, reward,
                          discount, threshold)

    if stochastic:
        # Get Q using equation 9.2 from Ziebart's thesis.
        Q = np.zeros((n_states, n_actions))
        for i in range(n_states):
            for j in range(n_actions):
                p = transition_probabilities[i, j, :]
                Q[i, j] = p.dot(reward + discount * v)
        Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
        Q = np.exp(Q) / np.exp(Q).sum(axis=1).reshape((n_states, 1))
        return Q

    def _policy(s):
        return max(range(n_actions),
                   key=lambda a: sum(transition_probabilities[s, a, k] *
                                     (reward[k] + discount * v[k])
                                     for k in range(n_states)))

    policy = np.array([_policy(s) for s in range(n_states)])
    return policy


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
    # assert np.isclose(v,
    #                   [5.7194282, 6.46706692, 6.42589811,
    #                    6.46706692, 7.47058224, 7.96505174,
    #                    6.42589811, 7.96505174, 8.19268666], 1).all()
    opt_v = optimal_value(env.possibleStates,
                          env.action_space.n,
                          env.transition_probability,
                          rewards,
                          gamma)
    assert np.isclose(v, opt_v).all()
