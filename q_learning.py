from collections import defaultdict
import numpy as np
import gym
import itertools
import sys
from utils import plotting
import dill
from matplotlib import pyplot as plt
import pandas as pd
import envs.gridworld

# Defining actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

options_policy = ["FourRoomsO1", "FourRoomsO2"]


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.1, epsilon=0.1):
    number_of_actions = 4

    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(number_of_actions))

    actions = np.arange(number_of_actions)
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, number_of_actions)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 500 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()
        state = str(state)

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():

            # Take a step
            action_probs = policy(state)
            action = np.random.choice(actions, p=action_probs)

            next_state, reward, done, _ = env.step(action)
            next_state = str(next_state)
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            state = next_state

    return Q, stats


def q_learning_with_option(env, num_episodes, discount_factor=1.0, alpha=0.1, epsilon=0.1):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    loadedPolicy = []
    for p in options_policy:
        loadedPolicy.append(dill.load(open(p, 'rb')))

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    actions = np.arange(env.action_space.n)
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 500 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()
        state = str(state)

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():

            # Take a step
            action_probs = policy(state)
            action = np.random.choice(actions, p=action_probs)

            if action >= 4:
                # Passing option policy to environment
                env.options_policy = loadedPolicy[action - 4]

            next_state, reward, done, _ = env.step(action)
            next_state = str(next_state)
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            state = next_state

    return Q, stats


def q_learning_with_options_only(env, num_episodes, discount_factor=1.0, alpha=0.1, epsilon=0.1):
    number_of_options = 2

    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    loadedPolicy = []
    for p in options_policy:
        loadedPolicy.append(dill.load(open(p, 'rb')))

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(number_of_options))

    actions = np.arange(number_of_options)
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, number_of_options)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 500 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()
        state = str(state)
        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():

            # Take a step
            action_probs = policy(state)
            action = np.random.choice(actions, p=action_probs)

            env.options_policy = loadedPolicy[action]
            next_state, reward, done, _ = env.step(action + 4)
            next_state = str(next_state)
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break

            state = next_state

    return Q, stats


def q_to_policy(q, offset=0):
    optimalPolicy = {}
    for state in q:
        optimalPolicy[state] = np.argmax(q[state]) + offset
    return optimalPolicy


def main():
    goal = "G2"
    if goal == "G2":
        env_name = "FourRooms-v2"
    else:
        env_name = "FourRooms-v1"
        goal = "G1"

    env = gym.make(env_name)
    runs = 30
    number_of_eps = 1000
    discount_factor = 1.0
    alpha = 0.1
    epsilon = 0.1

    # env.render()
    # actions = [5, 5]
    # for a in actions:
    #     if a > 3:
    #         env.options_policy = dill.load(open(options_policy[a - 4], 'rb'))
    #     next_observation, reward, done, _ = env.step(a)  # Taking option
    #     print(next_observation)
    #     env.render()

    stats_actions = {}
    stats_options = {}
    stats_options_action = {}
    for run in range(runs):
        print('\n---Actions ' + str(run) + ' ---')
        Q_actions, stats_actions[run] = q_learning(env, number_of_eps, discount_factor, alpha, epsilon)
        # print('\n')
        # env.render(drawArrows=True, policy=q_to_policy(Q_actions))
        # plotting.plot_episode_stats(stats_actions, 'Four Rooms - only A')

        print('\n---Options ' + str(run) + ' ---')
        Q_options, stats_options[run] = q_learning_with_options_only(env, number_of_eps, discount_factor, alpha,
                                                                     epsilon)
        # print('\n')
        # env.render(drawArrows=True, policy=q_to_policy(Q_options, 4))
        # plotting.plot_episode_stats(stats_options, 'Four Rooms - only O')

        print('\n---Options And Actions ' + str(run) + ' ---')
        Q_options_action, stats_options_action[run] = q_learning_with_option(env, number_of_eps, discount_factor, alpha,
                                                                             epsilon)
        # print('\n')
        # env.render(drawArrows=True, policy=q_to_policy(Q_options_action))
        # plotting.plot_episode_stats(stats_options_action, 'Four Rooms - with O and A')

    stats = {}
    stats_actions_mean = {'episode_lengths': np.zeros(number_of_eps),
                          'episode_rewards': np.zeros(number_of_eps)}
    stats_options_mean = {'episode_lengths': np.zeros(number_of_eps),
                          'episode_rewards': np.zeros(number_of_eps)}
    stats_options_action_mean = {'episode_lengths': np.zeros(number_of_eps),
                                 'episode_rewards': np.zeros(number_of_eps)}

    for eps in range(number_of_eps):
        a_episode_lengths = 0
        o_episode_lengths = 0
        a_o_episode_lengths = 0

        a_episode_rewards = 0
        o_episode_rewards = 0
        a_o_episode_rewards = 0

        for r in range(runs):
            a_episode_lengths = a_episode_lengths + stats_actions[r].episode_lengths[eps]
            o_episode_lengths = o_episode_lengths + stats_options[r].episode_lengths[eps]
            a_o_episode_lengths = a_o_episode_lengths + stats_options_action[r].episode_lengths[eps]

            a_episode_rewards = a_episode_rewards + stats_actions[r].episode_rewards[eps]
            o_episode_rewards = o_episode_rewards + stats_options[r].episode_rewards[eps]
            a_o_episode_rewards = a_o_episode_rewards + stats_options_action[r].episode_rewards[eps]

        stats_actions_mean['episode_lengths'][eps] = a_episode_lengths / runs
        stats_actions_mean['episode_rewards'][eps] = a_episode_rewards / runs

        stats_options_mean['episode_lengths'][eps] = o_episode_lengths / runs
        stats_options_mean['episode_rewards'][eps] = o_episode_rewards / runs

        stats_options_action_mean['episode_lengths'][eps] = a_o_episode_lengths / runs
        stats_options_action_mean['episode_rewards'][eps] = a_o_episode_rewards / runs

    stats[0] = stats_actions_mean
    stats[1] = stats_options_mean
    stats[2] = stats_options_action_mean

    fig1 = plt.figure(figsize=(10, 5))
    for i in range(3):
        plt.plot(stats[i]['episode_lengths'])
    plt.legend(['Actions only', 'Options only', 'Actions and Options'])
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title(env_name + " (" + goal + ") " +
              "Episode Length over Time (Averaged over " + str(runs) + ' runs)')
    plt.show()

    fig2 = plt.figure(figsize=(10, 5))
    for i in range(3):
        rewards_smoothed = pd.Series(stats[i]['episode_rewards']).rolling(10, min_periods=10).mean()
        plt.plot(rewards_smoothed)
    plt.legend(['Actions only', 'Options only', 'Actions and Options'])
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed with window={})".format(10))
    plt.title(env_name + " (" + goal + ") " + "Episode Reward over Time (Averaged over " + str(runs) + ' runs)')
    plt.show()


def view_policies():
    goal = "G1"
    if goal == "G2":
        env_name = "FourRooms-v2"
    else:
        env_name = "FourRooms-v1"
        goal = "G1"

    env = gym.make(env_name)

    number_of_eps = 1000
    discount_factor = 1.0
    alpha = 0.1
    epsilon = 0.1
    print('\n---Actions---')
    Q_actions, _ = q_learning(env, number_of_eps, discount_factor, alpha, epsilon)
    env.render(drawArrows=True, policy=q_to_policy(Q_actions), name_prefix=env_name + " (" + goal + ")")

    # print('\n---Options---')
    # Q_options, _ = q_learning_with_options_only(env, number_of_eps, discount_factor, alpha,
    #                                             epsilon)
    # env.render(drawArrows=True, policy=q_to_policy(Q_options, 4), name_prefix=env_name + " (" + goal + ")")

    print('\n---Options And Actions---')
    Q_options_action, _ = q_learning_with_option(env, 1000, discount_factor, alpha,
                                                 epsilon)
    env.render(drawArrows=True, policy=q_to_policy(Q_options_action), name_prefix=env_name + " (" + goal + ")")


if __name__ == '__main__':
    view_policies()
