import shutil
import os

import gym
import numpy as np
from matplotlib import pyplot as plt

from utils.options import load_option
from agents.qlearning.qlearning import QLearningAgent, QLearningWithOptionsAgent
from environments.fourrooms.envs.fourrooms_env import FourRoomsEnv


def process_stats(stats_list, reward):
    '''
    returns: episode_lengths, episode_rewards, None, None or
             mean_episode_lengths, mean_episode_rewards, std_episode_lengths, std_episode_rewards
    '''

    if len(stats_list) == 1:
        return stats_list[0].episode_lengths, stats_list[0].episode_rewards, 0, 0
    elif len(stats_list) > 1:
        all_episode_lengths = []
        all_episode_rewards = []

        for s in stats_list:
            all_episode_lengths.append(s['episode_lengths'])
            all_episode_rewards.append(s['episode_rewards'])

        all_episode_lengths = np.array(all_episode_lengths)
        all_episode_rewards = np.array(all_episode_rewards)

        if reward:
            q1 = np.percentile(all_episode_rewards, 25,
                               axis=0, interpolation='midpoint')
            q3 = np.percentile(all_episode_rewards, 75,
                               axis=0, interpolation='midpoint')
            return all_episode_rewards.mean(axis=0), all_episode_rewards.std(axis=0), q1, q3

        else:
            q1 = np.percentile(all_episode_lengths, 25,
                               axis=0, interpolation='midpoint')
            q3 = np.percentile(all_episode_lengths, 75,
                               axis=0, interpolation='midpoint')
            return all_episode_lengths.mean(axis=0), all_episode_lengths.std(axis=0), q1, q3


def get_plot(plt, x, stats, env_index, name, colour, type, line="-"):
    stats_ = stats[env_index][name]
    reward = type == 'reward'
    mean, std, q1, q3 = process_stats(stats_, reward)
    mean, std, q1, q3 = mean[:len(x)], std[:len(x)], q1[:len(x)], q3[:len(x)]
    # mean_smoothed = pd.Series(mean).rolling(5, min_periods=5).mean()
    cumsum = type == 'cumsum'
    if cumsum:
        plt.plot(np.cumsum(mean), x, line, color=colour)
    else:
        plt.plot(x, mean, line, color=colour)
        plt.fill_between(x, q1, q3, color=colour, alpha=0.2)


def plot_rewards(goal_index, stats, max_length, prefix, postfix, save_folder):
    fig1 = plt.figure(figsize=(10, 5))
    x = np.arange(1, max_length + 1)
    get_plot(plt, x, stats, goal_index, 'standard', 'red', 'reward')
    get_plot(plt, x, stats, goal_index, 'options', 'green', 'reward')
    # get_plot(plt, x, stats, goal_index, 'optimal', 'black', 'reward', line='--')

    plt.legend(['No options', 'With options'], loc='lower right')
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title(prefix + "Episode Reward over Time" + postfix)
    file_name = str(goal_index) + '_' + (prefix + "Episode Reward over Time" + postfix).replace(' ', '_').replace(':',
                                                                                                                  '_')
    fig1.savefig(save_folder + file_name)


def plot_episode_length(goal_index, stats, max_length, prefix, postfix, save_folder):
    fig1 = plt.figure(figsize=(10, 5))
    x = np.arange(1, max_length + 1)
    get_plot(plt, x, stats, goal_index, 'standard', 'red', '')
    get_plot(plt, x, stats, goal_index, 'options', 'green', '')
    # get_plot(plt, x, stats, goal_index, 'optimal', 'black', '', line='--')

    plt.legend(['No options', 'With options'], loc='upper right')
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title(prefix + "Episode Length over Time" + postfix)
    file_name = str(goal_index) + '_' + (prefix + "Episode Length over Time" + postfix).replace(' ', '_').replace(':',
                                                                                                                  '_')
    fig1.savefig(save_folder + file_name)


def plot_episode_time_step(goal_index, stats, max_length, prefix, postfix, save_folder):
    fig1 = plt.figure(figsize=(10, 5))
    x = np.arange(1, max_length + 1)
    get_plot(plt, x, stats, goal_index, 'standard', 'red', 'cumsum')
    get_plot(plt, x, stats, goal_index, 'options', 'green', 'cumsum')
    # get_plot(plt, x, stats, goal_index, 'optimal', 'black', 'cumsum', line='--')

    plt.legend(['No options', 'With options'], loc='lower right')
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title(prefix + "Episode per time step" + postfix)
    file_name = str(goal_index) + '_' + (prefix + "Episode per time step" + postfix).replace(' ', '_').replace(':',
                                                                                                               '_')
    fig1.savefig(save_folder + file_name)


def plot_all(params, stats, max_length=None):
    goals = params['goals']
    save_plots_folder = "plots/"

    if (os.path.isdir(save_plots_folder)):
        shutil.rmtree(save_plots_folder)
    os.makedirs(save_plots_folder)

    if max_length is None:
        max_length = len(stats[0]['standard'][0]['episode_lengths'])

    number_of_runs = len(stats[0])
    for i, goal in enumerate(goals):
        prefix = "FourRooms: " + goal
        postfix = " (averaged over " + str(number_of_runs) + " runs)"
        plot_rewards(i, stats, max_length, prefix, postfix, save_plots_folder)
        plot_episode_length(i, stats, max_length, prefix, postfix, save_plots_folder)
        plot_episode_time_step(i, stats, max_length, prefix, postfix, save_plots_folder)


def run(parameters):
    num_runs = int(parameters['number_of_runs'])
    num_episodes = int(parameters['episodes'])
    gamma = parameters['gamma']
    alpha = parameters['alpha']
    epsilon = parameters['epsilon']
    goals = parameters['goals']

    # load hand-crafted options
    options = [load_option('FourRoomsO1'), load_option('FourRoomsO2')]

    all_stats = []
    for goal in goals:
        print('For Goal:', goal)
        run_stats_standard = []
        run_stats_options = []

        for i in range(num_runs):
            print('\nRun:', i)
            env = gym.make(goal + "-v0")
            # with out options
            _ = env.reset()
            standard_agent = QLearningAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon)
            standard_stats = standard_agent.train(num_episodes)
            run_stats_standard.append(standard_stats)

            # with options
            _ = env.reset()
            agent = QLearningWithOptionsAgent(env, options, gamma=gamma, alpha=alpha, epsilon=epsilon,
                                              intra_options=False)
            options_stats = agent.train(num_episodes)
            run_stats_options.append(options_stats)

            env.close()

        all_stats.append({"standard": run_stats_standard, 'options': run_stats_options})

    return all_stats


def main():
    parameters = {'episodes': 200,
                  'gamma': 0.9,
                  'alpha': 0.125,
                  'epsilon': 0.1,
                  'goals': ['G1', 'G2'],
                  'number_of_runs': 10}

    all_stats = run(parameters)
    plot_all(parameters, all_stats)


if __name__ == '__main__':
    main()
