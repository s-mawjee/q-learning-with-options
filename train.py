import os
import shutil

import dill
import gym
import numpy as np
from matplotlib import pyplot as plt
from qbstyles import mpl_style

from agents.qlearning.qlearning import QLearningAgent, QLearningWithOptionsAgent
from utils.options import load_option

from environments.fourrooms.envs.fourrooms_env import FourRoomsEnv

mpl_style(dark=False, minor_ticks=True)
figure_size = (10, 6)
legend_alpha = 0.5


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
        plt.plot(x, np.cumsum(mean), line, color=colour)
    else:
        plt.plot(x, mean, line, color=colour)
        plt.fill_between(x, q1, q3, color=colour, alpha=0.2)


def plot_rewards(goal_index, stats, max_length, prefix, postfix, save_folder):
    fig1, ax = plt.subplots(figsize=figure_size)
    x = np.arange(1, max_length + 1)
    get_plot(plt, x, stats, goal_index, 'standard', 'lightcoral', 'reward')
    get_plot(plt, x, stats, goal_index, 'options', 'mediumseagreen', 'reward')

    plt.legend(['No options', 'With options'],
               loc='lower right', framealpha=legend_alpha)

    ax.set_xlim(xmin=0)
    ax.set_xlim(xmax=max_length)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title(prefix + "Episode Reward over Time" + postfix)
    file_name = str(goal_index + 1) + '_' + (prefix + "Episode Reward over Time" + postfix).replace(' ', '_').replace(
        ':',
        '_')
    fig1.savefig(save_folder + file_name)
    plt.show()


def plot_episode_length(goal_index, stats, max_length, prefix, postfix, save_folder):
    fig1, ax = plt.subplots(figsize=figure_size)
    x = np.arange(1, max_length + 1)
    get_plot(plt, x, stats, goal_index, 'standard', 'lightcoral', '')
    get_plot(plt, x, stats, goal_index, 'options', 'mediumseagreen', '')

    plt.legend(['No options', 'With options'],
               loc='upper right', framealpha=legend_alpha)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    ax.set_xlim(xmax=max_length)
    plt.title(prefix + "Episode Length over Time" + postfix)
    file_name = str(goal_index + 1) + '_' + (prefix + "Episode Length over Time" + postfix).replace(' ', '_').replace(
        ':',
        '_')
    fig1.savefig(save_folder + file_name)
    plt.show()


def plot_episode_time_step(goal_index, stats, max_length, prefix, postfix, save_folder):
    fig1, ax = plt.subplots(figsize=figure_size)
    x = np.arange(1, max_length + 1)
    get_plot(plt, x, stats, goal_index, 'standard', 'lightcoral', 'cumsum')
    get_plot(plt, x, stats, goal_index, 'options', 'mediumseagreen', 'cumsum')
    # get_plot(plt, x, stats, goal_index, 'optimal', 'black', 'cumsum', line='--')

    plt.legend(['No options', 'With options'],
               loc='upper left', framealpha=legend_alpha)

    plt.ylabel("Time Steps")
    plt.xlabel("Number of tasks")
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    ax.set_xlim(xmax=max_length)
    plt.title(prefix + "Number of time-steps to solve n-episodes" + postfix)
    file_name = str(goal_index + 1) + '_' + (prefix + "Number of time-steps to solve n-episodes").replace(' ',
                                                                                                          '_').replace(
        ':',
        '_')

    fig1.savefig(save_folder + file_name)
    plt.show()


def plot_all(params, stats, output_dir, max_length=None):
    goals = params['goals']

    if max_length is None:
        max_length = len(stats[0]['standard'][0]['episode_lengths'])

    number_of_runs = len(stats[0]['standard'])
    for i, goal in enumerate(goals):
        prefix = "FourRooms " + goal + ": "
        postfix = " (averaged over " + str(number_of_runs) + " runs)"
        plot_rewards(i, stats, max_length, prefix, postfix, output_dir)
        plot_episode_length(i, stats, max_length, prefix, postfix, output_dir)
        plot_episode_time_step(i, stats, max_length,
                               prefix, postfix, output_dir)


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
        print('\nFor Goal:', goal)
        run_stats_standard = []
        run_stats_options = []

        for i in range(num_runs):
            print('\nRun:', i)
            env = gym.make(goal + "-v0")
            # with out options
            standard_agent = QLearningAgent(
                env, gamma=gamma, alpha=alpha, epsilon=epsilon)
            standard_stats = standard_agent.train(num_episodes)
            run_stats_standard.append(standard_stats)

            # with options
            agent = QLearningWithOptionsAgent(env, options, gamma=gamma, alpha=alpha, epsilon=epsilon,
                                              intra_options=False)
            options_stats = agent.train(num_episodes)
            run_stats_options.append(options_stats)

            env.close()

        all_stats.append({"standard": run_stats_standard,
                          'options': run_stats_options})

    return all_stats


def plot_env(output_dir):
    for goal in ['G1', 'G2']:
        env = gym.make(goal + "-v0")
        env.reset(0)
        fig = env.render(title='Four Rooms: Goal ' + goal)
        fig.savefig(output_dir + 'fourrooms_' + goal)


def plot_options(output_dir, goal='G1'):
    env = gym.make(goal + "-v0")
    options = [load_option('FourRoomsO1'), load_option('FourRoomsO2')]
    for i, option in enumerate(options):
        title_post = ''
        if i == 0:
            title_post = 'Option 1 (clockwise)'
        elif i == 1:
            title_post = 'Option 2 (anti-clockwise)'

        fig = env.render(title='Four Rooms: ' + title_post,
                         draw_arrows=True,
                         policy=option.policy,
                         init_states=option.initialisation_set,
                         plot_option=True,
                         termination_states=option.termination_set)

        fig.savefig(output_dir + 'option_' + str(i + 1) + '_fourrooms')


def main():
    output_folder = "./output/"
    parameters = {'episodes': 1000,
                  'gamma': 0.9,
                  'alpha': 0.125,
                  'epsilon': 0.1,
                  'goals': ['G1', 'G2'],
                  'number_of_runs': 25}

    if (os.path.isdir(output_folder)):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    plot_env(output_folder)
    plot_options(output_folder)

    all_stats = run(parameters)

    # save stats
    output = open(output_folder + 'all_stats' + ".pkl", 'wb')
    dill.dump(all_stats, output)
    output.close()

    # plot
    plot_all(parameters, all_stats, output_folder)

    return all_stats


if __name__ == '__main__':
    main()

    # output_folder = "./output/"
    # parameters = {'episodes': 1000,
    #               'gamma': 0.9,
    #               'alpha': 0.125,
    #               'epsilon': 0.1,
    #               'goals': ['G1', 'G2'],
    #               'number_of_runs': 25}
    #
    # input_file = open(output_folder + 'all_stats' + ".pkl", 'rb')
    # all_stats = dill.load(input_file)
    # # plot
    # plot_all(parameters, all_stats, output_folder, max_length=500)
