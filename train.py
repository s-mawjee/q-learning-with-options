import time
import gym

from utils.options import load_option

from agents.qlearning.qlearning import QLearningAgent, QLearningWithOptionsAgent
from environments.fourrooms.envs.fourrooms_env import FourRoomsEnv


def run(parameters, withOptions=False, intra_options=False, ):
    num_episodes = int(parameters['episodes'])
    gamma = parameters['gamma']
    alpha = parameters['alpha']
    epsilon = parameters['epsilon']
    goal = parameters['goal']

    env = gym.make(goal + "-v0")

    if withOptions:
        options = [load_option('FourRoomsO1'), load_option('FourRoomsO2')]
        agent = QLearningWithOptionsAgent(env, options, gamma=gamma, alpha=alpha, epsilon=epsilon,
                                          intra_options=intra_options)
    else:
        agent = QLearningAgent(env, gamma=gamma, alpha=alpha, epsilon=epsilon)

    stats = agent.train(num_episodes)

    env.close()

    return stats


def main():
    parameters = {'episodes': 10000,
                  'gamma': 0.9,
                  'alpha': 0.12,
                  'epsilon': 0.1,
                  'goal': 'G1'}

    all_stats = []
    for i in range(1):
        start = time.time()
        stats = run(parameters, withOptions=True, intra_options=False)
        end = time.time()
        print()
        print('Time taken (', parameters['episodes'], 'episodes ):', end - start)
        all_stats.append(stats)


if __name__ == '__main__':
    main()
