import time
import gym
import numpy as np
import envs.gridworld

from utils.options import load_option

from agents.qlearning.qlearning_agent import QLearningAgent, QLearningWithOptionsAgent


def train(parameters, withOptions=False):
    num_episodes = int(parameters['episodes'])
    gamma = parameters['gamma']
    alpha = parameters['alpha']
    epsilon = parameters['epsilon']

    goal = "G1"
    if goal == "G2":
        env_name = "FourRooms-v2"
    else:
        env_name = "FourRooms-v1"
        goal = "G1"
    env = gym.make(env_name)

    if withOptions:
        options = [load_option('FourRoomsO1'), load_option('FourRoomsO2')]
        agent = QLearningWithOptionsAgent(env, options, discount_factor=gamma, alpha=alpha, epsilon=epsilon)
    else:
        agent = QLearningAgent(env, discount_factor=gamma, alpha=alpha, epsilon=epsilon)

    average_eps_reward = agent.train(num_episodes)

    env.render(drawArrows=True, policy=q_to_policy(agent.q), name_prefix=env_name + " (" + goal + ")")
    env.close()
    return average_eps_reward


def q_to_policy(q, offset=0):
    optimalPolicy = {}
    for state in q:
        optimalPolicy[state] = np.argmax(q[state]) + offset
    return optimalPolicy


def main():
    parameters = {'episodes': 1, 'gamma': 0.9, 'alpha': 0.12, 'epsilon': 0.1}
    print('---Start---')
    start = time.time()
    average_reward = train(parameters, withOptions=True)
    end = time.time()
    print('\nAverage reward:', average_reward)
    print('Time (', parameters['episodes'], 'episodes ):', end - start)
    print('---End---')


if __name__ == '__main__':
    main()
# env = gym.make('FourRooms-v1')
# actions = [0, 3, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 0, 0, 0]
# for a in actions:
#     next_observation, reward, done, _ = env.step(a)  # Taking option
#     # print(next_observation)
# env.render()
