import itertools
import sys

import gym
import envs.gridworld

from agents.qlearning.qlearning_agent import QLearningAgent


def train(parameters, withOptions=False):
    num_episodes = int(parameters['episodes'])
    gamma = parameters['gamma']
    alpha = parameters['alpha']
    epsilon = parameters['epsilon']

    env = gym.make('FourRooms-v1')
    if withOptions:
        options = []
        agent = QLearningAgent(env.action_space.n + len(options), discount_factor=gamma, alpha=alpha, epsilon=epsilon)
    else:
        agent = QLearningAgent(env.action_space.n, discount_factor=gamma, alpha=alpha, epsilon=epsilon)

    total_total_reward = 0.0
    for i_episode in range(num_episodes):
        # global_step.assign_add(1)

        # Print out which episode we're on.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()

        # One step in the environment
        total_reward = 0.0
        for t in itertools.count():
            # Take a step
            action = agent.step(state)
            next_state, reward, done, _ = env.step(action)

            total_reward += reward

            # Update
            agent.update(state, action, reward, next_state)

            if done:
                total_total_reward += reward
                break

            state = next_state

    env.close()

    return total_total_reward / num_episodes


def main():
    parameters = {'episodes': 500, 'gamma': 0.9, 'alpha': 0.12, 'epsilon': 0.1}
    print('---Start---')
    train(parameters)
    print('---End---')


if __name__ == '__main__':
    main()
