from collections import defaultdict
import numpy as np
import gym
import itertools
import sys
from utils import plotting
import dill
import envs.gridworld

# Defining actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

options_policy = ["FourRoomsO1.pkl", "FourRoomsO2.pkl"]


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


def q_learning(env, num_episodes, discount_factor=0.9, alpha=0.5, epsilon=0.5):
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
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset()

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():

            # Take a step
            action_probs = policy(state)
            action = np.random.choice(actions, p=action_probs)

            if action >= 4:
                # Passing option policy to enviroment
                env.options_policy = loadedPolicy[action - 4]

            next_state, reward, done, _ = env.step(action)
            next_state = next_state
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


def main():
    env = gym.make("FourRooms-v1")

    # env.render()
    # actions = [5, 5]
    # for a in actions:
    #     if a > 3:
    #         env.options_policy = dill.load(open(options_policy[a - 4], 'rb'))
    #     next_observation, reward, done, _ = env.step(a)  # Taking option
    #     print(next_observation)
    #     env.render()

    print('---start---')
    Q, stats = q_learning(env, 500)
    print('\n---end---')
    env.render()

    optimalPolicy = {}
    for state in Q:
        optimalPolicy[state] = np.argmax(Q[state])

    env.render(drawArrows=True, policy=optimalPolicy)
    # plotting.plot_episode_stats(stats)


if __name__ == '__main__':
    main()
