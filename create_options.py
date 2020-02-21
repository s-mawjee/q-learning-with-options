from utils.options import load_option, save_option
import gym
from option import QOption
import environments.fourrooms.envs.fourrooms_env

# Defining actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


def create_options(env):
    policy_option_1 = {

        # ROOM 0
        0: RIGHT, 1: RIGHT, 2: RIGHT, 3: RIGHT, 4: DOWN,
        10: RIGHT, 11: RIGHT, 12: RIGHT, 13: RIGHT, 14: DOWN,
        20: RIGHT, 21: RIGHT, 22: RIGHT, 23: RIGHT, 24: RIGHT,
        31: RIGHT, 32: RIGHT, 33: RIGHT, 34: RIGHT, 35: UP,
        41: RIGHT, 42: RIGHT, 43: RIGHT, 44: RIGHT, 45: UP,

        # ROOM 1
        5: DOWN, 6: DOWN, 7: DOWN, 8: DOWN, 9: DOWN,
        15: DOWN, 16: DOWN, 17: DOWN, 18: DOWN, 19: DOWN,
        26: DOWN, 27: DOWN, 28: DOWN, 29: DOWN, 30: DOWN,
        36: DOWN, 37: DOWN, 38: DOWN, 39: DOWN, 40: DOWN,
        46: DOWN, 47: DOWN, 48: DOWN, 49: DOWN, 50: DOWN,
        52: RIGHT, 53: RIGHT, 54: DOWN, 55: LEFT, 56: LEFT,

        # ROOM 2
        68: DOWN, 69: DOWN, 70: DOWN, 71: DOWN, 72: DOWN,
        78: DOWN, 79: DOWN, 80: DOWN, 81: DOWN, 82: DOWN,
        89: LEFT, 90: LEFT, 91: LEFT, 92: LEFT, 93: LEFT,
        99: UP, 100: UP, 101: UP, 102: UP, 103: UP,

        # ROOM 3
        57: RIGHT, 58: UP, 59: LEFT, 60: LEFT, 61: LEFT,
        63: UP, 64: UP, 65: UP, 66: UP, 67: UP,
        73: UP, 74: UP, 75: UP, 76: UP, 77: UP,
        83: UP, 84: UP, 85: UP, 86: UP, 87: UP,
        94: UP, 95: UP, 96: UP, 97: UP, 98: UP,

        25: RIGHT,
        51: UP,
        62: DOWN,
        88: LEFT

    }

    term_set = [25, 62, 88, 51]

    def policy_selection(policy, state):
        return policy[state]

    def termination_condition(termination_set, state):

        if state in termination_set:
            return True
        return False

    # termination_condition = lambda self, state: state == self.termination_set[int(state[1])]

    option1 = QOption(env.observation_space.n, policy_option_1,
                      term_set, policy_selection,
                      termination_condition, env.action_space.n)

    save_option('FourRoomsO1', option1)

    # Option 2 - anti clockwise hall way points

    policy_option_2 = {
        # ROOM 0
        0: DOWN, 1: DOWN, 2: DOWN, 3: DOWN, 4: DOWN,
        10: DOWN, 11: DOWN, 12: DOWN, 13: DOWN, 14: DOWN,
        20: DOWN, 21: DOWN, 22: DOWN, 23: DOWN, 24: DOWN,
        31: DOWN, 32: DOWN, 33: DOWN, 34: DOWN, 35: DOWN,
        41: RIGHT, 42: DOWN, 43: LEFT, 44: LEFT, 45: LEFT,

        # ROOM 1
        5: DOWN, 6: DOWN, 7: DOWN, 8: DOWN, 9: DOWN,
        15: DOWN, 16: DOWN, 17: DOWN, 18: DOWN, 19: DOWN,
        26: LEFT, 27: LEFT, 28: LEFT, 29: LEFT, 30: LEFT,
        36: UP, 37: UP, 38: UP, 39: UP, 40: UP,
        46: UP, 47: UP, 48: UP, 49: UP, 50: UP,
        52: UP, 53: UP, 54: UP, 55: UP, 56: UP,

        # ROOM 2
        68: RIGHT, 69: RIGHT, 70: UP, 71: LEFT, 72: LEFT,
        78: UP, 79: UP, 80: UP, 81: UP, 82: UP,
        89: UP, 90: UP, 91: UP, 92: UP, 93: UP,
        99: UP, 100: UP, 101: UP, 102: UP, 103: UP,

        # ROOM 3
        57: DOWN, 58: DOWN, 59: DOWN, 60: DOWN, 61: DOWN,
        63: DOWN, 64: DOWN, 65: DOWN, 66: DOWN, 67: DOWN,
        73: DOWN, 74: DOWN, 75: DOWN, 76: DOWN, 77: DOWN,
        83: RIGHT, 84: RIGHT, 85: RIGHT, 86: RIGHT, 87: RIGHT,
        94: UP, 95: UP, 96: UP, 97: UP, 98: UP,

        25: LEFT,
        51: DOWN,
        62: UP,
        88: RIGHT

    }

    term_set = [51, 88, 62, 25]

    def termination_condition_2(termination_set, state):
        if state in termination_set:
            return True
        return False

    option2 = QOption(env.observation_space.n, policy_option_2,
                      term_set, policy_selection,
                      termination_condition_2, env.action_space.n)

    save_option('FourRoomsO2', option2)

    # return option1, option2


def view_option_policy(env, option):
    policy = option.policy
    env.render(draw_arrows=True, policy=policy)


if __name__ == '__main__':
    env = gym.make("G1-v0")
    create_options(env)

    # test load
    o1 = load_option('FourRoomsO1')
    view_option_policy(env, o1)
    o2 = load_option('FourRoomsO2')
    view_option_policy(env, o2)
