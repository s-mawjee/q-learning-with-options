import pickle
import gym
import envs.gridworld

# Defining actions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


def create_options():
    policy_option_1 = {

        # ROOM 0
        (1, 1): DOWN, (1, 2): DOWN, (1, 3): DOWN, (1, 4): DOWN, (1, 5): DOWN,
        (2, 1): DOWN, (2, 2): DOWN, (2, 3): DOWN, (2, 4): DOWN, (2, 5): DOWN,
        (3, 1): RIGHT, (3, 2): RIGHT, (3, 3): RIGHT, (3, 4): RIGHT, (3, 5): RIGHT,
        (4, 1): UP, (4, 2): UP, (4, 3): UP, (4, 4): UP, (4, 5): UP,
        (5, 1): UP, (5, 2): UP, (5, 3): UP, (5, 4): UP, (5, 5): UP,

        # ROOM 1
        (1, 7): DOWN, (1, 8): DOWN, (1, 9): DOWN, (1, 10): DOWN, (1, 11): DOWN,
        (2, 7): DOWN, (2, 8): DOWN, (2, 9): DOWN, (2, 10): DOWN, (2, 11): DOWN,
        (3, 7): DOWN, (3, 8): DOWN, (3, 9): DOWN, (3, 10): DOWN, (3, 11): DOWN,
        (4, 7): DOWN, (4, 8): DOWN, (4, 9): DOWN, (4, 10): DOWN, (4, 11): DOWN,
        (5, 7): DOWN, (5, 8): DOWN, (5, 9): DOWN, (5, 10): DOWN, (5, 11): DOWN,
        (6, 7): RIGHT, (6, 8): RIGHT, (6, 9): DOWN, (6, 10): LEFT, (6, 11): LEFT,

        # ROOM 2
        (8, 7): DOWN, (8, 8): DOWN, (8, 9): DOWN, (8, 10): DOWN, (8, 11): DOWN,
        (9, 7): DOWN, (9, 8): DOWN, (9, 9): DOWN, (9, 10): DOWN, (9, 11): DOWN,
        (10, 7): LEFT, (10, 8): LEFT, (10, 9): LEFT, (10, 10): LEFT, (10, 11): LEFT,
        (11, 7): UP, (11, 8): UP, (11, 9): UP, (11, 10): UP, (11, 11): UP,

        # ROOM 3
        (7, 1): RIGHT, (7, 2): UP, (7, 3): LEFT, (7, 4): LEFT, (7, 5): LEFT,
        (8, 1): UP, (8, 2): UP, (8, 3): UP, (8, 4): UP, (8, 5): UP,
        (9, 1): UP, (9, 2): UP, (9, 3): UP, (9, 4): UP, (9, 5): UP,
        (10, 1): UP, (10, 2): UP, (10, 3): UP, (10, 4): UP, (10, 5): UP,
        (11, 1): UP, (11, 2): UP, (11, 3): UP, (11, 4): UP, (11, 5): UP,

        (3, 6): RIGHT,
        (6, 2): UP,
        (7, 9): DOWN,
        (10, 6): LEFT,

    }
    output = open('FourRoomsO1.pkl', 'wb')
    pickle.dump(policy_option_1, output)
    output.close()
    
    # Option 2 - anti clockwise hall way points
    policy_option_2 = {
        # ROOM 0
        (1, 1): DOWN, (1, 2): DOWN, (1, 3): DOWN, (1, 4): DOWN, (1, 5): DOWN,
        (2, 1): DOWN, (2, 2): DOWN, (2, 3): DOWN, (2, 4): DOWN, (2, 5): DOWN,
        (3, 1): DOWN, (3, 2): DOWN, (3, 3): DOWN, (3, 4): DOWN, (3, 5): DOWN,
        (4, 1): DOWN, (4, 2): DOWN, (4, 3): DOWN, (4, 4): DOWN, (4, 5): DOWN,
        (5, 1): RIGHT, (5, 2): DOWN, (5, 3): LEFT, (5, 4): LEFT, (5, 5): LEFT,

        # ROOM 1
        (1, 7): DOWN, (1, 8): DOWN, (1, 9): DOWN, (1, 10): DOWN, (1, 11): DOWN,
        (2, 7): DOWN, (2, 8): DOWN, (2, 9): DOWN, (2, 10): DOWN, (2, 11): DOWN,
        (3, 7): LEFT, (3, 8): LEFT, (3, 9): LEFT, (3, 10): LEFT, (3, 11): LEFT,
        (4, 7): UP, (4, 8): UP, (4, 9): UP, (4, 10): UP, (4, 11): UP,
        (5, 7): UP, (5, 8): UP, (5, 9): UP, (5, 10): UP, (5, 11): UP,
        (6, 7): UP, (6, 8): UP, (6, 9): UP, (6, 10): UP, (6, 11): UP,

        # ROOM 2
        (8, 7): RIGHT, (8, 8): RIGHT, (8, 9): UP, (8, 10): LEFT, (8, 11): LEFT,
        (9, 7): UP, (9, 8): UP, (9, 9): UP, (9, 10): UP, (9, 11): UP,
        (10, 7): UP, (10, 8): UP, (10, 9): UP, (10, 10): UP, (10, 11): UP,
        (11, 7): UP, (11, 8): UP, (11, 9): UP, (11, 10): UP, (11, 11): UP,
        # ROOM 3
        (7, 1): DOWN, (7, 2): DOWN, (7, 3): DOWN, (7, 4): DOWN, (7, 5): DOWN,
        (8, 1): DOWN, (8, 2): DOWN, (8, 3): DOWN, (8, 4): DOWN, (8, 5): DOWN,
        (9, 1): DOWN, (9, 2): DOWN, (9, 3): DOWN, (9, 4): DOWN, (9, 5): DOWN,
        (10, 1): RIGHT, (10, 2): RIGHT, (10, 3): RIGHT, (10, 4): RIGHT, (10, 5): RIGHT,
        (11, 1): UP, (11, 2): UP, (11, 3): UP, (11, 4): UP, (11, 5): UP,

        (3, 6): LEFT,
        (6, 2): DOWN,
        (7, 9): UP,
        (10, 6): RIGHT,

    }
    output = open('FourRoomsO2.pkl', 'wb')
    pickle.dump(policy_option_2, output)
    output.close()

    return policy_option_1, policy_option_2


def view_option_policy(policy):
    env = gym.make("FourRooms-v1")
    env.render(drawArrows=True, policy=policy)


if __name__ == '__main__':
    o1, o2 = create_options()
    view_option_policy(o1)
    view_option_policy(o2)
