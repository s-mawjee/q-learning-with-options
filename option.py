from collections import defaultdict
import numpy as np


class Option:
    """
    Option 

    """

    def __init__(self, init_set, policy, term_set, policy_selection, term_condition):
        self.initialisation_set = init_set
        self.policy = policy
        self.termination_set = term_set
        self._policy_selection = policy_selection
        self._termination_condition = term_condition

    def policy_selection(self, state):
        return self._policy_selection(self.policy, state)

    def termination_condition(self, state):
        return self._termination_condition(self.termination_set, state)

    def step(self, state):
        return self.policy_selection(state), self.termination_condition(state)


class QOption(Option):
    def __init__(self, init_set, policy, term_set, policy_selection, term_condition, number_of_action):
        super().__init__(init_set, policy, term_set, policy_selection, term_condition)
        self.number_of_action = number_of_action
        self.q = defaultdict(lambda: np.zeros(self.number_of_action))
        self.policy_to_q()

    def policy_to_q(self):
        for state, action in self.policy.items():
            self.q[state][action] = 1

    def update_policy(self):
        for state in self.q:
            self.policy[state] = np.argmax(self.q[state])
