class Option:
    """
    Option 

    """

    def __init__(self, init_set, policy, term_set, policy_selection, term_condition):
        self.initialisation_set = init_set
        self.policy = policy
        self.termination_set = term_set

        self.policy_selection = policy_selection
        self.termination_condition = term_condition

    def step(self, state):
        return self.policy_selection(self, state), self.termination_condition(self, state)
