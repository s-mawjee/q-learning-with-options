class Agent:
    """
    Agent 

    """

    def step(self, observation, reward):
        raise NotImplementedError()

    def update(self, state, action, reward, next_state):
        raise NotImplementedError()
