class Agent:
    """
    Agent 

    """

    def act(self, observation, reward):
        raise NotImplementedError()

    def update(self, state, action, reward, next_state):
        raise NotImplementedError()
