from gym.envs.registration import register

register(
    id='G1-v0',
    entry_point='environments.fourrooms.envs:FourRoomsEnv',
    kwargs={'terminal_states': [62], 'terminal_reward': 1.0, 'step_reward': -1.0}
)

register(
    id='G2-v0',
    entry_point='environments.fourrooms.envs:FourRoomsEnv',
    kwargs={'terminal_states': [80], 'terminal_reward': 1.0, 'step_reward': -1.0}
)
