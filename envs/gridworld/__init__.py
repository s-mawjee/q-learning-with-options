from gym.envs.registration import register

# Four rooms with G1
register(
    id='FourRooms-v1',
    entry_point='envs.gridworld.envs.FourRooms:FourRooms',
    kwargs={'goal': 'G1'}

)
# Four rooms with G2
register(
    id='FourRooms-v2',
    entry_point='envs.gridworld.envs.FourRooms:FourRooms',
    kwargs={'goal': 'G2'}
)
