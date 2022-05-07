# register a new deterministic environment
from gym.envs.registration import register
register(
    id='FrozenLake-Deterministic-v1',
    # entry_point='gym.envs.toy_text:FrozenLakeEnv',
    entry_point='lib.envs.myFrozenLake:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
)