from gym.envs.registration import register

register(
    id='Rx_Env/RxEnv-v0',
    entry_point='Rx_Env.envs:RxEnv',
    max_episode_steps=3000,
)