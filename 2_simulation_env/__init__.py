from gymnasium.envs.registration import register

register(
    id="OmniCellBioreactor-v0",
    entry_point="bioreactor_gym_env:BioreactorTwinEnv",
    max_episode_steps=500,
)
