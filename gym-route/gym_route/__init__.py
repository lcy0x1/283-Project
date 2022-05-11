from gym.envs.registration import register

register(
    id='route-v1',
    entry_point='gym_route.envs:VehicleEnv',
)