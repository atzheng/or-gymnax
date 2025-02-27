#!/usr/bin/env python3
from picard.environments.rideshare_dispatch import RideshareDispatch, ManhattanRideshareDispatch


def make(env_id: str, **env_kwargs):
    if env_id == "RideshareDispatch-v0":
        env = RideshareDispatch(**env_kwargs)
    elif env_id == "ManhattanRideshareDispatch-v0":
        env = ManhattanRideshareDispatch(**env_kwargs)
    else:
        raise ValueError("Environment ID is not registered.")

    return env, env.default_params
